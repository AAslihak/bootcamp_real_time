#!/usr/bin/env python3
"""WAV transmit / reference hydrophone monitor and recorder GUI (NI USB-6259).

AO : plays the wav at its native sample rate; every shot is followed by a
     guard interval (GI) of silence. The GI can never be shorter than the
     transmit duration (GI >= 50% of the period).
AI : continuously acquires the reference hydrophone, shows voltage vs time
     and spectrum (in volts, or in pressure using a hydrophone sensitivity
     curve from hydrophone_rvr.csv), and can record to txt or wav.

Without the NI-DAQmx driver or a device, the "Simulated" device lets you
try the interface (simulated AI = noise + attenuated AO signal).

Run:  python daq_gui.py
"""
import csv
import datetime as _dt
import math
import os
import queue
import struct
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.io import wavfile

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogSingleChannelReader
    from nidaqmx.stream_writers import AnalogSingleChannelWriter

    NIDAQMX_ERROR = None
except Exception as exc:  # package or driver not installed
    nidaqmx = None
    NIDAQMX_ERROR = exc

SIM_NAME = "Simulated (no hardware)"
AO_VMAX = 10.0  # USB-6259 AO range is +/-10 V
MAX_AO_SAMPLES = 100_000_000  # whole sequence is written at once (~800 MB cap)
TICK_MS = 50
RVR_FILE = "hydrophone_rvr.csv"  # hydrophone sensitivity curves, next to this file
SPECTRUM_FMAX = 20e3  # spectra are displayed from 0 to this frequency (Hz)
SPECTRUM_FMIN_LOG = 10.0  # lower edge of the log frequency axis (Hz)
PLOT_POINTS = 4000
PREVIEW_MAX_SAMPLES = 2_000_000  # transmit preview uses at most this many samples
TX_TITLE = "Transmit signal — one period (shaded: GI)"


# --------------------------------------------------------------------------
# Pure logic: wav and transmit sequence
# --------------------------------------------------------------------------
def load_wav(path):
    """Read a wav; return (fs, float64 array of shape (n, channels)), integers scaled to +/-1."""
    fs, data = wavfile.read(path)
    if data.ndim == 1:
        data = data[:, None]
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        x = data.astype(np.float64)
    return int(fs), x


def min_gi_samples(n_signal):
    """GI is at least as long as the signal (GI >= 50% of the period)."""
    return int(n_signal)


def build_sequence(wav, fs, peak_v, duration_s, gi_s, repeats):
    """Build the full AO sequence: [signal, GI (0 V)] x repeats.

    The wav peak is normalised to peak_v; a duration longer than the wav loops
    it, a shorter one truncates it. Returns (sequence, n_signal, n_gi).
    """
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 0:
        raise ValueError("The wav is all zeros; nothing to transmit")
    if not 0 < peak_v <= AO_VMAX:
        raise ValueError("Peak output voltage must be between 0 and %.0f V" % AO_VMAX)
    n_sig = int(round(duration_s * fs))
    if n_sig < 2:
        raise ValueError("Transmit duration is too short")
    n_gi = max(int(math.ceil(gi_s * fs)), min_gi_samples(n_sig))
    total = (n_sig + n_gi) * int(repeats)
    if total > MAX_AO_SAMPLES:
        raise ValueError(
            "Sequence has %.0f M samples, above the %.0f M limit; reduce repeats or durations"
            % (total / 1e6, MAX_AO_SAMPLES / 1e6)
        )
    shot = np.resize(wav, n_sig) * (peak_v / peak)  # np.resize loops / truncates
    period = np.zeros(n_sig + n_gi)
    period[:n_sig] = shot
    return np.tile(period, int(repeats)), n_sig, n_gi


# --------------------------------------------------------------------------
# DAQ backends
# --------------------------------------------------------------------------
class SimBackend:
    """Hardware-free test backend: AI = noise + 0.1 x AO signal."""

    ao_channels = ["ao0", "ao1", "ao2", "ao3"]
    ai_channels = ["ai%d" % i for i in range(16)]
    ao_max_rate = 2.86e6
    ai_max_rate = 1.25e6

    def __init__(self):
        self._ai_fs = None
        self._ai_t0 = 0.0
        self._ai_n = 0
        self._ao = None  # (wave, fs, AI sample index at AO start)
        self._ao_t0 = 0.0
        self._rng = np.random.default_rng()

    def start_ai(self, chan, fs, vmax, term):
        self._ai_fs = float(fs)
        self._ai_t0 = time.perf_counter()
        self._ai_n = 0
        return self._ai_fs

    def read_ai(self):
        fs = self._ai_fs
        n = int((time.perf_counter() - self._ai_t0) * fs) - self._ai_n
        if n <= 0:
            return np.empty(0)
        idx = self._ai_n + np.arange(n)
        x = 2e-3 * self._rng.standard_normal(n)
        if self._ao is not None:
            wave, ao_fs, start = self._ao
            k = np.round((idx - start) * (ao_fs / fs)).astype(np.int64)
            ok = (k >= 0) & (k < wave.size)
            x[ok] += 0.1 * wave[k[ok]]
        self._ai_n += n
        return x

    def stop_ai(self):
        self._ai_fs = None

    def start_ao(self, chan, fs, wave):
        start = self._ai_n if self._ai_fs else 0
        if self._ai_fs:
            start = int((time.perf_counter() - self._ai_t0) * self._ai_fs)
        self._ao = (wave, float(fs), start)
        self._ao_t0 = time.perf_counter()
        return float(fs)

    def ao_done(self):
        if self._ao is None:
            return True
        wave, fs, _ = self._ao
        return time.perf_counter() - self._ao_t0 >= wave.size / fs

    def stop_ao(self):
        if self._ao is not None:
            wave, fs, start = self._ao
            # aborted: keep only what was already played
            n_out = int((time.perf_counter() - self._ao_t0) * fs)
            self._ao = (wave[: max(0, min(n_out, wave.size))], fs, start)

    def close(self):
        self.stop_ai()
        self._ao = None


class NiBackend:
    """NI-DAQmx backend (USB-6259 and other M series)."""

    def __init__(self, device_name):
        self.dev = device_name
        dev = nidaqmx.system.Device(device_name)
        self.ao_channels = [c.name.split("/")[-1] for c in dev.ao_physical_chans]
        self.ai_channels = [c.name.split("/")[-1] for c in dev.ai_physical_chans]
        self.ao_max_rate = float(dev.ao_max_rate)
        self.ai_max_rate = float(dev.ai_max_single_chan_rate)
        self._ai = None
        self._reader = None
        self._ao = None
        self._ao_chan = None

    # ---- AI ----
    def start_ai(self, chan, fs, vmax, term):
        self.stop_ai()
        cfg = getattr(TerminalConfiguration, term, None)
        if cfg is None and term == "DIFF":  # name used by older nidaqmx versions
            cfg = TerminalConfiguration.DIFFERENTIAL
        task = nidaqmx.Task()
        try:
            task.ai_channels.add_ai_voltage_chan(
                "%s/%s" % (self.dev, chan),
                terminal_config=cfg,
                min_val=-vmax,
                max_val=vmax,
            )
            task.timing.cfg_samp_clk_timing(
                fs, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(fs * 10)
            )
            task.in_stream.input_buf_size = int(fs * 10)  # about 10 s of buffer
            actual = float(task.timing.samp_clk_rate)
            self._reader = AnalogSingleChannelReader(task.in_stream)
            task.start()
        except Exception:
            task.close()
            raise
        self._ai = task
        return actual

    def read_ai(self):
        n = int(self._ai.in_stream.avail_samp_per_chan)
        if n <= 0:
            return np.empty(0)
        buf = np.empty(n, dtype=np.float64)
        got = self._reader.read_many_sample(
            buf, number_of_samples_per_channel=n, timeout=1.0
        )
        return buf[:got]

    def stop_ai(self):
        if self._ai is not None:
            try:
                self._ai.stop()
            finally:
                self._ai.close()
                self._ai = None

    # ---- AO ----
    def start_ao(self, chan, fs, wave):
        self._close_ao()
        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                "%s/%s" % (self.dev, chan), min_val=-AO_VMAX, max_val=AO_VMAX
            )
            task.timing.cfg_samp_clk_timing(
                fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=wave.size
            )
            actual = float(task.timing.samp_clk_rate)
            AnalogSingleChannelWriter(task.out_stream, auto_start=False).write_many_sample(
                np.ascontiguousarray(wave, dtype=np.float64), timeout=60.0
            )
            task.start()
        except Exception:
            task.close()
            raise
        self._ao = task
        self._ao_chan = chan
        return actual

    def ao_done(self):
        return self._ao is None or self._ao.is_task_done()

    def _close_ao(self):
        if self._ao is not None:
            try:
                self._ao.stop()
            finally:
                self._ao.close()
                self._ao = None

    def stop_ao(self):
        """Stop the output and drive AO back to 0 V."""
        if self._ao is None:
            return
        chan = self._ao_chan
        try:
            self._close_ao()
        finally:
            with nidaqmx.Task() as t:
                t.ao_channels.add_ao_voltage_chan("%s/%s" % (self.dev, chan))
                t.write(0.0, auto_start=True)

    def close(self):
        try:
            self.stop_ao()
        finally:
            self.stop_ai()


def list_ni_devices():
    if nidaqmx is None:
        return []
    try:
        return [(d.name, d.product_type) for d in nidaqmx.system.System.local().devices]
    except Exception:
        return []


# --------------------------------------------------------------------------
# txt recording (written by a background thread)
# --------------------------------------------------------------------------
class Recorder:
    """Streams the hydrophone voltage to disk from a background thread.

    fmt "txt": '#' header lines, then one voltage (V) per line.
    fmt "wav": mono 32-bit float WAV whose sample values are volts (1.0 = 1 V);
               the header lines go to a separate *_info.txt file.
    """

    WAV_HEADER_BYTES = 58  # RIFF(12) + fmt(26) + fact(12) + data(8)

    def __init__(self, path, header_lines, fmt, fs):
        self.path = path
        self.fmt = fmt
        self.n_samples = 0
        self.error = None
        self._q = queue.Queue()
        if fmt == "wav":
            with open(os.path.splitext(path)[0] + "_info.txt", "w", encoding="utf-8") as info:
                for line in header_lines:
                    info.write("# %s\n" % line)
            self._fs = int(round(fs))
            self._f = open(path, "wb")
            self._f.write(self._wav_header(0))
        else:
            self._f = open(path, "w", encoding="utf-8")
            for line in header_lines:
                self._f.write("# %s\n" % line)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _wav_header(self, n):
        data_bytes = 4 * n
        return b"".join([
            b"RIFF", struct.pack("<I", self.WAV_HEADER_BYTES - 8 + data_bytes), b"WAVE",
            # format 3 = IEEE float, 1 channel, 32 bit
            b"fmt ", struct.pack("<IHHIIHHH", 18, 3, 1, self._fs, self._fs * 4, 4, 32, 0),
            b"fact", struct.pack("<II", 4, n),
            b"data", struct.pack("<I", data_bytes),
        ])

    def put(self, chunk):
        self.n_samples += chunk.size
        self._q.put(chunk)

    def _run(self):
        while True:
            chunk = self._q.get()
            if chunk is None:
                return
            if self.error is None:
                try:
                    if self.fmt == "wav":
                        self._f.write(chunk.astype("<f4").tobytes())
                    else:
                        np.savetxt(self._f, chunk, fmt="%.6e")
                except Exception as exc:  # e.g. disk full
                    self.error = exc

    def close(self):
        self._q.put(None)
        self._thread.join()
        try:
            if self.fmt == "wav":  # fill in the real sizes
                self._f.seek(0)
                self._f.write(self._wav_header(self.n_samples))
        finally:
            self._f.close()


# Built-in hydrophone sensitivity curves (dB re 1 V/uPa), used when
# hydrophone_rvr.csv is not next to this file. A curve with the same name in
# the CSV replaces the built-in one. These values were read off the RVR plot
# by eye (about +/-0.1 dB); put the exact calibration data in the CSV.
_RVR_FREQ_KHZ = [3.00, 3.85, 4.60, 5.40, 6.30, 7.25, 7.95, 8.75, 10.00, 10.65, 11.35,
                 12.20, 13.00, 14.10, 14.95, 15.65, 16.70, 17.55, 18.20, 19.25, 20.00]
BUILTIN_RVR = {
    "Orca 1": [-185.80, -184.00, -183.95, -186.55, -186.90, -186.80, -185.80, -185.15,
               -185.10, -185.50, -185.90, -185.80, -185.35, -184.95, -184.60, -184.00,
               -183.65, -183.20, -182.80, -182.15, -182.10],
    "Narwhal 2": [-185.45, -184.55, -182.50, -184.10, -185.05, -184.75, -184.10, -183.55,
                  -183.40, -183.75, -183.85, -183.85, -183.30, -182.95, -182.65, -182.00,
                  -181.60, -181.15, -180.60, -179.95, -179.70],
}


def builtin_rvr():
    f = np.array(_RVR_FREQ_KHZ) * 1e3
    return {name: (f, np.array(db)) for name, db in BUILTIN_RVR.items()}


def load_rvr(path):
    """Read hydrophone sensitivity curves: {name: (freq_Hz, dB re 1 V/uPa)}."""
    # utf-8-sig: Excel on Windows saves CSV files with a byte-order mark
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(line for line in fh
                                      if line.strip() and not line.lstrip().startswith("#"))]
    names = [n.strip() for n in rows[0][1:]]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    data = data[np.argsort(data[:, 0])]
    return {name: (data[:, 0] * 1e3, data[:, i + 1]) for i, name in enumerate(names)}


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def fourier_spectrum(x, fs):
    """Amplitude spectrum of a time record by direct Fourier transform.

    S(f) = integral s(t) exp(-i 2 pi f t) dt  ~=  dt * FFT(s)
    (transform pair as in Jensen et al., Computational Ocean Acoustics; their
    exp(+i w t) sign convention gives the same magnitude). No segmenting, no
    window, no averaging, no mean removal. Returns (f, |S(f)| in V/Hz) for
    0 <= f <= fs/2; zero-padded to a power of two, which only interpolates.
    """
    nfft = 1 << max(1, int(np.ceil(np.log2(x.size))))
    spec = np.abs(np.fft.rfft(x, nfft)) / fs
    return np.fft.rfftfreq(nfft, 1.0 / fs), spec


def spectrum_db(spec):
    """dB re 1 V/Hz with a 160 dB display floor below the peak."""
    peak = float(spec.max()) if spec.size else 0.0
    return 20 * np.log10(np.maximum(spec, max(peak * 1e-8, 1e-300)))


def minmax_decimate(t0, dt, y, n_points=PLOT_POINTS):
    """Min/max decimation for plotting; keeps the peaks."""
    n = y.size
    step = n // (n_points // 2)
    if step <= 1:
        return t0 + dt * np.arange(n), y
    m = (n // step) * step
    blocks = y[n - m :].reshape(-1, step)
    out = np.empty(blocks.shape[0] * 2)
    out[0::2] = blocks.min(axis=1)
    out[1::2] = blocks.max(axis=1)
    tb = t0 + dt * ((n - m) + step * np.arange(blocks.shape[0]))
    return np.repeat(tb, 2), out


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WAV Transmit / Reference Hydrophone (NI USB-6259)")
        self.backend = None
        self.wav = None  # float array of the selected channel
        self.wav_all = None
        self.wav_fs = None
        self.wav_path = None
        self.ai_running = False
        self.ai_fs = 0.0
        self.ring = np.zeros(1)
        self.ring_fill = 0
        self.recorder = None
        self.rec_auto = False
        self.tx_active = False
        self.tx_t0 = 0.0
        self.tx_info = None
        # next to this file, so it does not depend on where the program is launched from
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

        self._build_ui()
        self._populate_devices()
        self._update_tx_info()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

    # ---------------- UI ----------------
    def _build_ui(self):
        W = QtWidgets
        central = W.QWidget()
        self.setCentralWidget(central)
        root = W.QHBoxLayout(central)
        left = W.QVBoxLayout()
        panel = W.QWidget()
        panel.setLayout(left)
        left.setContentsMargins(0, 0, 18, 0)  # room for the scrollbar
        left.setSpacing(4)
        # scroll area so the window can be made smaller than the control column
        scroll = W.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)
        root.addWidget(scroll, 0)

        line = self.fontMetrics().lineSpacing()

        def grid_box(title):
            g = W.QGroupBox(title)
            lay = W.QGridLayout(g)
            lay.setVerticalSpacing(3)
            lay.setHorizontalSpacing(6)
            lay.setContentsMargins(8, 4, 8, 4)
            lay.setColumnStretch(1, 1)
            lay.setColumnStretch(3, 1)
            left.addWidget(g)
            return lay

        def pair(lay, row, label1, w1, label2=None, w2=None, tip1=None, tip2=None):
            """One row holding two labelled widgets side by side."""
            for col, (label, w, tip) in enumerate(((label1, w1, tip1), (label2, w2, tip2))):
                if w is None:
                    continue
                lb = W.QLabel(label)
                lay.addWidget(lb, row, 2 * col)
                lay.addWidget(w, row, 2 * col + 1)
                if tip:
                    lb.setToolTip(tip)
                    w.setToolTip(tip)

        def full(lay, row, w):
            if isinstance(w, W.QLayout):
                lay.addLayout(w, row, 0, 1, 4)
            else:
                lay.addWidget(w, row, 0, 1, 4)

        def note(lines=2):
            lb = W.QLabel()
            lb.setWordWrap(True)
            lb.setMinimumHeight(lines * line + 2)
            return lb

        # --- device ---
        f = grid_box("DAQ device")
        self.cb_dev = W.QComboBox()
        self.cb_dev.currentIndexChanged.connect(self._on_device_changed)
        b = W.QPushButton("Rescan")
        b.clicked.connect(self._populate_devices)
        row = W.QHBoxLayout()
        row.addWidget(self.cb_dev, 1)
        row.addWidget(b)
        full(f, 0, row)

        # --- transmit ---
        f = grid_box("Transmit (AO)")
        self.btn_wav = W.QPushButton("Select WAV…")
        self.btn_wav.clicked.connect(self._choose_wav)
        full(f, 0, self.btn_wav)
        self.lb_wav = note()
        self.lb_wav.setText("No file selected")
        full(f, 1, self.lb_wav)
        self.cb_wavch = W.QComboBox()
        self.cb_wavch.currentIndexChanged.connect(self._on_wav_channel)
        self.cb_ao = W.QComboBox()
        pair(f, 2, "WAV ch", self.cb_wavch, "AO ch", self.cb_ao,
             "Channel of the WAV file to transmit", "Analog output channel")
        self.sp_amp = W.QDoubleSpinBox()
        self.sp_amp.setRange(0.001, AO_VMAX)
        self.sp_amp.setDecimals(3)
        self.sp_amp.setSingleStep(0.1)
        self.sp_amp.setValue(1.0)
        self.sp_amp.setSuffix(" V")
        self.sp_amp.valueChanged.connect(self._update_tx_preview)
        self.sp_rep = W.QSpinBox()
        self.sp_rep.setRange(1, 100000)
        self.sp_rep.setValue(1)
        self.sp_rep.valueChanged.connect(self._update_tx_info)
        pair(f, 3, "Peak", self.sp_amp, "Repeats", self.sp_rep,
             "Peak output voltage (the WAV peak is scaled to this)",
             "Number of shots")
        self.sp_dur = W.QDoubleSpinBox()
        self.sp_dur.setRange(0.001, 3600.0)
        self.sp_dur.setDecimals(3)
        self.sp_dur.setValue(1.0)
        self.sp_dur.setSuffix(" s")
        self.sp_dur.valueChanged.connect(self._on_duration_changed)
        self.sp_gi = W.QDoubleSpinBox()
        self.sp_gi.setRange(1.0, 36000.0)
        self.sp_gi.setDecimals(3)
        self.sp_gi.setValue(1.0)
        self.sp_gi.setSuffix(" s")
        self.sp_gi.valueChanged.connect(self._update_tx_info)
        pair(f, 4, "Duration", self.sp_dur, "Guard", self.sp_gi,
             "Transmit duration of each shot",
             "Guard interval after each shot (cannot be shorter than the duration)")
        self.lb_txinfo = note()
        full(f, 5, self.lb_txinfo)
        self.ck_autorec = W.QCheckBox("Auto-record during transmit")
        self.ck_autorec.setToolTip("Record the hydrophone to txt for the whole transmit sequence")
        full(f, 6, self.ck_autorec)
        row = W.QHBoxLayout()
        self.btn_tx = W.QPushButton("Transmit")
        font = self.btn_tx.font()
        font.setBold(True)
        self.btn_tx.setFont(font)
        self.btn_tx.setMinimumHeight(30)
        self.btn_tx.clicked.connect(self._start_tx)
        self.btn_stop = W.QPushButton("Abort")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(30)
        self.btn_stop.clicked.connect(lambda: self._finish_tx(aborted=True))
        row.addWidget(self.btn_tx, 2)
        row.addWidget(self.btn_stop, 1)
        full(f, 7, row)
        self.pb = W.QProgressBar()
        self.pb.setRange(0, 1000)
        full(f, 8, self.pb)
        self.lb_txstate = W.QLabel("Idle")
        full(f, 9, self.lb_txstate)
        self.tx_widgets = [
            self.btn_wav, self.cb_wavch, self.cb_ao, self.sp_amp,
            self.sp_dur, self.sp_gi, self.sp_rep, self.ck_autorec,
        ]

        # --- hydrophone ---
        f = grid_box("Reference hydrophone (AI)")
        self.cb_ai = W.QComboBox()
        self.cb_term = W.QComboBox()
        self.cb_term.addItems(["RSE", "DIFF", "NRSE"])
        pair(f, 0, "AI ch", self.cb_ai, "Terminal", self.cb_term,
             "Analog input channel", "Terminal configuration")
        self.cb_range = W.QComboBox()
        for v in (10, 5, 2, 1, 0.5, 0.2, 0.1):
            self.cb_range.addItem("±%g V" % v, float(v))
        self.sp_aifs = W.QDoubleSpinBox()
        self.sp_aifs.setRange(1000, 1.25e6)
        self.sp_aifs.setDecimals(0)
        self.sp_aifs.setSingleStep(10000)
        self.sp_aifs.setValue(100000)
        pair(f, 1, "Range", self.cb_range, "Rate (Hz)", self.sp_aifs,
             "Input range", "Sample rate")
        self.btn_mon = W.QPushButton("Start monitoring")
        self.btn_mon.clicked.connect(self._toggle_monitor)
        full(f, 2, self.btn_mon)
        self.sp_win = W.QDoubleSpinBox()
        self.sp_win.setRange(0.01, 10.0)
        self.sp_win.setDecimals(2)
        self.sp_win.setValue(2.0)
        self.sp_win.setSuffix(" s")
        self.sp_win.valueChanged.connect(self._resize_ring)
        self.cb_nfft = W.QComboBox()
        for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
            self.cb_nfft.addItem(str(n), n)
        self.cb_nfft.setCurrentIndex(3)
        self.cb_nfft.currentIndexChanged.connect(self._resize_ring)
        pair(f, 3, "Window", self.sp_win, "FFT", self.cb_nfft,
             "Length of the time plot",
             "Number of most recent samples transformed for the hydrophone spectrum")
        self.cb_hyd = W.QComboBox()
        self.cb_hyd.addItem("None (V)", None)
        self.cb_hyd.currentIndexChanged.connect(self._on_hydrophone_changed)
        self.sp_gain = W.QDoubleSpinBox()
        self.sp_gain.setRange(-60.0, 120.0)
        self.sp_gain.setDecimals(1)
        self.sp_gain.setValue(0.0)
        self.sp_gain.setSuffix(" dB")
        pair(f, 4, "Sensor", self.cb_hyd, "Gain", self.sp_gain,
             "Hydrophone sensitivity curve used to convert the spectrum to pressure "
             "(curves come from %s)" % RVR_FILE,
             "Gain of any amplifier between the hydrophone and the DAQ that is not "
             "already included in the sensitivity curve; it is removed from the spectrum")
        self.ck_logf = W.QCheckBox("Log frequency axis")
        self.ck_logf.toggled.connect(self._set_log_freq)
        full(f, 5, self.ck_logf)
        row = W.QHBoxLayout()
        self.btn_rec = W.QPushButton("● Start recording")
        self.btn_rec.clicked.connect(self._toggle_record)
        self.cb_fmt = W.QComboBox()
        self.cb_fmt.addItems(["txt", "wav"])
        self.cb_fmt.setToolTip("Recording file format. wav = 32-bit float, sample values in volts")
        b = W.QPushButton("Folder…")
        b.setToolTip("Choose the folder recordings are saved to")
        b.clicked.connect(self._choose_dir)
        row.addWidget(self.btn_rec, 1)
        row.addWidget(self.cb_fmt)
        row.addWidget(b)
        full(f, 6, row)
        self.lb_rec = note()
        self._show_save_dir()
        full(f, 7, self.lb_rec)
        self.ai_widgets = [self.cb_ai, self.cb_term, self.cb_range, self.sp_aifs]
        left.addStretch(1)
        # let inputs shrink so two fit per row inside the fixed-width column
        for cb in panel.findChildren(W.QComboBox):
            cb.setSizeAdjustPolicy(
                W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(4)
        for sp in panel.findChildren(W.QAbstractSpinBox):
            sp.setMinimumWidth(84)

        # --- plots ---
        # top row: transmit signal, bottom row: hydrophone; left: time, right: spectrum
        plots = W.QGridLayout()
        root.addLayout(plots, 1)

        def make_plot(title, xlabel, xunit, ylabel, yunit, color, row, col):
            pw = pg.PlotWidget(title=title)
            pw.setLabel("bottom", xlabel, units=xunit)
            pw.setLabel("left", ylabel, units=yunit)
            pw.showGrid(x=True, y=True, alpha=0.3)
            plots.addWidget(pw, row, col)
            return pw, pw.plot(pen=pg.mkPen(color, width=1))

        psd = "|S(f)| (dB re 1 V/Hz)"
        self.plot_txt, self.curve_txt = make_plot(
            TX_TITLE, "time", "s", "voltage", "V", "#3aa655", 0, 0)
        self.gi_region = pg.LinearRegionItem(movable=False, brush=(255, 255, 255, 30))
        self.gi_region.setVisible(False)
        self.plot_txt.addItem(self.gi_region)
        self.plot_txf, self.curve_txf = make_plot(
            "Transmit signal — spectrum", "frequency", "Hz", psd, None, "#c9a227", 0, 1)
        self.plot_t, self.curve_t = make_plot(
            "Reference hydrophone — voltage vs time", "time", "s", "voltage", "V",
            "#2a78d6", 1, 0)
        self.plot_f, self.curve_f = make_plot(
            "Reference hydrophone — spectrum", "frequency", "Hz", psd, None, "#d6542a", 1, 1)
        self._set_log_freq(False)  # fix the frequency axis to 0..SPECTRUM_FMAX
        self._load_hydrophones()
        self.resize(1200, 820)

    # ---------------- device ----------------
    def _populate_devices(self):
        self.cb_dev.blockSignals(True)
        self.cb_dev.clear()
        for name, ptype in list_ni_devices():
            self.cb_dev.addItem("%s (%s)" % (name, ptype), name)
        self.cb_dev.addItem(SIM_NAME, None)
        self.cb_dev.blockSignals(False)
        self._on_device_changed()
        if nidaqmx is None:
            self.statusBar().showMessage(
                "nidaqmx not available (%s) — only the simulated device can be used" % NIDAQMX_ERROR
            )

    def _on_device_changed(self):
        self._shutdown_backend()
        name = self.cb_dev.currentData()
        try:
            self.backend = NiBackend(name) if name else SimBackend()
        except Exception as exc:
            self._error("Cannot open device %s" % name, exc)
            self.backend = SimBackend()
            self.cb_dev.blockSignals(True)
            self.cb_dev.setCurrentIndex(self.cb_dev.count() - 1)
            self.cb_dev.blockSignals(False)
        for cb, items in ((self.cb_ao, self.backend.ao_channels),
                          (self.cb_ai, self.backend.ai_channels)):
            cb.clear()
            cb.addItems(items)
        self.sp_aifs.setMaximum(self.backend.ai_max_rate)

    def _shutdown_backend(self):
        if self.backend is None:
            return
        if self.tx_active:
            self._finish_tx(aborted=True)
        self._stop_monitor()
        try:
            self.backend.close()
        except Exception as exc:
            self._error("Error while closing the device", exc)
        self.backend = None

    # ---------------- WAV / transmit settings ----------------
    def _choose_wav(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select WAV", os.path.dirname(self.wav_path or ""), "WAV (*.wav *.WAV)"
        )
        if path:
            self.load_wav_file(path)

    def load_wav_file(self, path):
        try:
            fs, data = load_wav(path)
        except Exception as exc:
            self._error("Cannot read WAV", exc)
            return
        self.wav_path, self.wav_fs, self.wav_all = path, fs, data
        self.cb_wavch.blockSignals(True)
        self.cb_wavch.clear()
        self.cb_wavch.addItems(["ch %d" % (i + 1) for i in range(data.shape[1])])
        self.cb_wavch.blockSignals(False)
        self.wav = data[:, 0]
        self.sp_dur.setValue(max(self.sp_dur.minimum(), data.shape[0] / fs))
        self._update_tx_info()

    def _on_wav_channel(self, i):
        if self.wav_all is not None and i >= 0:
            self.wav = self.wav_all[:, i]
            self._update_tx_preview()

    def _on_duration_changed(self, dur):
        # GI minimum = transmit duration; it may be longer, never shorter
        self.sp_gi.setMinimum(dur)
        self._update_tx_info()

    def _update_tx_info(self, *_):
        dur, gi, rep = self.sp_dur.value(), self.sp_gi.value(), self.sp_rep.value()
        period = dur + gi
        text = "Period %.3f s (GI %.0f%%), total %.3f s" % (
            period, 100 * gi / period, period * rep)
        if self.wav is not None:
            wav_len = self.wav.size / self.wav_fs
            self.lb_wav.setText("%s\n%d Hz, %.3f s, %d channel(s)" % (
                os.path.basename(self.wav_path), self.wav_fs, wav_len,
                self.wav_all.shape[1]))
            if dur > wav_len + 0.5 / self.wav_fs:
                text += "\nLonger than the WAV → WAV loops %.2f times" % (dur / wav_len)
            elif dur < wav_len - 0.5 / self.wav_fs:
                text += "\nShorter than the WAV → WAV is truncated"
        self.lb_txinfo.setText(text)
        self._update_tx_preview()

    def _update_tx_preview(self, *_):
        """Plot the waveform that will be sent (in volts) and its spectrum."""
        peak = float(np.max(np.abs(self.wav))) if self.wav is not None else 0.0
        if peak <= 0:
            self.curve_txt.setData([], [])
            self.curve_txf.setData([], [])
            self.gi_region.setVisible(False)
            return
        fs = self.wav_fs
        n_sig = max(2, int(round(self.sp_dur.value() * fs)))
        n = min(n_sig, PREVIEW_MAX_SAMPLES)
        shot = np.resize(self.wav, n) * (self.sp_amp.value() / peak)
        t, y = minmax_decimate(0.0, 1.0 / fs, shot)
        truncated = n < n_sig
        if truncated:
            self.plot_txt.setTitle("Transmit signal — first %.2f s shown" % (n / fs))
        else:
            t_end = n_sig / fs
            gi = max(self.sp_gi.value(), t_end)
            t = np.concatenate([t, [t_end, t_end + gi]])
            y = np.concatenate([y, [0.0, 0.0]])
            self.gi_region.setRegion((t_end, t_end + gi))
            self.plot_txt.setTitle(TX_TITLE)
        self.gi_region.setVisible(not truncated)
        self.curve_txt.setData(t, y)
        f, spec = fourier_spectrum(shot, fs)  # one transform of the whole shot
        self.curve_txf.setData(*minmax_decimate(f[1], f[1], spectrum_db(spec[1:])))

    def _set_log_freq(self, on):
        for pw in (self.plot_f, self.plot_txf):
            pw.setLogMode(x=on, y=False)
            if on:  # in log mode the view range is in log10(Hz)
                pw.setXRange(np.log10(SPECTRUM_FMIN_LOG), np.log10(SPECTRUM_FMAX), padding=0)
            else:
                pw.setXRange(0, SPECTRUM_FMAX, padding=0)

    # ---------------- transmit ----------------
    def _start_tx(self):
        if self.tx_active:
            return
        if self.wav is None:
            self._error("Select a WAV file first")
            return
        fs = self.wav_fs
        if fs > self.backend.ao_max_rate:
            self._error("WAV sample rate %d Hz exceeds the AO limit of %.0f Hz" % (fs, self.backend.ao_max_rate))
            return
        dur, gi, rep = self.sp_dur.value(), self.sp_gi.value(), self.sp_rep.value()
        try:
            seq, n_sig, n_gi = build_sequence(self.wav, fs, self.sp_amp.value(), dur, gi, rep)
        except Exception as exc:
            self._error("Cannot build the transmit sequence", exc)
            return

        self.tx_info = {
            "wav": self.wav_path, "peak_V": self.sp_amp.value(),
            "duration_s": n_sig / fs, "gi_s": n_gi / fs, "repeats": rep,
            "ao_channel": self.cb_ao.currentText(),
        }
        if self.ck_autorec.isChecked() and self.recorder is None:
            if not self._start_record(auto=True):
                return
        try:
            actual = self.backend.start_ao(self.cb_ao.currentText(), fs, seq)
        except Exception as exc:
            self._error("AO failed to start", exc)
            if self.rec_auto:
                self._stop_record()
            return
        self.tx_t0 = time.perf_counter()
        self.tx_info.update(fs=actual, period=(n_sig + n_gi) / actual,
                            sig=n_sig / actual, total=seq.size / actual)
        self.tx_active = True
        for w in self.tx_widgets:
            w.setEnabled(False)
        self.btn_tx.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cb_dev.setEnabled(False)
        self.statusBar().showMessage("Actual AO sample rate %.3f Hz (WAV %d Hz)" % (actual, fs))

    def _update_tx_progress(self):
        info = self.tx_info
        el = time.perf_counter() - self.tx_t0
        try:
            done = self.backend.ao_done()
        except Exception as exc:
            self._finish_tx(aborted=True)
            self._error("AO output error", exc)
            return
        if done:
            self._finish_tx(aborted=False)
            return
        el = min(el, info["total"])
        k = min(int(el // info["period"]), info["repeats"] - 1)
        phase = "transmitting" if (el - k * info["period"]) < info["sig"] else "guard interval"
        self.pb.setValue(int(1000 * el / info["total"]))
        self.lb_txstate.setText("Shot %d / %d — %s  (%.1f / %.1f s)" % (
            k + 1, info["repeats"], phase, el, info["total"]))

    def _finish_tx(self, aborted):
        if not self.tx_active:
            return
        self.tx_active = False
        try:
            self.backend.stop_ao()
        except Exception as exc:
            self._error("Error while stopping AO", exc)
        if self.rec_auto:
            self._stop_record()
        for w in self.tx_widgets:
            w.setEnabled(True)
        self.btn_tx.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cb_dev.setEnabled(True)
        self.pb.setValue(0 if aborted else 1000)
        self.lb_txstate.setText("Aborted (AO reset to 0 V)" if aborted else "Done")

    # ---------------- monitor ----------------
    def _toggle_monitor(self):
        if self.ai_running:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _start_monitor(self):
        if self.ai_running:
            return True
        try:
            self.ai_fs = self.backend.start_ai(
                self.cb_ai.currentText(), self.sp_aifs.value(),
                self.cb_range.currentData(), self.cb_term.currentText())
        except Exception as exc:
            self._error("AI failed to start", exc)
            return False
        self.ai_running = True
        self._resize_ring()
        for w in self.ai_widgets:
            w.setEnabled(False)
        self.btn_mon.setText("Stop monitoring")
        return True

    def _stop_monitor(self):
        if not self.ai_running:
            return
        if self.recorder is not None:
            self._stop_record()
        self.ai_running = False
        try:
            self.backend.stop_ai()
        except Exception as exc:
            self._error("Error while stopping AI", exc)
        for w in self.ai_widgets:
            w.setEnabled(True)
        self.btn_mon.setText("Start monitoring")

    def _resize_ring(self, *_):
        if not self.ai_running:
            return
        n = max(int(self.sp_win.value() * self.ai_fs), self.cb_nfft.currentData())
        self.ring = np.zeros(n)
        self.ring_fill = 0

    def _push(self, x):
        n = self.ring.size
        if x.size >= n:
            self.ring[:] = x[-n:]
        else:
            self.ring[: n - x.size] = self.ring[x.size :]
            self.ring[n - x.size :] = x
        self.ring_fill = min(n, self.ring_fill + x.size)

    def _update_plots(self):
        if self.ring_fill < 2:
            return
        fs = self.ai_fs
        n_win = min(self.ring_fill, max(2, int(self.sp_win.value() * fs)))
        y = self.ring[-n_win:]
        t, yd = minmax_decimate(-n_win / fs, 1.0 / fs, y)
        self.curve_t.setData(t, yd)
        nfft = self.cb_nfft.currentData()
        if self.ring_fill >= nfft:
            # one transform of the most recent nfft samples
            f, spec = fourier_spectrum(self.ring[-nfft:], fs)
            df, f, db = f[1], f[1:], spectrum_db(spec[1:])
            cal = self.cb_hyd.currentData()
            if cal is not None:
                # volts -> pressure: |P(f)| = |S(f)| / sensitivity, only where calibrated
                keep = (f >= cal[0][0]) & (f <= cal[0][-1])
                f = f[keep]
                db = db[keep] - np.interp(f, cal[0], cal[1]) - self.sp_gain.value()
            if f.size:
                self.curve_f.setData(*minmax_decimate(f[0], df, db))
            else:
                self.curve_f.setData([], [])

    def _load_hydrophones(self):
        """Fill the Sensor list: built-in curves, replaced/extended by the CSV."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RVR_FILE)
        curves = builtin_rvr()
        if not os.path.exists(path):
            self.rvr_source = "built-in curves (%s not found next to the program)" % RVR_FILE
        else:
            try:
                curves.update(load_rvr(path))
                self.rvr_source = path
            except Exception as exc:
                self.rvr_source = "built-in curves (%s could not be read: %s)" % (RVR_FILE, exc)
        for name, curve in curves.items():
            self.cb_hyd.addItem(name, curve)
        self.cb_hyd.setToolTip("Hydrophone sensitivity curve used to convert the spectrum "
                               "to pressure.\nSource: %s" % self.rvr_source)
        self._on_hydrophone_changed()

    def _on_hydrophone_changed(self, *_):
        cal = self.cb_hyd.currentData()
        self.sp_gain.setEnabled(cal is not None)
        if cal is None:
            self.plot_f.setTitle("Reference hydrophone — spectrum")
            self.plot_f.setLabel("left", "|S(f)| (dB re 1 V/Hz)")
        else:
            self.plot_f.setTitle("Hydrophone spectrum — %s (%.0f–%.0f kHz)" % (
                self.cb_hyd.currentText(), cal[0][0] / 1e3, cal[0][-1] / 1e3))
            self.plot_f.setLabel("left", "|P(f)| (dB re 1 µPa/Hz)")
        self.plot_f.enableAutoRange(axis="y")

    # ---------------- recording ----------------
    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save folder", self.save_dir)
        if d:
            self.save_dir = d
            if self.recorder is None:
                self._show_save_dir()

    def _show_save_dir(self):
        self.lb_rec.setText("Save to: …%s%s" % (os.sep, os.path.basename(self.save_dir)))
        self.lb_rec.setToolTip(self.save_dir)

    def _toggle_record(self):
        if self.recorder is None:
            self._start_record(auto=False)
        else:
            self._stop_record()

    def _start_record(self, auto):
        if not self._start_monitor():
            return False
        now = _dt.datetime.now()
        fmt = self.cb_fmt.currentText()
        path = os.path.join(self.save_dir, "hydrophone_%s.%s" % (
            now.strftime("%Y%m%d_%H%M%S"), fmt))
        header = [
            "reference hydrophone recording",
            "start_time: %s" % now.isoformat(timespec="milliseconds"),
            "fs_Hz: %.6f" % self.ai_fs,
            "device: %s" % self.cb_dev.currentText(),
            "ai_channel: %s, terminal: %s, range: %s" % (
                self.cb_ai.currentText(), self.cb_term.currentText(),
                self.cb_range.currentText()),
        ]
        if auto and self.tx_info:
            header.append("tx: " + ", ".join(
                "%s=%s" % kv for kv in self.tx_info.items()))
        header.append("hydrophone: %s, gain_dB: %.1f (used for the on-screen spectrum only; "
                      "the recorded data are raw volts)" % (
                          self.cb_hyd.currentText(), self.sp_gain.value()))
        if fmt == "wav":
            header.append("data: %s, mono 32-bit float, sample value = volts, "
                          "wav header sample rate = %d Hz" % (
                              os.path.basename(path), int(round(self.ai_fs))))
        else:
            header.append("column: voltage_V (one sample per line, t = index / fs_Hz)")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            self.recorder = Recorder(path, header, fmt, self.ai_fs)
        except Exception as exc:
            self._error("Cannot create the recording file", exc)
            return False
        self.rec_auto = auto
        self.btn_rec.setText("■ Stop recording")
        self.cb_fmt.setEnabled(False)
        self.btn_rec.setStyleSheet("color: #c0392b; font-weight: bold;")
        return True

    def _stop_record(self):
        rec, self.recorder, self.rec_auto = self.recorder, None, False
        if rec is None:
            return
        rec.close()
        self.btn_rec.setText("● Start recording")
        self.cb_fmt.setEnabled(True)
        self.btn_rec.setStyleSheet("")
        self.lb_rec.setText("Saved: %s\n%d samples, %.2f s" % (
            os.path.basename(rec.path), rec.n_samples, rec.n_samples / self.ai_fs))
        self.lb_rec.setToolTip(rec.path)
        if rec.error is not None:
            self._error("Write error; the file may be incomplete", rec.error)

    # ---------------- main loop ----------------
    def _tick(self):
        if self.ai_running:
            try:
                x = self.backend.read_ai()
            except Exception as exc:
                self._stop_monitor()
                self._error("AI read error; monitoring stopped", exc)
            else:
                if x.size:
                    self._push(x)
                    if self.recorder is not None:
                        self.recorder.put(x)
                        self.lb_rec.setText("Recording: %s — %.1f s" % (
                            os.path.basename(self.recorder.path),
                            self.recorder.n_samples / self.ai_fs))
                self._update_plots()
        if self.tx_active:
            self._update_tx_progress()

    def _error(self, title, exc=None):
        text = title if exc is None else "%s\n\n%s" % (title, exc)
        self.statusBar().showMessage(text.replace("\n", " "))
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.critical(self, "Error", text)

    def closeEvent(self, ev):
        self.timer.stop()
        self._shutdown_backend()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()

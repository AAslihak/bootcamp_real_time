#!/usr/bin/env python3
"""WAV transmit / reference hydrophone monitor and recorder GUI (NI USB-6366).

AO : plays the wav at its native sample rate, scaled so that its largest
     sample equals the Peak setting (default 1 V); every shot is followed by a
     guard interval (GI) of silence. The GI can never be shorter than the
     transmit duration (GI >= 50% of the period).
AI : continuously acquires the reference hydrophone, shows voltage vs time
     and spectrum (in volts, or in pressure using a hydrophone sensitivity
     curve from hydrophone_rvr.csv), and can record to txt or wav.

USB-6366 specifics (X series): 2 analog outputs (ao0-ao1), 8 simultaneously
sampled analog inputs (ai0-ai7) that are differential only (each BNC/terminal
pair is AI n+ / AI n-), input ranges +/-10, 5, 2, 1 V, up to 2 MS/s per channel.

Without the NI-DAQmx driver or a device, the "Simulated" device lets you
try the interface (simulated AI = the AO signal itself, like a loopback cable).

Run:  python daq_gui_6366.py
"""
import collections
import csv
import datetime as _dt
import fractions
import math
import os
import queue
import struct
import sys
import threading
import time
import traceback

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.io import wavfile
from scipy.signal import resample_poly

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
AO_VMAX = 10.0  # USB-6366 AO range is +/-10 V
MAX_AO_SAMPLES = 100_000_000  # whole sequence is written at once (~800 MB cap)
TICK_MS = 50
ERROR_LOG = "daq_gui_error.log"  # full error details are appended here, next to this file
DEFAULT_SENSOR = "Orca 1"  # hydrophone selected when the program starts
RVR_FILE = "hydrophone_rvr.csv"  # hydrophone sensitivity curves, next to this file
SPECTRUM_FMAX = 20e3  # spectra are displayed from 0 to this frequency (Hz)
SPECTRUM_FMIN_LOG = 10.0  # lower edge of the log frequency axis (Hz)
SPECTROGRAM_MIN_DT = 0.02  # smallest time step between spectrogram rows (s)
SPECTROGRAM_MAX_ROWS = 1000  # longer histories use a coarser time step instead of more rows
SPECTROGRAM_DYN_DB = 80.0  # colour scale spans this many dB below the maximum
SPECTROGRAM_FLOOR = -400.0  # value of "no data" cells (dB)
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
    """Hardware-free test backend: AI = the AO signal (ideal loopback, no noise)."""

    ao_channels = ["ao0", "ao1"]
    ai_channels = ["ai%d" % i for i in range(8)]
    ao_max_rate = 3.33e6
    ai_max_rate = 2.0e6

    def __init__(self):
        self._ai_fs = None
        self._ai_t0 = 0.0
        self._ai_n = 0
        self._ao = None  # (wave, fs, AI sample index at AO start)
        self._ao_at_ai_rate = None  # (AO wave resampled to the AI rate, that rate)
        self._ao_t0 = 0.0

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
        x = np.zeros(n)
        if self._ao is not None:
            wave, ao_fs, start = self._ao
            if self._ao_at_ai_rate is None or self._ao_at_ai_rate[1] != fs:
                # band-limited resampling; picking the nearest sample instead
                # would add false tones at multiples of the rate difference
                ratio = fractions.Fraction(fs / ao_fs).limit_denominator(1000)
                resampled = wave if ratio == 1 else resample_poly(
                    wave, ratio.numerator, ratio.denominator)
                self._ao_at_ai_rate = (resampled, fs)
            resampled = self._ao_at_ai_rate[0]
            k = idx - start
            n_played = min(resampled.size, int(wave.size * fs / ao_fs))  # shorter after abort
            ok = (k >= 0) & (k < n_played)
            x[ok] = resampled[k[ok]]
        self._ai_n += n
        return x

    def stop_ai(self):
        self._ai_fs = None

    def start_ao(self, chan, fs, wave):
        start = self._ai_n if self._ai_fs else 0
        if self._ai_fs:
            start = int((time.perf_counter() - self._ai_t0) * self._ai_fs)
        self._ao = (wave, float(fs), start)
        self._ao_at_ai_rate = None
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
    """NI-DAQmx backend (USB-6366 and other simultaneous-sampling X series)."""

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
        self.setWindowTitle("WAV Transmit / Reference Hydrophone (NI USB-6366)")
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
        small = panel.font()  # compact control column: slightly smaller text
        small.setPointSizeF(small.pointSizeF() * 0.85)
        panel.setFont(small)
        panel.setLayout(left)
        left.setContentsMargins(0, 0, 12, 0)  # room for the scrollbar
        left.setSpacing(4)
        # scroll area so the window can be made smaller than the control column
        scroll = W.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll, 0)

        line = self.fontMetrics().lineSpacing()

        def grid_box(title):
            g = W.QGroupBox(title)
            lay = W.QGridLayout(g)
            lay.setVerticalSpacing(3)
            lay.setHorizontalSpacing(4)
            lay.setContentsMargins(6, 3, 6, 3)
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
        self.sp_amp.setValue(1.0)  # default peak output: 1 V
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
        self.cb_term.addItems(["DIFF"])  # USB-6366 inputs are differential only
        pair(f, 0, "AI ch", self.cb_ai, "Terminal", self.cb_term,
             "Analog input channel",
             "USB-6366 inputs are differential only (AI n+ / AI n-)")
        self.cb_range = W.QComboBox()
        for v in (10, 5, 2, 1):  # USB-6366 input ranges
            self.cb_range.addItem("±%g V" % v, float(v))
        self.sp_aifs = W.QDoubleSpinBox()
        self.sp_aifs.setRange(1000, 2.0e6)
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
        self.cb_hyd.currentIndexChanged.connect(self._reset_spectrogram)
        self.sp_gain = W.QDoubleSpinBox()
        self.sp_gain.setRange(-60.0, 120.0)
        self.sp_gain.setDecimals(1)
        self.sp_gain.setValue(0.0)
        self.sp_gain.setSuffix(" dB")
        self.sp_gain.valueChanged.connect(self._reset_spectrogram)
        pair(f, 4, "Sensor", self.cb_hyd, "Gain", self.sp_gain,
             "Hydrophone sensitivity curve used to convert the spectrum to pressure "
             "(curves come from %s)" % RVR_FILE,
             "Gain of any amplifier between the hydrophone and the DAQ that is not "
             "already included in the sensitivity curve; it is removed from the spectrum")
        self.ck_logf = W.QCheckBox("Log freq")
        self.ck_logf.setToolTip("Logarithmic frequency axis for both spectra")
        self.ck_logf.toggled.connect(self._set_log_freq)
        self.ck_sg = W.QCheckBox("Spectrogram")
        self.ck_sg.setToolTip(
            "Show the hydrophone spectrum over time instead of the single spectrum:\n"
            "x = frequency, y = time (now at the top), colour = level")
        self.ck_sg.toggled.connect(self._set_spectrogram_view)
        self.sp_sgspan = W.QDoubleSpinBox()
        self.sp_sgspan.setRange(1.0, 120.0)
        self.sp_sgspan.setDecimals(0)
        self.sp_sgspan.setValue(10.0)
        self.sp_sgspan.setSuffix(" s")
        self.sp_sgspan.setToolTip("Length of the spectrogram time axis (how far back it shows)")
        self.sp_sgspan.valueChanged.connect(self._resize_ring)
        row = W.QHBoxLayout()
        row.addWidget(self.ck_logf)
        row.addWidget(self.ck_sg)
        row.addWidget(self.sp_sgspan, 1)
        full(f, 6, row)
        row = W.QHBoxLayout()
        self.btn_rec = W.QPushButton("● Start recording")
        self.btn_rec.clicked.connect(self._toggle_record)
        self.cb_fmt = W.QComboBox()
        self.cb_fmt.addItems(["wav", "txt"])  # wav is the default
        self.cb_fmt.setToolTip("Recording file format. wav = 32-bit float, sample values in volts")
        b = W.QPushButton("Folder…")
        b.setToolTip("Choose the folder recordings are saved to")
        b.clicked.connect(self._choose_dir)
        row.addWidget(self.btn_rec, 1)
        row.addWidget(self.cb_fmt)
        row.addWidget(b)
        full(f, 7, row)
        self.lb_rec = note()
        self._show_save_dir()
        full(f, 8, self.lb_rec)
        self.ai_widgets = [self.cb_ai, self.cb_term, self.cb_range, self.sp_aifs]
        left.addStretch(1)
        # let inputs shrink so two fit per row inside the fixed-width column
        for cb in panel.findChildren(W.QComboBox):
            cb.setSizeAdjustPolicy(
                W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(3)
        for sp in panel.findChildren(W.QAbstractSpinBox):
            sp.setMinimumWidth(68)
        # make the column exactly as wide as its contents need with this
        # system's widget style and font, so nothing is cut off on any OS
        scroll.setFixedWidth(panel.minimumSizeHint().width() + 4)

        # --- plots ---
        # top row: transmit signal, bottom row: hydrophone; left: time, right: spectrum
        plots = W.QGridLayout()
        root.addLayout(plots, 1)

        def make_plot(title, xlabel, xunit, ylabel, yunit, color, row, col):
            pw = pg.PlotWidget(title=title)
            pw.setLabel("bottom", xlabel, units=xunit)
            pw.setLabel("left", ylabel, units=yunit)
            pw.showGrid(x=True, y=True, alpha=0.3)
            plots.addWidget(pw, 2 * row, col)  # grid row 1 holds the level readout
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
        # the hydrophone spectrum shares its cell with the spectrogram view
        plots.removeWidget(self.plot_f)
        self.sg_stack = W.QStackedWidget()
        self.sg_stack.addWidget(self.plot_f)
        self.plot_sg = pg.PlotWidget()
        self.plot_sg.setLabel("bottom", "frequency", units="Hz")
        self.plot_sg.setLabel("left", "time", units="s")
        self.sg_img = pg.ImageItem(axisOrder="row-major")  # image[row = time, col = freq]
        self.sg_img.setAutoDownsample(True)
        self.plot_sg.addItem(self.sg_img)
        self.sg_bar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"), interactive=False)
        self.sg_bar.setImageItem(self.sg_img, insert_in=self.plot_sg.getPlotItem())
        self.sg_stack.addWidget(self.plot_sg)
        plots.addWidget(self.sg_stack, 2, 1)
        # received level (SPL) readout: a strip between the two rows of plots
        self.lb_spl = W.QLabel()
        font = self.lb_spl.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.lb_spl.setFont(font)
        self.lb_spl.setStyleSheet(
            "background: #101820; color: #ffd166; padding: 5px 8px; border-radius: 3px;")
        self.lb_spl.setToolTip(
            "Received level, independent of the FFT size.\n"
            "With a Sensor: sound pressure level (SPL) in dB re 1 µPa rms over the "
            "calibrated band, using the sensitivity curve at every frequency.\n"
            "Without a Sensor: voltage level in dB re 1 V rms (all frequencies above 0 Hz).\n"
            "'now' is the most recent FFT-size block; 'last transmit' is the highest "
            "level averaged over one transmit duration.")
        self.lb_spl.setWordWrap(True)  # wrap instead of forcing the window wider
        plots.addWidget(self.lb_spl, 1, 0, 1, 2)
        plots.setRowStretch(0, 1)
        plots.setRowStretch(2, 1)
        self.sg = None  # spectrogram state, created when monitoring starts
        self.lv = None  # received-level state, created when monitoring starts
        self._set_log_freq(False)  # fix the frequency axis to 0..SPECTRUM_FMAX
        self._load_hydrophones()
        self._draw_level()  # show the readout strip (with its hint) from the start
        # wide window: use most of the screen width (the plots take the extra room)
        width, height = 1500, 760
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = max(1150, min(1800, int(avail.width() * 0.96)))
            height = min(height, avail.height() - 40)
        self.resize(width, height)

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
            self._update_tx_info()  # refreshes the peak read-out and the preview

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
        self._begin_shot_level(self.tx_info["sig"])
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
        if self.lv is not None and self.lv["blocks"] is not None:
            self.lv["tail"] = self.lv["blocks"].maxlen  # catch a shot still in the buffer
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
        # must hold the time plot, one FFT and one spectrogram row, plus half a
        # second so spectrogram rows can be taken slightly in the past
        nfft = self.cb_nfft.currentData()
        n = max(int(self.sp_win.value() * self.ai_fs), nfft,
                nfft * self._spectrogram_step()[1])
        self.ring = np.zeros(n + int(0.5 * self.ai_fs))
        self.ring_fill = 0
        self._reset_spectrogram()

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
        if self.ck_sg.isChecked():
            self._draw_spectrogram()
        elif self.ring_fill >= nfft:
            # one transform of the most recent nfft samples
            f, spec = fourier_spectrum(self.ring[-nfft:], fs)
            df, f, db = f[1], f[1:], spectrum_db(spec[1:])
            cal = self.cb_hyd.currentData()
            if cal is not None:
                # volts -> pressure: |P(f)| = |S(f)| / sensitivity, only where calibrated
                keep = (f >= cal[0][0]) & (f <= cal[0][-1])
                f = f[keep]
                db = db[keep] - np.interp(f, cal[0], cal[1]) - self.sp_gain.value()
            if f.size and np.any(spec):  # an all-zero block has no spectrum to draw
                self.curve_f.setData(*minmax_decimate(f[0], df, db))
            else:
                self.curve_f.setData([], [])

    # ---------------- spectrogram ----------------
    def _set_spectrogram_view(self, on):
        self.sg_stack.setCurrentIndex(1 if on else 0)
        self.ck_logf.setEnabled(not on)  # the image uses a linear frequency axis
        if on and self.sg is not None:
            self.sg["dirty"] = True

    def _spectrogram_step(self):
        """(samples between rows, FFT segments averaged per row)."""
        fs, nfft = self.ai_fs, self.cb_nfft.currentData()
        dt = max(SPECTROGRAM_MIN_DT, self.sp_sgspan.value() / SPECTROGRAM_MAX_ROWS)
        hop = max(nfft // 2, int(fs * dt))
        return hop, max(1, hop // nfft)

    def _reset_spectrogram(self, *_):
        """(Re)create the rolling image for the current rate / FFT size / sensor."""
        cal = self.cb_hyd.currentData()
        unit = "dB re 1 V²/Hz" if cal is None else "dB re 1 µPa²/Hz"
        self.plot_sg.setTitle("Hydrophone spectrogram" if cal is None
                              else "Hydrophone spectrogram — %s" % self.cb_hyd.currentText())
        self.sg_bar.axis.setLabel("PSD (%s)" % unit)
        self._reset_level()
        if not self.ai_running:
            return
        fs, nfft = self.ai_fs, self.cb_nfft.currentData()
        hop, navg = self._spectrogram_step()
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        ncol = max(2, int(np.searchsorted(f, SPECTRUM_FMAX, side="right")))
        nrow = max(2, int(round(self.sp_sgspan.value() * fs / hop)))
        win = np.hanning(nfft)
        offset = np.zeros(ncol)  # dB subtracted from every row
        if cal is not None:
            fc = f[:ncol]
            inside = (fc >= cal[0][0]) & (fc <= cal[0][-1])
            offset = np.where(inside, np.interp(fc, cal[0], cal[1]) + self.sp_gain.value(),
                              np.inf)  # outside the calibrated band: no data
        self.sg = {
            "hop": hop, "navg": navg, "nfft": nfft, "win": win, "ncol": ncol,
            "offset": offset,
            # one-sided power spectral density of a Hann-windowed segment
            "scale": 2.0 / (fs * np.sum(win ** 2)),
            "pending": 0, "dirty": True,
            "img": np.full((nrow, ncol), SPECTROGRAM_FLOOR, dtype=np.float32),
        }
        span = nrow * hop / fs
        self.sg_img.setImage(self.sg["img"], autoLevels=False)
        self.sg_img.setRect(QtCore.QRectF(0.0, -span, ncol * f[1], span))
        self.plot_sg.setXRange(0, min(SPECTRUM_FMAX, ncol * f[1]), padding=0)
        self.plot_sg.setYRange(-span, 0, padding=0)

    def _feed_spectrogram(self, n_new):
        """Add one row per `hop` new samples; rows are Hann-windowed PSDs in dB."""
        sg = self.sg
        if sg is None:
            return
        sg["pending"] += n_new
        nfft, hop, navg, size = sg["nfft"], sg["hop"], sg["navg"], self.ring.size
        while sg["pending"] >= hop:
            end = size - (sg["pending"] - hop)  # this row's segment ends here
            sg["pending"] -= hop
            start = end - navg * nfft
            if start < size - self.ring_fill or start < 0:
                continue  # not enough samples yet (or fell too far behind)
            # when rows are further apart than one FFT, average the segments in
            # between so that nothing between two rows is skipped
            segs = self.ring[start:end].reshape(navg, nfft) * sg["win"]
            spec = np.abs(np.fft.rfft(segs, axis=1)[:, : sg["ncol"]]) ** 2
            psd = sg["scale"] * spec.mean(axis=0)
            row = 10 * np.log10(psd + 1e-300) - sg["offset"]
            img = sg["img"]
            img[:-1] = img[1:]  # scroll: oldest row at the bottom, newest at the top
            img[-1] = np.maximum(row, SPECTROGRAM_FLOOR)
            sg["dirty"] = True

    # ---------------- received level (SPL) ----------------
    def _reset_level(self):
        """(Re)create the level meter for the current rate / FFT size / sensor."""
        self.lv = None
        if not self.ai_running:
            self._draw_level()
            return
        fs, nfft = self.ai_fs, self.cb_nfft.currentData()
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        cal = self.cb_hyd.currentData()
        if cal is None:
            weight = np.ones(f.size)
            weight[0] = 0.0  # ignore the DC offset
            name, unit = "Level", "re 1 V"
        else:
            inside = (f >= cal[0][0]) & (f <= cal[0][-1])
            # sensitivity in V per µPa at every bin (curve plus external gain)
            sens = 10 ** ((np.interp(f, cal[0], cal[1]) + self.sp_gain.value()) / 20)
            weight = np.where(inside, 1.0 / sens ** 2, 0.0)
            name = "SPL"
            unit = "re 1 µPa, %.0f–%.0f kHz" % (cal[0][0] / 1e3, cal[0][-1] / 1e3)
        # Parseval: mean square of a block = (2 / N^2) * sum over positive bins |X_k|^2
        weight = weight * 2.0 / nfft ** 2
        if nfft % 2 == 0:
            weight[-1] /= 2.0  # the Nyquist bin is not doubled
        self.lv = {"nfft": nfft, "weight": weight, "block_s": nfft / fs, "pending": 0,
                   "name": name, "unit": unit, "now": None,
                   "blocks": None, "tail": None, "shot_e": None, "shot_dur": None}
        self._draw_level()

    def _begin_shot_level(self, duration_s):
        """Start looking for the highest level averaged over one transmit duration."""
        lv = self.lv
        if lv is None:
            return
        # enough whole blocks to be sure one complete shot fits inside
        n = int(math.ceil(duration_s / lv["block_s"])) + 1
        lv.update(blocks=collections.deque(maxlen=n), tail=None, shot_e=0.0,
                  shot_dur=duration_s)

    def _feed_level(self, n_new):
        lv = self.lv
        if lv is None:
            return
        lv["pending"] += n_new
        nfft, size = lv["nfft"], self.ring.size
        while lv["pending"] >= nfft:
            end = size - (lv["pending"] - nfft)
            lv["pending"] -= nfft
            start = end - nfft
            if start < size - self.ring_fill or start < 0:
                continue
            mean_square = float(np.sum(
                lv["weight"] * np.abs(np.fft.rfft(self.ring[start:end])) ** 2))
            lv["now"] = mean_square
            if lv["blocks"] is not None:
                lv["blocks"].append(mean_square * lv["block_s"])  # energy of this block
                lv["shot_e"] = max(lv["shot_e"], sum(lv["blocks"]))
                if lv["tail"] is not None:  # transmit finished: a few more blocks, then stop
                    lv["tail"] -= 1
                    if lv["tail"] <= 0:
                        lv["blocks"] = None

    def _draw_level(self):
        lv = self.lv
        sensor = self.cb_hyd.currentData() is not None
        head = "SPL (%s)" % self.cb_hyd.currentText() if sensor else "Received level"
        hint = "" if sensor else "   ·   select a Sensor for SPL in dB re 1 µPa"
        if lv is None:
            self.lb_spl.setText("%s — press Start monitoring%s" % (head, hint))
            return

        def db(v):
            return "%.1f dB" % (10 * np.log10(v)) if v and v > 0 else "no signal"

        if lv["shot_dur"] is None:
            shot = "—"
        else:
            shot = "%s (%.3f s avg)" % (db(lv["shot_e"] / lv["shot_dur"]), lv["shot_dur"])
        self.lb_spl.setText("%s — now: %s  ·  last transmit: %s  [%s]%s" % (
            head, db(lv["now"]), shot, lv["unit"], hint))

    def _draw_spectrogram(self):
        sg = self.sg
        if sg is None or not sg["dirty"]:
            return
        sg["dirty"] = False
        top = float(sg["img"].max())
        self.sg_img.setImage(sg["img"], autoLevels=False)
        if top > SPECTROGRAM_FLOOR:
            self.sg_bar.setLevels((top - SPECTROGRAM_DYN_DB, top))

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
        default = self.cb_hyd.findText(DEFAULT_SENSOR)
        if default >= 0:
            self.cb_hyd.setCurrentIndex(default)
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
                    self._feed_spectrogram(x.size)
                    self._feed_level(x.size)
                    if self.recorder is not None:
                        self.recorder.put(x)
                        self.lb_rec.setText("Recording: %s — %.1f s" % (
                            os.path.basename(self.recorder.path),
                            self.recorder.n_samples / self.ai_fs))
                self._update_plots()
                self._draw_level()
        if self.tx_active:
            self._update_tx_progress()

    def _error(self, title, exc=None):
        text = title if exc is None else "%s\n\n%s" % (title, exc)
        self.statusBar().showMessage(text.replace("\n", " "))
        box = self._error_box(title, text, exc)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            box.exec()

    def _error_box(self, title, text, exc):
        """Build the error dialog and append the full details to the log file."""
        details = ""
        if exc is not None:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            settings = "device=%s ao=%s ai=%s terminal=%s range=%s rate=%s" % (
                self.cb_dev.currentText(), self.cb_ao.currentText(), self.cb_ai.currentText(),
                self.cb_term.currentText(), self.cb_range.currentText(), self.sp_aifs.value())
            details = "%s\n%s\n\n%s" % (title, settings, details)
            log = os.path.join(os.path.dirname(os.path.abspath(__file__)), ERROR_LOG)
            try:
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write("===== %s =====\n%s\n" % (
                        _dt.datetime.now().isoformat(timespec="seconds"), details))
                text += "\n\nFull details were saved to %s" % ERROR_LOG
            except OSError:
                pass
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("Error")
        box.setText(text)
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        if details:
            box.setDetailedText(details)
        return box

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

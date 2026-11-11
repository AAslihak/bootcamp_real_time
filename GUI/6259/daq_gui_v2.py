#!/usr/bin/env python3
"""WAV transmit / reference hydrophone monitor and recorder GUI (NI USB-6259).

AO : plays the wav at its native sample rate, scaled so that its largest
     sample equals the Peak setting (default 1 V); every shot is followed by a
     guard interval (GI) of silence. The GI can never be shorter than the
     transmit duration (GI >= 50% of the period).
AI : continuously acquires the reference hydrophone, shows voltage vs time
     and spectrum (in volts, or in pressure using a hydrophone sensitivity
     curve from hydrophone_rvr.csv), and can record to txt or wav.

Version 2.1 adds Distance: the hydrophone-to-source distance (default 0.30 m).
Pressure levels are referred back to 1 m assuming spherical spreading (1/r):
level at 1 m = measured level + 20 log10(distance / 1 m).

Without the NI-DAQmx driver or a device, the "Simulated" device lets you
try the interface (simulated AI = the AO signal itself, like a loopback cable).

Run:  python daq_gui.py
"""
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
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy import fft as sfft
from scipy.io import wavfile
from scipy.signal import resample_poly, welch

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogSingleChannelReader
    from nidaqmx.stream_writers import AnalogSingleChannelWriter

    NIDAQMX_ERROR = None
except Exception as exc:  # package or driver not installed
    nidaqmx = None
    NIDAQMX_ERROR = exc

VERSION = "2.1"  # shown in the window title; see GUI/CHANGELOG.md
SIM_NAME = "Simulated (no hardware)"
AO_VMAX = 10.0  # USB-6259 AO range is +/-10 V
MAX_AO_SAMPLES = 100_000_000  # whole sequence is written at once (~800 MB cap)
TICK_MS = 50
ERROR_LOG = "daq_gui_error.log"  # full error details are appended here, next to this file
DEFAULT_SENSOR = "Orca 1"  # hydrophone selected when the program starts
# Level alarm on "SL" = the maximum of the hydrophone PSD (dB re 1 µPa²/Hz):
# below SL_WARN_DB nothing happens, from SL_WARN_DB up to (not including)
# SL_ALARM_DB the window gets an orange frame, at SL_ALARM_DB and above a red one.
SL_WARN_DB = 156.0
SL_ALARM_DB = 159.0
SL_WARN_COLOR = "#ff9500"
SL_ALARM_COLOR = "#ff2d2d"
SL_FRAME_PX = 8
SL_LOG_MAX = 1000  # entries kept in the red-alarm list
NFFT_DEFAULT = 131072  # FFT length of the hydrophone PSD when the program starts (points)
NFFT_MAX = 4194304  # longest FFT accepted (2^22 points)
SG_NFFT_DEFAULT = 8192  # FFT length of each spectrogram segment at start (points)
SG_NFFT_MAX = 65536  # longest spectrogram segment accepted
SPECTRUM_XPAD = 1.03  # the frequency axis runs a little past SPECTRUM_FMAX so its label shows
PSD_YMIN_DEFAULT, PSD_YMAX_DEFAULT = 60.0, 160.0  # manual y range offered first (dB)
SL_PERIOD_S = 1.0  # SL (and the frame colour) is updated once per this many seconds
RVR_FILE = "hydrophone_rvr.csv"  # hydrophone sensitivity curves, next to this file
DEFAULT_DISTANCE_M = 0.30  # hydrophone-to-source distance when the program starts
REF_DISTANCE_M = 1.0  # pressure levels are referred to this distance (1/r spreading)
PRESSURE_UNIT = "dB re 1 µPa²/Hz @ 1 m"
TX_PSD_SEGMENT_S = 1.0  # segment length of the transmit PSD: 1 s gives 1 Hz resolution
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


def distance_correction_db(distance_m):
    """dB added to a level measured at distance_m to refer it to 1 m (1/r spreading)."""
    return 20.0 * math.log10(distance_m / REF_DISTANCE_M)


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

    ao_channels = ["ao0", "ao1", "ao2", "ao3"]
    ai_channels = ["ai%d" % i for i in range(16)]
    ao_max_rate = 2.86e6
    ai_max_rate = 1.25e6

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
def power_spectral_density(x, fs, nseg, nfft=None, hann=True):
    """One-sided power spectral density (V^2/Hz) by Welch's method.

    The record is cut into segments of nseg samples with 50% overlap; each
    periodogram is 2 |FFT(w x)|^2 / (fs sum(w^2)) and the segments are averaged.
    w is a Hann window, or all ones (no window) when hann is False.
    A record shorter than nseg is treated as one segment.
    nfft (>= segment length) zero-pads each segment, which sets the spacing of
    the frequency points to fs / nfft. The mean is not removed. Integrating
    the result over frequency gives the mean-square voltage. Returns (f, psd).
    """
    nseg = int(min(nseg, x.size))
    nfft = None if nfft is None else int(max(nfft, nseg))
    # np.hanning: the same window the spectrogram rows use
    window = np.hanning(nseg) if hann else np.ones(nseg)
    return welch(x, fs=fs, window=window, noverlap=nseg // 2, nfft=nfft,
                 detrend=False, scaling="density")


def record_psd(x, fs, win, win_power):
    """One-sided PSD (V^2/Hz) of one whole record: a single Hann-windowed periodogram.

    psd = 2 |FFT(win * x)|^2 / (fs * sum(win^2)); the 0 Hz and Nyquist bins are
    not doubled. Frequency resolution is fs / len(x), i.e. 1 / record duration.
    Returns (f, psd).
    """
    psd = np.abs(sfft.rfft(x * win)) ** 2 * (2.0 / (fs * win_power))
    psd[0] *= 0.5
    if x.size % 2 == 0:
        psd[-1] *= 0.5
    return sfft.rfftfreq(x.size, 1.0 / fs), psd


def psd_db(psd):
    """10 log10 of a PSD, with a 160 dB display floor below the peak."""
    peak = float(psd.max()) if psd.size else 0.0
    return 10 * np.log10(np.maximum(psd, max(peak * 1e-16, 1e-300)))


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


class FrameWidget(QtWidgets.QWidget):
    """Central widget that can draw a coloured frame around the whole window.

    The frame is painted directly. (A style sheet would also work, but changing
    a style sheet makes Qt re-style every child and lose the smaller font of
    the control column.)
    """

    def __init__(self):
        super().__init__()
        self._color = None

    def set_frame(self, color):
        """color: e.g. "#ff2d2d", or None for no frame."""
        if color != self._color:
            self._color = color
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._color is None:
            return
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(self._color))
        pen.setWidth(SL_FRAME_PX)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        half = SL_FRAME_PX / 2.0
        painter.drawRect(QtCore.QRectF(self.rect()).adjusted(half, half, -half, -half))
        painter.end()


class LengthBox(QtCore.QObject):
    """An FFT length entered either as a number of points ("nfft") or in seconds.

    `spin` and `unit` are the two widgets; `points()` is the length in samples.
    Switching the unit converts the number so that the length stays the same.
    """

    changed = QtCore.Signal()

    def __init__(self, default_points, max_points, rate_fn, parent=None):
        super().__init__(parent)
        self._rate = rate_fn  # callable returning the sample rate used for conversion
        self._max = int(max_points)
        self._unit = "nfft"
        self.spin = QtWidgets.QDoubleSpinBox()
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["nfft", "s"])  # nfft is the default unit
        self._configure(default_points)
        self.spin.valueChanged.connect(lambda *_: self.changed.emit())
        self.unit.currentTextChanged.connect(self._on_unit)

    def _configure(self, value):
        self.spin.blockSignals(True)
        if self._unit == "nfft":
            self.spin.setDecimals(0)
            self.spin.setRange(16, self._max)
            self.spin.setSingleStep(1024)
        else:
            self.spin.setDecimals(4)
            self.spin.setRange(0.0001, 60.0)
            self.spin.setSingleStep(0.1)
        self.spin.setValue(value)
        self.spin.blockSignals(False)

    def _on_unit(self, unit):
        if unit == self._unit:
            return
        fs = self._rate()
        value = self.spin.value() / fs if unit == "s" else round(self.spin.value() * fs)
        self._unit = unit
        self._configure(value)
        self.changed.emit()

    def points(self):
        if self._unit == "nfft":
            n = int(round(self.spin.value()))  # exactly what was entered
        else:
            n = int(round(self.spin.value() * self._rate()))
            n = sfft.next_fast_len(max(n, 16), real=True)  # a nearby length the FFT handles quickly
        return int(min(max(n, 16), self._max))

    def set_seconds(self, seconds):
        """Switch the unit to seconds and set the length."""
        self._unit = "s"
        self.unit.blockSignals(True)
        self.unit.setCurrentText("s")
        self.unit.blockSignals(False)
        self._configure(seconds)
        self.changed.emit()

    def set_enabled(self, on):
        self.spin.setEnabled(on)
        self.unit.setEnabled(on)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "WAV Transmit / Reference Hydrophone (NI USB-6259) — v%s" % VERSION)
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
        central = FrameWidget()
        self.central = central
        self._psd_win = None  # cached Hann window for the current record length
        self._psd_next = 0.0  # earliest time for the next hydrophone PSD
        self.sl_now = None  # displayed SL: (peak of the hydrophone PSD in dB, its frequency in Hz)
        self.sl_best = None  # highest PSD peak seen so far in the current SL period
        self.sl_t0 = 0.0  # start of the current SL period
        self.sl_state = None
        self.setCentralWidget(central)
        root = W.QHBoxLayout(central)
        root.setContentsMargins(SL_FRAME_PX + 4, SL_FRAME_PX + 4, SL_FRAME_PX + 4, SL_FRAME_PX + 4)
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
             "Input range: the largest voltage the input can measure. Pick the smallest "
             "range that still covers the signal; anything beyond it is clipped",
             "Sample rate you request; after Start monitoring it shows the rate the "
             "device actually uses")
        self.btn_mon = W.QPushButton("Start monitoring")
        self.btn_mon.clicked.connect(self._toggle_monitor)
        full(f, 2, self.btn_mon)
        # FFT length of the hydrophone PSD (= length of the time plot), entered
        # either as a number of points (nfft) or in seconds
        self.fft_len = LengthBox(NFFT_DEFAULT, NFFT_MAX, self._fft_rate, self)
        self.sp_win, self.cb_winunit = self.fft_len.spin, self.fft_len.unit
        self.fft_len.changed.connect(self._resize_ring)
        tip = ("FFT length of the hydrophone PSD; the time plot shows exactly this record.\n"
               "nfft: number of points.  s: seconds (points = seconds x Rate).\n"
               "Frequency resolution = Rate / nfft = 1 / seconds.")
        lb = W.QLabel("FFT length")
        for wdg in (lb, self.sp_win, self.cb_winunit):
            wdg.setToolTip(tip)
        row = W.QHBoxLayout()
        row.addWidget(lb)
        row.addWidget(self.sp_win, 1)
        row.addWidget(self.cb_winunit)
        full(f, 3, row)
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
        # distance between the hydrophone and the source: levels are referred to 1 m
        self.sp_dist = W.QDoubleSpinBox()
        self.sp_dist.setRange(0.01, 10000.0)
        self.sp_dist.setDecimals(2)
        self.sp_dist.setSingleStep(0.05)
        self.sp_dist.setValue(DEFAULT_DISTANCE_M)
        self.sp_dist.setSuffix(" m")
        self.sp_dist.valueChanged.connect(self._on_distance_changed)
        self.lb_dist = W.QLabel()
        pair(f, 5, "Distance", self.sp_dist, "→ 1 m", self.lb_dist,
             "Distance between the hydrophone and the source. The pressure spectrum and the\n"
             "Acoustic level are referred back to 1 m assuming spherical spreading (1/r):\n"
             "level at 1 m = measured level + 20 log10(Distance / 1 m).\n"
             "Set 1 m to show the level as measured at the hydrophone.",
             "Correction applied to the measured level to refer it to 1 m")
        # y range of the hydrophone PSD plot (and colour scale of the spectrogram)
        self.ck_yauto = W.QCheckBox("Auto")
        self.ck_yauto.setChecked(True)
        self.sp_ymin = W.QDoubleSpinBox()
        self.sp_ymax = W.QDoubleSpinBox()
        for sp, val in ((self.sp_ymin, PSD_YMIN_DEFAULT), (self.sp_ymax, PSD_YMAX_DEFAULT)):
            sp.setRange(-400.0, 400.0)
            sp.setDecimals(0)
            sp.setSingleStep(5.0)
            sp.setValue(val)
            sp.setEnabled(False)
            sp.valueChanged.connect(self._apply_yrange)
        self.ck_yauto.toggled.connect(self._apply_yrange)
        tip = ("Vertical range of the hydrophone PSD plot in dB (also the colour scale of the "
               "spectrogram).\nAuto: fit the data. Untick to enter the minimum and maximum.")
        lb = W.QLabel("PSD y")
        for wdg in (lb, self.sp_ymin, self.sp_ymax, self.ck_yauto):
            wdg.setToolTip(tip)
        row = W.QHBoxLayout()
        row.addWidget(lb)
        row.addWidget(self.sp_ymin, 1)
        row.addWidget(W.QLabel("to"))
        row.addWidget(self.sp_ymax, 1)
        row.addWidget(self.ck_yauto)
        full(f, 6, row)
        self.ck_logf = W.QCheckBox("Log freq")
        self.ck_logf.setToolTip("Logarithmic frequency axis for both spectra")
        self.ck_logf.toggled.connect(self._set_log_freq)
        self.ck_sg = W.QCheckBox("Spectrogram")
        self.ck_sg.setToolTip(
            "Show the hydrophone PSD over time instead of the single PSD:\n"
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
        full(f, 7, row)
        # FFT length of each spectrogram segment: same kind of input as "FFT length";
        # it can only be changed while the Spectrogram box is ticked
        self.sg_len = LengthBox(SG_NFFT_DEFAULT, SG_NFFT_MAX, self._fft_rate, self)
        self.sp_sgfft, self.cb_sgunit = self.sg_len.spin, self.sg_len.unit
        self.sg_len.changed.connect(self._resize_ring)
        self.sg_len.set_enabled(False)
        tip = ("FFT length of each spectrogram segment, in points (nfft) or seconds.\n"
               "Can be changed only while 'Spectrogram' is ticked.\n"
               "Shorter = finer time resolution, longer = finer frequency resolution.")
        lb = W.QLabel("Sg FFT")
        for wdg in (lb, self.sp_sgfft, self.cb_sgunit):
            wdg.setToolTip(tip)
        row = W.QHBoxLayout()
        row.addWidget(lb)
        row.addWidget(self.sp_sgfft, 1)
        row.addWidget(self.cb_sgunit)
        full(f, 8, row)
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
        full(f, 9, row)
        self.lb_rec = note()
        self._show_save_dir()
        full(f, 10, self.lb_rec)
        self.ai_widgets = [self.cb_ai, self.cb_term, self.cb_range, self.sp_aifs]
        # --- red alarm log: every SL above the red threshold, newest first ---
        g = W.QGroupBox("Red alarms (Acoustic level ≥ %g dB)" % SL_ALARM_DB)
        lay = W.QVBoxLayout(g)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(3)
        self.lw_alarm = W.QListWidget()
        self.lw_alarm.setMinimumHeight(2 * line + 6)
        # take whatever height is left in the column instead of asking for more
        self.lw_alarm.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Ignored)
        self.lw_alarm.setToolTip(
            "One line each time the Acoustic level shown above is in the red range: "
            "time, level (dB re 1 µPa²/Hz @ 1 m) and frequency. Newest at the top.")
        lay.addWidget(self.lw_alarm, 1)
        row = W.QHBoxLayout()
        self.lb_alarm = W.QLabel("0 alarms")
        b = W.QPushButton("Clear")
        b.clicked.connect(self._clear_alarms)
        row.addWidget(self.lb_alarm, 1)
        row.addWidget(b)
        lay.addLayout(row)
        left.addWidget(g, 1)
        self.n_alarms = 0
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

        psd = "PSD (dB re 1 V²/Hz)"
        self.plot_txt, self.curve_txt = make_plot(
            TX_TITLE, "time", "s", "voltage", "V", "#3aa655", 0, 0)
        self.gi_region = pg.LinearRegionItem(movable=False, brush=(255, 255, 255, 30))
        self.gi_region.setVisible(False)
        self.plot_txt.addItem(self.gi_region)
        self.plot_txf, self.curve_txf = make_plot(
            "Transmit signal — PSD", "frequency", "Hz", psd, None, "#c9a227", 0, 1)
        self.plot_t, self.curve_t = make_plot(
            "Reference hydrophone — voltage vs time", "time", "s", "voltage", "V",
            "#2a78d6", 1, 0)
        self.plot_f, self.curve_f = make_plot(
            "Hydrophone PSD", "frequency", "Hz", psd, None, "#d6542a", 1, 1)
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
        # SL read-out: a strip between the two rows of plots
        self.lb_spl = W.QLabel()
        font = self.lb_spl.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.lb_spl.setFont(font)
        self.lb_spl.setStyleSheet(
            "background: #101820; color: #ffd166; padding: 5px 8px; border-radius: 3px;")
        self.lb_spl.setToolTip(
            "Acoustic level: the maximum of the hydrophone PSD (the lower-right plot) and the "
            "frequency where it occurs, in dB re 1 µPa²/Hz with a Sensor selected,\n"
            "referred to 1 m from the source (see Distance).\n"
            "It is updated once per second and shows the highest value reached "
            "during the previous second.\n"
            "Frame of the window: none below %g dB, orange from %g dB, red from %g dB "
            "(only with a Sensor selected)." % (
                SL_WARN_DB, SL_WARN_DB, SL_ALARM_DB))
        self.lb_spl.setWordWrap(True)  # wrap instead of forcing the window wider
        plots.addWidget(self.lb_spl, 1, 0, 1, 2)
        plots.setRowStretch(0, 1)
        plots.setRowStretch(2, 1)
        self.sg = None  # spectrogram state, created when monitoring starts
        self._set_log_freq(False)  # fix the frequency axis to 0..SPECTRUM_FMAX
        self._load_hydrophones()
        self._draw_level()  # show the readout strip (with its hint) from the start
        # keep the "nfft = seconds, Δf" note in the PSD title up to date
        self.fft_len.changed.connect(self._on_hydrophone_changed)
        self.sp_aifs.valueChanged.connect(self._on_hydrophone_changed)
        # wide window: use most of the screen width (the plots take the extra room)
        width, height = 1500, 860
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = max(1150, min(1800, int(avail.width() * 0.96)))
            height = min(height, avail.height() - 30)
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
        wav_len = data.shape[0] / fs
        self.sp_dur.setValue(max(self.sp_dur.minimum(), wav_len))
        # guard interval: defaults to the WAV length and can never be shorter than it
        self._on_duration_changed(self.sp_dur.value())
        self.sp_gi.setValue(wav_len)
        # hydrophone sample rate: defaults to twice the WAV sample rate
        if self.ai_running:
            self.statusBar().showMessage(
                "Rate left at %.0f Hz: stop monitoring to change it (2 x WAV rate = %.0f Hz)"
                % (self.ai_fs, 2.0 * fs))
        else:
            self.sp_aifs.setValue(2.0 * fs)  # limited to the device maximum by the box itself
        # FFT length of the hydrophone PSD: defaults to the length of the transmitted
        # signal, in seconds so that it does not depend on the sample rate
        self.fft_len.set_seconds(self.sp_dur.value())
        self._update_tx_info()

    def _on_wav_channel(self, i):
        if self.wav_all is not None and i >= 0:
            self.wav = self.wav_all[:, i]
            self._update_tx_info()  # refreshes the peak read-out and the preview

    def _on_duration_changed(self, dur):
        # GI minimum = transmit duration (at least 50% of each period is silence),
        # and never less than the WAV length; it may be longer, never shorter
        wav_len = self.wav_all.shape[0] / self.wav_fs if self.wav_all is not None else 0.0
        self.sp_gi.setMinimum(max(dur, wav_len))
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
        # Welch PSD of the whole shot with 1 Hz frequency resolution: segments of
        # 1 second (fs points), 50% overlap, NO window (a window would taper the two
        # ends of a short pulse, i.e. the two ends of the band of a swept signal).
        # A shot shorter than 1 s is a single segment, zero-padded to fs points so
        # the points are still 1 Hz apart.
        nseg = int(round(fs * TX_PSD_SEGMENT_S))
        f, spec = power_spectral_density(shot, fs, nseg, nfft=nseg, hann=False)
        self.curve_txf.setData(*minmax_decimate(f[1], f[1], psd_db(spec[1:])))

    def _set_log_freq(self, on):
        for pw in (self.plot_f, self.plot_txf):
            pw.setLogMode(x=on, y=False)
            self._style_freq_axis(pw, on)

    @staticmethod
    def _style_freq_axis(pw, log=False, fmax=SPECTRUM_FMAX):
        """0..fmax with a label every 2 kHz, drawn slightly past fmax so its label is visible."""
        axis = pw.getAxis("bottom")
        if log:  # in log mode the view range is in log10(Hz)
            axis.setTickSpacing()
            pw.setXRange(np.log10(SPECTRUM_FMIN_LOG), np.log10(fmax * SPECTRUM_XPAD), padding=0)
        else:
            axis.setTickSpacing(major=2000.0, minor=1000.0)
            pw.setXRange(0, fmax * SPECTRUM_XPAD, padding=0)

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
        # the device may adjust the requested rate slightly: show what it really uses
        self.sp_aifs.blockSignals(True)
        self.sp_aifs.setValue(self.ai_fs)
        self.sp_aifs.blockSignals(False)
        self.statusBar().showMessage("Actual AI sample rate %.3f Hz" % self.ai_fs)
        self.sl_now = self.sl_best = None
        self.sl_t0 = time.perf_counter()
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
        self._draw_level()  # clears the SL read-out and the coloured frame
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
        # must hold the FFT record and one spectrogram row, plus half a
        # second so spectrogram rows can be taken slightly in the past
        nfft = self.sg_len.points()
        n = max(self._record_len(), nfft, nfft * self._spectrogram_step()[1])
        self.ring = np.zeros(n + int(0.5 * self.ai_fs))
        self.ring_fill = 0
        self._reset_spectrogram()
        self._on_hydrophone_changed()  # title shows the FFT length in points and seconds

    def _push(self, x):
        n = self.ring.size
        if x.size >= n:
            self.ring[:] = x[-n:]
        else:
            self.ring[: n - x.size] = self.ring[x.size :]
            self.ring[n - x.size :] = x
        self.ring_fill = min(n, self.ring_fill + x.size)

    def _fft_rate(self):
        """Sample rate used to convert between points and seconds."""
        return self.ai_fs if self.ai_running else self.sp_aifs.value()

    def _record_len(self):
        """Points in the FFT of the hydrophone PSD; the time plot shows the same record."""
        return self.fft_len.points()

    def _update_plots(self):
        if self.ring_fill < 2:
            return
        fs = self.ai_fs
        n_rec = self._record_len()
        n_win = min(self.ring_fill, n_rec)
        y = self.ring[-n_win:]
        t, yd = minmax_decimate(-n_win / fs, 1.0 / fs, y)
        self.curve_t.setData(t, yd)
        now = time.perf_counter()
        if self.ring_fill >= n_rec and now >= self._psd_next:
            # PSD of exactly the record shown in the time plot (one FFT of FFT-length points)
            if self._psd_win is None or self._psd_win[0].size != n_rec:
                win = np.hanning(n_rec)
                self._psd_win = (win, float(np.sum(win ** 2)))
            f, spec = record_psd(self.ring[-n_rec:], fs, *self._psd_win)
            df, f, db = f[1], f[1:], psd_db(spec[1:])
            cal = self.cb_hyd.currentData()
            if cal is not None:
                # V^2/Hz -> µPa^2/Hz: divide by the sensitivity squared, only where calibrated
                keep = (f >= cal[0][0]) & (f <= cal[0][-1])
                f = f[keep]
                db = (db[keep] - np.interp(f, cal[0], cal[1]) - self.sp_gain.value()
                      + distance_correction_db(self.sp_dist.value()))  # referred to 1 m
            has_signal = bool(f.size) and bool(np.any(spec))  # all-zero record: nothing to show
            if has_signal:
                i = int(np.argmax(db))
                if self.sl_best is None or db[i] > self.sl_best[0]:
                    self.sl_best = (float(db[i]), float(f[i]))  # highest peak this period
            if not self.ck_sg.isChecked():
                if has_signal:
                    self.curve_f.setData(*minmax_decimate(f[0], df, db))
                else:
                    self.curve_f.setData([], [])
            # a long FFT takes a while: leave time for the rest of the program
            spent = time.perf_counter() - now
            self._psd_next = time.perf_counter() + 2.0 * spent
        if self.ck_sg.isChecked():
            self._draw_spectrogram()
        # once per SL_PERIOD_S: publish the highest PSD peak of the period as SL
        if time.perf_counter() - self.sl_t0 >= SL_PERIOD_S:
            self.sl_now, self.sl_best = self.sl_best, None
            self.sl_t0 = time.perf_counter()
            self._log_alarm()

    # ---------------- spectrogram ----------------
    def _set_spectrogram_view(self, on):
        self.sg_stack.setCurrentIndex(1 if on else 0)
        self.ck_logf.setEnabled(not on)  # the image uses a linear frequency axis
        self.sg_len.set_enabled(on)  # the spectrogram FFT can only be changed while it is shown
        if on and self.sg is not None:
            self.sg["dirty"] = True

    def _spectrogram_step(self):
        """(samples between rows, FFT segments averaged per row)."""
        fs, nfft = self.ai_fs, self.sg_len.points()
        dt = max(SPECTROGRAM_MIN_DT, self.sp_sgspan.value() / SPECTROGRAM_MAX_ROWS)
        hop = max(nfft // 2, int(fs * dt))
        return hop, max(1, hop // nfft)

    def _reset_spectrogram(self, *_):
        """(Re)create the rolling image for the current rate / FFT size / sensor."""
        cal = self.cb_hyd.currentData()
        unit = "dB re 1 V²/Hz" if cal is None else PRESSURE_UNIT
        self.plot_sg.setTitle("Hydrophone spectrogram" if cal is None
                              else "Hydrophone spectrogram — %s" % self.cb_hyd.currentText())
        self.sg_bar.axis.setLabel("PSD (%s)" % unit)
        self._draw_level()
        if not self.ai_running:
            return
        fs, nfft = self.ai_fs, self.sg_len.points()
        hop, navg = self._spectrogram_step()
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        ncol = max(2, int(np.searchsorted(f, SPECTRUM_FMAX, side="right")))
        nrow = max(2, int(round(self.sp_sgspan.value() * fs / hop)))
        win = np.hanning(nfft)
        offset = np.zeros(ncol)  # dB subtracted from every row
        if cal is not None:
            fc = f[:ncol]
            inside = (fc >= cal[0][0]) & (fc <= cal[0][-1])
            offset = np.where(inside, np.interp(fc, cal[0], cal[1]) + self.sp_gain.value()
                              - distance_correction_db(self.sp_dist.value()),
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
        self._style_freq_axis(self.plot_sg, False, min(SPECTRUM_FMAX, ncol * f[1]))
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

    # ---------------- SL read-out ----------------
    def _draw_level(self):
        """Show SL = the maximum of the hydrophone PSD and the frequency where it occurs."""
        sensor = self.cb_hyd.currentData() is not None
        head = "Acoustic level (%s)" % self.cb_hyd.currentText() if sensor else "Peak (no Sensor)"
        hint = "" if sensor else "   ·   select a Sensor for Acoustic level in " + PRESSURE_UNIT
        if not self.ai_running:
            self.sl_now = self.sl_best = None
        self._set_sl_frame()
        if not self.ai_running:
            now = "press Start monitoring"
        elif self.sl_now is None:
            now = "now: no signal"
        else:
            unit = PRESSURE_UNIT if sensor else "dB re 1 V²/Hz"
            now = "now: %.1f %s, at %.0f Hz" % (self.sl_now[0], unit, self.sl_now[1])
        self.lb_spl.setText("%s — %s%s" % (head, now, hint))

    def _log_alarm(self):
        """Add the newly published SL to the list if it is in the red range."""
        if (self.sl_now is None or self.cb_hyd.currentData() is None
                or self.sl_now[0] < SL_ALARM_DB):
            return
        self.n_alarms += 1
        self.lw_alarm.insertItem(0, "%s   %.1f dB re 1 µPa²/Hz @ 1 m, %.0f Hz" % (
            _dt.datetime.now().strftime("%H:%M:%S"), self.sl_now[0], self.sl_now[1]))
        while self.lw_alarm.count() > SL_LOG_MAX:
            self.lw_alarm.takeItem(self.lw_alarm.count() - 1)
        self.lb_alarm.setText("%d alarm%s" % (self.n_alarms, "" if self.n_alarms == 1 else "s"))

    def _clear_alarms(self):
        self.lw_alarm.clear()
        self.n_alarms = 0
        self.lb_alarm.setText("0 alarms")

    def _set_sl_frame(self):
        """Colour the outer frame of the window according to the current SL."""
        state = None
        if self.sl_now is not None and self.cb_hyd.currentData() is not None:
            level = self.sl_now[0]
            if level >= SL_ALARM_DB:
                state = SL_ALARM_COLOR
            elif level >= SL_WARN_DB:
                state = SL_WARN_COLOR
        if state == self.sl_state:
            return
        self.sl_state = state
        self.central.set_frame(state)

    def _draw_spectrogram(self):
        sg = self.sg
        if sg is None or not sg["dirty"]:
            return
        sg["dirty"] = False
        top = float(sg["img"].max())
        self.sg_img.setImage(sg["img"], autoLevels=False)
        lo, hi = self.sp_ymin.value(), self.sp_ymax.value()
        if not self.ck_yauto.isChecked() and hi > lo:
            self.sg_bar.setLevels((lo, hi))
        elif top > SPECTROGRAM_FLOOR:
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

    def _on_distance_changed(self, *_):
        self.lb_dist.setText("%+.1f dB" % distance_correction_db(self.sp_dist.value()))
        self.sl_best = None  # peaks found with the old distance no longer apply
        self._reset_spectrogram()

    def _on_hydrophone_changed(self, *_):
        cal = self.cb_hyd.currentData()
        self.sp_gain.setEnabled(cal is not None)
        self.sp_dist.setEnabled(cal is not None)
        self.lb_dist.setText("%+.1f dB" % distance_correction_db(self.sp_dist.value()))
        n, fs = self._record_len(), self._fft_rate()
        fft = "nfft %d = %.4g s, Δf %.3g Hz" % (n, n / fs, fs / n)
        if cal is None:
            self.plot_f.setTitle("Hydrophone PSD — %s" % fft)
            self.plot_f.setLabel("left", "PSD (dB re 1 V²/Hz)")
        else:
            self.plot_f.setTitle("Hydrophone PSD — %s, %s" % (self.cb_hyd.currentText(), fft))
            self.plot_f.setLabel("left", "PSD (%s)" % PRESSURE_UNIT)
        self._apply_yrange()

    def _apply_yrange(self, *_):
        """Automatic or user-set vertical range of the hydrophone PSD (and spectrogram colours)."""
        auto = self.ck_yauto.isChecked()
        self.sp_ymin.setEnabled(not auto)
        self.sp_ymax.setEnabled(not auto)
        lo, hi = self.sp_ymin.value(), self.sp_ymax.value()
        if auto or hi <= lo:
            self.plot_f.enableAutoRange(axis="y")
        else:
            self.plot_f.setYRange(lo, hi, padding=0)
        if self.sg is not None:
            self.sg["dirty"] = True  # redraw the spectrogram with the new colour scale

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
        header.append("hydrophone: %s, gain_dB: %.1f, distance_m: %.2f (used for the on-screen "
                      "spectrum only; the recorded data are raw volts)" % (
                          self.cb_hyd.currentText(), self.sp_gain.value(),
                          self.sp_dist.value()))
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
        return True

    def _stop_record(self):
        rec, self.recorder, self.rec_auto = self.recorder, None, False
        if rec is None:
            return
        rec.close()
        self.btn_rec.setText("● Start recording")
        self.cb_fmt.setEnabled(True)
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

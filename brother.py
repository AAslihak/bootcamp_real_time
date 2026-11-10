"""
readIntfUI.py  (v3)

Real-time monitor and recorder for NI DAQ analog inputs.

Acquisition settings (need a restart, use Apply):
    device, channels, sample rate, input range, terminal configuration

Live settings (take effect immediately while running):
    view (time / spectrum / spectrogram), display window, refresh rate,
    y scale, FFT length, frequency limits, dynamic range, visible channels,
    live plot on/off, freeze

Recording (independent of acquisition):
    Record / Stop recording at any time while acquiring. Each recording gets a
    unique time-stamped name, so file names never need to be changed by hand.
    Data are written to disk as they arrive (not held in memory) and split into
    parts every N seconds. "Save last N s" exports a snapshot without
    interrupting acquisition or recording.

Device name "SIM" runs a simulated 3-15 kHz chirp source so the UI can be
tested without hardware.

WAV files are 32-bit float, in volts, readable with MATLAB audioread().
"""

import json
import os
import queue
import struct
import threading
import time
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal


APP_TITLE = "NI DAQ Multi-Channel Monitor v3.2"
RATE_PRESETS = ["44000", "50000", "100000", "200000", "500000", "1000000", "2000000"]
SAMPS_PRESETS = ["Auto", "500", "1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000"]
RANGE_PRESETS = ["0.2", "0.5", "1", "2", "5", "10"]
TERMINAL_CHOICES = ["DIFF", "PSEUDO_DIFF", "RSE", "NRSE"]
NFFT_CHOICES = ["256", "512", "1024", "2048", "4096", "8192", "16384"]
VIEW_CHOICES = ["Time", "Spectrum", "Spectrogram"]
LAYOUT_CHOICES = ["Per-channel files", "One multichannel file"]

RING_SECONDS = 10.0             # history kept for display and snapshots
RING_MAX_VALUES = 64_000_000    # cap on stored float32 values (~256 MB)
WAV_MAX_BYTES = 4_000_000_000   # WAV size fields are 32-bit
OVERFLOW_CODE = -200279         # "application is not able to keep up"
MAX_PLOT_BINS = 1500            # min/max bins per trace in the time view
MAX_SPEC_COLUMNS = 1500         # time columns in the spectrogram image


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def parse_channels(text):
    """'ai0:3' -> [ai0, ai1, ai2, ai3]; 'ai0, ai2' -> [ai0, ai2]."""
    names = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            prefix = a.rstrip("0123456789")
            start = int(a[len(prefix):])
            end = int(b.lstrip("abcdefghijklmnopqrstuvwxyz"))
            step = 1 if end >= start else -1
            names += [f"{prefix}{i}" for i in range(start, end + step, step)]
        else:
            names.append(part)
    if not names:
        raise ValueError("No channels given.")
    return names


def minmax_decimate(y, fs, bins=MAX_PLOT_BINS):
    """Shrink a long trace for plotting while keeping peaks visible."""
    n = len(y)
    t_end = n / fs
    if n <= 2 * bins:
        return np.arange(n) / fs - t_end, y
    step = n // bins
    m = step * bins
    blocks = y[n - m:].reshape(bins, step)
    out = np.empty(2 * bins, dtype=np.float32)
    out[0::2] = blocks.min(axis=1)
    out[1::2] = blocks.max(axis=1)
    idx = (n - m) + np.repeat(np.arange(bins) * step, 2) + np.tile([0, step // 2], bins)
    return idx / fs - t_end, out


class Ring:
    """Thread-safe circular buffer of the most recent samples, shape (frames, channels)."""

    def __init__(self, frames, n_channels):
        self.buf = np.zeros((frames, n_channels), dtype=np.float32)
        self.cap = frames
        self.pos = 0
        self.count = 0
        self.lock = threading.Lock()

    def push(self, block):
        n = block.shape[0]
        if n >= self.cap:
            block = block[-self.cap:]
            n = self.cap
        with self.lock:
            end = self.pos + n
            if end <= self.cap:
                self.buf[self.pos:end] = block
            else:
                k = self.cap - self.pos
                self.buf[self.pos:] = block[:k]
                self.buf[:n - k] = block[k:]
            self.pos = end % self.cap
            self.count = min(self.count + n, self.cap)

    def last(self, n):
        with self.lock:
            n = int(min(n, self.count))
            if n <= 0:
                return self.buf[:0].copy()
            start = (self.pos - n) % self.cap
            if start + n <= self.cap:
                return self.buf[start:start + n].copy()
            return np.concatenate((self.buf[start:], self.buf[:self.pos]))


# ----------------------------------------------------------------------------
# streaming WAV writing
# ----------------------------------------------------------------------------

class WavStreamWriter:
    """32-bit float WAV written incrementally. Header sizes are patched as it goes."""

    def __init__(self, path, n_channels, sample_rate):
        self.path = path
        self.nch = n_channels
        self.frames = 0
        fs = int(round(sample_rate))
        align = 4 * n_channels
        self.f = open(path, "wb")
        self.f.write(b"RIFF" + struct.pack("<I", 0) + b"WAVE")
        self.f.write(b"fmt " + struct.pack("<IHHIIHHH", 18, 3, n_channels, fs, fs * align, align, 32, 0))
        self.f.write(b"fact" + struct.pack("<I", 4))
        self._fact_pos = self.f.tell()
        self.f.write(struct.pack("<I", 0))
        self.f.write(b"data")
        self._data_pos = self.f.tell()
        self.f.write(struct.pack("<I", 0))

    def write(self, frames):
        self.f.write(np.ascontiguousarray(frames, dtype="<f4").tobytes())
        self.frames += frames.shape[0]

    def update_header(self):
        if self.f is None:
            return
        end = self.f.tell()
        self.f.seek(4)
        self.f.write(struct.pack("<I", end - 8))
        self.f.seek(self._fact_pos)
        self.f.write(struct.pack("<I", self.frames))
        self.f.seek(self._data_pos)
        self.f.write(struct.pack("<I", self.frames * 4 * self.nch))
        self.f.seek(end)
        self.f.flush()

    def close(self):
        if self.f is None:
            return
        self.update_header()
        self.f.close()
        self.f = None


def write_wav_set(stem, data, channels, rate, layout):
    """Write a whole array at once (used for snapshots). Returns file paths."""
    if layout == LAYOUT_CHOICES[1]:
        paths = [stem + ".wav"]
        w = WavStreamWriter(paths[0], len(channels), rate)
        w.write(data)
        w.close()
    else:
        paths = []
        for k, ch in enumerate(channels):
            p = f"{stem}_{ch}.wav"
            w = WavStreamWriter(p, 1, rate)
            w.write(data[:, k:k + 1])
            w.close()
            paths.append(p)
    return paths


class Recorder:
    """Writes one recording, split into parts. Runs inside the writer thread."""

    def __init__(self, folder, base, channels, rate, layout, split_s, meta):
        self.folder = folder
        self.base = base
        self.channels = channels
        self.rate = rate
        self.layout = layout
        self.meta = dict(meta)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bytes_per_frame = 4 * (len(channels) if layout == LAYOUT_CHOICES[1] else 1)
        max_frames = (WAV_MAX_BYTES - 1000) // bytes_per_frame
        self.split_frames = min(int(split_s * rate), max_frames) if split_s > 0 else max_frames
        self.part = 0
        self.part_frames = 0
        self.total_frames = 0
        self.writers = []
        self.files = []
        self.gaps = []
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(folder, exist_ok=True)
        self.meta_path = os.path.join(folder, f"{base}_{stamp_safe(self.stamp)}.json")
        self._open_part()
        self.write_metadata()

    def _open_part(self):
        self.part += 1
        stem = os.path.join(self.folder, f"{self.base}_{self.stamp}_part{self.part:03d}")
        if self.layout == LAYOUT_CHOICES[1]:
            paths = [stem + ".wav"]
            self.writers = [WavStreamWriter(paths[0], len(self.channels), self.rate)]
        else:
            paths = [f"{stem}_{ch}.wav" for ch in self.channels]
            self.writers = [WavStreamWriter(p, 1, self.rate) for p in paths]
        self.files += paths
        self.part_frames = 0

    def _close_part(self):
        for w in self.writers:
            w.close()
        self.writers = []

    def write(self, block):
        i = 0
        n = block.shape[0]
        while i < n:
            room = self.split_frames - self.part_frames
            if room <= 0:
                self._close_part()
                self._open_part()
                self.write_metadata()
                continue
            chunk = block[i:i + room]
            if self.layout == LAYOUT_CHOICES[1]:
                self.writers[0].write(chunk)
            else:
                for k, w in enumerate(self.writers):
                    w.write(chunk[:, k:k + 1])
            self.part_frames += chunk.shape[0]
            self.total_frames += chunk.shape[0]
            i += chunk.shape[0]

    def flush_headers(self):
        for w in self.writers:
            w.update_header()

    def note_gap(self):
        self.gaps.append({
            "after_seconds": self.total_frames / self.rate,
            "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def current_file(self):
        return os.path.basename(self.files[-1]) if self.files else ""

    def write_metadata(self, final=False, notes=None):
        if notes is not None:
            self.meta["notes"] = notes
        meta = dict(self.meta)
        meta.update({
            "sample_rate_hz": self.rate,
            "channels": self.channels,
            "layout": self.layout,
            "started_at": self.started_at,
            "stopped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if final else None,
            "duration_s": self.total_frames / self.rate,
            "files": [os.path.basename(f) for f in self.files],
            "data_gaps_from_buffer_overflow": self.gaps,
            "format": "32-bit float WAV, volts at the DAQ input",
        })
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def close(self, notes=None):
        self._close_part()
        self.write_metadata(final=True, notes=notes)


def stamp_safe(s):
    return s.replace(":", "")


# ----------------------------------------------------------------------------
# data sources
# ----------------------------------------------------------------------------

class NISource:
    """NI-DAQmx input task with fallbacks for settings the device or driver rejects."""

    def __init__(self, cfg):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        self._nidaqmx = nidaqmx
        self._AcqType = AcquisitionType
        self._Term = TerminalConfiguration
        self.cfg = cfg
        self.notes = []
        self.task = None

        # try the requested terminal type first, then the device default
        attempts = [cfg["terminal"]]
        if cfg["terminal"] != "DEFAULT":
            attempts.append("DEFAULT")
        last_exc = None
        for term_name in attempts:
            try:
                self._build(term_name)
                if term_name != cfg["terminal"]:
                    self.notes.append(f"terminal {cfg['terminal']} rejected, using DEFAULT")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                self._close_task()
        if last_exc is not None:
            raise last_exc

        self.info = "; ".join(self.notes)

    def _terminal(self, name):
        term = getattr(self._Term, name, None)
        if term is None:  # older nidaqmx names
            old = {"DIFF": "DIFFERENTIAL", "PSEUDO_DIFF": "PSEUDODIFFERENTIAL"}
            term = getattr(self._Term, old.get(name, "DEFAULT"), None)
        return term

    def _build(self, term_name):
        cfg = self.cfg
        self.notes = []
        self.task = self._nidaqmx.Task()
        term = self._terminal(term_name)
        for ch in cfg["channels"]:
            kwargs = dict(min_val=-cfg["v_range"], max_val=cfg["v_range"])
            if term is not None and term_name != "DEFAULT":
                kwargs["terminal_config"] = term
            self.task.ai_channels.add_ai_voltage_chan(f"{cfg['device']}/{ch}", **kwargs)

        # clamp the rate to what this device allows with this many channels
        rate = float(cfg["rate"])
        try:
            max_rate = float(self.task.timing.samp_clk_max_rate)
            if rate > max_rate:
                self.notes.append(f"requested {rate:,.0f} S/s is above the device limit, using {max_rate:,.0f}")
                rate = max_rate
        except Exception:
            pass

        self.task.timing.cfg_samp_clk_timing(
            rate,
            sample_mode=self._AcqType.CONTINUOUS,
            samps_per_chan=cfg.get("samps") or max(1000, int(rate * 0.05)),
        )
        try:
            self.rate = float(self.task.timing.samp_clk_rate)  # actual (coerced) rate
        except Exception:
            self.rate = rate
        self.block = cfg.get("samps") or max(1000, int(self.rate * 0.05))  # samples per channel per read

        buf_seconds = 10.0 if self.rate <= 500e3 else 4.0
        try:
            self.task.in_stream.input_buf_size = int(max(self.rate * buf_seconds, 4 * self.block))
        except Exception:
            self.notes.append("could not enlarge driver buffer")
        try:
            self.buf_frames = int(self.task.in_stream.input_buf_size)
        except Exception:
            self.buf_frames = int(self.rate * buf_seconds)

        # validate everything now so bad settings fail here, not mid-run
        try:
            from nidaqmx.constants import TaskMode
            self.task.control(TaskMode.TASK_COMMIT)
        except ImportError:
            pass

        nch = len(cfg["channels"])
        self.data = np.zeros((nch, self.block), dtype=np.float64)
        try:
            from nidaqmx.stream_readers import AnalogMultiChannelReader
            self.reader = AnalogMultiChannelReader(self.task.in_stream)
        except Exception:
            self.reader = None  # fall back to task.read()

    def _close_task(self):
        if self.task is not None:
            try:
                self.task.close()
            except Exception:
                pass
        self.task = None

    def start(self):
        self.task.start()

    def read(self):
        if self.reader is not None:
            self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.block, timeout=5.0)
            return self.data.T.astype(np.float32)
        raw = np.asarray(self.task.read(number_of_samples_per_channel=self.block, timeout=5.0),
                         dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return raw.T.copy()

    def backlog(self):
        try:
            return self.task.in_stream.avail_samp_per_chan / self.buf_frames
        except Exception:
            return 0.0

    def recover(self):
        self.task.stop()
        self.task.start()

    def close(self):
        if self.task is not None:
            try:
                self.task.stop()
            except Exception:
                pass
        self._close_task()


class SimSource:
    """Fake hardware: 0.1 s 3-15 kHz chirps every 0.3 s with channel delays and noise."""

    def __init__(self, cfg):
        self.rate = float(cfg["rate"])
        self.nch = len(cfg["channels"])
        self.block = cfg.get("samps") or max(1000, int(self.rate * 0.05))
        self.buf_frames = int(max(self.rate * 10, 4 * self.block))
        self.n = 0
        self.t0 = None
        try:
            self.rng = np.random.default_rng()
        except AttributeError:  # numpy older than 1.17
            self.rng = np.random.RandomState()
        self.f0, self.f1, self.T, self.period = 3e3, 15e3, 0.1, 0.3
        self.delays = np.arange(self.nch) * 0.5e-3
        self.gains = 0.5 / (1 + 0.3 * np.arange(self.nch))
        self.info = "simulated source"

    def start(self):
        self.t0 = time.perf_counter()

    def read(self):
        target = (self.n + self.block) / self.rate
        wait = target - (time.perf_counter() - self.t0)
        if wait > 0:
            time.sleep(wait)
        t = (self.n + np.arange(self.block)) / self.rate
        k = (self.f1 - self.f0) / self.T
        out = np.empty((self.block, self.nch), dtype=np.float32)
        for c in range(self.nch):
            tm = (t - self.delays[c]) % self.period
            on = tm < self.T
            ph = 2 * np.pi * (self.f0 * tm + 0.5 * k * tm ** 2)
            out[:, c] = self.gains[c] * np.sin(ph) * on + 0.01 * self.rng.standard_normal(self.block)
        self.n += self.block
        return out

    def backlog(self):
        lag = (time.perf_counter() - self.t0) - self.n / self.rate
        return max(0.0, lag * self.rate / self.buf_frames)

    def recover(self):
        pass

    def close(self):
        pass


DAQ_HINTS = {
    -50103: "The DAQ is being used by another program. Close any other copy of this script, "
            "NI MAX test panels, or other scripts using this device. Then press Start.",
    -200220: "Device not found. Open NI MAX, check the device name (for example Dev1), "
             "and type it in the Device box.",
    -201003: "The device is not responding. Check the USB cable and power, then press Start.",
    -200170: "A channel name is not valid for this device. Check the Channels box (for example ai0:3).",
    -200077: "A setting is not supported by this device. Try a lower sample rate or "
             "a different input range.",
    -200279: "The computer could not keep up with the DAQ. Lower the sample rate or turn off Live plot.",
}


def explain_error(exc):
    code = getattr(exc, "error_code", None)
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    hint = DAQ_HINTS.get(code)
    if hint:
        return f"{hint}\n\n(NI error {code}: {first})"
    return f"{type(exc).__name__}: {first}"


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readIntfUI_errors.log")


def log_exception(where, exc_info=None):
    import traceback
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} | {where} =====\n")
            if exc_info is None:
                f.write(traceback.format_exc())
            else:
                f.write("".join(traceback.format_exception(*exc_info)))
    except Exception:
        pass


def is_overflow(exc):
    return getattr(exc, "error_code", None) == OVERFLOW_CODE or str(OVERFLOW_CODE) in str(exc)


# ----------------------------------------------------------------------------
# application
# ----------------------------------------------------------------------------

class DAQMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1250x900")

        # acquisition state
        self.running = False
        self.stop_event = None
        self.acq_thread = None
        self.cfg = None
        self.ring = None
        self.actual_rate = None
        self.backlog_frac = 0.0
        self.overflows = 0
        self.pending_restart = False
        self.resume_recording = False

        # recording state (writer thread owns the Recorder)
        self.recording = False
        self.recorder = None
        self.write_q = queue.Queue()
        self.ui_q = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        # plot state
        self.layout_key = None
        self.axes = []
        self.lines = []
        self.img = None
        self.cbar = None
        self.vis_vars = {}
        self.level_labels = {}

        self._make_vars()
        self._build_ui()
        self._init_plot()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._callback_error
        self.root.after(200, self._tick)

    # ------------------------------------------------------------------ vars
    def _make_vars(self):
        v = tk.StringVar
        self.device_var = v(value="Dev1")
        self.channels_var = v(value="ai0:3")
        self.rate_var = v(value="200000")
        self.samps_var = v(value="Auto")
        self.range_var = v(value="5")
        self.terminal_var = v(value="DIFF")

        self.view_var = v(value="Time")
        self.window_var = v(value="0.5")
        self.refresh_var = v(value="200")
        self.yscale_var = v(value="Auto")
        self.yfixed_var = v(value="1.0")
        self.nfft_var = v(value="2048")
        self.fmin_var = v(value="0")
        self.fmax_var = v(value="30")
        self.dyn_var = v(value="80")
        self.spec_ch_var = v(value="")
        self.live_var = tk.BooleanVar(value=True)
        self.freeze_var = tk.BooleanVar(value=False)

        self.save_dir_var = v(value=os.path.abspath("daq_recordings"))
        self.base_var = v(value="daq_capture")
        self.layout_var = v(value=LAYOUT_CHOICES[0])
        self.split_var = v(value="300")
        self.snap_var = v(value="10")

        self.status_var = v(value="Idle")
        self.rec_status_var = v(value="Not recording")

    # -------------------------------------------------------------------- ui
    def _build_ui(self):
        pad = dict(padx=4, pady=3)

        acq = ttk.LabelFrame(self.root, text="Acquisition (press Apply or Enter to change while running)", padding=6)
        acq.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(acq, text="Device:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(acq, textvariable=self.device_var, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(acq, text="Channels:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(acq, textvariable=self.channels_var, width=12).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(acq, text="Input range (+/- V):").grid(row=0, column=4, sticky="w", **pad)
        ttk.Combobox(acq, textvariable=self.range_var, values=RANGE_PRESETS, width=6).grid(row=0, column=5, sticky="w", **pad)
        ttk.Label(acq, text="Terminal:").grid(row=0, column=6, sticky="w", **pad)
        ttk.Combobox(acq, textvariable=self.terminal_var, values=TERMINAL_CHOICES, width=11,
                     state="readonly").grid(row=0, column=7, sticky="w", **pad)

        # sample rate and samples per channel: arrows step through presets, or type any value
        ttk.Label(acq, text="Sample rate (S/s/ch):").grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        rate_box = tk.Spinbox(acq, values=RATE_PRESETS, textvariable=self.rate_var, width=10)
        rate_box.grid(row=1, column=2, sticky="w", **pad)
        ttk.Label(acq, text="Samples/ch per read:").grid(row=1, column=3, columnspan=2, sticky="w", **pad)
        samps_box = tk.Spinbox(acq, values=SAMPS_PRESETS, textvariable=self.samps_var, width=10)
        samps_box.grid(row=1, column=5, sticky="w", **pad)
        self.rate_var.set("200000")   # Spinbox resets its variable to the first preset on creation
        self.samps_var.set("Auto")
        for box in (rate_box, samps_box):
            box.bind("<Return>", lambda e: self.apply_settings())
        self.read_info_label = tk.Label(acq, text="", anchor="w", width=38)
        self.read_info_label.grid(row=1, column=6, columnspan=3, sticky="w", **pad)

        btns = ttk.Frame(acq)
        btns.grid(row=0, column=9, rowspan=2, sticky="e", padx=(12, 0))
        self.start_btn = ttk.Button(btns, text="Start", command=self.start_acquisition)
        self.start_btn.grid(row=0, column=0, **pad)
        ttk.Button(btns, text="Stop", command=self.stop_acquisition).grid(row=0, column=1, **pad)
        ttk.Button(btns, text="Apply", command=self.apply_settings).grid(row=1, column=0, columnspan=2,
                                                                         sticky="ew", **pad)

        disp = ttk.LabelFrame(self.root, text="Display (changes take effect immediately)", padding=6)
        disp.pack(fill="x", padx=8, pady=4)
        ttk.Label(disp, text="View:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(disp, textvariable=self.view_var, values=VIEW_CHOICES, width=11,
                     state="readonly").grid(row=0, column=1, **pad)
        ttk.Label(disp, text="Window (s):").grid(row=0, column=2, sticky="w", **pad)
        tk.Spinbox(disp, textvariable=self.window_var, from_=0.01, to=RING_SECONDS, increment=0.1,
                    width=7).grid(row=0, column=3, **pad)
        ttk.Label(disp, text="Refresh (ms):").grid(row=0, column=4, sticky="w", **pad)
        tk.Spinbox(disp, textvariable=self.refresh_var, from_=50, to=2000, increment=50,
                    width=6).grid(row=0, column=5, **pad)
        ttk.Label(disp, text="Y scale:").grid(row=0, column=6, sticky="w", **pad)
        ttk.Combobox(disp, textvariable=self.yscale_var, values=["Auto", "Fixed"], width=6,
                     state="readonly").grid(row=0, column=7, **pad)
        ttk.Label(disp, text="Fixed +/- V:").grid(row=0, column=8, sticky="w", **pad)
        ttk.Entry(disp, textvariable=self.yfixed_var, width=6).grid(row=0, column=9, **pad)
        ttk.Checkbutton(disp, text="Live plot", variable=self.live_var).grid(row=0, column=10, **pad)
        ttk.Checkbutton(disp, text="Freeze", variable=self.freeze_var).grid(row=0, column=11, **pad)

        ttk.Label(disp, text="FFT length:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Combobox(disp, textvariable=self.nfft_var, values=NFFT_CHOICES, width=8).grid(row=1, column=1, **pad)
        ttk.Label(disp, text="f min (kHz):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(disp, textvariable=self.fmin_var, width=7).grid(row=1, column=3, **pad)
        ttk.Label(disp, text="f max (kHz):").grid(row=1, column=4, sticky="w", **pad)
        ttk.Entry(disp, textvariable=self.fmax_var, width=6).grid(row=1, column=5, **pad)
        ttk.Label(disp, text="Dynamic range (dB):").grid(row=1, column=6, sticky="w", **pad)
        ttk.Entry(disp, textvariable=self.dyn_var, width=6).grid(row=1, column=7, **pad)
        ttk.Label(disp, text="Spectrogram ch:").grid(row=1, column=8, sticky="w", **pad)
        self.spec_ch_box = ttk.Combobox(disp, textvariable=self.spec_ch_var, values=[], width=6,
                                        state="readonly")
        self.spec_ch_box.grid(row=1, column=9, **pad)

        self.vis_frame = ttk.Frame(disp)
        self.vis_frame.grid(row=2, column=0, columnspan=13, sticky="w", **pad)
        ttk.Label(self.vis_frame, text="Show channels: (press Start)").pack(side="left")

        rec = ttk.LabelFrame(self.root, text="Recording (start and stop any time while acquiring)", padding=6)
        rec.pack(fill="x", padx=8, pady=4)
        ttk.Label(rec, text="Folder:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(rec, textvariable=self.save_dir_var, width=28).grid(row=0, column=1, columnspan=3, sticky="ew", **pad)
        ttk.Button(rec, text="Browse", command=self._browse_save_dir).grid(row=0, column=4, **pad)
        ttk.Label(rec, text="Base name:").grid(row=0, column=5, sticky="w", **pad)
        ttk.Entry(rec, textvariable=self.base_var, width=16).grid(row=0, column=6, **pad)
        ttk.Label(rec, text="Files:").grid(row=0, column=7, sticky="w", **pad)
        ttk.Combobox(rec, textvariable=self.layout_var, values=LAYOUT_CHOICES, width=18,
                     state="readonly").grid(row=0, column=8, **pad)
        ttk.Label(rec, text="Split every (s):").grid(row=0, column=9, sticky="w", **pad)
        ttk.Entry(rec, textvariable=self.split_var, width=6).grid(row=0, column=10, **pad)

        self.rec_btn = ttk.Button(rec, text="Record", command=self.toggle_recording)
        self.rec_btn.grid(row=1, column=0, **pad)
        ttk.Label(rec, textvariable=self.rec_status_var, foreground="darkred").grid(
            row=1, column=1, columnspan=5, sticky="w", **pad)
        ttk.Label(rec, text="Snapshot length (s):").grid(row=1, column=6, sticky="e", **pad)
        ttk.Entry(rec, textvariable=self.snap_var, width=6).grid(row=1, column=7, sticky="w", **pad)
        ttk.Button(rec, text="Save last N s", command=self.save_snapshot).grid(row=1, column=8, sticky="w", **pad)

        notes = ttk.LabelFrame(self.root, text="Data notes (saved in each recording's .json)", padding=6)
        notes.pack(fill="x", padx=8, pady=4)
        self.notes_box = tk.Text(notes, height=2, width=140)
        self.notes_box.insert("1.0", "DAQ capture notes\n")
        self.notes_box.pack(fill="x")

        stat = ttk.Frame(self.root, padding=(8, 0))
        stat.pack(fill="x")
        ttk.Label(stat, textvariable=self.status_var, foreground="darkblue").pack(side="left")
        self.levels_frame = ttk.Frame(self.root, padding=(8, 2))
        self.levels_frame.pack(fill="x")

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _init_plot(self):
        self.fig = Figure(figsize=(11, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _browse_save_dir(self):
        folder = filedialog.askdirectory(initialdir=self.save_dir_var.get() or os.path.abspath("."))
        if folder:
            self.save_dir_var.set(folder)

    def _rebuild_channel_widgets(self, channels):
        for w in self.vis_frame.winfo_children():
            w.destroy()
        for w in self.levels_frame.winfo_children():
            w.destroy()
        ttk.Label(self.vis_frame, text="Show channels:").pack(side="left", padx=(0, 6))
        old = self.vis_vars
        self.vis_vars = {}
        self.level_labels = {}
        for ch in channels:
            var = old.get(ch) or tk.BooleanVar(value=True)
            self.vis_vars[ch] = var
            ttk.Checkbutton(self.vis_frame, text=ch, variable=var).pack(side="left", padx=4)
            lab = tk.Label(self.levels_frame, text=f"{ch}: --", width=26, anchor="w")
            lab.pack(side="left")
            self.level_labels[ch] = lab
        self.spec_ch_box["values"] = channels
        if self.spec_ch_var.get() not in channels:
            self.spec_ch_var.set(channels[0])
        self.layout_key = None

    # ------------------------------------------------------------- numbers
    @staticmethod
    def _num(var, default, lo=None, hi=None):
        try:
            x = float(var.get())
        except (ValueError, tk.TclError):
            return default
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return x

    @staticmethod
    def _parse_samps(text):
        text = str(text).strip()
        if text == "" or text.lower().startswith("auto"):
            return None
        n = int(float(text))
        if n < 10:
            raise ValueError("Samples per channel must be at least 10 (or Auto).")
        return n

    def _update_read_info(self):
        """Live hint next to the rate and samples boxes."""
        try:
            rate = float(self.rate_var.get())
            samps = self._parse_samps(self.samps_var.get())
            if rate <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            self.read_info_label.config(text="Enter a number (samples: number or Auto)", fg="red")
            return
        n = samps if samps is not None else max(1000, int(rate * 0.05))
        txt = f"= {n:,} samples, {1000 * n / rate:.1f} ms per read"
        color = "black"
        if self.running and self.cfg is not None:
            if rate != self.cfg["rate"] or samps != self.cfg.get("samps"):
                txt += "  (press Apply)"
                color = "darkorange"
        elif rate / n > 100:
            txt += "  (too small, may overflow)"
            color = "darkorange"
        self.read_info_label.config(text=txt, fg=color)

    def _read_cfg(self):
        channels = parse_channels(self.channels_var.get())
        rate = float(self.rate_var.get())
        v_range = float(self.range_var.get())
        samps = self._parse_samps(self.samps_var.get())
        if rate <= 0 or v_range <= 0:
            raise ValueError("Sample rate and input range must be positive.")
        return {
            "device": self.device_var.get().strip() or "Dev1",
            "channels": channels,
            "rate": rate,
            "samps": samps,
            "v_range": v_range,
            "terminal": self.terminal_var.get(),
        }

    # --------------------------------------------------------- acquisition
    def start_acquisition(self):
        if self.running:
            return
        try:
            cfg = self._read_cfg()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.cfg = cfg
        self._rebuild_channel_widgets(cfg["channels"])
        self.running = True
        self.overflows = 0
        self.backlog_frac = 0.0
        self.ring = None
        self.actual_rate = None
        self.stop_event = threading.Event()
        self.status_var.set("Starting...")
        self.acq_thread = threading.Thread(target=self._acq_loop, args=(cfg, self.stop_event), daemon=True)
        self.acq_thread.start()

    def stop_acquisition(self):
        if not self.running or self.stop_event is None:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def apply_settings(self):
        try:
            self._read_cfg()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        if not self.running:
            self.status_var.set("Settings OK. Press Start.")
            return
        self.pending_restart = True
        self.resume_recording = self.recording
        self.stop_acquisition()

    def _acq_loop(self, cfg, stop_event):
        src = None
        try:
            src = SimSource(cfg) if cfg["device"].upper() == "SIM" else NISource(cfg)
            nch = len(cfg["channels"])
            ring_frames = int(min(RING_SECONDS * src.rate, RING_MAX_VALUES // nch))
            self.ring = Ring(ring_frames, nch)
            self.actual_rate = src.rate
            self.ui_q.put(("started", {
                "rate": src.rate,
                "block": src.block,
                "buf_frames": src.buf_frames,
                "ring_s": ring_frames / src.rate,
                "info": src.info,
            }))
            src.start()
            while not stop_event.is_set():
                try:
                    block = src.read()
                except Exception as exc:
                    if is_overflow(exc):
                        self.overflows += 1
                        if self.recording:
                            self.write_q.put(("gap",))
                        src.recover()
                        continue
                    raise
                self.ring.push(block)
                if self.recording:
                    self.write_q.put(("data", block))
                self.backlog_frac = src.backlog()
        except Exception as exc:
            log_exception("acquisition")
            self.ui_q.put(("error", explain_error(exc)))
        finally:
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
            self.ui_q.put(("stopped", None))

    # ----------------------------------------------------------- recording
    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.running or self.actual_rate is None:
            messagebox.showinfo("Not acquiring", "Start acquisition first, then press Record.")
            return
        base = self.base_var.get().strip() or "daq_capture"
        args = dict(
            folder=self.save_dir_var.get().strip() or os.path.abspath("."),
            base=base,
            channels=self.cfg["channels"],
            rate=self.actual_rate,
            layout=self.layout_var.get(),
            split_s=self._num(self.split_var, 300.0, lo=0.0),
            meta={
                "device": self.cfg["device"],
                "input_range_v": self.cfg["v_range"],
                "terminal_config": self.cfg["terminal"],
                "notes": self.notes_box.get("1.0", "end-1c").strip(),
            },
        )
        self.write_q.put(("open", args))
        self.recording = True
        self.rec_btn.config(text="Stop recording")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.write_q.put(("close", self.notes_box.get("1.0", "end-1c").strip()))
        self.rec_btn.config(text="Record")

    def _writer_loop(self):
        rec = None
        last_hdr = time.time()
        while True:
            try:
                item = self.write_q.get(timeout=1.0)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    kind = item[0]
                    if kind == "quit":
                        if rec is not None:
                            rec.close()
                        return
                    if kind == "open":
                        if rec is not None:
                            rec.close()
                        rec = Recorder(**item[1])
                        self.recorder = rec
                    elif kind == "data" and rec is not None:
                        rec.write(item[1])
                    elif kind == "gap" and rec is not None:
                        rec.note_gap()
                    elif kind == "close" and rec is not None:
                        rec.close(notes=item[1])
                        self.ui_q.put(("rec_done", f"Saved {rec.total_frames / rec.rate:.1f} s to "
                                                   f"{os.path.basename(rec.meta_path)} and its WAV files"))
                        rec = None
                        self.recorder = None
                if rec is not None and time.time() - last_hdr > 5.0:
                    rec.flush_headers()  # keeps files readable if the program crashes
                    last_hdr = time.time()
            except Exception as exc:
                log_exception("recording")
                self.ui_q.put(("rec_error", f"Recording error: {exc}"))
                try:
                    if rec is not None:
                        rec.close()
                except Exception:
                    pass
                rec = None
                self.recorder = None

    def save_snapshot(self):
        if self.ring is None or self.actual_rate is None:
            messagebox.showinfo("No data", "Start acquisition first.")
            return
        secs = self._num(self.snap_var, 10.0, lo=0.01)
        data = self.ring.last(int(secs * self.actual_rate))
        if data.shape[0] == 0:
            return
        folder = self.save_dir_var.get().strip() or os.path.abspath(".")
        base = self.base_var.get().strip() or "daq_capture"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.join(folder, f"{base}_snapshot_{stamp}")
        meta = {
            "device": self.cfg["device"],
            "sample_rate_hz": self.actual_rate,
            "channels": self.cfg["channels"],
            "input_range_v": self.cfg["v_range"],
            "terminal_config": self.cfg["terminal"],
            "duration_s": data.shape[0] / self.actual_rate,
            "notes": self.notes_box.get("1.0", "end-1c").strip(),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format": "32-bit float WAV, volts at the DAQ input",
        }
        args = (stem, data, list(self.cfg["channels"]), self.actual_rate, self.layout_var.get(), meta, folder)
        threading.Thread(target=self._write_snapshot, args=args, daemon=True).start()

    def _write_snapshot(self, stem, data, channels, rate, layout, meta, folder):
        try:
            os.makedirs(folder, exist_ok=True)
            paths = write_wav_set(stem, data, channels, rate, layout)
            meta["files"] = [os.path.basename(p) for p in paths]
            with open(stem + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self.ui_q.put(("info", f"Snapshot saved: {os.path.basename(stem)} ({data.shape[0] / rate:.1f} s)"))
        except Exception as exc:
            log_exception("snapshot")
            self.ui_q.put(("rec_error", f"Snapshot error: {exc}"))

    # ------------------------------------------------------------ ui loop
    def _tick(self):
        self._drain_ui_queue()
        self._update_read_info()
        if self.running and self.ring is not None and self.actual_rate:
            self._update_status()
            if self.live_var.get() and not self.freeze_var.get():
                try:
                    self._update_plot()
                except Exception as exc:  # never let a plotting problem stop the loop
                    if not getattr(self, "_plot_error_logged", False):
                        log_exception("plot")
                        self._plot_error_logged = True
                    self.status_var.set(f"Plot error (acquisition still running): {exc}")
        refresh = int(self._num(self.refresh_var, 200, lo=50, hi=5000))
        self.root.after(refresh, self._tick)

    def _drain_ui_queue(self):
        while True:
            try:
                kind, payload = self.ui_q.get_nowait()
            except queue.Empty:
                return
            if kind == "started":
                self.start_info = payload
                self.rate_var.set(f"{payload['rate']:.0f}")
                if self.cfg is not None:
                    self.cfg["rate"] = float(f"{payload['rate']:.0f}")
                if self.resume_recording:
                    self.resume_recording = False
                    self.start_recording()
            elif kind == "stopped":
                self.running = False
                self.stop_event = None
                if self.recording and not self.pending_restart:
                    self.stop_recording()
                elif self.recording and self.pending_restart:
                    # close the old file; a new one opens after restart with the new settings
                    self.stop_recording()
                if self.pending_restart:
                    self.pending_restart = False
                    self.start_acquisition()
                elif not self.status_var.get().startswith("Error"):
                    self.status_var.set("Stopped")
            elif kind == "error":
                self.pending_restart = False
                self.resume_recording = False
                self.status_var.set("Error: " + payload.splitlines()[0])
                messagebox.showerror("DAQ error", payload + f"\n\nFull details: {LOG_PATH}")
            elif kind == "rec_done":
                self.rec_status_var.set(payload)
            elif kind == "rec_error":
                self.recording = False
                self.rec_btn.config(text="Record")
                self.rec_status_var.set(payload)
                messagebox.showerror("Recording error", payload)
            elif kind == "info":
                self.rec_status_var.set(payload)

    def _update_status(self):
        info = getattr(self, "start_info", None)
        if not info:
            return
        txt = (f"Running | {info['rate']:,.0f} S/s per ch (actual) | {info['block']:,} samples/ch per read | "
               f"driver buffer {info['buf_frames']:,}/ch ({info['buf_frames'] / info['rate']:.1f} s) | "
               f"backlog {100 * self.backlog_frac:.0f}% | overflows {self.overflows}")
        if info.get("info"):
            txt += f" | {info['info']}"
        self.status_var.set(txt)
        rec = self.recorder
        if self.recording and rec is not None:
            secs = rec.total_frames / rec.rate
            self.rec_status_var.set(
                f"RECORDING {int(secs // 60):02d}:{secs % 60:04.1f} | {rec.current_file()} | "
                f"write queue {self.write_q.qsize()}"
            )

    def _visible_channels(self):
        return [ch for ch in self.cfg["channels"] if self.vis_vars.get(ch) is not None and self.vis_vars[ch].get()]

    def _rebuild_axes(self, view, visible):
        self.fig.clf()
        self.axes, self.lines, self.img, self.cbar = [], [], None, None
        if view == "Time":
            n = max(1, len(visible))
            for i, ch in enumerate(visible):
                ax = self.fig.add_subplot(n, 1, i + 1, sharex=self.axes[0] if self.axes else None)
                line, = ax.plot([], [], lw=0.8)
                ax.set_ylabel(f"{ch} (V)")
                ax.grid(True, alpha=0.25)
                self.axes.append(ax)
                self.lines.append(line)
            if self.axes:
                self.axes[-1].set_xlabel("Time relative to latest sample (s)")
        elif view == "Spectrum":
            ax = self.fig.add_subplot(111)
            for ch in visible:
                line, = ax.plot([], [], lw=1.0, label=ch)
                self.lines.append(line)
            ax.set_xlabel("Frequency (kHz)")
            ax.set_ylabel("PSD (dB re 1 V\u00b2/Hz)")
            ax.grid(True, alpha=0.25)
            if visible:
                ax.legend(loc="upper right")
            self.axes = [ax]
        else:
            ax = self.fig.add_subplot(111)
            ax.set_xlabel("Time relative to latest sample (s)")
            ax.set_ylabel("Frequency (kHz)")
            self.axes = [ax]
        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.layout_key = (view, tuple(visible))

    def _update_levels(self, y):
        v_range = self.cfg["v_range"]
        for k, ch in enumerate(self.cfg["channels"]):
            lab = self.level_labels.get(ch)
            if lab is None or y.shape[0] == 0:
                continue
            col = y[:, k]
            peak = float(np.max(np.abs(col)))
            vpp = float(np.max(col) - np.min(col))
            clip = peak >= 0.98 * v_range
            lab.config(text=f"{ch}: {vpp:.3f} Vpp ({100 * peak / v_range:.0f}% range){'  CLIP' if clip else ''}",
                       fg="red" if clip else "black")

    def _update_plot(self):
        fs = self.actual_rate
        view = self.view_var.get()
        visible = self._visible_channels()
        if (view, tuple(visible)) != self.layout_key:
            self._rebuild_axes(view, visible)

        window = self._num(self.window_var, 0.5, lo=0.001, hi=RING_SECONDS)
        y_all = self.ring.last(int(window * fs))
        if y_all.shape[0] < 2:
            return
        self._update_levels(y_all)
        if not visible:
            self.canvas.draw_idle()
            return
        idx = [self.cfg["channels"].index(ch) for ch in visible]
        y = y_all[:, idx]
        t_span = y.shape[0] / fs

        nfft = int(self._num(self.nfft_var, 2048, lo=64, hi=262144))
        fmin = self._num(self.fmin_var, 0.0, lo=0.0) * 1e3
        fmax = self._num(self.fmax_var, fs / 2e3, lo=0.001) * 1e3
        fmax = min(fmax, fs / 2)
        if fmin >= fmax:
            fmin = 0.0
        dyn = self._num(self.dyn_var, 80.0, lo=5.0, hi=200.0)

        if view == "Time":
            fixed = self.yscale_var.get() == "Fixed"
            ylim = self._num(self.yfixed_var, 1.0, lo=1e-6)
            for k, (ax, line) in enumerate(zip(self.axes, self.lines)):
                t, yy = minmax_decimate(y[:, k], fs)
                line.set_data(t, yy)
                ax.set_xlim(-t_span, 0)
                if fixed:
                    ax.set_ylim(-ylim, ylim)
                else:
                    lo, hi = float(yy.min()), float(yy.max())
                    pad = max(1e-4, 0.1 * (hi - lo))
                    ax.set_ylim(lo - pad, hi + pad)

        elif view == "Spectrum":
            seg = min(nfft, y.shape[0])
            yy = y[-min(y.shape[0], 4_000_000):]
            f, p = signal.welch(yy, fs, window="hann", nperseg=seg, noverlap=seg // 2, axis=0)
            pdb = 10 * np.log10(p + 1e-30)
            band = (f >= fmin) & (f <= fmax)
            if not band.any():
                return
            for k, line in enumerate(self.lines):
                line.set_data(f[band] / 1e3, pdb[band, k])
            ax = self.axes[0]
            ax.set_xlim(fmin / 1e3, fmax / 1e3)
            top = float(pdb[band].max())
            ax.set_ylim(top - dyn, top + 5)

        else:
            ch = self.spec_ch_var.get()
            if ch not in self.cfg["channels"]:
                return
            x = y_all[:, self.cfg["channels"].index(ch)]
            seg = min(nfft, len(x))
            f, tt, p = signal.spectrogram(x, fs, window="hann", nperseg=seg, noverlap=seg // 2, mode="psd")
            band = (f >= fmin) & (f <= fmax)
            if not band.any() or p.shape[1] == 0:
                return
            p = p[band]
            f = f[band]
            g = int(np.ceil(p.shape[1] / MAX_SPEC_COLUMNS))
            if g > 1:
                m = (p.shape[1] // g) * g
                p = p[:, :m].reshape(p.shape[0], -1, g).mean(axis=2)
                tt = tt[:m].reshape(-1, g).mean(axis=1)
            pdb = 10 * np.log10(p + 1e-30)
            top = float(pdb.max())
            tt = tt - len(x) / fs
            extent = [tt[0], tt[-1] if len(tt) > 1 else tt[0] + 1e-3, f[0] / 1e3, f[-1] / 1e3]
            ax = self.axes[0]
            if self.img is None:
                self.img = ax.imshow(pdb, origin="lower", aspect="auto", extent=extent, cmap="viridis",
                                     interpolation="nearest")
                self.cbar = self.fig.colorbar(self.img, ax=ax)
                self.cbar.set_label("dB re 1 V\u00b2/Hz")
            else:
                self.img.set_data(pdb)
                self.img.set_extent(extent)
            self.img.set_clim(top - dyn, top)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(f"{ch}  |  FFT {seg}  |  {fs / seg:.1f} Hz bins")

        self.canvas.draw_idle()

    def _callback_error(self, exc_type, exc, tb):
        log_exception("ui callback", (exc_type, exc, tb))
        self.status_var.set(f"UI error logged to readIntfUI_errors.log: {exc_type.__name__}: {exc}")

    # --------------------------------------------------------------- close
    def _on_close(self):
        if self.recording:
            self.stop_recording()
        if self.running and self.stop_event is not None:
            self.stop_event.set()
            if self.acq_thread is not None:
                self.acq_thread.join(timeout=6.0)
        self.write_q.put(("quit",))
        self.writer_thread.join(timeout=10.0)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DAQMonitorApp(root)
    root.mainloop()

"""
sendChirpUI.py

GUI for transmitting tones, chirps, and phase-coded M sequences.

Features:
- select device and AO output channel
- configure signal type, cycle length, duty cycle, sample rate, and amplitude
- start/stop transmit
- live waveform, time-averaged PSD, spectrogram, and autocorrelation previews
- playback through the default computer audio output without a DAQ
"""

import threading
import re
import time
from io import BytesIO
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk, messagebox, filedialog

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import spectrogram, resample_poly, max_len_seq, correlate, firwin, hilbert
from scipy.fft import next_fast_len
from scipy.signal.windows import tukey


SPEAKER_DEVICE = "default (laptop speaker)"
DEFAULT_HYPOTHETICAL_FS = 2_000_000


class ChirpTransmitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Signal Transmit")
        self.root.geometry("1400x820")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.running = False
        self.thread = None
        self.stop_event = None
        self.prepared_output = None
        self.active_output = None
        self.hypothetical_output = None

        self.available_devices = self._discover_devices()
        default_device = self.available_devices[0]
        self.device_var = tk.StringVar(value=default_device)
        self.available_channels = self._discover_ao_channels(default_device)
        self.ao_channel_var = tk.StringVar(value=self.available_channels[0] if self.available_channels else "Not used")

        self.f0_var = tk.StringVar(value="8000")
        self.f1_var = tk.StringVar(value="12000")
        self.signal_type_var = tk.StringVar(value="Chirp")
        self.t_cycle_var = tk.StringVar(value="0.300")
        self.duty_cycle_var = tk.StringVar(value="100")
        self.order_var = tk.StringVar(value="7")
        self.sequence_info_var = tk.StringVar()
        self.fs_var = tk.StringVar(value="44000")
        self.amplitude_var = tk.StringVar(value="0.5")
        self.repeats_var = tk.StringVar(value="1")
        self.output_info_var = tk.StringVar()
        self.lag_half_width_var = tk.StringVar(value="5")
        self.hypothetical_preview_var = tk.BooleanVar(value=False)
        self.hypothetical_fs_var = tk.StringVar(value=str(DEFAULT_HYPOTHETICAL_FS))
        self.plot_duration_ms_var = tk.DoubleVar(value=1.0)
        self.plot_start_ms_var = tk.DoubleVar(value=0.0)
        self.plot_window_info_var = tk.StringVar()
        self.plot_control_key = None
        self.adjusting_plot_controls = False
        self.wav_path_var = tk.StringVar(value="chirp.wav")
        self.status_var = tk.StringVar(value="Idle")

        self.preview_after_id = None
        self.help_window = None
        self._build_ui()
        self._on_device_change()
        self._on_signal_change()
        for variable in (self.f0_var, self.f1_var, self.t_cycle_var, self.duty_cycle_var, self.order_var, self.fs_var, self.amplitude_var, self.repeats_var, self.lag_half_width_var, self.hypothetical_fs_var):
            variable.trace_add("write", self._schedule_preview)
        self._update_preview()

    def _discover_devices(self):
        try:
            from nidaqmx.system import System
            devices = list(System.local().devices.device_names)
        except Exception:
            devices = []
        devices.append(SPEAKER_DEVICE)
        return devices

    def _discover_ao_channels(self, device_name):
        if device_name == SPEAKER_DEVICE:
            return []
        try:
            from nidaqmx.system import System
            dev = System.local().devices[device_name]
            channels = []
            for ch in dev.ao_physical_chans:
                name = str(ch).split("/")[-1].strip().lower()
                match = re.search(r"\bao(\d+)\b", name)
                if match:
                    channel_name = f"ao{match.group(1)}"
                    if channel_name not in channels:
                        channels.append(channel_name)
            return channels
        except Exception:
            return []

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="Device:").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.device_combo = ttk.Combobox(main, textvariable=self.device_var, values=self.available_devices, state="readonly", width=30)
        self.device_combo.grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_change)
        ttk.Button(main, text="Help", command=self._show_help).grid(row=0, column=2, padx=5, pady=4, sticky="ne")

        ttk.Label(main, text="AO channel:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.ao_combo = ttk.Combobox(main, textvariable=self.ao_channel_var, values=self.available_channels, state="readonly", width=18)
        self.ao_combo.grid(row=1, column=1, padx=5, pady=4, sticky="ew")

        ttk.Label(main, text="Signal type:").grid(row=2, column=0, sticky="w", padx=5, pady=4)
        self.signal_combo = ttk.Combobox(main, textvariable=self.signal_type_var, values=("Tone", "Chirp", "M sequences"), state="readonly")
        self.signal_combo.grid(row=2, column=1, padx=5, pady=4, sticky="ew")
        self.signal_combo.bind("<<ComboboxSelected>>", self._on_signal_change)

        lag_controls = ttk.Frame(main)
        lag_controls.grid(row=2, column=2, sticky="e", padx=5, pady=4)
        ttk.Label(lag_controls, text="Lag half-width (ms):").pack(side="left", padx=5)
        ttk.Entry(lag_controls, textvariable=self.lag_half_width_var, width=7).pack(side="left")

        self.preview_frame = ttk.LabelFrame(main, text="Intended signal — waveform, PSD, spectrogram, and autocorrelation", padding=5)
        self.preview_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=8)
        self.preview_figure = Figure(figsize=(13, 2.8), dpi=100, constrained_layout=True)
        self.waveform_axes = self.preview_figure.add_subplot(141)
        self.psd_axes = self.preview_figure.add_subplot(142)
        self.preview_axes = self.preview_figure.add_subplot(143)
        self.autocorrelation_axes = self.preview_figure.add_subplot(144)
        plot_controls = ttk.Frame(self.preview_frame, width=200)
        plot_controls.pack(side="left", fill="y", padx=(0, 8))
        plot_controls.pack_propagate(False)
        ttk.Label(plot_controls, text="Plot time window", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 12))
        ttk.Checkbutton(plot_controls, text="Hypothetical preview", variable=self.hypothetical_preview_var, command=self._on_hypothetical_change).pack(anchor="w", pady=(0, 4))
        ttk.Label(plot_controls, text="Hypothetical FS (Hz)").pack(anchor="w")
        self.hypothetical_fs_entry = ttk.Entry(plot_controls, textvariable=self.hypothetical_fs_var, state="disabled", width=18)
        self.hypothetical_fs_entry.pack(fill="x", pady=(0, 8))
        ttk.Label(plot_controls, text="Unchecked: exact output", wraplength=190).pack(anchor="w", pady=(0, 12))
        ttk.Label(plot_controls, text="Duration (ms)").pack(anchor="w")
        self.plot_duration_scale = tk.Scale(plot_controls, variable=self.plot_duration_ms_var, from_=1, to=1, resolution=-1, showvalue=False, orient="horizontal", command=self._on_plot_control_change)
        self.plot_duration_scale.pack(fill="x")
        ttk.Label(plot_controls, text="Start (ms)").pack(anchor="w", pady=(12, 0))
        self.plot_start_scale = tk.Scale(plot_controls, variable=self.plot_start_ms_var, from_=0, to=0, resolution=-1, showvalue=False, orient="horizontal", command=self._on_plot_control_change)
        self.plot_start_scale.pack(fill="x")
        ttk.Label(plot_controls, textvariable=self.plot_window_info_var, wraplength=190).pack(anchor="w", pady=(12, 0))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=self.preview_frame)
        self.preview_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.preview_colorbar = None

        parameters = ttk.Frame(main)
        parameters.grid(row=4, column=0, columnspan=3, sticky="ew")
        parameters.columnconfigure(1, weight=1)
        self.frequency_label = ttk.Label(parameters, text="Start frequency (Hz):")
        self.frequency_label.grid(row=0, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(parameters, textvariable=self.f0_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        self.stop_frequency_label = ttk.Label(parameters, text="Stop frequency (Hz):")
        self.stop_frequency_label.grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.stop_frequency_entry = ttk.Entry(parameters, textvariable=self.f1_var)
        self.stop_frequency_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=4)
        self.order_label = ttk.Label(parameters, text="Sequence order:")
        self.order_label.grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.order_entry = ttk.Spinbox(parameters, from_=2, to=16, textvariable=self.order_var)
        self.order_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=4)
        self.sequence_info_label = ttk.Label(parameters, textvariable=self.sequence_info_var, wraplength=1000)
        self.sequence_info_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        ttk.Label(parameters, text="T_cycle (s):").grid(row=3, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(parameters, textvariable=self.t_cycle_var).grid(row=3, column=1, sticky="ew", padx=5, pady=4)
        ttk.Label(parameters, text="Duty-cycle %:").grid(row=4, column=0, sticky="w", padx=5, pady=4)
        ttk.Spinbox(parameters, from_=0, to=100, increment=1, textvariable=self.duty_cycle_var).grid(row=4, column=1, sticky="ew", padx=5, pady=4)
        shared = ttk.Frame(parameters)
        shared.grid(row=5, column=0, columnspan=2, sticky="ew")
        for column in (1, 3, 5):
            shared.columnconfigure(column, weight=1)
        ttk.Label(shared, text="Synthesis FS (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(shared, textvariable=self.fs_var, width=12).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        self.amplitude_label = ttk.Label(shared, text="Amplitude (V peak):")
        self.amplitude_label.grid(row=0, column=2, sticky="w", padx=5, pady=4)
        ttk.Entry(shared, textvariable=self.amplitude_var, width=12).grid(row=0, column=3, sticky="ew", padx=5, pady=4)
        self.repeats_label = ttk.Label(shared, text="Repeats:")
        self.repeats_label.grid(row=0, column=4, sticky="w", padx=5, pady=4)
        ttk.Entry(shared, textvariable=self.repeats_var, width=8).grid(row=0, column=5, sticky="ew", padx=5, pady=4)

        ttk.Label(main, textvariable=self.output_info_var, wraplength=1300).grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=4)

        output = ttk.Frame(main)
        output.grid(row=6, column=0, columnspan=3, sticky="ew")
        output.columnconfigure(1, weight=1)
        ttk.Label(output, text="WAV file:").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(output, textvariable=self.wav_path_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        ttk.Button(output, text="Browse...", command=self._browse_wav_path).grid(row=0, column=2, padx=5, pady=4)
        ttk.Button(output, text="Save WAV + TXT + PNG", command=self.save_output).grid(row=0, column=3, padx=5, pady=4)
        button_row = ttk.Frame(main)
        button_row.grid(row=7, column=0, columnspan=3, pady=6, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Start", command=self.start_output).grid(row=0, column=0, padx=5, sticky="ew")
        ttk.Button(button_row, text="Stop", command=self.stop_output).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Label(main, textvariable=self.status_var, foreground="darkblue", wraplength=1000).grid(row=8, column=0, columnspan=3, sticky="w", padx=5, pady=4)

    def _on_signal_change(self, event=None):
        signal_type = self.signal_type_var.get()
        for widget in (self.stop_frequency_label, self.stop_frequency_entry, self.order_label, self.order_entry, self.sequence_info_label):
            widget.grid_remove()
        if signal_type == "Chirp":
            self.frequency_label.config(text="Start frequency (Hz):")
            self.stop_frequency_label.grid()
            self.stop_frequency_entry.grid()
        elif signal_type == "Tone":
            self.frequency_label.config(text="Fundamental frequency (Hz):")
        else:
            self.frequency_label.config(text="Carrier frequency (Hz):")
            self.order_label.grid()
            self.order_entry.grid()
            self.sequence_info_label.grid()
        self._schedule_preview()

    def _schedule_preview(self, *args):
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(250, self._update_preview)

    def _on_plot_control_change(self, value):
        if not self.adjusting_plot_controls:
            self._schedule_preview()

    def _on_hypothetical_change(self):
        self.hypothetical_fs_entry.config(state="normal" if self.hypothetical_preview_var.get() else "disabled")
        self._schedule_preview()

    def _plot_window_samples(self, prepared):
        total_samples = prepared['total_samples']
        fs = prepared['fs']
        total_ms = 1000 * total_samples / fs
        min_duration_ms = min(1.0, total_ms)
        key = (total_samples, fs)
        self.adjusting_plot_controls = True
        try:
            new_signal_duration = self.plot_control_key != key
            if new_signal_duration:
                self.plot_duration_scale.configure(from_=min_duration_ms, to=total_ms)
            if new_signal_duration:
                self.plot_control_key = key
                self.plot_duration_ms_var.set(total_ms)
                self.plot_start_ms_var.set(0.0)
            duration_ms = min(total_ms, max(min_duration_ms, self.plot_duration_ms_var.get()))
            if self.plot_duration_ms_var.get() != duration_ms:
                self.plot_duration_ms_var.set(duration_ms)
            max_start_ms = max(0.0, total_ms - duration_ms)
            self.plot_start_scale.configure(from_=0, to=max_start_ms)
            start_ms = min(max_start_ms, max(0.0, self.plot_start_ms_var.get()))
            if self.plot_start_ms_var.get() != start_ms:
                self.plot_start_ms_var.set(start_ms)
        finally:
            self.adjusting_plot_controls = False
        window_samples = min(total_samples, max(2, int(round(duration_ms * fs / 1000))))
        start_sample = min(total_samples - window_samples, max(0, int(round(start_ms * fs / 1000))))
        end_sample = start_sample + window_samples
        self.plot_window_info_var.set(f"Showing {start_sample / fs:.4g}–{end_sample / fs:.4g} s of {total_ms / 1000:.4g} s")
        return start_sample, end_sample

    def _show_help(self):
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            return
        self.help_window = tk.Toplevel(self.root)
        self.help_window.title("Signal parameters — Help")
        self.help_window.geometry("820x650")
        notebook = ttk.Notebook(self.help_window)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        common = (
            "COMMON SETTINGS\n\n"
            "Device: choose an NI DAQ or 'default (laptop speaker)'. The speaker option uses the macOS default audio output; select MacBook Speakers in System Settings if needed.\n\n"
            "AO channel: the DAQ analog output connected to your source amplifier. Not used for speaker playback.\n\n"
            "Signal type: selects Tone, Chirp, or M sequences and shows only the relevant fields.\n\n"
            "T_cycle (s): length of one complete probe-plus-silence cycle. Repeats includes both portions. All signal types use nearest-sample rounding, round(FS × T_cycle). M sequences at 100% duty then shorten the cycle to exactly one sequence period and report the actual source cycle duration.\n\n"
            "Duty-cycle %: requested active fraction at the start of each cycle; the remainder is zero. 0% gives silence. Tone and Chirp use round(cycle samples × duty-cycle / 100) active samples. M sequences fit a complete, equal-chip sequence inside the nearest-sample active budget. At 100% duty the cycle is exactly one sequence period, so repeats form a continuous, gap-free loop; below 100% the leftover samples are silent.\n\n"
            "Synthesis FS (Hz): whole-number sample rate used to generate the actual output cycle. Frequencies must be below synthesis FS / 2. For a DAQ, output FS equals synthesis FS. For speakers, one full cycle is resampled to the selected audio device's native rate BEFORE cycle repetition, saving, and playback. The cycle length rounds up to a whole output sample, so actual timing can differ slightly from the entered duration. The separate hypothetical preview uses the entered Hypothetical FS.\n\n"
            "Output FS / plotted Nyquist: the actual prepared output rate is used by the WAV and playback. With hypothetical preview unchecked, all plots use that same rate and show frequencies to output FS / 2. With it checked, the plots use the entered Hypothetical FS and extend to half that rate; the WAV and playback rates do not change.\n\n"
            "Resampling filter: an explicit low-pass FIR with 80 × max(up, down) + 1 taps and a Kaiser window (beta = 10) suppresses interpolation images and filters before downsampling. Its cutoff is half the lower of synthesis FS and output FS; components near that cutoff can be attenuated. This does not undo aliasing already present in the synthesized cycle or eliminate genuine finite-burst and phase-coding sidebands. The full prepared spectrum remains visible for debugging.\n\n"
            "Amplitude: peak volts for DAQ output (0–10), or digital full-scale amplitude for speakers (0–1). Speaker volume also depends on macOS volume; this is not calibrated acoustic pressure. No automatic normalization is applied.\n\n"
            "Repeats: positive integer number of identical probe-plus-silence cycles. Every cycle restarts the waveform and internal M-sequence phase.\n\n"
            "Total playback duration: prepared cycle samples × repeats / output rate. The displayed duration uses actual sample counts and remains available when a sequence exceeds the preview limit. Entire sequences are prepared in memory.\n\n"
            "Save WAV + TXT + PNG: prepares the signal from the current controls and writes the complete float32 REAL output buffer without playing it. A same-name .txt file records the selected settings, plot mode, plot window, lag width, and actual output timing. A same-name .png directly exports waveform, PSD, and spectrogram plots for the selected time window and preview mode. The WAV always contains the full real output signal. Choose the WAV path with Browse. Save is unavailable during playback.\n\n"
            "Start / Stop: Start captures one prepared buffer and its output device and plays it without saving files. Plots stay tied to that buffer while it is playing, even if signal parameters are edited; the new settings are prepared and shown after playback stops. The lag zoom control still works while playing. Stop interrupts playback. Speaker playback explicitly uses the captured audio-device index.\n\n"
            "Previews, left to right: waveform (amplitude versus time, including repeats and silence); time-averaged PSD; spectrogram (frequency versus time); autocorrelation. All four use the selected preview mode. The first three use the selected time window; autocorrelation retains its full-signal or one-cycle definition. The PSD averages the selected window's spectrogram power densities, including silence, before conversion to dB. Segment detrending is disabled: subtracting an unweighted segment mean before Hann windowing can create a spurious near-DC feature. Finite probe duration, gating, and phase coding still produce real spectral spread outside nominal frequencies. These plots describe electrical/digital waveforms, not the transducer or speaker response. Preview limit: 2,000,000 samples in the selected mode, so the available hypothetical duration depends on the entered Hypothetical FS.\n\n"
            "Hypothetical preview and Hypothetical FS (Hz): unchecked plots the exact float32 output samples that the WAV contains and Start sends to the selected device; the rate field is disabled. Checked enables the rate field and synthesizes a separate float32 waveform at that positive whole-number sample rate from the captured Tone, Chirp, or M-sequence parameters. The default is 2,000,000 Hz. The hypothetical waveform is for inspection only and is never sent to the device or saved as the WAV. M-sequence chip counts stay the same, but integer chip lengths and effective chip rate can differ at the entered rate. The TXT identifies the plot mode and rate and records hypothetical chip timing when relevant. The hypothetical preview does not model a speaker's acoustic response. Saving a hypothetical PNG requires its full sequence to fit the 2,000,000-sample preview limit.\n\n"
            "Plot time window: Duration and Start sliders to the left of the figure select which samples appear in the waveform, PSD, spectrogram, and saved PNG. Duration ranges from 1 ms to the selected preview's full duration; for shorter signals it is fixed at the full duration. Start ranges from zero to the latest start that keeps the selected window inside that preview. The TXT records the exact selected sample indices and times at the plot rate. The WAV always holds the complete real output buffer.\n\n"
            "Autocorrelation: Tone and Chirp show the ENVELOPE of the selected preview mode's complete signal aperiodic autocorrelation, including repeats and silence, normalized to its zero-lag value. M sequences show the circular (periodic) autocorrelation ENVELOPE of ONE full cycle in the selected mode; any duty-cycle silence is included, and at 100% duty a dashed line marks the ideal 1/N sidelobe floor. The plot title and vertical-axis label identify the mode. Both use a −100 dB display floor and the same zero-lag zoom. Silent signals are labeled rather than normalized.\n\n"
            "Lag half-width (ms): positive width on EACH side of zero lag in the autocorrelation plot. The default 5 gives a fixed view from −5 to +5 ms. Change it to inspect a narrower central peak or wider sidelobe region. The same range is retained when changing signal type or repeats; this is a display setting and does not change playback or WAV samples. Lags are displayed in milliseconds.\n\n"
            "Edge taper: Tone and Chirp use a Tukey window with alpha = 0.1 to smooth the first and last 5% of their active interval. M sequences have NO burst taper: the first and last chips retain the same source amplitude as all other chips. Speaker resampling still filters transitions and may introduce ringing; it is separate from amplitude tapering."
        )
        descriptions = dict()
        descriptions["Tone"] = "TONE\n\nFundamental frequency (Hz): frequency of a sinusoidal probe.\n\nThe tone occupies the first duty-cycle fraction of T_cycle. For T_cycle = 0.100 s and duty-cycle = 50%, a 0.050 s tapered tone is followed by 0.050 s silence.\n\n"
        descriptions["Chirp"] = "CHIRP\n\nStart frequency (Hz): frequency at the beginning of the cycle.\n\nStop frequency (Hz): frequency that the sweep reaches at the end of the ACTIVE pulse. Both upward and downward sweeps are supported.\n\nThe full start-to-stop sweep always fits inside the active interval, so the transmitted bandwidth does not depend on duty-cycle. The instantaneous frequency is start + (stop − start) × t / active duration.\n\nFor start = 5000 Hz, stop = 4000 Hz, T_cycle = 0.100 s, and duty-cycle = 50%, the sweep runs from 5000 Hz to 4000 Hz in 0.050 s, then the remaining 0.050 s is silent.\n\n"
        descriptions["M sequences"] = "M SEQUENCES\n\nCarrier frequency (Hz): sinusoid whose sign switches according to a repeatable maximum-length binary sequence. Signs +1 and −1 correspond to carrier phase 0° and 180° (binary phase-shift keying). No whole-burst taper is applied.\n\nSequence order: integer from 2 to 16. The sequence contains N = 2^order − 1 chips. SciPy's default taps and all-ones initial state produce the same code on every cycle.\n\nM-sequence timing: cycle samples = round(FS × T_cycle), active budget = round(cycle samples × duty-cycle / 100), and samples per chip = floor(active budget / N), rounded down to a length that holds a WHOLE NUMBER OF CARRIER CYCLES. Every chip has exactly that many synthesis samples, with at least two samples per chip required for a nonzero duty. Whole carrier cycles per chip keep the carrier continuous across chips and across repeated periods, which the ideal periodic autocorrelation requires. Actual active samples = N × samples per chip. At 100% duty the cycle equals the active samples (a continuous loop); otherwise the remaining cycle samples are silent. Python's nearest-sample round uses ties to even.\n\nFor carrier 10000 Hz, order 7, synthesis FS = 44000 Hz, T_cycle = 0.0635 s, and duty-cycle = 100%, a chip must be a multiple of 22 samples (5 carrier cycles). The 2794-sample budget fits 127 chips × 22 samples exactly, giving 2000 chips/s, a main lobe from 8 to 12 kHz, and a 63.5 ms period that loops with no gap. The readout displays actual timing alongside the requested duration and duty.\n\nMain-lobe checks: carrier − actual chip rate must be above 0 Hz, and carrier + actual chip rate must be below min(synthesis FS, output FS) / 2. These checks place the nominal main lobe between DC and both Nyquist limits; rectangular chip transitions still have spectral sidelobes outside that lobe, so these checks do not guarantee zero aliasing.\n\nSource and output timing: chip lengths are equal integers at synthesis FS. Speaker resampling may place chip transitions between native output samples and filters the rectangular transitions; it does not guarantee equal integer output-sample counts per chip. Exact previews, the WAV, and playback use the same final prepared samples. The hypothetical mode computes separate chip timing at the entered Hypothetical FS.\n\nCircular envelope: one full cycle of the selected preview mode is used. Compute R = real(ifft(fft(x) × conj(fft(x)))), then the Hilbert envelope abs(hilbert(R)). Normalize the envelope by its zero-lag value, center the lag axis, and plot 20 × log10(normalized envelope). Additional repeats do not participate; duty-cycle silence does. For the textbook result (a single peak one chip wide on a flat floor 1/N below it) use 100% duty and at least two repeats, and process any received period after the first.\n\nThis is a carrier-modulated waveform diagnostic. Speaker resampling and filtering can change its shape from the textbook MLS result; DAQ output is not resampled. No M-sequence burst taper is applied. At 0% duty the output is silent and chip rate is shown as zero. Use the actual saved waveform as the matched-filter template. Physical source/receiver bandwidth further shapes the received signal.\n\n"
        for signal_type in ("Tone", "Chirp", "M sequences"):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=signal_type)
            scrollbar = ttk.Scrollbar(tab, orient="vertical")
            scrollbar.pack(side="right", fill="y")
            text = tk.Text(tab, wrap="word", padx=16, pady=12, font="TkDefaultFont", yscrollcommand=scrollbar.set)
            text.pack(fill="both", expand=True)
            scrollbar.config(command=text.yview)
            text.insert("1.0", descriptions[signal_type] + common)
            text.bold_font = tkfont.nametofont("TkDefaultFont").copy()
            text.bold_font.configure(weight="bold")
            text.tag_configure("parameter", font=text.bold_font)
            for match in re.finditer(r"^(?:Fundamental frequency \(Hz\)|Start frequency \(Hz\)|Stop frequency \(Hz\)|Carrier frequency \(Hz\)|Sequence order|M-sequence timing|Circular envelope|Main-lobe checks|Source and output timing|Device|AO channel|Signal type|T_cycle \(s\)|Duty-cycle %|Synthesis FS \(Hz\)|Output FS / plotted Nyquist|Resampling filter|Amplitude|Repeats|Total playback duration|Save WAV \+ TXT \+ PNG|Start / Stop|Previews, left to right|Hypothetical preview and Hypothetical FS \(Hz\)|Plot time window|Autocorrelation|Lag half-width \(ms\)|Edge taper):", text.get("1.0", "end-1c"), re.MULTILINE):
                text.tag_add("parameter", f"1.0+{match.start()}c", f"1.0+{match.end() - 1}c")
            text.config(state="disabled")
            if signal_type == self.signal_type_var.get():
                notebook.select(tab)

    def _update_preview(self):
        self.preview_after_id = None
        self.output_info_var.set("")
        if self.preview_colorbar is not None:
            self.preview_colorbar.remove()
            self.preview_colorbar = None
        self.preview_axes.clear()
        self.preview_axes.set_xlabel("Time (s)")
        self.preview_axes.set_ylabel("Frequency (kHz)")
        self.preview_axes.set_title("Spectrogram")
        self.psd_axes.clear()
        self.psd_axes.set_title("Time-averaged PSD")
        self.psd_axes.set_xlabel("Frequency (kHz)")
        speaker = self.device_var.get() == SPEAKER_DEVICE
        self.waveform_axes.clear()
        self.waveform_axes.set_title("Time-signal waveform")
        self.waveform_axes.set_xlabel("Time (s)")
        self.waveform_axes.set_ylabel("Amplitude (full scale)" if speaker else "Voltage (V)")
        self.waveform_axes.grid(alpha=0.25)
        self.autocorrelation_axes.clear()
        self.autocorrelation_axes.set_title("Autocorrelation")
        self.autocorrelation_axes.set_xlabel("Lag (ms)")
        self.autocorrelation_axes.set_ylabel("Normalized magnitude (dB)")
        self.autocorrelation_axes.set_ylim(-100, 3)
        self.autocorrelation_axes.grid(alpha=0.25)
        psd_label = "PSD (dB re 1 full-scale²/Hz)" if speaker else "PSD (dB re 1 V²/Hz)"
        self.psd_axes.set_ylabel(psd_label)

        try:
            lag_half_width_ms = float(self.lag_half_width_var.get())
            if not np.isfinite(lag_half_width_ms) or lag_half_width_ms <= 0:
                raise ValueError("Lag half-width must be a finite, positive number of milliseconds.")
            real_prepared = self.active_output if self.running else self._prepare_output()
            self.output_info_var.set(real_prepared['description'])
            prepared = self._plot_prepared_output(real_prepared)
            self.sequence_info_var.set(prepared['sequence_info'])
            if prepared['total_samples'] > 2_000_000:
                raise ValueError("Preview limited to 2,000,000 prepared output samples; reduce duration or repeats.")
        except Exception as exc:
            self.preview_frame.config(text="Output preview unavailable")
            self.output_info_var.set(str(exc))
            self.preview_axes.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True, transform=self.preview_axes.transAxes)
            self.preview_canvas.draw_idle()
            return

        data = prepared['data']
        fs = prepared['fs']
        start_sample, end_sample = self._plot_window_samples(prepared)
        window_data = data[start_sample:end_sample]
        speaker = prepared['device'] == SPEAKER_DEVICE
        psd_label = "PSD (dB re 1 full-scale²/Hz)" if speaker else "PSD (dB re 1 V²/Hz)"
        self.waveform_axes.set_ylabel("Amplitude (full scale)" if speaker else "Voltage (V)")
        self.psd_axes.set_ylabel(psd_label)
        label = "Hypothetical preview — not sent or saved as WAV" if self.hypothetical_preview_var.get() else ("Playing output" if self.running else "Prepared output")
        self.preview_frame.config(text=f"{label} — {prepared['output_name']} — Plot FS: {fs} Hz; real output FS: {real_prepared['fs']} Hz — waveform, PSD, spectrogram, autocorrelation")
        sample_times = np.arange(start_sample, end_sample) / fs
        self.waveform_axes.plot(sample_times, window_data, linewidth=0.7)
        self.waveform_axes.set_xlim(start_sample / fs, end_sample / fs)
        nperseg = self._spectrogram_nperseg(prepared, len(window_data))
        # Preserve the original segments: mean subtraction before Hann windowing can introduce near-DC power.
        frequencies, times, psd = spectrogram(window_data, fs=fs, window="hann", nperseg=nperseg, noverlap=int(0.75 * nperseg), scaling="density", mode="psd", detrend=False)
        times = times + start_sample / fs
        psd = np.asarray(psd, dtype=np.float64)
        psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
        peak_db = max(float(np.max(psd_db)), -160.0)
        image = self.preview_axes.pcolormesh(times, frequencies / 1000, psd_db, shading="auto", cmap="viridis", vmin=peak_db - 80, vmax=peak_db)
        self.preview_axes.set_xlim(start_sample / fs, end_sample / fs)
        self.preview_axes.set_ylim(0, fs / 2000)
        self.preview_colorbar = self.preview_figure.colorbar(image, ax=self.preview_axes)
        self.preview_colorbar.set_label(psd_label)
        mean_psd = np.mean(psd, axis=1)
        mean_psd_db = 10 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny))
        self.psd_axes.plot(frequencies / 1000, mean_psd_db)
        self.psd_axes.set_xlim(0, fs / 2000)
        mean_peak_db = max(float(np.max(mean_psd_db)), -160.0)
        self.psd_axes.set_ylim(mean_peak_db - 80, mean_peak_db + 5)
        self.psd_axes.grid(alpha=0.25)
        self._plot_autocorrelation(prepared, lag_half_width_ms)
        self.preview_canvas.draw_idle()

    def _plot_autocorrelation(self, prepared, lag_half_width_ms):
        self.autocorrelation_axes.set_xlim(-lag_half_width_ms, lag_half_width_ms)
        fs = prepared['fs']
        m_sequence = prepared['m_sequence_timing'] is not None
        if m_sequence:
            # One full transmitted cycle: this is the periodic autocorrelation a looped transmission really has.
            data = prepared['cycle']
            self.autocorrelation_axes.set_title("MLS periodic autocorrelation")
            self.autocorrelation_axes.set_ylabel("Normalized envelope (dB)")
        else:
            data = prepared['data']
        # Promote the prepared float32 sample values for accurate correlation arithmetic.
        analysis_data = np.asarray(data, dtype=np.float64)
        if not np.any(analysis_data):
            self.autocorrelation_axes.text(0.5, 0.5, "Silent signal\nAutocorrelation is zero", ha="center", va="center", transform=self.autocorrelation_axes.transAxes)
            return
        if m_sequence:
            spectrum = np.fft.fft(analysis_data)
            correlation = np.fft.ifft(spectrum * np.conj(spectrum)).real
            envelope = np.abs(hilbert(correlation))
            normalized_envelope = envelope / envelope[0]
            centered_envelope = np.fft.fftshift(normalized_envelope)
            lags = 1000 * (np.arange(len(correlation)) - len(correlation) // 2) / fs
            self.autocorrelation_axes.plot(lags, 20 * np.log10(np.maximum(centered_envelope, 1e-5)), label="One cycle, looped")
            if len(analysis_data) == prepared['m_sequence_timing'].get('output_active_samples'):
                ideal_floor_db = -20 * np.log10(prepared['m_sequence_timing']['chips'])
                self.autocorrelation_axes.axhline(ideal_floor_db, color="gray", linestyle="--", linewidth=0.8, label=f"Ideal floor {ideal_floor_db:.1f} dB")
        else:
            correlation = correlate(analysis_data, analysis_data, mode="full", method="fft")
            lags = 1000 * (np.arange(len(correlation)) - (len(data) - 1)) / fs
            # Plot the envelope, not the raw carrier-frequency oscillation, so the pulse-compression shape is visible.
            envelope = np.abs(hilbert(correlation, N=next_fast_len(2 * len(correlation))))[:len(correlation)]
            normalized_envelope = envelope / envelope[len(data) - 1]
            self.autocorrelation_axes.set_title("Autocorrelation envelope")
            self.autocorrelation_axes.set_ylabel("Normalized envelope (dB)")
            self.autocorrelation_axes.plot(lags, 20 * np.log10(np.maximum(normalized_envelope, 1e-5)), label="Full signal")
        self.autocorrelation_axes.legend(loc="lower right")

    def _browse_wav_path(self):
        path = filedialog.asksaveasfilename(title="Save signal WAV", defaultextension=".wav", initialfile=self.wav_path_var.get(), filetypes=[("WAV files", "*.wav")])
        if path:
            self.wav_path_var.set(path)

    def _on_device_change(self, event=None):
        device = self.device_var.get()
        channels = self._discover_ao_channels(device)
        self.available_channels = channels
        self.ao_combo.config(values=channels)
        if channels:
            self.ao_channel_var.set(channels[0])
        else:
            self.ao_channel_var.set("Not used" if device == SPEAKER_DEVICE else "No AO channels")
        self.ao_combo.config(state="disabled" if device == SPEAKER_DEVICE else "readonly")
        self.amplitude_label.config(text="Amplitude (full scale, 0–1):" if device == SPEAKER_DEVICE else "Amplitude (V peak):")
        self._schedule_preview()

    def _normalize_ao_channel(self, channel_name):
        if channel_name is None:
            return "ao0"
        channel = str(channel_name).strip().lower()
        if not channel.startswith("ao"):
            channel = "ao" + channel.lstrip("ao")
        return channel

    def _read_parameters(self):
        signal_type = self.signal_type_var.get()
        f0 = float(self.f0_var.get())
        f1 = float(self.f1_var.get()) if signal_type == "Chirp" else f0
        t_cycle = float(self.t_cycle_var.get())
        duty_cycle = float(self.duty_cycle_var.get())
        fs = float(self.fs_var.get())
        amplitude = float(self.amplitude_var.get())
        repeats = int(self.repeats_var.get())
        order = int(self.order_var.get()) if signal_type == "M sequences" else 7
        if signal_type not in ("Tone", "Chirp", "M sequences"):
            raise ValueError("Select Tone, Chirp, or M sequences.")
        if not np.all(np.isfinite([f0, f1, t_cycle, duty_cycle, fs, amplitude])):
            raise ValueError("Enter finite signal parameters.")
        if fs <= 0 or t_cycle <= 0 or repeats < 1:
            raise ValueError("Use positive FS and T_cycle, at least two samples per cycle, and Repeats ≥ 1.")
        if not fs.is_integer():
            raise ValueError("FS must be a whole number of samples per second so the WAV and playback rates agree exactly.")
        if duty_cycle < 0 or duty_cycle > 100:
            raise ValueError("Duty-cycle must be between 0 and 100%.")
        if signal_type == "M sequences":
            if order < 2 or order > 16:
                raise ValueError("Sequence order must be an integer from 2 to 16.")
            n_cycle, active_budget, samples_per_chip, n_active = self._m_sequence_timing(f0, t_cycle, duty_cycle, fs, order)
            if duty_cycle > 0 and samples_per_chip < 2:
                chip_step = self._m_sequence_chip_step(f0, fs)
                raise ValueError(f"M sequence needs equal chips of at least two samples that each hold a whole number of carrier cycles. At this carrier and synthesis FS a chip must be a multiple of {chip_step} samples, and {2**order - 1} such chips do not fit in the active interval. Increase T_cycle or duty-cycle, lower the order, or choose a carrier that divides the synthesis FS more evenly.")
        else:
            n_cycle = int(round(t_cycle * fs))
            n_active = int(round(n_cycle * duty_cycle / 100))
            if duty_cycle > 0 and n_active < 2:
                raise ValueError("The active duration must contain at least two samples; increase duty-cycle, T_cycle, or FS.")
        if n_cycle < 2:
            raise ValueError("T_cycle must contain at least two synthesis samples.")
        if min(f0, f1) < 0 or max(f0, f1) >= fs / 2:
            raise ValueError("Frequencies must be nonnegative and below FS / 2.")
        amplitude_limit = 1 if self.device_var.get() == SPEAKER_DEVICE else 10
        if amplitude < 0 or amplitude > amplitude_limit:
            raise ValueError(f"Amplitude must be between 0 and {amplitude_limit} for this output.")
        if signal_type == "M sequences":
            if n_active:
                chip_rate = fs / samples_per_chip
                self._check_m_sequence_band(f0, chip_rate, fs, fs)
        return signal_type, f0, f1, t_cycle, duty_cycle, fs, amplitude, repeats, order

    def _m_sequence_chip_step(self, f0, fs):
        # Smallest chip length (in samples) that holds a whole number of carrier cycles.
        if f0 <= 0:
            return 1
        return (Fraction(f0).limit_denominator(1000) / Fraction(int(fs))).denominator

    def _m_sequence_timing(self, f0, t_cycle, duty_cycle, fs, order):
        n_cycle = int(round(t_cycle * fs))
        active_budget = int(round(n_cycle * duty_cycle / 100))
        n_chips = 2**order - 1
        # Textbook phase-coded MLS: every chip holds the same whole number of carrier cycles, so the
        # carrier is continuous from one period to the next and the periodic autocorrelation is ideal.
        chip_step = self._m_sequence_chip_step(f0, fs)
        samples_per_chip = (active_budget // n_chips) // chip_step * chip_step
        n_active = n_chips * samples_per_chip
        if duty_cycle >= 100 and n_active:
            # Continuous transmission: the cycle is exactly one sequence period, with no silent remainder.
            n_cycle = n_active
        return n_cycle, active_budget, samples_per_chip, n_active

    def _check_m_sequence_band(self, carrier_frequency, chip_rate, source_fs, output_fs):
        if carrier_frequency - chip_rate <= 0:
            raise ValueError(f"Carrier frequency − chip rate must be above 0 Hz; actual chip rate is {chip_rate:.6g} chips/s. Increase carrier frequency or active duration, or lower sequence order.")
        nyquist = min(source_fs, output_fs) / 2
        if carrier_frequency + chip_rate >= nyquist:
            raise ValueError(f"Carrier frequency + chip rate must be below {nyquist:.6g} Hz (the lower synthesis/output Nyquist limit); actual chip rate is {chip_rate:.6g} chips/s.")

    def _make_signal(self, signal_type, f0, f1, t_cycle, duty_cycle, fs, amplitude, order):
        if signal_type == "M sequences":
            n_cycle, active_budget, samples_per_chip, n_active = self._m_sequence_timing(f0, t_cycle, duty_cycle, fs, order)
        else:
            n_cycle = int(round(t_cycle * fs))
            n_active = int(round(n_cycle * duty_cycle / 100))
        cycle = np.zeros(n_cycle)
        if n_active == 0:
            return cycle
        t = np.arange(n_active) / fs
        if signal_type == "Chirp":
            # Textbook linear FM pulse: the sweep runs from start to stop frequency across the active pulse.
            t_active = n_active / fs
            probe = amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t_active)))
        elif signal_type == "Tone":
            probe = amplitude * np.sin(2 * np.pi * f0 * t)
        else:
            sequence, state = max_len_seq(order)
            bipolar_sequence = 2 * sequence.astype(float) - 1
            # Every source chip has exactly the same integer sample count, including the first and last.
            chip_values = np.repeat(bipolar_sequence, samples_per_chip)
            probe = amplitude * chip_values * np.sin(2 * np.pi * f0 * t)
        if signal_type in ("Tone", "Chirp") and n_active > 2:
            probe *= tukey(n_active, alpha=0.1)
        cycle[:n_active] = probe
        return cycle

    def _make_output(self, cycle, repeats):
        return np.tile(cycle, repeats)

    def _prepare_output(self, for_playback=False):
        parameters = self._read_parameters()
        signal_type, f0, f1, t_cycle, duty_cycle, source_fs, amplitude, repeats, order = parameters
        device = self.device_var.get()
        channel = self._normalize_ao_channel(self.ao_channel_var.get())
        source_fs = int(source_fs)
        output_fs = source_fs
        audio_device = None
        output_name = device
        if device == SPEAKER_DEVICE:
            import sounddevice as sd
            output = sd.query_devices(kind="output")
            native_fs = float(output["default_samplerate"])
            if not np.isfinite(native_fs) or native_fs <= 0 or not native_fs.is_integer():
                raise ValueError("The default audio output must provide a positive whole-number sample rate for WAV/playback agreement.")
            output_fs = int(native_fs)
            audio_device = output["index"]
            output_name = output["name"]
            sd.check_output_settings(device=audio_device, channels=1, dtype="float32", samplerate=output_fs)
        m_sequence_timing = None
        if signal_type == "M sequences":
            source_cycle_samples, active_budget, samples_per_chip, n_active = self._m_sequence_timing(f0, t_cycle, duty_cycle, source_fs, order)
            n_chips = 2**order - 1
            chip_rate = source_fs / samples_per_chip if n_active else 0
            if n_active:
                self._check_m_sequence_band(f0, chip_rate, source_fs, output_fs)
            m_sequence_timing = dict()
            m_sequence_timing['chips'] = n_chips
            m_sequence_timing['samples_per_chip'] = samples_per_chip
            m_sequence_timing['active_budget_samples'] = active_budget
            m_sequence_timing['active_samples'] = n_active
            m_sequence_timing['active_duration'] = n_active / source_fs
            m_sequence_timing['effective_duty'] = 100 * n_active / source_cycle_samples
            m_sequence_timing['chip_rate'] = chip_rate
            m_sequence_timing['carrier_cycles_per_chip'] = f0 * samples_per_chip / source_fs
        else:
            source_cycle_samples = int(round(t_cycle * source_fs))
        key = (device, channel, parameters, output_fs, audio_device, output_name)
        if self.prepared_output is not None and self.prepared_output['key'] == key:
            if self.prepared_output['data'] is not None or not for_playback:
                return self.prepared_output

        ratio = Fraction(output_fs, source_fs)
        cycle_samples = (source_cycle_samples * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        if m_sequence_timing is not None:
            output_active_samples = (m_sequence_timing['active_samples'] * ratio.numerator + ratio.denominator - 1) // ratio.denominator
            m_sequence_timing['output_active_samples'] = output_active_samples
            m_sequence_timing['output_active_duration'] = output_active_samples / output_fs
        total_samples = cycle_samples * repeats
        sequence_description = f"{repeats} cycle(s)"
        prepared = dict()
        prepared['key'] = key
        prepared['parameters'] = parameters
        prepared['device'] = device
        prepared['channel'] = channel
        prepared['audio_device'] = audio_device
        prepared['output_name'] = output_name
        prepared['source_fs'] = source_fs
        prepared['source_cycle_samples'] = source_cycle_samples
        prepared['fs'] = output_fs
        prepared['cycle_samples'] = cycle_samples
        prepared['total_samples'] = total_samples
        prepared['description'] = f"Synthesis FS: {source_fs} Hz | Real output FS (WAV/playback): {output_fs} Hz | Real output Nyquist: {output_fs / 2000:g} kHz | {sequence_description} | Total playback duration: {total_samples / output_fs:.6g} s"
        prepared['sequence_info'] = ""
        prepared['m_sequence_timing'] = m_sequence_timing
        if signal_type == "M sequences":
            prepared['sequence_info'] = f"Source MLS: {n_chips} chips × {samples_per_chip} samples/chip | {m_sequence_timing['carrier_cycles_per_chip']:.6g} carrier cycles/chip | Chip rate: {chip_rate:.6g} chips/s | Active: {1000 * n_active / source_fs:.6g} ms (requested {1000 * t_cycle * duty_cycle / 100:.6g} ms) | Duty: {m_sequence_timing['effective_duty']:.6g}% (requested {duty_cycle:g}%) | Source cycle: {1000 * source_cycle_samples / source_fs:.6g} ms (requested {1000 * t_cycle:.6g} ms)"
        prepared['data'] = None
        prepared['cycle'] = None
        prepared['m_sequence_period'] = None
        if total_samples > 2_000_000 and not for_playback:
            self.prepared_output = prepared
            return prepared

        cycle = self._make_signal(signal_type, f0, f1, t_cycle, duty_cycle, source_fs, amplitude, order)
        if output_fs != source_fs:
            max_rate = max(ratio.numerator, ratio.denominator)
            half_length = 40 * max_rate
            # A longer, high-rejection interpolation filter retains the source band while suppressing images.
            resampling_filter = firwin(2 * half_length + 1, 1 / max_rate, window=("kaiser", 10.0))
            cycle = resample_poly(cycle, ratio.numerator, ratio.denominator, window=resampling_filter)
        cycle = np.asarray(cycle, dtype=np.float32)
        if device == SPEAKER_DEVICE and np.max(np.abs(cycle)) > 1:
            raise ValueError("Prepared speaker samples exceed full scale after resampling; lower the amplitude.")
        data = np.ascontiguousarray(self._make_output(cycle, repeats), dtype=np.float32)
        data.setflags(write=False)
        prepared['data'] = data
        prepared['cycle'] = data[:cycle_samples]
        if m_sequence_timing is not None:
            prepared['m_sequence_period'] = data[:output_active_samples]
        self.prepared_output = prepared
        return prepared

    def _plot_prepared_output(self, prepared):
        if not self.hypothetical_preview_var.get():
            return prepared
        hypothetical_fs_value = float(self.hypothetical_fs_var.get())
        if not np.isfinite(hypothetical_fs_value) or hypothetical_fs_value <= 0 or not hypothetical_fs_value.is_integer():
            raise ValueError("Hypothetical FS must be a positive whole number of samples per second.")
        hypothetical_fs = int(hypothetical_fs_value)
        if self.hypothetical_output is not None and self.hypothetical_output['real_key'] == prepared['key'] and self.hypothetical_output['fs'] == hypothetical_fs:
            return self.hypothetical_output
        signal_type, f0, f1, t_cycle, duty_cycle, source_fs, amplitude, repeats, order = prepared['parameters']
        if max(f0, f1) >= hypothetical_fs / 2:
            raise ValueError(f"Hypothetical preview frequencies must be below {hypothetical_fs / 2:g} Hz (half of Hypothetical FS).")
        if signal_type == "M sequences":
            cycle_samples, active_budget, samples_per_chip, active_samples = self._m_sequence_timing(f0, t_cycle, duty_cycle, hypothetical_fs, order)
            if duty_cycle > 0 and samples_per_chip < 2:
                raise ValueError(f"The selected M-sequence settings cannot fit equal chips at Hypothetical FS = {hypothetical_fs} Hz.")
            chip_rate = hypothetical_fs / samples_per_chip if active_samples else 0
            if active_samples:
                self._check_m_sequence_band(f0, chip_rate, hypothetical_fs, hypothetical_fs)
        else:
            cycle_samples = int(round(t_cycle * hypothetical_fs))
        if cycle_samples < 2:
            raise ValueError("T_cycle must contain at least two hypothetical samples; increase Hypothetical FS or T_cycle.")
        if signal_type in ("Tone", "Chirp") and duty_cycle > 0 and int(round(cycle_samples * duty_cycle / 100)) < 2:
            raise ValueError("The active pulse must contain at least two hypothetical samples; increase Hypothetical FS or duty-cycle.")
        total_samples = cycle_samples * repeats
        if total_samples > 2_000_000:
            raise ValueError(f"The hypothetical preview at {hypothetical_fs} Hz exceeds 2,000,000 samples. Reduce T_cycle, Repeats, or Hypothetical FS, or switch to exact output preview.")
        cycle = np.asarray(self._make_signal(signal_type, f0, f1, t_cycle, duty_cycle, hypothetical_fs, amplitude, order), dtype=np.float32)
        data = np.ascontiguousarray(self._make_output(cycle, repeats), dtype=np.float32)
        data.setflags(write=False)
        hypothetical = dict(prepared)
        hypothetical['real_key'] = prepared['key']
        hypothetical['fs'] = hypothetical_fs
        hypothetical['source_fs'] = hypothetical_fs
        hypothetical['source_cycle_samples'] = cycle_samples
        hypothetical['cycle_samples'] = cycle_samples
        hypothetical['total_samples'] = total_samples
        hypothetical['cycle'] = data[:cycle_samples]
        hypothetical['data'] = data
        hypothetical['output_name'] = f"Hypothetical {hypothetical_fs} Hz synthesis — not sent"
        hypothetical['reference_fs'] = prepared['fs']
        if signal_type == "M sequences":
            hypothetical['m_sequence_timing'] = dict(prepared['m_sequence_timing'])
            hypothetical['m_sequence_timing']['chips'] = 2**order - 1
            hypothetical['m_sequence_timing']['samples_per_chip'] = samples_per_chip
            hypothetical['m_sequence_timing']['active_samples'] = active_samples
            hypothetical['m_sequence_timing']['active_budget_samples'] = active_budget
            hypothetical['m_sequence_timing']['active_duration'] = active_samples / hypothetical_fs
            hypothetical['m_sequence_timing']['effective_duty'] = 100 * active_samples / cycle_samples
            hypothetical['m_sequence_timing']['chip_rate'] = chip_rate
            hypothetical['m_sequence_timing']['carrier_cycles_per_chip'] = f0 * samples_per_chip / hypothetical_fs
            hypothetical['m_sequence_timing']['output_active_samples'] = active_samples
            hypothetical['m_sequence_timing']['output_active_duration'] = active_samples / hypothetical_fs
            hypothetical['m_sequence_period'] = data[:active_samples]
            hypothetical['sequence_info'] = f"Hypothetical MLS: {2**order - 1} chips × {samples_per_chip} samples/chip | {f0 * samples_per_chip / hypothetical_fs:.6g} carrier cycles/chip | Chip rate: {chip_rate:.6g} chips/s | Active: {1000 * active_samples / hypothetical_fs:.6g} ms | Duty: {100 * active_samples / cycle_samples:.6g}% | Cycle: {1000 * cycle_samples / hypothetical_fs:.6g} ms"
        self.hypothetical_output = hypothetical
        return hypothetical

    def _play_speaker(self, prepared, stop_event):
        import sounddevice as sd

        if stop_event.is_set():
            return
        sd.play(prepared['data'], samplerate=prepared['fs'], device=prepared['audio_device'], blocking=False)
        try:
            while sd.get_stream().active:
                if stop_event.wait(0.02):
                    sd.stop()
                    break
            sd.wait()
        finally:
            sd.stop()

    def _finish_output(self, error):
        self.running = False
        self.active_output = None
        self.stop_event = None
        self.thread = None
        self.status_var.set(f"Error: {error}" if error else "Stopped")
        self._update_preview()
        if error:
            messagebox.showerror("Transmit Error", error)

    def _wav_parameter_text(self, prepared, plot_prepared, wav_path, lag_half_width_ms, start_sample, end_sample, hypothetical_fs_setting):
        signal_type, f0, f1, t_cycle, duty_cycle, requested_fs, amplitude, repeats, order = prepared['parameters']
        source_fs = prepared['source_fs']
        source_cycle_samples = prepared['source_cycle_samples']
        output_fs = prepared['fs']
        cycle_samples = prepared['cycle_samples']
        total_samples = prepared['total_samples']
        m_sequence_timing = prepared['m_sequence_timing']
        if m_sequence_timing is None:
            active_samples = int(round(source_cycle_samples * duty_cycle / 100))
        else:
            active_samples = m_sequence_timing['active_samples']
        lines = ["Acoustic signal WAV parameters", f"Created: {datetime.now().astimezone().isoformat(timespec='seconds')}", f"WAV file: {wav_path}", f"Plot PNG file: {Path(wav_path).with_suffix('.png')}", "WAV encoding: mono float32", "", "Requested signal settings", f"Signal type: {signal_type}"]
        if signal_type == "Tone":
            lines.append(f"Fundamental frequency (Hz): {f0}")
        elif signal_type == "Chirp":
            lines.append(f"Start frequency (Hz): {f0}")
            lines.append(f"Stop frequency (Hz): {f1}")
            lines.append("Chirp sweep: linear from start to stop across the active pulse")
        else:
            lines.append(f"Carrier frequency (Hz): {f0}")
            lines.append(f"Sequence order: {order}")
            lines.append("M-sequence code: SciPy max_len_seq default taps and initial state")
        lines.extend([f"T_cycle requested (s): {t_cycle}", f"Duty-cycle requested (%): {duty_cycle}", f"Synthesis FS (Hz): {source_fs}", f"Amplitude ({'digital full scale' if prepared['device'] == SPEAKER_DEVICE else 'V peak'}): {amplitude}", f"Repeats: {repeats}", "", "Prepared WAV and output buffer", f"Output device: {prepared['device']}", f"Output device name: {prepared['output_name']}", f"AO channel: {prepared['channel'] if prepared['device'] != SPEAKER_DEVICE else 'not used'}", f"Output FS / WAV sample rate (Hz): {output_fs}", f"Source cycle samples: {source_cycle_samples}", f"Source active samples per cycle: {active_samples}", f"Source active duration per cycle (s): {active_samples / source_fs}", f"Source cycle duration (s): {source_cycle_samples / source_fs}", f"Effective source duty-cycle (%): {100 * active_samples / source_cycle_samples}", f"Output cycle samples: {cycle_samples}", f"Output cycle duration (s): {cycle_samples / output_fs}", f"Total WAV samples: {total_samples}", f"Total WAV duration (s): {total_samples / output_fs}"])
        if prepared['device'] == SPEAKER_DEVICE:
            lines.append(f"Speaker output device index: {prepared['audio_device']}")
        if m_sequence_timing is None:
            lines.append("Active-pulse taper: Tukey window, alpha = 0.1, when active pulse exceeds 2 samples")
        else:
            lines.extend([f"M-sequence chips per cycle: {m_sequence_timing['chips']}", f"Requested active budget (source samples): {m_sequence_timing['active_budget_samples']}", f"Synthesis samples per chip: {m_sequence_timing['samples_per_chip']}", f"Chip rate (chips/s): {m_sequence_timing['chip_rate']}", f"Carrier cycles per chip: {m_sequence_timing['carrier_cycles_per_chip']}", f"Output active samples per cycle: {m_sequence_timing['output_active_samples']}", f"Output active duration per cycle (s): {m_sequence_timing['output_active_duration']}", "Active-pulse taper: none"])
        if source_fs == output_fs:
            lines.append("Speaker resampling: none")
        else:
            lines.append(f"Speaker resampling ratio (output/synthesis): {Fraction(output_fs, source_fs)}")
            lines.append("Speaker resampling: scipy.signal.resample_poly with FIR low-pass, 80 * max(up, down) + 1 taps, Kaiser beta = 10")
        lines.append(f"Autocorrelation lag half-width (ms): {lag_half_width_ms}")
        lines.append("Saved plot panels: waveform, time-averaged PSD, spectrogram")
        lines.append(f"Plot mode: {'hypothetical synthesis (not sent)' if plot_prepared is not prepared else 'exact prepared output'}")
        lines.append(f"Hypothetical FS setting (Hz): {hypothetical_fs_setting}")
        lines.append(f"Plot sample rate (Hz): {plot_prepared['fs']}")
        lines.append(f"Plot window start sample: {start_sample}")
        lines.append(f"Plot window end sample (exclusive): {end_sample}")
        lines.append(f"Plot window start (s): {start_sample / plot_prepared['fs']}")
        lines.append(f"Plot window end (s): {end_sample / plot_prepared['fs']}")
        lines.append(f"Plot window duration (s): {(end_sample - start_sample) / plot_prepared['fs']}")
        if plot_prepared is not prepared and signal_type == "M sequences":
            lines.append(f"Hypothetical samples per chip: {plot_prepared['m_sequence_timing']['samples_per_chip']}")
            lines.append(f"Hypothetical chip rate (chips/s): {plot_prepared['m_sequence_timing']['chip_rate']}")
            lines.append(f"Hypothetical carrier cycles per chip: {plot_prepared['m_sequence_timing']['carrier_cycles_per_chip']}")
            lines.append(f"Hypothetical active duration (s): {plot_prepared['m_sequence_timing']['active_duration']}")
            lines.append(f"Hypothetical effective duty-cycle (%): {plot_prepared['m_sequence_timing']['effective_duty']}")
            lines.append(f"Hypothetical cycle duration (s): {plot_prepared['cycle_samples'] / plot_prepared['fs']}")
        lines.append("The WAV contains the complete prepared buffer. Saving does not start playback.")
        return "\n".join(lines) + "\n"

    def _spectrogram_nperseg(self, prepared, window_samples):
        reference_fs = prepared.get('reference_fs', prepared['fs'])
        segment_samples = max(2, int(round(512 * prepared['fs'] / reference_fs)))
        return min(segment_samples, len(prepared['cycle']), window_samples)

    def _make_plot_png(self, prepared, start_sample, end_sample):
        data = prepared['data']
        fs = prepared['fs']
        window_data = data[start_sample:end_sample]
        speaker = prepared['device'] == SPEAKER_DEVICE
        psd_label = "PSD (dB re 1 full-scale²/Hz)" if speaker else "PSD (dB re 1 V²/Hz)"
        figure = Figure(figsize=(13, 4), dpi=150, constrained_layout=True)
        FigureCanvasAgg(figure)
        waveform_axes = figure.add_subplot(131)
        psd_axes = figure.add_subplot(132)
        spectrogram_axes = figure.add_subplot(133)
        waveform_axes.plot(np.arange(start_sample, end_sample) / fs, window_data, linewidth=0.7)
        waveform_axes.set_title("Time-signal waveform")
        waveform_axes.set_xlabel("Time (s)")
        waveform_axes.set_ylabel("Amplitude (full scale)" if speaker else "Voltage (V)")
        waveform_axes.set_xlim(start_sample / fs, end_sample / fs)
        waveform_axes.grid(alpha=0.25)
        nperseg = self._spectrogram_nperseg(prepared, len(window_data))
        frequencies, times, psd = spectrogram(window_data, fs=fs, window="hann", nperseg=nperseg, noverlap=int(0.75 * nperseg), scaling="density", mode="psd", detrend=False)
        times = times + start_sample / fs
        psd = np.asarray(psd, dtype=np.float64)
        psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
        peak_db = max(float(np.max(psd_db)), -160.0)
        image = spectrogram_axes.pcolormesh(times, frequencies / 1000, psd_db, shading="auto", cmap="viridis", vmin=peak_db - 80, vmax=peak_db)
        spectrogram_axes.set_title("Spectrogram")
        spectrogram_axes.set_xlabel("Time (s)")
        spectrogram_axes.set_ylabel("Frequency (kHz)")
        spectrogram_axes.set_xlim(start_sample / fs, end_sample / fs)
        spectrogram_axes.set_ylim(0, fs / 2000)
        figure.colorbar(image, ax=spectrogram_axes, label=psd_label)
        mean_psd = np.mean(psd, axis=1)
        mean_psd_db = 10 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny))
        psd_axes.plot(frequencies / 1000, mean_psd_db)
        psd_axes.set_title("Time-averaged PSD")
        psd_axes.set_xlabel("Frequency (kHz)")
        psd_axes.set_ylabel(psd_label)
        psd_axes.set_xlim(0, fs / 2000)
        mean_peak_db = max(float(np.max(mean_psd_db)), -160.0)
        psd_axes.set_ylim(mean_peak_db - 80, mean_peak_db + 5)
        psd_axes.grid(alpha=0.25)
        png = BytesIO()
        figure.savefig(png, format="png")
        return png.getvalue()

    def _save_output_files(self, prepared, plot_prepared, wav_path, lag_half_width_ms, start_sample, end_sample, hypothetical_fs_setting):
        from scipy.io import wavfile
        plot_png = self._make_plot_png(plot_prepared, start_sample, end_sample)
        parameter_text = self._wav_parameter_text(prepared, plot_prepared, wav_path, lag_half_width_ms, start_sample, end_sample, hypothetical_fs_setting)
        wavfile.write(wav_path, prepared['fs'], prepared['data'])
        Path(wav_path).with_suffix(".txt").write_text(parameter_text, encoding="utf-8")
        Path(wav_path).with_suffix(".png").write_bytes(plot_png)

    def save_output(self):
        if self.running:
            messagebox.showerror("Save error", "Stop playback before saving the current settings.")
            return
        try:
            wav_path = self.wav_path_var.get().strip()
            if not wav_path:
                raise ValueError("Choose a WAV file path before saving.")
            if not wav_path.lower().endswith(".wav"):
                wav_path += ".wav"
            prepared = self._prepare_output(for_playback=True)
            plot_prepared = self._plot_prepared_output(prepared)
            hypothetical_fs_setting = self.hypothetical_fs_var.get().strip()
            lag_half_width_ms = float(self.lag_half_width_var.get())
            if not np.isfinite(lag_half_width_ms) or lag_half_width_ms <= 0:
                raise ValueError("Lag half-width must be a finite, positive number of milliseconds.")
            start_sample, end_sample = self._plot_window_samples(plot_prepared)
            self.status_var.set("Saving WAV, TXT, and PNG...")
            self.root.update_idletasks()
            self._save_output_files(prepared, plot_prepared, wav_path, lag_half_width_ms, start_sample, end_sample, hypothetical_fs_setting)
        except Exception as exc:
            self.status_var.set(f"Save error: {exc}")
            messagebox.showerror("Save error", str(exc))
            return
        self.status_var.set(f"Saved {wav_path}, {Path(wav_path).with_suffix('.txt')}, and {Path(wav_path).with_suffix('.png')}")

    def _transmit_loop(self, prepared, stop_event):
        error = None
        try:
            data = prepared['data']
            fs = prepared['fs']
            if prepared['device'] == SPEAKER_DEVICE:
                self._play_speaker(prepared, stop_event)
            else:
                import nidaqmx
                from nidaqmx.constants import AcquisitionType

                with nidaqmx.Task() as ao:
                    ao.ao_channels.add_ao_voltage_chan(f"{prepared['device']}/{prepared['channel']}", min_val=-10.0, max_val=10.0)
                    ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(data))
                    ao.write(np.asarray(data, dtype=np.float64), auto_start=False)
                    if not stop_event.is_set():
                        ao.start()
                        deadline = time.monotonic() + 10.0 + len(data) / fs
                        while not ao.is_task_done():
                            if time.monotonic() > deadline:
                                raise TimeoutError("DAQ output did not finish within the expected duration plus 10 seconds.")
                            if stop_event.wait(0.02):
                                ao.stop()
                                break
                # A finished or stopped task holds its last voltage; return the output to 0 V.
                with nidaqmx.Task() as zero:
                    zero.ao_channels.add_ao_voltage_chan(f"{prepared['device']}/{prepared['channel']}", min_val=-10.0, max_val=10.0)
                    zero.write(0.0, auto_start=True)
        except Exception as exc:
            error = str(exc)
        finally:
            self.root.after(0, self._finish_output, error)

    def start_output(self):
        if self.running:
            return

        try:
            prepared = self._prepare_output(for_playback=True)
            if prepared['device'] != SPEAKER_DEVICE and self.ao_channel_var.get() not in self.available_channels:
                raise ValueError("Select an available AO channel.")
        except Exception as exc:
            messagebox.showerror("Output preparation error", str(exc))
            return

        self.active_output = prepared
        self.running = True
        self.stop_event = threading.Event()
        self.status_var.set(f"Playing {prepared['output_name']} at {prepared['fs']} Hz. Previews show this buffer; parameter edits apply after playback stops.")
        self._update_preview()
        self.thread = threading.Thread(target=self._transmit_loop, args=(prepared, self.stop_event), daemon=True)
        self.thread.start()

    def stop_output(self):
        if not self.running:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def _close(self):
        if self.running:
            self.stop_output()
            self.root.after(50, self._close)
            return
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChirpTransmitApp(root)
    root.mainloop()

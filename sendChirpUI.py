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
from fractions import Fraction
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk, messagebox, filedialog

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import spectrogram, resample_poly, max_len_seq, correlate
from scipy.signal.windows import tukey


SPEAKER_DEVICE = "default (laptop speaker)"


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
        self.save_wav_var = tk.BooleanVar(value=False)
        self.wav_path_var = tk.StringVar(value="chirp.wav")
        self.status_var = tk.StringVar(value="Idle")

        self.preview_after_id = None
        self.help_window = None
        self._build_ui()
        self._on_device_change()
        self._on_signal_change()
        for variable in (self.f0_var, self.f1_var, self.t_cycle_var, self.duty_cycle_var, self.order_var, self.fs_var, self.amplitude_var, self.repeats_var, self.lag_half_width_var):
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
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=self.preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
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
        ttk.Label(shared, text="FS (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=4)
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
        ttk.Checkbutton(output, text="Save generated signal to WAV", variable=self.save_wav_var).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=4)
        ttk.Label(output, text="WAV file:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(output, textvariable=self.wav_path_var).grid(row=1, column=1, sticky="ew", padx=5, pady=4)
        ttk.Button(output, text="Browse...", command=self._browse_wav_path).grid(row=1, column=2, padx=5, pady=4)
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
            "T_cycle (s): length of one complete probe-plus-silence cycle. Repeats includes both portions. The cycle sample count is floor(FS × T_cycle).\n\n"
            "Duty-cycle %: active fraction at the start of every cycle; the remainder is zero. 0% gives silence; 100% has no appended silent interval. Active samples = floor(cycle samples × duty-cycle / 100).\n\n"
            "FS (Hz): whole-number synthesis sample rate. Frequencies must be below FS / 2. For a DAQ, the prepared rate is FS. For speakers, one full cycle is resampled to the selected audio device's native rate BEFORE cycle repetition, plotting, saving, and playback. The cycle length rounds up to a whole output sample, so actual timing can differ slightly from the entered durations. Resampling applies anti-alias filtering; output bandwidth is limited by the native rate. The timing readout shows both synthesis and output rates.\n\n"
            "Amplitude: peak volts for DAQ output (0–10), or digital full-scale amplitude for speakers (0–1). Speaker volume also depends on macOS volume; this is not calibrated acoustic pressure. No automatic normalization is applied.\n\n"
            "Repeats: positive integer number of identical probe-plus-silence cycles. Every cycle restarts the waveform and internal M-sequence phase.\n\n"
            "Total playback duration: prepared cycle samples × repeats / output rate. The displayed duration uses actual sample counts and remains available when a sequence exceeds the preview limit. Entire sequences are prepared in memory.\n\n"
            "Save generated signal to WAV: writes the same prepared float32 buffer used by the previews and playback, including all repeats, duty-cycle silence, and speaker resampling. The WAV header uses the PREPARED OUTPUT rate, which can differ from synthesis FS for speakers. Playback reads this buffer directly; it does not reload the WAV. Choose its path with Browse.\n\n"
            "Start / Stop: Start captures one prepared buffer and its output device. Plots stay tied to that buffer while it is playing, even if signal parameters are edited; the new settings are prepared and shown after playback stops. The lag zoom control still works while playing. Stop interrupts playback. The WAV, when enabled, contains the FULL prepared sequence even if playback is stopped early. Speaker playback explicitly uses the captured audio-device index.\n\n"
            "Previews, left to right: waveform (amplitude versus time, including repeats and silence); time-averaged PSD; spectrogram (frequency versus time); autocorrelation. All four plots use samples from the single prepared output buffer after float32 conversion and speaker resampling. The PSD averages the spectrogram's linear power densities, including silence, before conversion to dB. Segment detrending is disabled: subtracting an unweighted segment mean before Hann windowing can create a spurious near-DC feature. Finite probe duration, gating, and phase coding still produce real spectral spread outside nominal frequencies. These plots describe the intended electrical/digital waveform, not the transducer or speaker response. Preview limit: 2,000,000 prepared output samples. Larger sequences can still be saved/played but display a preview-limit message. Entire sequences are prepared in memory.\n\n"
            "Autocorrelation: shows the complete prepared output buffer's aperiodic autocorrelation, including repeats and silence. It is normalized to its zero-lag value and shown as 20 × log10(abs(R) / R(0)), with a display floor of −100 dB. This normalization does not show processing gain. Silent signals are labeled rather than normalized.\n\n"
            "Lag half-width (ms): positive width on EACH side of zero lag in the autocorrelation plot. The default 5 gives a fixed view from −5 to +5 ms. Change it to inspect a narrower central peak or wider sidelobe region. The same range is retained when changing signal type or repeats; this is a display setting and does not change playback or WAV samples. Lags are displayed in milliseconds.\n\n"
            "Edge taper: a Tukey window with alpha = 0.1 smooths the first and last 5% of the active interval. This preserves the phase/frequency law while reducing abrupt burst edges. It does not smooth internal M-sequence phase switches."
        )
        descriptions = dict()
        descriptions["Tone"] = "TONE\n\nFundamental frequency (Hz): frequency of a sinusoidal probe.\n\nThe tone occupies the first duty-cycle fraction of T_cycle. For T_cycle = 0.100 s and duty-cycle = 50%, a 0.050 s tapered tone is followed by 0.050 s silence.\n\n"
        descriptions["Chirp"] = "CHIRP\n\nStart frequency (Hz): frequency at the beginning of the cycle.\n\nStop frequency (Hz): frequency that the sweep would reach at the end of the FULL T_cycle. Both upward and downward sweeps are supported.\n\nDuty-cycle gates the sweep; it does not compress the entire sweep into the active interval. The instantaneous frequency is start + (stop − start) × t / T_cycle.\n\nFor start = 5000 Hz, stop = 4000 Hz, T_cycle = 0.100 s, and duty-cycle = 50%, the active sweep approaches 4500 Hz at 0.050 s, then the remaining 0.050 s is silent. The last active sample is just before that boundary.\n\n"
        descriptions["M sequences"] = "M SEQUENCES\n\nCarrier frequency (Hz): sinusoid whose sign switches according to a repeatable maximum-length binary sequence. Signs +1 and −1 correspond to carrier phase 0° and 180° (binary phase-shift keying).\n\nSequence order: integer from 2 to 16. One sequence contains 2^order − 1 chips. SciPy's default taps and all-ones initial state produce the same sequence on every cycle.\n\nOne COMPLETE sequence fits into the active interval, followed by silence. Chip rate is derived, not independently entered: (2^order − 1) / actual active duration. For order 7, T_cycle = 0.100 s, and duty-cycle = 50%, 127 chips occupy 0.050 s: 2540 chips/s.\n\nReducing duty-cycle increases chip rate and broadens the spectrum. Each chip spans an integer number of samples, so chip durations may differ by one sample. At least two samples per chip are required. Carrier + chip rate must be below FS / 2 so the upper main-lobe edge fits; rectangular chips still have sidelobes and are not strictly band-limited.\n\nThe read-only line shows chip count, chip rate, and active duration. At 0% duty the output is silent and the displayed rate is 0. The tapered, finite carrier waveform does not retain the exact ideal periodic MLS autocorrelation; use the actual saved waveform as the matched-filter template. The physical source and receiver bandwidth further shape it.\n\n"
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
            for match in re.finditer(r"^(?:Fundamental frequency \(Hz\)|Start frequency \(Hz\)|Stop frequency \(Hz\)|Carrier frequency \(Hz\)|Sequence order|Device|AO channel|Signal type|T_cycle \(s\)|Duty-cycle %|FS \(Hz\)|Amplitude|Repeats|Total playback duration|Save generated signal to WAV|Start / Stop|Previews, left to right|Autocorrelation|Lag half-width \(ms\)|Edge taper):", text.get("1.0", "end-1c"), re.MULTILINE):
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
            prepared = self.active_output if self.running else self._prepare_output()
            self.output_info_var.set(prepared['description'])
            self.sequence_info_var.set(prepared['sequence_info'])
            if prepared['total_samples'] > 2_000_000:
                raise ValueError("Preview limited to 2,000,000 prepared output samples; reduce duration or repeats.")
        except Exception as exc:
            self.preview_frame.config(text="Output preview unavailable")
            if not self.output_info_var.get():
                self.output_info_var.set(str(exc))
            self.preview_axes.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True, transform=self.preview_axes.transAxes)
            self.preview_canvas.draw_idle()
            return

        data = prepared['data']
        cycle = prepared['cycle']
        fs = prepared['fs']
        speaker = prepared['device'] == SPEAKER_DEVICE
        psd_label = "PSD (dB re 1 full-scale²/Hz)" if speaker else "PSD (dB re 1 V²/Hz)"
        self.waveform_axes.set_ylabel("Amplitude (full scale)" if speaker else "Voltage (V)")
        self.psd_axes.set_ylabel(psd_label)
        label = "Playing output" if self.running else "Prepared output"
        self.preview_frame.config(text=f"{label} — {prepared['output_name']} at {fs} Hz — waveform, PSD, spectrogram, autocorrelation")
        sample_times = np.arange(len(data)) / fs
        self.waveform_axes.plot(sample_times, data, linewidth=0.7)
        self.waveform_axes.set_xlim(0, len(data) / fs)
        nperseg = min(512, len(cycle))
        # Preserve the original segments: mean subtraction before Hann windowing can introduce near-DC power.
        frequencies, times, psd = spectrogram(data, fs=fs, window="hann", nperseg=nperseg, noverlap=int(0.75 * nperseg), scaling="density", mode="psd", detrend=False)
        psd = np.asarray(psd, dtype=np.float64)
        psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
        peak_db = max(float(np.max(psd_db)), -160.0)
        image = self.preview_axes.pcolormesh(times, frequencies / 1000, psd_db, shading="auto", cmap="viridis", vmin=peak_db - 80, vmax=peak_db)
        self.preview_axes.set_xlim(0, len(data) / fs)
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
        self._plot_autocorrelation(data, fs, lag_half_width_ms)
        self.preview_canvas.draw_idle()

    def _plot_autocorrelation(self, data, fs, lag_half_width_ms):
        self.autocorrelation_axes.set_xlim(-lag_half_width_ms, lag_half_width_ms)
        # Promote the prepared float32 sample values for accurate correlation arithmetic.
        analysis_data = np.asarray(data, dtype=np.float64)
        if not np.any(analysis_data):
            self.autocorrelation_axes.text(0.5, 0.5, "Silent signal\nAutocorrelation is zero", ha="center", va="center", transform=self.autocorrelation_axes.transAxes)
            return
        correlation = correlate(analysis_data, analysis_data, mode="full", method="fft")
        lags = 1000 * (np.arange(len(correlation)) - (len(data) - 1)) / fs
        normalized_magnitude = np.abs(correlation) / correlation[len(data) - 1]
        self.autocorrelation_axes.plot(lags, 20 * np.log10(np.maximum(normalized_magnitude, 1e-5)), label="Full signal")
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
        if fs <= 0 or t_cycle <= 0 or repeats < 1 or int(t_cycle * fs) < 2:
            raise ValueError("Use positive FS and T_cycle, at least two samples per cycle, and Repeats ≥ 1.")
        if not fs.is_integer():
            raise ValueError("FS must be a whole number of samples per second so the WAV and playback rates agree exactly.")
        if duty_cycle < 0 or duty_cycle > 100:
            raise ValueError("Duty-cycle must be between 0 and 100%.")
        n_cycle = int(t_cycle * fs)
        n_active = int(n_cycle * duty_cycle / 100)
        if duty_cycle > 0 and n_active < 2:
            raise ValueError("The active duration must contain at least two samples; increase duty-cycle, T_cycle, or FS.")
        if min(f0, f1) < 0 or max(f0, f1) >= fs / 2:
            raise ValueError("Frequencies must be nonnegative and below FS / 2.")
        amplitude_limit = 1 if self.device_var.get() == SPEAKER_DEVICE else 10
        if amplitude < 0 or amplitude > amplitude_limit:
            raise ValueError(f"Amplitude must be between 0 and {amplitude_limit} for this output.")
        if signal_type == "M sequences":
            if order < 2 or order > 16:
                raise ValueError("Sequence order must be an integer from 2 to 16.")
            n_chips = 2**order - 1
            if n_active:
                chip_rate = n_chips * fs / n_active
                if n_active < 2 * n_chips:
                    raise ValueError("M sequence needs at least two samples per chip; lower order or increase active duration or FS.")
                if f0 + chip_rate >= fs / 2:
                    raise ValueError("Carrier frequency + chip rate must be below FS / 2; lower order or increase active duration or FS.")
        return signal_type, f0, f1, t_cycle, duty_cycle, fs, amplitude, repeats, order

    def _make_signal(self, signal_type, f0, f1, t_cycle, duty_cycle, fs, amplitude, order):
        n_cycle = int(t_cycle * fs)
        n_active = int(n_cycle * duty_cycle / 100)
        cycle = np.zeros(n_cycle)
        if n_active == 0:
            return cycle
        t = np.arange(n_active) / fs
        if signal_type == "Chirp":
            # Sweep slope uses the full T_cycle, even when the active interval is shorter.
            probe = amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t_cycle)))
        elif signal_type == "Tone":
            probe = amplitude * np.sin(2 * np.pi * f0 * t)
        else:
            sequence, state = max_len_seq(order)
            bipolar_sequence = 2 * sequence.astype(float) - 1
            # Fit every chip into the active interval; boundaries are quantized to samples.
            chip_indices = np.arange(n_active, dtype=np.int64) * len(sequence) // n_active
            probe = amplitude * bipolar_sequence[chip_indices] * np.sin(2 * np.pi * f0 * t)
        if n_active > 2:
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
        key = (device, channel, parameters, output_fs, audio_device, output_name)
        if self.prepared_output is not None and self.prepared_output['key'] == key:
            if self.prepared_output['data'] is not None or not for_playback:
                return self.prepared_output

        ratio = Fraction(output_fs, source_fs)
        source_cycle_samples = int(t_cycle * source_fs)
        cycle_samples = (source_cycle_samples * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        total_samples = cycle_samples * repeats
        sequence_description = f"{repeats} cycle(s)"
        prepared = dict()
        prepared['key'] = key
        prepared['device'] = device
        prepared['channel'] = channel
        prepared['audio_device'] = audio_device
        prepared['output_name'] = output_name
        prepared['source_fs'] = source_fs
        prepared['fs'] = output_fs
        prepared['cycle_samples'] = cycle_samples
        prepared['total_samples'] = total_samples
        prepared['description'] = f"{sequence_description} | Total playback duration: {total_samples / output_fs:.6g} s | Output: {output_fs} Hz (synthesis: {source_fs} Hz)"
        prepared['sequence_info'] = ""
        if signal_type == "M sequences":
            n_active = int(source_cycle_samples * duty_cycle / 100)
            n_chips = 2**order - 1
            chip_rate = n_chips * source_fs / n_active if n_active else 0
            prepared['sequence_info'] = f"Source code: {n_chips} chips/active interval | {chip_rate:.2f} chips/s | Active: {n_active / source_fs:.6g} s"
        prepared['data'] = None
        prepared['cycle'] = None
        if total_samples > 2_000_000 and not for_playback:
            self.prepared_output = prepared
            return prepared

        cycle = self._make_signal(signal_type, f0, f1, t_cycle, duty_cycle, source_fs, amplitude, order)
        if output_fs != source_fs:
            cycle = resample_poly(cycle, ratio.numerator, ratio.denominator)
        cycle = np.asarray(cycle, dtype=np.float32)
        if device == SPEAKER_DEVICE and np.max(np.abs(cycle)) > 1:
            raise ValueError("Prepared speaker samples exceed full scale after resampling; lower the amplitude.")
        data = np.ascontiguousarray(self._make_output(cycle, repeats), dtype=np.float32)
        data.setflags(write=False)
        prepared['data'] = data
        prepared['cycle'] = data[:cycle_samples]
        self.prepared_output = prepared
        return prepared

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

    def _transmit_loop(self, prepared, wav_path, stop_event):
        error = None
        try:
            data = prepared['data']
            fs = prepared['fs']
            if wav_path:
                from scipy.io import wavfile
                wavfile.write(wav_path, fs, data)

            if prepared['device'] == SPEAKER_DEVICE:
                self._play_speaker(prepared, stop_event)
            else:
                import nidaqmx
                from nidaqmx.constants import AcquisitionType

                with nidaqmx.Task() as ao:
                    ao.ao_channels.add_ao_voltage_chan(f"{prepared['device']}/{prepared['channel']}", min_val=-10.0, max_val=10.0)
                    ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(data))
                    ao.write(data, auto_start=False)
                    if not stop_event.is_set():
                        ao.start()
                        deadline = time.monotonic() + 10.0 + len(data) / fs
                        while not ao.is_task_done():
                            if time.monotonic() > deadline:
                                raise TimeoutError("DAQ output did not finish within the expected duration plus 10 seconds.")
                            if stop_event.wait(0.02):
                                ao.stop()
                                break
        except Exception as exc:
            error = str(exc)
        finally:
            self.root.after(0, self._finish_output, error)

    def start_output(self):
        if self.running:
            return

        try:
            wav_path = None
            if self.save_wav_var.get():
                wav_path = self.wav_path_var.get().strip()
                if not wav_path:
                    raise ValueError("Choose a WAV file path or uncheck WAV saving.")
                if not wav_path.lower().endswith(".wav"):
                    wav_path += ".wav"
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
        self.thread = threading.Thread(target=self._transmit_loop, args=(prepared, wav_path, self.stop_event), daemon=True)
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

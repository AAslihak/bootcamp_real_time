"""
sendChirpUI.py

GUI for transmitting a chirp from F0 to F1 on a selectable AO channel.

Features:
- select device and AO output channel
- configure F0, F1, T_chirp, sample rate, and amplitude
- start/stop transmit
"""

import threading
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np


class ChirpTransmitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chirp Transmit")
        self.root.geometry("600x400")

        self.running = False
        self.thread = None
        self.stop_event = None

        self.available_devices = self._discover_devices()
        default_device = self.available_devices[0] if self.available_devices else "Dev1"
        self.device_var = tk.StringVar(value=default_device)
        self.available_channels = self._discover_ao_channels(default_device)
        self.ao_channel_var = tk.StringVar(value=self.available_channels[0] if self.available_channels else "ao0")

        self.f0_var = tk.StringVar(value="8000")
        self.f1_var = tk.StringVar(value="12000")
        self.t_chirp_var = tk.StringVar(value="0.300")
        self.fs_var = tk.StringVar(value="44000")
        self.amplitude_var = tk.StringVar(value="0.5")
        self.repeats_var = tk.StringVar(value="1")
        self.save_wav_var = tk.BooleanVar(value=False)
        self.wav_path_var = tk.StringVar(value="chirp.wav")
        self.status_var = tk.StringVar(value="Idle")

        self._build_ui()

    def _discover_devices(self):
        try:
            from nidaqmx.system import System
            return list(System.local().devices.device_names) or ["Dev1"]
        except Exception:
            return ["Dev1"]

    def _discover_ao_channels(self, device_name):
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
            return channels[:4] or [f"ao{i}" for i in range(4)]
        except Exception:
            return ["ao0", "ao1", "ao2", "ao3"]

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Device:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.device_combo = ttk.Combobox(main, textvariable=self.device_var, values=self.available_devices, state="readonly", width=18)
        self.device_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        ttk.Label(main, text="AO channel:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ao_combo = ttk.Combobox(main, textvariable=self.ao_channel_var, values=self.available_channels, state="readonly", width=18)
        self.ao_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="F0 (Hz):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.f0_var, width=18).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="F1 (Hz):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.f1_var, width=18).grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="T_chirp (s):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.t_chirp_var, width=18).grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="FS (Hz):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.fs_var, width=18).grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="Amplitude (V):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.amplitude_var, width=18).grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="Repeats:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.repeats_var, width=18).grid(row=7, column=1, padx=5, pady=5, sticky="ew")

        ttk.Checkbutton(main, text="Save generated chirp to WAV", variable=self.save_wav_var).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        ttk.Label(main, text="WAV file:").grid(row=9, column=0, sticky="w", padx=5, pady=5)
        wav_path_frame = ttk.Frame(main)
        wav_path_frame.grid(row=9, column=1, padx=5, pady=5, sticky="ew")
        wav_path_frame.columnconfigure(0, weight=1)
        ttk.Entry(wav_path_frame, textvariable=self.wav_path_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(wav_path_frame, text="Browse...", command=self._browse_wav_path).grid(row=0, column=1, padx=(5, 0))

        button_row = ttk.Frame(main)
        button_row.grid(row=10, column=0, columnspan=2, pady=10, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(button_row, text="Start", command=self.start_output).grid(row=0, column=0, padx=5, sticky="ew")
        ttk.Button(button_row, text="Stop", command=self.stop_output).grid(row=0, column=1, padx=5, sticky="ew")

        ttk.Label(main, textvariable=self.status_var, foreground="darkblue").grid(row=11, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

    def _browse_wav_path(self):
        path = filedialog.asksaveasfilename(
            title="Save chirp WAV",
            defaultextension=".wav",
            initialfile="chirp.wav",
            filetypes=[("WAV files", "*.wav")],
        )
        if path:
            self.wav_path_var.set(path)

    def _on_device_change(self, event=None):
        device = self.device_var.get()
        channels = self._discover_ao_channels(device)
        self.available_channels = channels
        self.ao_combo.config(values=channels)
        if channels:
            self.ao_channel_var.set(channels[0])

    def _normalize_ao_channel(self, channel_name):
        if channel_name is None:
            return "ao0"
        channel = str(channel_name).strip().lower()
        if not channel.startswith("ao"):
            channel = "ao" + channel.lstrip("ao")
        return channel

    def _make_chirp(self, f0, f1, t_chirp, fs, amplitude):
        t = np.arange(int(t_chirp * fs)) / fs
        chirp_sig = amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t_chirp)))
        if len(chirp_sig) > 2:
            from scipy.signal.windows import tukey
            chirp_sig *= tukey(len(chirp_sig), alpha=0.1)
        return chirp_sig

    def _transmit_loop(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType

            device = self.device_var.get().strip() or "Dev1"
            channel = self._normalize_ao_channel(self.ao_channel_var.get())
            f0 = float(self.f0_var.get())
            f1 = float(self.f1_var.get())
            t_chirp = float(self.t_chirp_var.get())
            fs = float(self.fs_var.get())
            amplitude = float(self.amplitude_var.get())
            repeats = int(self.repeats_var.get())

            chirp_sig = self._make_chirp(f0, f1, t_chirp, fs, amplitude)
            total_samples = len(chirp_sig) * repeats

            with nidaqmx.Task() as ao:
                ao.ao_channels.add_ao_voltage_chan(f"{device}/{channel}", min_val=-10.0, max_val=10.0)
                ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples)

                data = np.tile(chirp_sig, repeats)
                if self.save_wav_var.get():
                    from scipy.io import wavfile

                    wav_path = self.wav_path_var.get().strip()
                    if not wav_path:
                        raise ValueError("Choose a WAV file path or uncheck WAV saving.")
                    if not wav_path.lower().endswith(".wav"):
                        wav_path += ".wav"
                    wavfile.write(wav_path, int(round(fs)), np.asarray(data, dtype=np.float32))

                ao.write(data)
                ao.start()
                ao.wait_until_done(timeout=10.0 + total_samples / fs)

            self.status_var.set("Stopped")

        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            messagebox.showerror("Transmit Error", str(exc))
        finally:
            self.running = False
            self.stop_event = None
            self.thread = None

    def start_output(self):
        if self.running:
            return

        try:
            float(self.f0_var.get())
            float(self.f1_var.get())
            float(self.t_chirp_var.get())
            float(self.fs_var.get())
            float(self.amplitude_var.get())
            int(self.repeats_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "All values must be numeric.")
            return

        self.running = True
        self.stop_event = threading.Event()
        self.status_var.set("Running...")
        self.thread = threading.Thread(target=self._transmit_loop, daemon=True)
        self.thread.start()

    def stop_output(self):
        if not self.running:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChirpTransmitApp(root)
    root.mainloop()

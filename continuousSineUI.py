"""
continuousSineUI.py

Small GUI for continuously outputting a selectable sine wave on an AO channel.

Features:
- start/stop transmit control
- output interface selection (device + AO channel)
- configurable sine frequency, sample rate, and amplitude
- looped output until user stops it
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np


class ContinuousSineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Continuous Sine Transmit")
        self.root.geometry("480x320")

        self.running = False
        self.thread = None
        self.stop_event = None

        self.available_devices = self._discover_devices()
        default_device = self.available_devices[0] if self.available_devices else "Dev1"
        self.device_var = tk.StringVar(value=default_device)
        self.available_channels = self._discover_ao_channels(default_device)
        self.ao_channel_var = tk.StringVar(value=self.available_channels[0] if self.available_channels else "ao0")
        self.freq_var = tk.StringVar(value="1000")
        self.fs_var = tk.StringVar(value="44000")
        self.amplitude_var = tk.StringVar(value="0.5")
        self.block_var = tk.StringVar(value="0.1")
        self.status_var = tk.StringVar(value="Idle")

        self._build_ui()

    def _discover_devices(self):
        try:
            import nidaqmx
            from nidaqmx.system import System

            system = System.local()
            devices = list(system.devices.device_names)
            return devices if devices else ["Dev1"]
        except Exception:
            return ["Dev1"]

    def _discover_ao_channels(self, device_name):
        try:
            import nidaqmx
            from nidaqmx.system import System

            system = System.local()
            dev = system.devices[device_name]
            ao_names = []
            for ch in dev.ao_physical_chans:
                name = str(ch)
                if "(" in name and ")" in name:
                    name = name.split("(", 1)[0].strip()
                name = name.split("/")[-1].strip().lower()
                if name and name not in ao_names:
                    ao_names.append(name)
            if not ao_names:
                for i in range(4):
                    ao_names.append(f"ao{i}")
            normalized = []
            for name in ao_names:
                if name.startswith("ao"):
                    normalized.append(name)
            if not normalized:
                normalized = [f"ao{i}" for i in range(4)]
            return normalized[:4]
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

        ttk.Label(main, text="Frequency (Hz):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.freq_var, width=18).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="Sample rate (Hz):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.fs_var, width=18).grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="Amplitude (V peak):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.amplitude_var, width=18).grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main, text="Block (s):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(main, textvariable=self.block_var, width=18).grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        button_row = ttk.Frame(main)
        button_row.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(button_row, text="Start", command=self.start_output).grid(row=0, column=0, padx=5, sticky="ew")
        ttk.Button(button_row, text="Stop", command=self.stop_output).grid(row=0, column=1, padx=5, sticky="ew")

        ttk.Label(main, textvariable=self.status_var, foreground="darkblue").grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

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

    @staticmethod
    def _generate_sine_wave(freq_hz, fs, amplitude, duration_sec):
        n_samples = int(fs * duration_sec)
        t = np.arange(n_samples) / fs
        return amplitude * np.sin(2 * np.pi * freq_hz * t)

    def _transmit_loop(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType

            device = self.device_var.get().strip() or "Dev1"
            channel = self._normalize_ao_channel(self.ao_channel_var.get())
            freq_hz = float(self.freq_var.get())
            fs = float(self.fs_var.get())
            amplitude = float(self.amplitude_var.get())
            block_duration = float(self.block_var.get())

            with nidaqmx.Task() as ao:
                ao.ao_channels.add_ao_voltage_chan(f"{device}/{channel}", min_val=-10.0, max_val=10.0)
                while not self.stop_event.is_set():
                    data = self._generate_sine_wave(freq_hz, fs, amplitude, block_duration)
                    ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(data))
                    ao.write(data)
                    ao.start()
                    ao.wait_until_done(timeout=10.0 + len(data) / fs)
                    ao.stop()

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
            float(self.freq_var.get())
            float(self.fs_var.get())
            float(self.amplitude_var.get())
            float(self.block_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Frequency, sample rate, amplitude, and block duration must be numeric.")
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
    app = ContinuousSineApp(root)
    root.mainloop()

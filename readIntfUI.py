"""
readIntfUI.py

Standalone GUI for continuously reading channels ai0-ai3 and plotting them in real time.

Features:
- adjustable sample rate
- device address entry
- start/stop controls
- optional WAV logging
- notes box saved with the data file metadata
"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile


MAX_BUFFER_SAMPLES = 8191


class DAQMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NI DAQ Multi-Channel Monitor")
        self.root.geometry("1100x800")

        self.channels = ["ai0", "ai1", "ai2", "ai3"]
        self.max_buffer_frames = MAX_BUFFER_SAMPLES // len(self.channels)
        self.device_name = "Dev1"  # USB-6366 default for this UI
        self.data_lock = threading.Lock()
        self.running = False
        self.stop_event = None
        self.thread = None
        self.history = np.empty((0, len(self.channels)))
        self.log_chunks = []
        self.log_path = None

        self.sample_rate_var = tk.StringVar(value="44000")
        self.device_var = tk.StringVar(value="Dev1")
        self.v_range_var = tk.StringVar(value="5.0")
        self.log_enabled_var = tk.BooleanVar(value=False)
        self.output_file_var = tk.StringVar(value="daq_capture.wav")
        self.save_dir_var = tk.StringVar(value=os.path.abspath("."))
        self.status_var = tk.StringVar(value="Idle")

        self._build_ui()
        self._init_plot()
        self.root.after(100, self._update_ui)

    def _build_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", side="top")

        ttk.Label(control_frame, text="Device:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.device_var, width=20).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Sample rate (Hz):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.sample_rate_var, width=12).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(control_frame, text="Input range (+/- V):").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.v_range_var, width=10).grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(control_frame, text="File name:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.output_file_var, width=24).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Save folder:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.save_dir_var, width=34).grid(row=1, column=3, columnspan=3, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Browse", command=self._browse_save_dir).grid(row=1, column=6, padx=5, pady=5)

        self.log_checkbox = ttk.Checkbutton(control_frame, text="Log to WAV", variable=self.log_enabled_var)
        self.log_checkbox.grid(row=1, column=7, padx=10, pady=5)

        ttk.Button(control_frame, text="Start", command=self.start_acquisition).grid(row=1, column=8, padx=10, pady=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_acquisition).grid(row=1, column=9, padx=10, pady=5)

        ttk.Label(control_frame, textvariable=self.status_var, foreground="darkblue").grid(row=2, column=0, columnspan=10, sticky="w", padx=5, pady=(0, 5))

        # notes box for metadata header
        notes_frame = ttk.LabelFrame(self.root, text="Data Notes / Header", padding=8)
        notes_frame.pack(fill="x", side="top", padx=10, pady=(0, 10))

        self.notes_box = tk.Text(notes_frame, height=4, width=140)
        self.notes_box.insert("1.0", "DAQ capture notes\n")
        self.notes_box.pack(fill="x")

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _init_plot(self):
        self.fig = Figure(figsize=(10.5, 6.5), dpi=100)
        self.axs = [self.fig.add_subplot(4, 1, idx + 1) for idx in range(4)]
        self.lines = []
        for idx, ch in enumerate(self.channels):
            ax = self.axs[idx]
            line, = ax.plot([], [], lw=1.2)
            ax.set_title(ch)
            ax.set_ylabel("V")
            ax.grid(True, alpha=0.25)
            self.lines.append(line)
        self.axs[-1].set_xlabel("Time (s)")
        self.fig.tight_layout(pad=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_ui(self):
        with self.data_lock:
            history = self.history.copy()

        if self.running and history.size > 0:
            display_n = min(history.shape[0], max(2000, int(float(self.sample_rate_var.get()) * 0.5)))
            y = history[-display_n:]
            sample_rate = max(float(self.sample_rate_var.get()), 1e-6)
            t = np.arange(y.shape[0]) / sample_rate
            time_window = max(1.0 / sample_rate, y.shape[0] / sample_rate)
            for idx, line in enumerate(self.lines):
                line.set_data(t, y[:, idx])
                self.axs[idx].set_xlim(0, time_window)
                if y.size:
                    ymin = np.min(y[:, idx])
                    ymax = np.max(y[:, idx])
                    pad = max(0.05, 0.1 * max(abs(ymin), abs(ymax), 1.0))
                    self.axs[idx].set_ylim(ymin - pad, ymax + pad)
            self.canvas.draw_idle()

        self.root.after(50, self._update_ui)

    def _append_to_history(self, data):
        if data.size == 0:
            return
        with self.data_lock:
            new_data = np.asarray(data, dtype=np.float64).reshape(-1, len(self.channels))
            combined = np.concatenate((self.history, new_data), axis=0)
            self.history = combined[-self.max_buffer_frames:]

    def _browse_save_dir(self):
        folder = filedialog.askdirectory(initialdir=self.save_dir_var.get() or os.path.abspath("."))
        if folder:
            self.save_dir_var.set(folder)

    def _save_logged_data(self):
        if not self.log_enabled_var.get():
            return

        if len(self.log_chunks) == 0:
            return

        save_dir = self.save_dir_var.get().strip() or os.path.abspath(".")
        os.makedirs(save_dir, exist_ok=True)

        fname = self.output_file_var.get().strip() or "daq_capture.wav"
        base_name = os.path.splitext(fname)[0]
        if not base_name:
            base_name = "daq_capture"

        data = np.concatenate(self.log_chunks, axis=0)
        if data.shape[1] != len(self.channels):
            data = data.reshape(-1, len(self.channels))

        data = np.asarray(data, dtype=np.float32)
        for idx, channel in enumerate(self.channels):
            channel_path = os.path.join(save_dir, f"{base_name}_{channel}.wav")
            wavfile.write(
                channel_path,
                int(float(self.sample_rate_var.get())),
                data[:, idx],
            )

            metadata = {
                "device": self.device_var.get(),
                "sample_rate_hz": float(self.sample_rate_var.get()),
                "channel": channel,
                "input_range_v": float(self.v_range_var.get()),
                "notes": self.notes_box.get("1.0", "end-1c").strip(),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_path = os.path.splitext(channel_path)[0] + ".json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        self.log_path = os.path.join(save_dir, base_name)

    def _acquire_loop(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, TerminalConfiguration

            device = self.device_var.get().strip() or "Dev1"
            sample_rate = float(self.sample_rate_var.get())
            v_range = float(self.v_range_var.get())
            block_size = min(
                max(1024, int(sample_rate * 0.1)),
                self.max_buffer_frames,
            )

            with nidaqmx.Task() as task:
                for ch in self.channels:
                    task.ai_channels.add_ai_voltage_chan(
                        f"{device}/{ch}",
                        min_val=-v_range,
                        max_val=v_range,
                        terminal_config=TerminalConfiguration.DIFF,
                    )

                task.timing.cfg_samp_clk_timing(
                    sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=block_size,
                )
                task.start()

                while not self.stop_event.is_set():
                    data = task.read(number_of_samples_per_channel=block_size, timeout=5.0)
                    arr = np.asarray(data, dtype=np.float64)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    if arr.shape[1] != len(self.channels):
                        # sometimes DAQ returns data as (samples,) or (channels, samples)
                        if arr.shape[0] == len(self.channels):
                            arr = arr.T
                        else:
                            arr = arr.reshape(-1, len(self.channels))

                    self._append_to_history(arr)
                    if self.log_enabled_var.get():
                        self.log_chunks.append(arr.copy())

                task.stop()

            self._save_logged_data()
            self.status_var.set("Stopped")

        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            messagebox.showerror("DAQ Error", str(exc))
        finally:
            self.running = False
            self.stop_event = None
            self.thread = None

    def start_acquisition(self):
        if self.running:
            return

        try:
            float(self.sample_rate_var.get())
            float(self.v_range_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Sample rate and input range must be numeric values.")
            return

        self.running = True
        self.stop_event = threading.Event()
        self.history = np.empty((0, len(self.channels)))
        self.log_chunks = []
        self.log_path = None

        if self.log_enabled_var.get():
            save_dir = self.save_dir_var.get().strip() or os.path.abspath(".")
            os.makedirs(save_dir, exist_ok=True)
            base_name = self.output_file_var.get().strip() or f"daq_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            if not base_name.lower().endswith(".wav"):
                base_name += ".wav"
            self.log_path = os.path.join(save_dir, base_name)

        self.status_var.set("Running...")
        self.thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self.thread.start()

    def stop_acquisition(self):
        if not self.running:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")


if __name__ == "__main__":
    root = tk.Tk()
    app = DAQMonitorApp(root)
    root.mainloop()

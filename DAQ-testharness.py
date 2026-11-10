"""
Standalone DAQ test harness for the NI USB-6366.

This script generates a 10-cycle, 1 kHz sine wave on AO0 and reads all four
analog input channels simultaneously:
    ai0, ai1, ai2, ai3

It is intended as a quick hardware check to confirm that each channel is
working and that the AO output is being seen on the inputs.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


DEVICE = "Dev1"
AO_CHAN = "ao1"
AI_CHANNELS = ["ai0", "ai1", "ai2", "ai3"]
FS = 100000           # samples/sec
FREQ_HZ = 1000        # output sine frequency
N_CYCLES = 10          # number of full sine cycles
DURATION_SEC = N_CYCLES / FREQ_HZ
AI_RANGE = 2.0         # input range in volts (+/-)
AO_AMPLITUDE = 1.0     # output amplitude in volts peak
SAVE_DIR = "daq_recordings"


def get_next_recording_prefix(save_dir=SAVE_DIR):
    """Create an incrementing recording prefix like 'record_0001' for each new run."""
    os.makedirs(save_dir, exist_ok=True)

    existing = []
    if os.path.isdir(save_dir):
        for name in os.listdir(save_dir):
            if name.startswith("record_") and name.endswith(".wav"):
                existing.append(name)

    used = set()
    for name in existing:
        stem = os.path.splitext(name)[0]
        if stem.startswith("record_"):
            try:
                used.add(int(stem.split("_")[-1]))
            except ValueError:
                pass

    idx = 1
    while idx in used:
        idx += 1
    return os.path.join(save_dir, f"record_{idx:04d}")


def save_channel_wavs(data, base_prefix):
    """Write each AI channel to its own WAV file using the same recording prefix."""
    for idx, name in enumerate(AI_CHANNELS):
        path = f"{base_prefix}_{name}.wav"
        wavfile.write(path, int(FS), np.asarray(data[:, idx], dtype=np.float32))
        print(f"Saved {path}")


def make_output_wave(fs, freq_hz, n_cycles, amplitude):
    """Create a 10-cycle 1 kHz sine wave on AO0."""
    n_samples = int(fs * (n_cycles / freq_hz))
    t = np.arange(n_samples) / fs
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return wave, n_samples


def read_four_inputs(device, channels, fs, n_samples, v_range):
    """Acquire voltage data from 4 analog input channels at the same time."""
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    except ImportError as exc:
        raise RuntimeError(
            "nidaqmx is not installed. Install the NI-DAQmx Python package first."
        ) from exc

    wave, _ = make_output_wave(fs, FREQ_HZ, N_CYCLES, AO_AMPLITUDE)

    with nidaqmx.Task() as ai, nidaqmx.Task() as ao:
        for channel in channels:
            ai.ai_channels.add_ai_voltage_chan(
                f"{device}/{channel}",
                min_val=-v_range,
                max_val=v_range,
                terminal_config=TerminalConfiguration.DIFF,
            )

        ao.ao_channels.add_ao_voltage_chan(
            f"{device}/{AO_CHAN}",
            min_val=-10,
            max_val=10,
        )

        ai.timing.cfg_samp_clk_timing(
            fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        ao.timing.cfg_samp_clk_timing(
            fs,
            source=f"/{device}/ai/SampleClock",
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )

        ao.write(wave, auto_start=False)
        ao.start()
        ai.start()

        data = np.array(ai.read(number_of_samples_per_channel=n_samples, timeout=10.0 + DURATION_SEC))
        ao.wait_until_done(timeout=10.0 + DURATION_SEC)

    data = np.asarray(data)
    if data.ndim == 2 and data.shape[0] == len(channels):
        data = data.T
    elif data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, wave


def print_channel_stats(data):
    """Print basic voltage statistics for each channel."""
    for idx, name in enumerate(AI_CHANNELS):
        y = data[:, idx]
        rms = np.sqrt(np.mean(np.square(y)))
        peak = np.max(np.abs(y))
        print(f"{name}: min={y.min(): .4f} V, max={y.max(): .4f} V, "
              f"peak={peak: .4f} V, RMS={rms: .4f} V")


def plot_channels(data, fs, wave):
    """Display the four input channels and the AO output waveform."""
    t_ms = np.arange(data.shape[0]) / fs * 1_000
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, channel in enumerate(AI_CHANNELS):
        ax = axes[idx]
        y = data[:, idx]
        ax.plot(t_ms, y, lw=1.0, color="C0", label=f"{channel}")
        ax.set_title(f"{channel}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)

    # Show the AO0 output in the top-left subplot as a reference if desired.
    axes[0].plot(np.arange(len(wave)) / fs * 1_000, wave, lw=1.0, color="C3", alpha=0.75, label="ao0 output")
    axes[0].legend(fontsize=8)

    fig.suptitle("DAQ test harness: AO0 sine + AI0-AI3 acquisition")
    fig.tight_layout()
    base_prefix = get_next_recording_prefix()
    fig.savefig(f"{base_prefix}_summary.png", dpi=150)
    save_channel_wavs(data, base_prefix)
    np.savez(f"{base_prefix}_data.npz", t=t_ms / 1000.0, data=data, fs=FS)
    print(f"Saved summary plot: {base_prefix}_summary.png")
    print(f"Saved raw data archive: {base_prefix}_data.npz")
    plt.show()


def main():
    n_samples = int(FS * DURATION_SEC)
    print(f"Generating {N_CYCLES} cycles of a {FREQ_HZ} Hz sine on {AO_CHAN} at {FS} Hz.")
    print(f"Sampling {len(AI_CHANNELS)} channels on device {DEVICE} for {DURATION_SEC:.3f} s...")

    try:
        data, wave = read_four_inputs(DEVICE, AI_CHANNELS, FS, n_samples, AI_RANGE)
    except Exception as exc:
        print("\nDAQ acquisition failed.")
        print(f"Reason: {exc}")
        print("\nCheck that:")
        print("  - the device name matches NI MAX (for example 'Dev1')")
        print("  - the DAQ is connected and powered")
        print("  - ao0 and ai0-ai3 are available")
        print("  - the NI-DAQmx Python package is installed")
        raise

    print("\nChannel statistics:")
    print_channel_stats(data)
    plot_channels(data, FS, wave)


if __name__ == "__main__":
    main()

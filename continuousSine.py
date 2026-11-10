"""
continuousSine.py

Continuously generate a sinusoid on AO0.
Defaults:
    frequency = 1 kHz
    amplitude = 0.5 V peak
    sample rate = 44 kHz

This script leaves the output running until Ctrl+C is pressed.
"""

import argparse
import time
import numpy as np


def generate_sine_wave(freq_hz=1000.0, fs=44000.0, amplitude=0.5, duration_sec=0.1):
    """Generate one block of sine samples."""
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def normalize_ao_channel(channel_name):
    """Normalize analog output channel names across NI devices (e.g., AO0 vs ao0)."""
    if channel_name is None:
        return "ao0"
    channel_name = str(channel_name).strip().lower()
    if not channel_name.startswith("ao"):
        channel_name = "ao" + channel_name.lstrip("aoAO")
    return channel_name


def continuous_output(device="Dev1", ao_channel="ao0", freq_hz=1000.0,
                     fs=44000.0, amplitude=0.5, block_duration_sec=0.1):
    """Continuously output a sine wave on the selected AO channel."""
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, RegenerationMode
    except ImportError as exc:
        raise RuntimeError("nidaqmx is not installed. Install the NI-DAQmx Python package first.") from exc

    ao_channel = normalize_ao_channel(ao_channel)

    print(f"Starting continuous sine output on {device}/{ao_channel}")
    print(f"  Frequency = {freq_hz:.1f} Hz")
    print(f"  Amplitude = {amplitude:.3f} V peak")
    print(f"  Sample rate = {fs:.1f} Hz")
    print("Press Ctrl+C to stop.")

    with nidaqmx.Task() as ao:
        ao.ao_channels.add_ao_voltage_chan(f"{device}/{ao_channel}", min_val=-10.0, max_val=10.0)
        ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.CONTINUOUS,
                                     samps_per_chan=max(1024, int(fs * 0.1)))

        try:
            ao.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGEN
        except Exception:
            pass

        while True:
            data = generate_sine_wave(freq_hz=freq_hz, fs=fs, amplitude=amplitude,
                                      duration_sec=block_duration_sec)
            ao.write(data, auto_start=False)
            ao.start()
            ao.wait_until_done(timeout=10.0 + len(data) / fs)
            ao.stop()


def main():
    parser = argparse.ArgumentParser(description="Continuously generate a selectable sine wave on AO0.")
    parser.add_argument("--device", default="Dev1", help="NI device name, e.g. Dev1")
    parser.add_argument("--ao-channel", default="ao0", help="AO output channel")
    parser.add_argument("--freq-hz", type=float, default=1000.0, help="Output sine frequency in Hz")
    parser.add_argument("--fs", type=float, default=44000.0, help="Output sample rate in Hz")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Output amplitude in volts peak")
    parser.add_argument("--block-duration-sec", type=float, default=0.1, help="Duration of each output block in seconds")
    args = parser.parse_args()

    continuous_output(
        device=args.device,
        ao_channel=args.ao_channel,
        freq_hz=args.freq_hz,
        fs=args.fs,
        amplitude=args.amplitude,
        block_duration_sec=args.block_duration_sec,
    )


if __name__ == "__main__":
    main()

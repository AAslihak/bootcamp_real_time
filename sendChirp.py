"""
sendChirp.py

Transmit a chirp from F0 to F1 over T_chirp seconds using the NI DAQ AO0 output.
The default configuration matches the requested test case:
    F0 = 8 kHz
    F1 = 12 kHz
    T_chirp = 300 ms
    FS = 44 kHz

The output is parameterized so different values can be used without editing the code.
"""

import argparse
import numpy as np


def normalize_ao_channel(channel_name):
    """Normalize analog output channel names across NI devices (AO0 vs ao0)."""
    if channel_name is None:
        return "ao0"
    channel_name = str(channel_name).strip().lower()
    if not channel_name.startswith("ao"):
        channel_name = "ao" + channel_name.lstrip("aoAO")
    return channel_name


def make_chirp(f0=8000.0, f1=12000.0, t_chirp=0.300, fs=44000.0, amplitude=0.5):
    """Create a linear chirp with a smooth taper."""
    t = np.arange(int(t_chirp * fs)) / fs
    chirp_sig = amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t_chirp)))

    # apply a short Tukey taper to reduce edge transients
    if len(chirp_sig) > 2:
        from scipy.signal.windows import tukey
        chirp_sig = chirp_sig * tukey(len(chirp_sig), alpha=0.1)
    return chirp_sig


def save_chirp_wav(path, fs, data):
    """Save the transmitted waveform as a float32 WAV file."""
    try:
        from scipy.io import wavfile
    except ImportError as exc:
        raise RuntimeError("scipy is required to save a WAV file.") from exc

    wavfile.write(path, int(round(fs)), np.asarray(data, dtype=np.float32))


def send_chirp(device="Dev1", ao_channel="ao0", f0=8000.0, f1=12000.0,
              t_chirp=0.300, fs=44000.0, amplitude=0.5, repeats=1, save_wav=None):
    """Write one or more chirps to the selected AO output channel."""
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
    except ImportError as exc:
        raise RuntimeError("nidaqmx is not installed. Install the NI-DAQmx Python package first.") from exc

    ao_channel = normalize_ao_channel(ao_channel)
    chirp_sig = make_chirp(f0=f0, f1=f1, t_chirp=t_chirp, fs=fs, amplitude=amplitude)
    total_samples = len(chirp_sig) * repeats
    data = np.tile(chirp_sig, repeats)

    if save_wav:
        save_chirp_wav(save_wav, fs, data)
        print(f"Saved WAV: {save_wav}")

    with nidaqmx.Task() as ao:
        ao.ao_channels.add_ao_voltage_chan(f"{device}/{ao_channel}", min_val=-10.0, max_val=10.0)
        ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples)

        ao.write(data)
        ao.start()
        ao.wait_until_done(timeout=10.0 + total_samples / fs)

    print(f"Sent {repeats} chirp(s) to {device}/{ao_channel}:")
    print(f"  F0 = {f0:.1f} Hz")
    print(f"  F1 = {f1:.1f} Hz")
    print(f"  T_chirp = {t_chirp:.3f} s")
    print(f"  FS = {fs:.1f} Hz")
    print(f"  Samples = {total_samples}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Send a chirp from F0 to F1 using DAQ AO0.")
    parser.add_argument("--device", default="Dev1", help="NI device name, e.g. Dev1")
    parser.add_argument("--ao-channel", default="ao0", help="AO output channel, e.g. ao0")
    parser.add_argument("--f0", type=float, default=8000.0, help="Start frequency in Hz")
    parser.add_argument("--f1", type=float, default=12000.0, help="Stop frequency in Hz")
    parser.add_argument("--t-chirp", type=float, default=0.300, help="Chirp duration in seconds")
    parser.add_argument("--fs", type=float, default=44000.0, help="Sample rate in Hz")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Output amplitude in volts peak")
    parser.add_argument("--repeats", type=int, default=1, help="Number of chirps to send")
    parser.add_argument("--save-wav", help="Optional path to save the generated chirp as a WAV file")
    args = parser.parse_args()

    send_chirp(
        device=args.device,
        ao_channel=args.ao_channel,
        f0=args.f0,
        f1=args.f1,
        t_chirp=args.t_chirp,
        fs=args.fs,
        amplitude=args.amplitude,
        repeats=args.repeats,
        save_wav=args.save_wav,
    )


if __name__ == "__main__":
    main()

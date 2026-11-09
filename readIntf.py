"""
readIntf.py

Read and plot ai0, ai1, ai2, ai3 in real time from the NI DAQ.
This is intended for quick live inspection of the four channels.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


def acquire_block(device="Dev1", channels=None, fs=44000.0, samples_per_block=4096, v_range=5.0):
    """Acquire one block of samples from the selected channels."""
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    except ImportError as exc:
        raise RuntimeError("nidaqmx is not installed. Install the NI-DAQmx Python package first.") from exc

    if channels is None:
        channels = ["ai0", "ai1", "ai2", "ai3"]

    with nidaqmx.Task() as task:
        for ch in channels:
            task.ai_channels.add_ai_voltage_chan(
                f"{device}/{ch}",
                min_val=-v_range,
                max_val=v_range,
                terminal_config=TerminalConfiguration.DIFF,
            )

        task.timing.cfg_samp_clk_timing(
            fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples_per_block,
        )

        data = task.read(number_of_samples_per_channel=samples_per_block, timeout=10.0)

    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[0] == len(channels):
        return arr.T
    return arr.reshape(-1, len(channels))


def live_plot(device="Dev1", channels=None, fs=44000.0, samples_per_block=4096,
              v_range=5.0, update_seconds=0.2, total_seconds=5.0):
    """Continuously plot the live AI data in a rolling window."""
    if channels is None:
        channels = ["ai0", "ai1", "ai2", "ai3"]

    n_channels = len(channels)
    n_blocks = max(1, int(total_seconds / update_seconds))
    block_count = 0
    x_axis = np.arange(samples_per_block) / fs

    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    lines = []
    for idx, ch in enumerate(channels):
        ax = axes[idx]
        ax.set_title(f"{ch}")
        ax.set_ylabel("Volts")
        ax.grid(alpha=0.3)
        ln, = ax.plot(x_axis, np.zeros(samples_per_block), label=ch)
        lines.append(ln)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    try:
        while block_count < n_blocks:
            data = acquire_block(device=device, channels=channels, fs=fs,
                                samples_per_block=samples_per_block, v_range=v_range)
            for idx, line in enumerate(lines):
                line.set_ydata(data[:, idx])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(update_seconds)
            block_count += 1
    except KeyboardInterrupt:
        print("\nLive plot stopped by user.")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live plot ai0-ai3 from the NI DAQ.")
    parser.add_argument("--device", default="Dev1", help="NI device name, e.g. Dev1")
    parser.add_argument("--channels", nargs="*", default=["ai0", "ai1", "ai2", "ai3"], help="Analog input channels to plot")
    parser.add_argument("--fs", type=float, default=44000.0, help="Sample rate in Hz")
    parser.add_argument("--samples-per-block", type=int, default=4096, help="Samples per acquisition block")
    parser.add_argument("--v-range", type=float, default=5.0, help="Input voltage range (+/- V)")
    parser.add_argument("--update-seconds", type=float, default=0.2, help="Plot update interval in seconds")
    parser.add_argument("--total-seconds", type=float, default=5.0, help="Total live display time in seconds")
    args = parser.parse_args()

    live_plot(
        device=args.device,
        channels=args.channels,
        fs=args.fs,
        samples_per_block=args.samples_per_block,
        v_range=args.v_range,
        update_seconds=args.update_seconds,
        total_seconds=args.total_seconds,
    )


if __name__ == "__main__":
    main()

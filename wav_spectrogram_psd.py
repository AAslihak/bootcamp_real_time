"""Spectrogram and PSD plots for four WAV files."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, welch

# ---- put your four files here ----
FILES = [
    "daq_recordings/record_0001_ai0.wav",
    "daq_recordings/record_0001_ai1.wav",
    "daq_recordings/record_0001_ai2.wav",
    "daq_recordings/record_0001_ai3.wav",
]
NFFT = 4096   # FFT length (frequency resolution = fs / NFFT)

fig_spec, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
fig_psd, ax_psd = plt.subplots(figsize=(9, 5), constrained_layout=True)

for ax, path in zip(axes.ravel(), FILES):
    fs, x = wavfile.read(path)          # x = samples (volts), fs = sample rate (Hz)
    if x.ndim > 1:
        x = x[:, 0]                     # multi-channel file: use the first channel
    n = min(NFFT, len(x) // 8)          # shorter window if the recording is short

    # Spectrogram (Hann window, 50% overlap), in dB re 1 V^2/Hz
    f, t, P = spectrogram(x, fs, window="hann", nperseg=n, noverlap=n // 2)
    P_dB = 10 * np.log10(P + 1e-30)
    im = ax.pcolormesh(t, f / 1e3, P_dB, shading="auto",
                       vmin=P_dB.max() - 80, vmax=P_dB.max())   # 80 dB range
    ax.set(title=path, xlabel="Time (s)", ylabel="Frequency (kHz)")
    fig_spec.colorbar(im, ax=ax, label="dB re 1 V$^2$/Hz")

    # PSD (Welch average of the same windows)
    f, Pxx = welch(x, fs, window="hann", nperseg=n, noverlap=n // 2)
    ax_psd.plot(f / 1e3, 10 * np.log10(Pxx + 1e-30), label=path)

ax_psd.set(title="Power spectral density", xlabel="Frequency (kHz)",
           ylabel="PSD (dB re 1 V$^2$/Hz)")
ax_psd.grid(alpha=0.3)
ax_psd.legend()

fig_spec.savefig("spectrograms.png", dpi=200)
fig_psd.savefig("psd.png", dpi=200)
plt.show()

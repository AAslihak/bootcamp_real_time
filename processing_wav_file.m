
% extract data from .wav file- inputs 
[x, fs] = audioread('chirp_8to12k.wav');     % x = samples, fs = sampling rate (Hz)

t_sig = (0:1/fs:2).';  

% --- Spectrogram settings ---
nfft     = 4096;                 % FFT length (sets frequency resolution fs/nfft)
win      = hann(nfft);           % window
noverlap = round(0.5*nfft);      % 50% overlap

[~, f, t, P] = spectrogram(x, win, noverlap, nfft, fs, 'psd');
% P is PSD in V^2/Hz, size [length(f) x length(t)]

% x will be our voltage values

PSD_dB = 10*log10(P);            % dB re 1 V^2/Hz

% --- Plot ---
figure
imagesc(t, f/1e3, PSD_dB)
axis xy
xlabel('Time (s)')
ylabel('Frequency (kHz)')
title('Power Spectral Density')
cb = colorbar; cb.Label.String = 'dB re 1 V^2/Hz';
clim([max(PSD_dB(:))-80, max(PSD_dB(:))])   % show an 80 dB dynamic range
% (use caxis instead of clim in older MATLAB versions)
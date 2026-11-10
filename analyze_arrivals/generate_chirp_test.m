function [receiverPath, templatePath, onsetSec] = generate_chirp_test(outDir)
% Generate a clean transmit template and a noisy four-channel receiver WAV.
% onsetSec is 33-by-4, in seconds from the beginning of the receiver WAV.
if nargin < 1, outDir = fileparts(mfilename('fullpath')); end
if ~exist(outDir,'dir'), mkdir(outDir); end

fs = 44000;
duration = 0.1;
gap = 0.2;
f0 = 3000;
f1 = 15000;
n = 10*fs;
t = (0:round(duration*fs)-1)'/fs;
sweepRate = (f1-f0)/duration;
tx = 0.5*cos(2*pi*(f0*t + sweepRate*t.^2/2));
edge = round(0.002*fs);
ramp = (1-cos(pi*(0:edge-1)'/(edge-1)))/2;
tx(1:edge) = tx(1:edge).*ramp;
tx(end-edge+1:end) = tx(end-edge+1:end).*flipud(ramp);

delays = [0 13 26 44];
gains = [1 0.85 0.70 0.55];
first = round(0.2*fs);
period = round((duration+gap)*fs);
starts = (first:period:n-numel(tx)-max(delays))';
onsetSec = (starts+delays)/fs;

oldRng = rng;
restoreRng = onCleanup(@()rng(oldRng));
rng(41,'twister');
% Independent Gaussian noise per channel, with RMS changing over time.
noiseRMS = 0.045 + 0.075*(0.5-0.5*cos(2*pi*(0:n-1)'/(4*fs)));
rx = noiseRMS.*randn(n,numel(delays));
for ch = 1:numel(delays)
    for start = starts'
        idx = start+delays(ch)+(1:numel(tx));
        rx(idx,ch) = rx(idx,ch)+gains(ch)*tx;
    end
end

assert(max(abs(rx(:)))<1,'Synthetic waveform would clip in the WAV.');
receiverPath = fullfile(outDir,'noisy_chirp_train.wav');
templatePath = fullfile(outDir,'chirp_template.wav');
audiowrite(receiverPath,rx,fs,'BitsPerSample',24);
audiowrite(templatePath,tx,fs,'BitsPerSample',24);
% Plot all four channels of the generated 10-second WAV.
timeSec = (0:n-1)' / fs;
figure('Name', 'Noisy chirp train');
plot(timeSec, rx);
xlabel('Time (s)');
ylabel('Amplitude');
title('Generated chirp train with noise');
legend('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4');
xlim([0 n/fs]);
grid on;
end

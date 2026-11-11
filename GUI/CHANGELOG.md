# Changelog — DAQ GUI

## v3.0-beta.2 — beta (tag `gui-v3.0-beta.2`, branch `gui-v3-beta`)

- No window function: the hydrophone PSD and the spectrogram now use a
  rectangular window instead of Hann (the transmit PSD already had none).
  Set `SPECTRUM_WINDOW = "hann"` at the top of the program to restore it.
  Without a window the noise floor next to a strong signal is higher
  (spectral leakage), and readings differ from v2 / beta.1.

## v3.0-beta.1 — beta (tag `gui-v3.0-beta.1`)

Files: `6259/daq_gui_v3.py`, `6366/daq_gui_6366_v3.py`. Only run with the
simulated device so far. Built on v2.0; the Distance feature below was later released in v2.1.

- Distance (default 0.30 m): pressure levels are referred back to 1 m
  assuming spherical spreading, level at 1 m = measured level
  + 20 log10(Distance / 1 m). Applies to the hydrophone PSD, the spectrogram,
  the Acoustic level and the alarms; the unit becomes dB re 1 µPa²/Hz @ 1 m.
  Note: the 156 / 159 dB thresholds are now compared with the 1 m value,
  which is 10.5 dB lower than the v2 reading at 0.30 m.
- "Equalise with TVR": inverse filtering of the transmit signal with the
  projector TVR (`projector_tvr.csv`, built-in copy as fallback), magnitude
  only, then scaled to Peak. The radiated spectrum becomes flat at the level
  of the weakest frequency in the band.
- "Auto-record during transmit" is now labelled "Auto-record".
- The version is shown in the window title; `distance_m` and `tvr_equalised`
  are written to the recording info.

## v2.1 — release (tag `gui-v2.1`, branch `main`)

Files: `6259/daq_gui_v2.py`, `6366/daq_gui_6366.py`.

- Distance (default 0.30 m): pressure levels are referred back to 1 m
  assuming spherical spreading, level at 1 m = measured level
  + 20 log10(Distance / 1 m). Applies to the hydrophone PSD, the spectrogram,
  the Acoustic level and the alarms; the unit becomes dB re 1 µPa²/Hz @ 1 m.
  Set Distance to 1 m to see the level as measured at the hydrophone.
- IMPORTANT: readings are 10.5 dB lower than in v2.0 with the hydrophone at
  0.30 m, and the 156 / 159 dB thresholds are now compared with the 1 m value.
- The version is shown in the window title; `distance_m` is written to the
  recording info.

## v2.0 — release (tag `gui-v2.0`)

Files: `6259/daq_gui_v2.py`, `6366/daq_gui_6366.py`.

- Hydrophone PSD in pressure units (dB re 1 µPa²/Hz) using the hydrophone
  sensitivity curve (`hydrophone_rvr.csv`, built-in copy as fallback) and a
  Gain setting.
- "Acoustic level" strip: maximum of the hydrophone PSD and its frequency,
  updated once per second. Window frame turns orange from 156 dB and red from
  159 dB; red events are listed under "Red alarms".
- FFT length entered in points or seconds; set to the signal length when a
  WAV is loaded. Rate defaults to 2 x the WAV sample rate, Guard to the WAV
  length.
- Spectrogram view with its own FFT length; fixed 0-20 kHz frequency axes.
- Recording to wav (32-bit float, volts, with an `_info.txt`) or txt.
- Errors are written to `daq_gui_error.log`; `check_daq.py` checks the DAQ
  connection step by step.
- The USB-6259 version has been used on the hardware (recordings of
  2026-11-11). The USB-6366 version has only been run with the simulated
  device.

## v1 — `6259/daq_gui.py`

First version: WAV transmit with guard interval, hydrophone monitor in
volts, recording.

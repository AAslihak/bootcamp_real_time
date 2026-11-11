# Changelog — DAQ GUI

## v2.0 — release (tag `gui-v2.0`, branch `main`)

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

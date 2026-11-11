# WAV transmit / reference hydrophone GUI

Plays a WAV file on an NI DAQ analog output and monitors / records the
reference hydrophone on an analog input. One folder per DAQ model.

## Which file to run

| DAQ | Release (v2) | Beta (v3, branch `gui-v3-beta`) |
|---|---|---|
| NI USB-6259 | `6259/daq_gui_v2.py` | `6259/daq_gui_v3.py` |
| NI USB-6366 | `6366/daq_gui_6366.py` | `6366/daq_gui_6366_v3.py` |

`6259/daq_gui.py` is the old v1, kept for reference only.

```
python -m pip install -r requirements.txt
python daq_gui_v2.py        # release
python daq_gui_v3.py        # beta
```

Without the NI-DAQmx driver or a device the program starts with a simulated
device, so the interface can be tried on any computer. If the DAQ does not
work, run `python check_daq.py` and report its full output together with
`daq_gui_error.log`.

## Versions

| Version | Status | Where | Tag |
|---|---|---|---|
| v2.0 | release | branch `main` | `gui-v2.0` |
| v3.0-beta.2 | beta, not yet tested on hardware | branch `gui-v3-beta` | `gui-v3.0-beta.2` |

- `main` only holds released versions. Get a specific release with
  `git checkout gui-v2.0`.
- New features are developed on a beta branch and merged into `main` when
  they have been tested on the hardware; the merge gets a new `gui-vX.Y` tag.
- Every change is listed in [CHANGELOG.md](CHANGELOG.md).

## Calibration files

- `hydrophone_rvr.csv`: receive voltage response of the reference hydrophones
  (dB re 1 V/µPa), used to show the spectrum in pressure units.

- `projector_tvr.csv` (v3): transmitting voltage response of the projector
  (dB re 1 µPa/V @ 1 m), used by "Equalise with TVR".

The values in these files were read off the calibration plots by eye (about
±0.1 dB). Replace them with the exact calibration data.

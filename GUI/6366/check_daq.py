#!/usr/bin/env python3
"""Step-by-step check of the NI DAQ connection, to locate where a problem is.

Run:  python check_daq.py            (checks every device found)
      python check_daq.py Dev1       (checks only Dev1)

Each step prints PASS or FAIL. The first FAIL tells you which layer is broken.
The analog output test only ever writes 0 V, so it is safe with an amplifier
connected. Send the full printed output when reporting a problem.
"""
import platform
import sys
import traceback


def step(n, title):
    print("\n[%d] %s" % (n, title))


def fail(msg, hint):
    print("    FAIL: %s" % msg)
    print("    -> %s" % hint)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    step(1, "Python")
    print("    Python %s on %s %s" % (platform.python_version(), platform.system(),
                                      platform.release()))
    if sys.version_info < (3, 9):
        fail("Python is older than 3.9", "Install Python 3.9 or newer.")
        return
    print("    PASS")

    step(2, "nidaqmx Python package")
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
    except Exception as exc:
        fail("cannot import nidaqmx (%s)" % exc,
             "Run:  pip install nidaqmx   (with the same Python that runs the GUI)")
        return
    print("    nidaqmx %s" % getattr(nidaqmx, "__version__", "?"))
    print("    PASS")

    step(3, "NI-DAQmx driver")
    try:
        system = nidaqmx.system.System.local()
        v = system.driver_version
        print("    driver %s.%s.%s" % (v.major_version, v.minor_version, v.update_version))
    except Exception as exc:
        fail("the package cannot reach the driver (%s)" % exc,
             "Install the NI-DAQmx driver from ni.com, then restart the computer.")
        return
    print("    PASS")

    step(4, "Devices seen by the driver")
    try:
        devices = list(system.devices)
    except Exception as exc:
        fail(str(exc), "Open NI MAX and check Devices and Interfaces.")
        return
    for d in devices:
        try:
            print("    %s : %s (serial %s)" % (d.name, d.product_type, d.serial_num))
        except Exception as exc:
            print("    %s : cannot read details (%s)" % (d.name, exc))
    if not devices:
        fail("the driver sees no device",
             "The problem is below Python: check the USB cable / power, then look in "
             "Windows Device Manager and NI MAX. If NI MAX does not list the device, "
             "the GUI cannot either.")
        return
    if only:
        devices = [d for d in devices if d.name == only]
        if not devices:
            fail("no device named %s" % only, "Use one of the names listed above.")
            return
    print("    PASS")

    all_ok = True
    n = 5
    for d in devices:
        step(n, "%s: channels" % d.name)
        n += 1
        try:
            ai = [c.name for c in d.ai_physical_chans]
            ao = [c.name for c in d.ao_physical_chans]
            print("    %d analog inputs, %d analog outputs" % (len(ai), len(ao)))
            print("    max rates: AI %.0f Hz, AO %.0f Hz" % (
                d.ai_max_single_chan_rate, d.ao_max_rate))
            print("    PASS")
        except Exception:
            traceback.print_exc()
            fail("cannot query the device",
                 "Run Self-Test and Reset Device in NI MAX. A simulated NI MAX device "
                 "or a disconnected device can cause this.")
            all_ok = False
            continue

        if ai:
            step(n, "%s: read 1000 samples from %s at 10 kHz" % (d.name, ai[0]))
            n += 1
            try:
                with nidaqmx.Task() as t:
                    t.ai_channels.add_ai_voltage_chan(ai[0])
                    t.timing.cfg_samp_clk_timing(
                        10000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)
                    data = t.read(number_of_samples_per_channel=1000, timeout=5.0)
                mean = sum(data) / len(data)
                rms = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
                print("    got %d samples, mean %.5f V, rms %.5f V" % (len(data), mean, rms))
                print("    PASS")
            except Exception:
                traceback.print_exc()
                fail("analog input does not work",
                     "Close NI MAX test panels and other programs using the device, "
                     "then try again.")
                all_ok = False

        if ao:
            step(n, "%s: write 1000 samples of 0 V to %s at 10 kHz" % (d.name, ao[0]))
            n += 1
            try:
                with nidaqmx.Task() as t:
                    t.ao_channels.add_ao_voltage_chan(ao[0])
                    t.timing.cfg_samp_clk_timing(
                        10000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)
                    t.write([0.0] * 1000, auto_start=False)
                    t.start()
                    t.wait_until_done(timeout=5.0)
                print("    PASS")
            except Exception:
                traceback.print_exc()
                fail("analog output does not work",
                     "Close NI MAX test panels and other programs using the device, "
                     "then try again.")
                all_ok = False

    print()
    if all_ok:
        print("All checks passed. The driver, the Python package and the device work.")
        print("If the GUI still fails, the problem is in the GUI program: report the")
        print("error text shown in its error window / status bar.")
    else:
        print("Some checks failed. Send this whole output when reporting.")


if __name__ == "__main__":
    main()

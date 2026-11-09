"""
RVR (receiving voltage response) of an uncalibrated hydrophone by comparison.

Both hydrophones sit at the same spot in front of the same source, so they
feel the same pressure p. Each one turns p into a voltage:

    V_ref  = p * M_ref          V_test = p * M_test

Divide one by the other and p cancels:

    M_test = M_ref * (V_test / V_ref)

In dB (RVR in dB re 1 V/uPa):

    RVR_test(f) = RVR_ref(f) + 20 log10( V_test(f) / V_ref(f) )

Any preamp gain on either channel is removed first.

INPUT: a CSV file with one row per measurement:
    freq_Hz, V_ref, V_test
Use the same kind of voltage for both columns (both peak, or both RMS).
Repeat a frequency on several rows to average several measurements.
If the file does not exist, an example file is written so you can see the format.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt

# ======================= SETUP: EDIT THESE ===========================
DATA_FILE    = "rvr_data.csv"     # your measured voltages
REF_RVR_DB   = -215.0             # RVR of the calibrated hydrophone (dB re 1 V/uPa)
REF_FMAX_HZ  = 20_000             # highest frequency where that calibration is valid
GAIN_REF_DB  = 0.0                # preamp gain on the calibrated hydrophone (dB)
GAIN_TEST_DB = 0.0                # preamp gain on the uncalibrated hydrophone (dB)
OUT_CSV      = "rvr_results.csv"
OUT_PNG      = "rvr_curve.png"
# =====================================================================


def ref_rvr(f):
    """RVR of the calibrated hydrophone at frequency f (dB re 1 V/uPa).
    Flat here. Replace with an interpolated calibration table if you have one."""
    return np.full_like(np.asarray(f, float), REF_RVR_DB)


def write_example(path):
    """Example data: test hydrophone 3-6 dB more sensitive, with a little noise."""
    rng = np.random.default_rng(1)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_Hz", "V_ref", "V_test"])
        for f in range(1000, 20001, 1000):
            true_offset = 3 + 3 * f / 20000                     # dB
            for _ in range(3):                                  # 3 repeats per frequency
                v_ref = 0.010 * (1 + 0.02 * rng.normal())
                v_test = v_ref * 10 ** (true_offset / 20) * (1 + 0.02 * rng.normal())
                w.writerow([f, f"{v_ref:.6f}", f"{v_test:.6f}"])
    print(f"No data file found. Wrote an example to {path}. Replace it with your data.")


def load(path):
    f, vr, vt = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            f.append(float(row["freq_Hz"]))
            vr.append(float(row["V_ref"]))
            vt.append(float(row["V_test"]))
    return np.array(f), np.array(vr), np.array(vt)


def main():
    if not os.path.exists(DATA_FILE):
        write_example(DATA_FILE)
    f_all, vr_all, vt_all = load(DATA_FILE)

    # remove preamp gains so both voltages are at the hydrophone terminals
    vr_all = vr_all / 10 ** (GAIN_REF_DB / 20)
    vt_all = vt_all / 10 ** (GAIN_TEST_DB / 20)

    # voltage ratio for every row, in dB
    ratio_db = 20 * np.log10(vt_all / vr_all)

    freqs = np.unique(f_all)
    rows = []
    for f in freqs:
        r = ratio_db[f_all == f]
        mean_ratio = r.mean()
        spread = r.std(ddof=1) if len(r) > 1 else np.nan      # repeatability (dB)
        rvr_test = ref_rvr(f) + mean_ratio
        rows.append((f, float(ref_rvr(f)), mean_ratio, float(rvr_test), spread, len(r)))
    rows = np.array(rows)

    print(f"{'freq (Hz)':>10} {'ref RVR':>9} {'ratio (dB)':>11} {'test RVR':>9} {'std (dB)':>9} {'n':>3}")
    for f, rr, d, rt, s, n in rows:
        flag = "  <- above ref calibration range" if f > REF_FMAX_HZ else ""
        print(f"{f:10.0f} {rr:9.1f} {d:11.2f} {rt:9.1f} {s:9.2f} {int(n):3d}{flag}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_Hz", "RVR_ref_dB", "V_ratio_dB", "RVR_test_dB_re_1V_per_uPa",
                    "std_dB", "n_measurements"])
        w.writerows(rows)

    ok = rows[:, 0] <= REF_FMAX_HZ
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    a = ax[0]
    a.plot(rows[:, 0] / 1e3, rows[:, 1], "k--", label="calibrated hydrophone (reference)")
    a.errorbar(rows[ok, 0] / 1e3, rows[ok, 3], yerr=np.nan_to_num(rows[ok, 4]),
               fmt="o-", ms=4, capsize=2, color="C0", label="uncalibrated hydrophone (result)")
    if (~ok).any():
        a.plot(rows[~ok, 0] / 1e3, rows[~ok, 3], "o", color="0.6", ms=4,
               label="outside reference calibration range")
    a.set_ylabel("RVR (dB re 1 V/µPa)")
    a.set_title("Receiving voltage response by comparison")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1]
    a.plot(rows[:, 0] / 1e3, rows[:, 2], "o-", ms=3, color="C1")
    a.set_xlabel("Frequency (kHz)")
    a.set_ylabel("V_test / V_ref (dB)")
    a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=200)
    print(f"Saved {OUT_CSV} and {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()

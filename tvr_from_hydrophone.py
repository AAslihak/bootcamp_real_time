"""
TVR (transmitting voltage response) of a source, measured with one calibrated
hydrophone placed at a known range r.

The source is driven with voltage V_drive. The hydrophone at range r gives
voltage V_hyd. Its calibration (RVR, in dB re 1 V/uPa) turns that voltage
into pressure:

    SPL at r  =  20 log10(V_hyd) - RVR                     [dB re 1 uPa]

The TVR is defined at 1 m, so add back the transmission loss from 1 m to r:

    TL(r)     =  20 log10(r / 1 m)  +  alpha * r            [dB]
                 spherical spreading   absorption

    TVR(f)    =  SPL at r + TL(r) - 20 log10(V_drive)       [dB re 1 uPa/V at 1 m]

alpha (dB/m) comes from the Francois-Garrison formula. Salinity 0 gives fresh
water (only the pure-water term). Salinity 35 gives seawater.

INPUT: a CSV file with one row per measurement:
    freq_Hz, V_drive, V_hyd
Use the same kind of voltage in both columns (both peak, or both RMS).
Repeat a frequency on several rows to average several measurements.
If the file does not exist, an example file is written so you can see the format.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt

# ======================= SETUP: EDIT THESE ===========================
DATA_FILE     = "tvr_data.csv"   # your measured voltages
RANGE_M       = 3.0              # source-to-hydrophone distance (m)
HYD_RVR_DB    = -215.0           # RVR of the calibrated hydrophone (dB re 1 V/uPa)
REF_FMAX_HZ   = 20_000           # highest frequency where that calibration is valid
HYD_GAIN_DB   = 0.0              # preamp gain after the hydrophone (dB)
DRIVE_RATIO   = 1.0              # V_source / V_measured (e.g. 10 for a 10:1 probe or divider)

# water properties for the absorption (Francois-Garrison)
TEMP_C        = 20.0             # temperature (C)
SALINITY      = 0.0              # practical salinity: 0 = fresh tank, 35 = seawater
DEPTH_M       = 1.0              # depth (m)
PH            = 8.0              # pH (only matters in seawater)

SOURCE_RADIUS_M = None           # largest source dimension / 2 (m). Used for a far-field check

OUT_CSV       = "tvr_results.csv"
OUT_PNG       = "tvr_curve.png"
# =====================================================================


def sound_speed(T, S, D):
    """Simple sound speed used inside Francois-Garrison (m/s)."""
    return 1412 + 3.21 * T + 1.19 * S + 0.0167 * D


def absorption_db_per_m(f_hz, T=TEMP_C, S=SALINITY, D=DEPTH_M, pH=PH):
    """Francois-Garrison absorption (1982), returned in dB/m."""
    f = np.asarray(f_hz, float) / 1000.0                     # kHz
    c = sound_speed(T, S, D)
    theta = T + 273.0
    # boric acid relaxation
    A1 = 8.86 / c * 10 ** (0.78 * pH - 5)
    f1 = 2.8 * np.sqrt(S / 35) * 10 ** (4 - 1245 / theta)
    # magnesium sulfate relaxation
    A2 = 21.44 * S / c * (1 + 0.025 * T)
    P2 = 1 - 1.37e-4 * D + 6.2e-9 * D ** 2
    f2 = 8.17 * 10 ** (8 - 1990 / theta) / (1 + 0.0018 * (S - 35))
    # pure water (viscosity)
    if T <= 20:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.50e-8 * T ** 3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3
    P3 = 1 - 3.83e-5 * D + 4.9e-10 * D ** 2
    a = (A1 * f1 * f ** 2 / (f1 ** 2 + f ** 2)
         + A2 * P2 * f2 * f ** 2 / (f2 ** 2 + f ** 2)
         + A3 * P3 * f ** 2)                                  # dB/km
    return a / 1000.0


def hyd_rvr(f):
    """RVR of the calibrated hydrophone (dB re 1 V/uPa). Flat here.
    Replace with an interpolated calibration table if you have one."""
    return np.full_like(np.asarray(f, float), HYD_RVR_DB)


def write_example(path):
    """Example data from a resonant source (true TVR peaks near 12 kHz)."""
    rng = np.random.default_rng(2)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_Hz", "V_drive", "V_hyd"])
        for f in range(1000, 20001, 1000):
            tvr_true = 135 - 10 * np.log10(1 + ((f - 12000) / 4000) ** 2)
            tl = 20 * np.log10(RANGE_M) + absorption_db_per_m(f) * RANGE_M
            for _ in range(3):
                v_drive = 10.0 * (1 + 0.01 * rng.normal())
                spl = tvr_true + 20 * np.log10(v_drive) - tl
                v_hyd = 10 ** ((spl + HYD_RVR_DB + HYD_GAIN_DB) / 20) * (1 + 0.02 * rng.normal())
                w.writerow([f, f"{v_drive / DRIVE_RATIO:.6f}", f"{v_hyd:.6g}"])
    print(f"No data file found. Wrote an example to {path}. Replace it with your data.")


def load(path):
    f, vd, vh = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            f.append(float(row["freq_Hz"]))
            vd.append(float(row["V_drive"]))
            vh.append(float(row["V_hyd"]))
    return np.array(f), np.array(vd), np.array(vh)


def main():
    if not os.path.exists(DATA_FILE):
        write_example(DATA_FILE)
    f_all, vd_all, vh_all = load(DATA_FILE)
    vd_all = vd_all * DRIVE_RATIO                         # volts at the source
    vh_all = vh_all / 10 ** (HYD_GAIN_DB / 20)            # volts at the hydrophone

    # every row: TVR = 20log(Vhyd) - RVR + TL - 20log(Vdrive)
    spread_db = 20 * np.log10(RANGE_M)
    absorb_all = absorption_db_per_m(f_all) * RANGE_M
    tvr_all = (20 * np.log10(vh_all) - hyd_rvr(f_all)
               + spread_db + absorb_all - 20 * np.log10(vd_all))

    rows = []
    for f in np.unique(f_all):
        sel = f_all == f
        t = tvr_all[sel]
        rows.append((f, t.mean(), t.std(ddof=1) if sel.sum() > 1 else np.nan,
                     spread_db, absorb_all[sel][0], sel.sum()))
    rows = np.array(rows)

    print(f"Range {RANGE_M} m: spreading correction {spread_db:.2f} dB at every frequency.")
    print(f"Water: T = {TEMP_C} C, S = {SALINITY}, depth = {DEPTH_M} m.\n")
    print(f"{'freq (Hz)':>10} {'TVR (dB)':>9} {'std':>6} {'absorption (dB)':>16} {'n':>3}")
    for f, tvr, s, sp, ab, n in rows:
        flag = "  <- above hydrophone calibration range" if f > REF_FMAX_HZ else ""
        print(f"{f:10.0f} {tvr:9.1f} {s:6.2f} {ab:16.5f} {int(n):3d}{flag}")

    if SOURCE_RADIUS_M is not None:
        c = sound_speed(TEMP_C, SALINITY, DEPTH_M)
        r_far = SOURCE_RADIUS_M ** 2 / (c / rows[:, 0].max())
        print(f"\nFar-field distance a^2/lambda at {rows[:,0].max()/1e3:.0f} kHz: {r_far:.2f} m. "
              + ("OK." if RANGE_M > 2 * r_far else "Range may be too short: near field."))

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_Hz", "TVR_dB_re_1uPa_per_V_at_1m", "std_dB",
                    "spreading_corr_dB", "absorption_corr_dB", "n_measurements"])
        w.writerows(rows)

    ok = rows[:, 0] <= REF_FMAX_HZ
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    a = ax[0]
    a.errorbar(rows[ok, 0] / 1e3, rows[ok, 1], yerr=np.nan_to_num(rows[ok, 2]),
               fmt="o-", ms=4, capsize=2, color="C0", label="TVR")
    if (~ok).any():
        a.plot(rows[~ok, 0] / 1e3, rows[~ok, 1], "o", color="0.6", ms=4,
               label="outside hydrophone calibration range")
    a.set_ylabel("TVR (dB re 1 µPa/V at 1 m)")
    a.set_title(f"Transmitting voltage response (hydrophone at {RANGE_M} m)")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1]
    a.plot(rows[:, 0] / 1e3, rows[:, 3], "k--", label="spreading, 20 log r")
    a.plot(rows[:, 0] / 1e3, rows[:, 4], "C3o-", ms=3, label="absorption, α r")
    a.set_xlabel("Frequency (kHz)"); a.set_ylabel("Correction to 1 m (dB)")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=200)
    print(f"\nSaved {OUT_CSV} and {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()

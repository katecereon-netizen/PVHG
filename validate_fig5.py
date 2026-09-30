"""
Fig. 5 validation of the five-node PV + Diaz-Marin hydrogel model against
C. Lai, H. Yang, Y. Gao, T. Ma, L. Lu, Appl. Energy 426 (2026) 128583 (indoor test, ~1040 W/m2).

Procedure -- nothing is fitted on the PV/HG data:
  1. Fig. 4a (IJHMT) bare PV  -> refit PV_CAP_MULT and v_wind with the physics corrections on.
     Same module in both papers, so PV_CAP_MULT is carried over unchanged.
  2. Fig. 5 bare PV           -> fit v_wind (lab air flow is not reported and is not the same test)
     and delta, the clock offset between panel (a) and panels (b)-(d).
  3. PV/HG                    -> prediction with the Appl. Energy Table 4 gel (4 mm, phi_s/l/g =
     0.1/0.6/0.3), omega0 = 0.20 (Sec. 3.1: freeze-dried gel re-swollen in 20 wt% CaCl2), RH = 70 %,
     HTC_conv = 2.8 + 3v (Eq. 24).
  4. Sensitivities / diagnostics on the PV/HG prediction.

Outputs: fig5_validation.png, fig5_forcing.png, fig5_results.csv and printed tables.
"""

import csv
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig4a_data as f4
import fig5_data as d
from pvhg_model import PVHG, metrics

K = 273.15
COLOR_HG, COLOR_BARE = "#2ABFFF", "#FF6A2A"            # same as the Fig. 4a script
COLOR_V1, COLOR_V2 = "#1baf7a", "#4a3aa7"              # diagnostic variants
COLOR_PAPER, INK, INK2 = "#7a7974", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "grid.alpha": 0.35, "savefig.dpi": 300, "savefig.bbox": "tight"})

# ------------------------------------------------------------------------------------------ setup
# corrections found in the review (see pvhg_model.py docstring); dilution kept, back sink = T_g
T_P_MIN = 5.0     # power is scored for t >= 5 min: before that panel (a)'s lamp ramp contradicts panel (c)
FIX = dict(lewis_exp=2 / 3, elec_area_frac=0.0064 / 0.01, iface_Rted=True, relin_pv=True)
ORIGINAL_FIG4 = dict(v_wind=0.32, PV_CAP_MULT=1.50)     # the Fig. 4a script's calibration


def G5(t):
    return float(np.interp(t / 60, d.t_a_min, d.G_a, left=0.0))


def Ta5(t):
    return float(np.interp(t / 60, d.t_a_min, d.Ta_a)) + K


def Tg5(t):
    return float(np.interp(t / 60, d.t_a_min, d.Tg_a)) + K


# panel (a) clock; both modules start at the room temperature read off panel (a) before lamp-on
FIG5 = dict(G=G5, Ta=Ta5, Tg=Tg5, RH=0.70, T_init_bare=22.4 + K, T_init_hg=22.4 + K,
            t_end=90 * 60.0, htc_a=2.8, htc_b=3.0, d_HG=4e-3, omega0=0.20)


def fit_fig4_bare(fix):
    def rmse(x):
        b = PVHG(t_end=3 * 3600.0, v_wind=x[0], PV_CAP_MULT=x[1], **fix).run(False)
        return metrics(b["t"] / 3600, b["T_top"] - K, f4.t_bare_fig4a, f4.v_bare_fig4a)["RMSE"]
    r = minimize(rmse, x0=[0.32, 1.5], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4))
    return dict(v_wind=r.x[0], PV_CAP_MULT=r.x[1])


def fit_fig5_bare(fix, cap, back):
    """v_wind and clock offset on the bare temperature, then eta_ref on the bare power (t >= 5 min,
    where panel (a)'s lamp ramp and the measured power no longer disagree), then v_wind/offset again."""
    eta = 0.13
    for _ in range(2):
        def rmse(x):
            b = PVHG(v_wind=x[0], PV_CAP_MULT=cap, back_sink=back, eta_ref=eta, **fix, **FIG5).run(False)
            return metrics(b["t"] / 60 - x[1], b["T_top"] - K, d.t_min, d.T_bare_exp)["RMSE"]
        x0 = min(itertools.product((0.3, 0.8, 1.5), (2.0, 4.0)), key=rmse)
        v, delta = minimize(rmse, x0=x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4)).x
        b = PVHG(v_wind=v, PV_CAP_MULT=cap, back_sink=back, eta_ref=eta, **fix, **FIG5).run(False)
        k = d.t_min >= T_P_MIN
        U = np.interp(d.t_min[k], b["t"] / 60 - delta, b["P"]) / eta  # P is linear in eta_ref at fixed T
        eta = float(np.sum(U * d.P_bare_exp[k]) / np.sum(U * U))
    return v, delta, eta


def run_case(kw):
    """kw: plain (picklable) PVHG keyword overrides on top of FIG5."""
    m = PVHG(**{**FIG5, **kw})
    return m.run(False), m.run(True), m.HTC


def evaluate(bare, hg, delta):
    tm = bare["t"] / 60 - delta                        # panel (b)-(d) clock
    kp = d.t_min >= T_P_MIN
    T_b, T_h = bare["T_top"] - K, hg["T_top"] - K
    return dict(
        tm=tm, T_bare=T_b, T_hg=T_h, P_bare=bare["P"], P_hg=hg["P"], dT=T_b - T_h,
        m_Tb=metrics(tm, T_b, d.t_min, d.T_bare_exp), m_Th=metrics(tm, T_h, d.t_min, d.T_hg_exp),
        m_Pb=metrics(tm, bare["P"], d.t_min[kp], d.P_bare_exp[kp]), m_Ph=metrics(tm, hg["P"], d.t_min[kp], d.P_hg_exp[kp]),
        m_dT=metrics(tm, T_b - T_h, d.t_min, d.dT_exp))


def main():
    # ---- 1. PV parameters from Fig. 4a bare (same module) -------------------------------------
    pv_fix = fit_fig4_bare(FIX)
    print(f"Fig. 4a bare refit, corrected physics: v_wind = {pv_fix['v_wind']:.3f} m/s, "
          f"PV_CAP_MULT = {pv_fix['PV_CAP_MULT']:.3f}   (original script: 0.32 m/s, 1.50)")

    # ---- 2. Fig. 5 bare: v_wind and clock offset ------------------------------------------------
    fits = {}
    for name, fix, cap, back in (("corrected", FIX, pv_fix["PV_CAP_MULT"], "ground"),
                                 ("original", {}, ORIGINAL_FIG4["PV_CAP_MULT"], "ground"),
                                 ("corrected, rear->T_a", FIX, pv_fix["PV_CAP_MULT"], "ambient")):
        v, delta, eta = fit_fig5_bare(fix, cap, back)
        fits[name] = (dict(v_wind=v, PV_CAP_MULT=cap, back_sink=back, eta_ref=eta, **fix), delta)
        print(f"Fig. 5 bare fit [{name:21s}]: v_wind = {v:.3f} m/s (h_conv = {2.8 + 3 * v:.2f} W/m2K), "
              f"clock offset delta = {delta:.2f} min, eta_ref = {eta:.4f}")

    # ---- 3./4. PV/HG prediction and sensitivities -----------------------------------------------
    base, delta = fits["corrected"]
    cases = {
        "corrected physics (Table 4 gel)": (base, delta),
        "original physics":                fits["original"],
        "omega0 = 0.30":                   ({**base, "omega0": 0.30}, delta),
        "gel 8.45 mm (Fig. 4a water load)": ({**base, "d_HG": 8.45e-3}, delta),
        "rear face -> T_a (refit)":        fits["corrected, rear->T_a"],
        "D_l x 100":                       ({**base, "Dl_mult": 100.0}, delta),
        "D_l x 100 + ECP on whole face":   ({**base, "Dl_mult": 100.0, "ecp_area": 1.0}, delta),
        "Lai shortcuts (vented pores + whole-face ECP)": ({**base, "vent_pores": True, "ecp_area": 1.0}, delta),
    }
    with ProcessPoolExecutor() as ex:
        runs = dict(zip(cases, ex.map(run_case, [kw for kw, _ in cases.values()])))
    res = {k: evaluate(runs[k][0], runs[k][1], cases[k][1]) for k in cases}
    for k in cases:
        res[k]["eta"] = cases[k][0]["eta_ref"]

    paper = dict(m_Tb=metrics(d.t_min, d.T_bare_paper, d.t_min, d.T_bare_exp),
                 m_Th=metrics(d.t_min, d.T_hg_paper, d.t_min, d.T_hg_exp),
                 m_dT=metrics(d.t_min, d.dT_paper, d.t_min, d.dT_exp))
    print("\nErrors vs Fig. 5 data (Er = Lai Eq. 43 mean relative error):")
    print(f"{'case':48s} {'T bare':>16s} {'T PV/HG':>16s} {'bias PV/HG':>10s} {'P bare*':>8s} {'P PV/HG*':>8s} {'dT MAE':>7s}")
    for k, r in res.items():
        bias = np.mean(np.interp(d.t_min, r["tm"], r["T_hg"]) - d.T_hg_exp)
        print(f"{k:48s} {r['m_Tb']['MAE']:5.2f}C {r['m_Tb']['Er']:5.2f}%   {r['m_Th']['MAE']:5.2f}C {r['m_Th']['Er']:5.2f}%"
              f"   {bias:+7.2f}C  {r['m_Pb']['Er']:6.2f}%  {r['m_Ph']['Er']:6.2f}%  {r['m_dT']['MAE']:5.2f}C")
    print(f"{'Lai et al. model (digitised from Fig. 5)':48s} {paper['m_Tb']['MAE']:5.2f}C {paper['m_Tb']['Er']:5.2f}%"
          f"   {paper['m_Th']['MAE']:5.2f}C {paper['m_Th']['Er']:5.2f}%{'':24s}{paper['m_dT']['MAE']:5.2f}C")
    print(f"* power scored for t >= {T_P_MIN:.0f} min")
    hg = runs["corrected physics (Table 4 gel)"][1]
    i = np.argmin(np.abs(res["corrected physics (Table 4 gel)"]["tm"] - 81))
    print(f"\ncorrected, Table 4 gel: peak MCP = {hg['MCP'].max():.0f} W/m2, peak ECP = {hg['ECP'].max():.0f} W/m2, "
          f"water lost by 81 min = {100 * (1 - hg['water'][i] / hg['water'][0]):.1f} %, "
          f"max |energy residual| = {np.abs(hg['res']).max():.1e} W/m2")

    # ---- table view -----------------------------------------------------------------------------
    main_r, orig_r = res["corrected physics (Table 4 gel)"], res["original physics"]
    with open("fig5_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_min", "T_bare_exp", "T_bare_model", "T_hg_exp", "T_hg_model", "T_hg_model_original",
                    "P_bare_exp", "P_bare_model", "P_hg_exp", "P_hg_model", "dT_exp", "dT_model"])
        at = lambda r, key: np.interp(d.t_min, r["tm"], r[key])
        cols = [d.t_min, d.T_bare_exp, at(main_r, "T_bare"), d.T_hg_exp, at(main_r, "T_hg"), at(orig_r, "T_hg"),
                d.P_bare_exp, at(main_r, "P_bare"), d.P_hg_exp, at(main_r, "P_hg"), d.dT_exp, at(main_r, "dT")]
        for row in zip(*cols):
            w.writerow([f"{v:.4g}" for v in row])

    plot_validation(res, delta)
    plot_forcing(delta)


def plot_validation(res, delta):
    r, ro = res["corrected physics (Table 4 gel)"], res["original physics"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    sel = lambda x: (x >= 0) & (x <= 82)

    ax = axs[0, 0]
    ax.plot(d.t_min, d.T_bare_exp, "o", ms=4.5, mfc="none", mec=COLOR_BARE, mew=1.0, label="W/O HG, exp.")
    ax.plot(d.t_min, d.T_hg_exp, "^", ms=4.5, mfc="none", mec=COLOR_HG, mew=1.0, label="W/ HG, exp.")
    k = sel(r["tm"])
    lab = lambda m: f" ($E_r$ {m['Er']:.2f}%, MAE {m['MAE']:.2f}$^\\circ$C)"
    ax.plot(r["tm"][k], r["T_bare"][k], "-", color=COLOR_BARE, lw=2, label="W/O HG, model" + lab(r["m_Tb"]))
    ax.plot(r["tm"][k], r["T_hg"][k], "-", color=COLOR_HG, lw=2, label="W/ HG, model" + lab(r["m_Th"]))
    ax.plot(ro["tm"][k], ro["T_hg"][k], "--", color=COLOR_HG, lw=1.2, label="W/ HG, original physics" + lab(ro["m_Th"]))
    ax.plot(d.t_min, d.T_hg_paper, ":", color=COLOR_PAPER, lw=1.5, label="W/ HG, Lai et al. model")
    ax.set_ylabel(r"PV top surface temperature [$^\circ$C]")
    ax.legend(loc="lower right", fontsize=7.8, frameon=True, framealpha=0.95, edgecolor="0.3", fancybox=False)
    ax.set_title("(a) temperature", loc="left", fontsize=11, fontweight="bold")

    ax = axs[0, 1]
    ax.plot(d.t_min, d.P_bare_exp, "v", ms=4, mfc="none", mec=COLOR_BARE, mew=0.9, label="W/O HG, exp.")
    ax.plot(d.t_min, d.P_hg_exp, "o", ms=4, mfc="none", mec=COLOR_HG, mew=0.9, label="W/ HG, exp.")
    labp = lambda m: f" ($E_r$ {m['Er']:.2f}%, MAE {m['MAE']:.3f} W)"
    ax.plot(r["tm"][k], r["P_bare"][k], "-", color=COLOR_BARE, lw=2, label="W/O HG, model" + labp(r["m_Pb"]))
    ax.plot(r["tm"][k], r["P_hg"][k], "-", color=COLOR_HG, lw=2, label="W/ HG, model" + labp(r["m_Ph"]))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Power [W]")
    ax.text(0.03, 0.04, r"model: $P = G A_{PV}\,\eta_{ref}[1-\beta(T_{cell}-25)]$, IJHMT Eq.(20)-(21)" "\n"
            rf"$\eta_{{ref}}$ = {res['corrected physics (Table 4 gel)']['eta']:.3f} fitted on W/O HG power; errors for $t \geq$ {T_P_MIN:.0f} min" "\n"
            r"dip at $t$ < 4 min: panel (a) lamp ramp lags the measured power",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3, color=INK2)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=7.8, frameon=True, framealpha=0.95,
              edgecolor="0.3", fancybox=False)
    ax.set_title("(b) power", loc="left", fontsize=11, fontweight="bold")

    ax = axs[1, 0]
    ax.plot(d.t_min, d.dT_exp, "o", ms=3.5, color=INK, label=r"$\Delta T$, exp.")
    ax.plot(r["tm"][k], r["dT"][k], "-", color=COLOR_HG, lw=2, label=r"$\Delta T$, model")
    ax.plot(ro["tm"][k], ro["dT"][k], "--", color=COLOR_HG, lw=1.2, label=r"$\Delta T$, model, original physics")
    ax.plot(d.t_min, d.dT_paper, ":", color=COLOR_PAPER, lw=1.5, label=r"$\Delta T$, Lai et al. model")
    ax.set_ylim(0, 30)
    ax.set_ylabel(r"$T_{W/O\,HG}-T_{W/\,HG}$ [$^\circ$C]")
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.95, edgecolor="0.3", fancybox=False)
    ax.set_title("(c) temperature reduction", loc="left", fontsize=11, fontweight="bold")

    ax = axs[1, 1]
    ax.plot(d.t_min, d.T_hg_exp, "^", ms=4.5, mfc="none", mec=INK, mew=0.9, label="W/ HG, exp.")
    variants = (("corrected physics (Table 4 gel)", COLOR_HG, "-", "Table 4 gel"),
                ("D_l x 100", COLOR_V1, "-", r"$D_l\times100$"),
                ("D_l x 100 + ECP on whole face", COLOR_V2, "-", r"$D_l\times100$ + ECP on whole face"))
    for key, c, ls, lab in variants:
        rv = res[key]
        ax.plot(rv["tm"][k], rv["T_hg"][k], ls, color=c, lw=1.8, label=f"{lab} (MAE {rv['m_Th']['MAE']:.1f}$^\\circ$C)")
    ax.plot(d.t_min, d.T_hg_paper, ":", color=COLOR_PAPER, lw=1.5, label="Lai et al. model")
    ax.set_ylabel(r"W/ HG top surface temperature [$^\circ$C]")
    ax.legend(loc="lower right", fontsize=7.8, frameon=True, framealpha=0.95, edgecolor="0.3", fancybox=False)
    ax.set_title("(d) what limits the PV/HG prediction", loc="left", fontsize=11, fontweight="bold")

    for ax in axs.flat:
        ax.set_xlim(0, 82)
        ax.set_xlabel("Time [min]")
        ax.grid(True)
        ax.minorticks_on()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09)
    fig.text(0.5, 0.012,
             rf"Fig. 5 conditions: $G$, $T_a$, $T_g$ from panel (a), shifted by $\delta$ = {delta:.2f} min (fitted on bare PV)  |  "
             r"gel 4 mm, $\omega_0$ = 0.20, $\phi_{s/l/g}$ = 0.1/0.6/0.3  |  PV/HG curves are predictions, not fits",
             ha="center", va="bottom", fontsize=8, color=INK2)
    fig.savefig("fig5_validation.png")
    plt.close(fig)


def plot_forcing(delta):
    t = d.t_a_min - delta
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5.2), sharex=True)
    a1.plot(t, d.G_a, "-", color=INK, lw=1.6)
    a1.set_ylabel(r"$G$ [W/m$^2$]")
    a1.set_title(r"Fig. 5(a) boundary conditions on the panel (b)-(d) clock", loc="left", fontsize=10.5, fontweight="bold")
    a2.plot(t, d.Tg_a, "-", color=COLOR_V2, lw=1.6)
    a2.plot(t, d.Ta_a, "-", color=COLOR_V1, lw=1.6)
    a2.text(80, d.Tg_a[-1] - 2.0, r"$T_g$ (platform)", ha="right", va="top", fontsize=9, color=INK)
    a2.text(80, d.Ta_a[-1] + 1.5, r"$T_a$ (air)", ha="right", va="bottom", fontsize=9, color=INK)
    a2.set_ylabel(r"Temperature [$^\circ$C]")
    a2.set_xlabel("Time [min]")
    for ax in (a1, a2):
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.grid(True)
        ax.set_xlim(-delta, 82)
    fig.tight_layout()
    fig.savefig("fig5_forcing.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

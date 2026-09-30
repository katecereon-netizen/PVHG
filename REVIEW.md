# Physics review of `diaz_hg_fig4a_5node_corrected.py` and Fig. 5 validation

**Short answer.** The numerical core is sound. There is no bug that invalidates the Fig. 4a result, but the review found
some real physics inconsistencies and one crash path. The Fig. 5 test (Lai et al., *Appl. Energy* 426 (2026) 128583)
is **not** passed. The bare module validates (Er 2.1 %), but the PV/HG temperature is predicted about 10 °C too hot.
Fig. 5(d) of the same paper is ~17 °C of sustained cooling at 80 min; the model gives ~5 °C. The miss comes from
gel transport physics and parameters, not from a coding error. The details are below.

Everything here is reproducible with `python3 validate_fig5.py` (about 30 s), which uses `pvhg_model.py`.
`pvhg_model.py` is a refactor of your script. With default arguments it reproduces your script's output to 1e-10 K
over the full 5 h run. Every correction below is a switch whose default is your original behaviour.

## 1. What was checked and is correct

| Check | Result |
|---|---|
| Discrete energy closure of the gel, `q_if - q_face - MCP - ECP - stored` | 3e-8 W/m2 |
| Water mass closure (inventory change vs. integrated face outflow, 1.5 h) | 1e-13 relative |
| Latent heat charged once: -h_sorb·R with R from Eq. (8), so the face J enters through the last cell's dn/dt | correct |
| Newton linearisation: da_w/dn = a_w'(w)·(-w² M f / m_salt), dE/dT, J(n, T) | correct |
| Material-grid storage per reference volume, current-width face conductances | correct |
| Grid N_HG 80 → 160, dt 30 → 10 s (Fig. 4a) | PV/HG MAE changes ≤ 0.04 °C |
| r_p = 2e-8 m with N_p = 1e10 m^-3 → N·S = 5.0e3 m^-2 | consistent with the size of Lai's reported MCP |

## 2. Physics issues found (effect on Fig. 4a before any refit)

| # | Issue | Where | Effect |
|---|---|---|---|
| 1 | **Electrical extraction per m² of footprint uses η·G.** η belongs to the cell area A_PV = 0.64·S_PV, but the 1-D balance is per m² of S_PV, so this over-removes 0.36·η·G of heat (≈30 W/m² in Fig. 4a, ≈45 W/m² in Fig. 5). Lai Eq. (15) has the same inconsistency. | `P_elec_heat_removed` | bare +1.2 °C, PV/HG +1.2 °C |
| 2 | **Lewis relation exponent.** The code uses h_m = h/(ρ c_p Le). Chilton–Colburn (laminar and turbulent flat plate) gives Le^(2/3), so h_m is 8 % too high. | `h_mass` | PV/HG +0.1 °C |
| 3 | **Dilution-enthalpy fit intercept.** `polyval(P_HDIL, 0)` = 47.58 kJ/mol, not `H_VAP_0` = 44.0, so pure water gets +3.6 kJ/mol. The fit also dips between w = 0 and 0.2, which is non-physical. | `h_sorb` | ≤ 0.1 °C at w = 0.2–0.4; wrong for dilute gels |
| 4 | Tedlar half-layer resistance missing from the Eq. (23) interface mismatch | `mismatch` | 0.03 °C |
| 5 | PV radiation and η(T) linearised at the old time level in the PV/HG run, but iterated in the bare run | `run` | < 0.01 °C at dt = 30 s |
| 6 | **Crash path.** If the gel Newton fails at the first secant guess, `abs(f1 - f0)` raises a `TypeError` | `run` | crash (seen with larger D_l) |
| 7 | `M_SALT_MEAS` "measured" is not a measurement. 16.9 g comes from the *simulated* salt leakage in IJHMT Sec. 4.3 (1.69 g = 10 %). That total is 50.7 g of water ≈ 5 mm × 0.01 m² × 1000 kg/m³, so Lai's model most likely loads water at ρ_L per total volume in a 5 mm gel. It does not imply an 8.45 mm gel at φ_l0 = 0.6. | parameters | 5 mm instead of 8.45 mm: PV/HG Er 2.7 % → 4.9 % |
| 8 | The rear face radiates to T_g, whereas Lai Eqs. (23)/(25) send all radiation to T_a. This is a modelling choice, but it is the biggest single sensitivity. | `HTC_rad2` | bare −7 °C if switched to T_a |

Items 1–5 are corrected in `validate_fig5.py`: `lewis_exp=2/3`, `elec_area_frac=0.64`, `iface_Rted`, `relin_pv`.
Item 3 is left as is, because Fig. S1's data are not available to refit it. Item 6 now raises a clear error.

With items 1, 2, 4 and 5 corrected, refitting the bare Fig. 4a curve gives v_wind = 0.469 m/s (h_conv = 3.38 W/m²K)
and PV_CAP_MULT = 1.575, versus 0.32 m/s and 1.50 originally. The Fig. 4a PV/HG error moves from MAE 1.08 °C to 1.53 °C,
so the original calibration had absorbed part of these errors.

## 3. Fig. 5 setup (what changes from Fig. 4a)

| Quantity | Fig. 4a (IJHMT) | Fig. 5 (Appl. Energy) |
|---|---|---|
| G, T_a, T_g | 642 W/m², 29.0 °C, 51.6 °C constant | digitised Fig. 5(a) time series, ≈1040 W/m², 33.6 °C, 66.5 °C |
| HTC_conv | 2.3 + 2.3v (Eq. 18) | 2.8 + 3v (Eq. 24) |
| Gel | 5 mm (Table 3), ω0 = 0.25 | 4 mm (Table 4); ω0 not in Table 4. Sec. 3.1 says the gel is re-swollen in **20 wt %** CaCl2 but also calls 30 wt % "employed", so 0.20 is used and 0.30 is a sensitivity |
| Duration | 3 h | 81 min |
| v_wind | not reported | not reported |

`fig5_data.py` holds the digitised data, traced by colour from the published raster. As a check, the paper's own
simulation curves traced the same way give Er = 3.88 % (bare) and 2.37 % (PV/HG), against 3.72 % and 2.39 % printed
in the figure. The **MAE values printed in Fig. 5(b) are inconsistent with the Er values**: 0.047 °C for PV/HG is
impossible at Er = 2.39 %. Do not use them as targets.

Panel (a) runs on its own clock. Its G ramp ends at ~7.5 min, while panel (c) already shows ~0.8 W at t = 1 min.
A single offset δ = 2.9 min, fitted on the bare temperature, aligns them. The remaining disagreement is why power
is scored for t ≥ 5 min only.

## 4. Fig. 5 results (PV/HG curves are predictions; only bare-PV data were used for fitting)

Bare-PV fits: v_wind = 1.27 m/s (h_conv = 6.6 W/m²K), δ = 2.92 min, η_ref = 0.121. PV_CAP_MULT = 1.575 is carried
over from Fig. 4a, since it is the same module.

| Case | T bare MAE / Er | T PV/HG MAE / Er | P bare Er* | P PV/HG Er* | ΔT MAE |
|---|---|---|---|---|---|
| **corrected physics, Table 4 gel** | 1.37 °C / 2.14 % | 9.89 °C / 19.6 % | 4.7 % | 11.7 % | 9.8 °C |
| original physics | 1.37 °C / 2.15 % | 9.40 °C / 18.6 % | 4.7 % | 11.4 % | 9.3 °C |
| ω0 = 0.30 | — | 11.25 °C | — | 12.5 % | 11.2 °C |
| gel 8.45 mm (Fig. 4a water load) | — | 7.95 °C | — | 10.6 % | 7.8 °C |
| rear face → T_a (refit: h_conv 3.7) | 1.18 °C / 1.77 % | 8.88 °C | 4.6 % | 11.2 % | 8.8 °C |
| D_l × 100 | — | 5.73 °C | — | 9.4 % | 5.7 °C |
| D_l × 100 + ECP over the whole face | — | 3.03 °C | — | 8.3 % | 2.9 °C |
| Lai shortcuts (pores vented to C_a + whole-face ECP) | — | 8.96 °C | — | 11.2 % | 8.9 °C |
| *Lai et al. model (digitised)* | 2.15 °C / 3.88 % | 1.09 °C / 2.37 % | | | 1.4 °C |

\* t ≥ 5 min. "—" means the same bare fit as the row above it.

## 5. Why the PV/HG prediction fails, and what it tells you

1. **The gel face salts up.** D_l from Eq. (9) with G = 1 kPa, K = 6.74e-18 m² and η = 8.5e-4 Pa·s is 1.8e-10 m²/s.
   In 80 min that reaches only √(D_l·t) ≈ 0.9 mm into the gel. With immobile salt, the face salt fraction rises from
   0.20 to 0.44 by 81 min while the PV side stays at 0.21. Face water activity drops from 0.83 to 0.31, and ECP falls
   from 254 to 84 W/m², with only 19 % of the water used. Fig. 4a (ΔT decaying 16 → 5 °C) was compatible with this;
   Fig. 5 (ΔT still ~17 °C at 80 min) is not.
2. **Early cooling is also short, by ~7 °C at 10–20 min, in every variant.** At t = 20 min the measured PV/HG top is
   45 °C, which requires ≈650 W/m² into the gel. Through the PV plus 4 mm of gel (R ≈ 0.014 m²K/W), that leaves the
   evaporating face at ~36 °C, too cold to evaporate much. So in a 1-D slab, heat must be removed close to the PV.
3. **Lai's model gets its curve from simplifications that yours correctly does not make.** Their micropore term
   (Eqs. 13 and 17) uses the *ambient* C_a inside the pores, so bulk evaporation escapes with no pore-diffusion or
   boundary-layer resistance. Their ECP (Eq. 26) acts on the whole face. Emulating these in your framework
   (`vent_pores`, `ecp_area=1`) does not close the gap by itself, because face resupply (item 1 above) dominates.
4. **v_wind does not transfer between the two tests.** With the rear face seeing T_g, the bare modules need
   h_conv ≈ 3.4 (Fig. 4a) and ≈ 6.6 W/m²K (Fig. 5). With the rear seeing T_a, they need ≈ 1.2 (below natural
   convection) and ≈ 3.7. The rear-face exchange (view factor to the heated platform, air temperature under the
   module) is not pinned down by either paper.

## 6. Suggested next steps (not done here, since each is a modelling decision)

* Treat D_l (or equivalently K and G) as the gel parameter to calibrate, jointly on Fig. 4a and Fig. 5. It is
  what controls face resupply.
* Consider salt back-diffusion. The immobile-salt assumption is what drives the face to w ≈ 0.44.
* Decide the water loading (ρ_L·δ vs. φ_l0·ρ_L·δ) consistently for both papers. The recipe (90 mL water per
  two 100×100 mm coupons) points to ≈4.5–5 kg/m², i.e. ~5 mm of water, not 2.4 kg/m².
* Keep the rear-face sink and h_conv per experiment as fitted quantities, and report them.

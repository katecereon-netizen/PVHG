"""
Fig. 4(a) validation, bare PV and PV/HG -- Diaz-Marin hydrogel WITH SURFACE EVAPORATION on the
five-node PV network (glass / EVA-1 / cell / EVA-2 / Tedlar).  x = 0 is the PV/HG interface,
x = H(t) the HG/air face.

Hydrogel: Diaz-Marin Eqs.(4), (8), (14) + SI (S33)-(S41), solved on a MATERIAL (solid-attached)
grid, so the thickness H(t) = sum dx0*(1 + eps_xx) follows the swelling strain (S36) exactly and
no grid-velocity term is needed.  Pores deform affinely: r_p ~ (1+eps)^(1/3), N_p ~ 1/(1+eps).

Surface evaporation at x = H(t): two routes through ONE air boundary layer, each weighted by the
area fraction it occupies on the face, so the exposed area is not counted twice:
  pore vapour   -D_eff dC/dx = phi_g*h_m*(C_H - C_inf)                   Robin, generalises (S43)
  free liquid   -D_l   dn/dx = J = phi_l*h_m*(a_w*P_sat(T_H)/(R*T_H) - C_inf)
  heat          -k dT/dx     = h*(T_H - T_a) + q_rad + h_sorb*J
  h_m = h/(rho_a*c_pa*Le), Le = Sc/Pr, Sc = nu/D_v      (Lewis analogy), h = Lai Eq.(18) at the face
Latent heat is charged ONCE: the volumetric source of (14) is written as -R*h_sorb with
R = -(1-phi_g0)*dn/dt + d/dx(D_l dn/dx) (Eq. 8), so liquid drawn toward the face carries no
latent heat and h_sorb*J at the face is the only charge for surface evaporation.

PV side, Lai Eqs.(18)-(21), unchanged except for two corrections needed once dt is refined:

1. TIME STEP.  The bare module's own thermal time constant is C/h ~ 265 s (C = layer heat
   capacity, h = total front+back loss coefficient), so dt = 300 s (Lai Table 3) does not
   resolve its transient -- refining dt from 300 to 30 s changes the early bare-PV curve by
   several degrees C (implicit Euler's numerical damping was masking this).  dt = 30 s is
   converged to within 0.05 C of dt = 10 s.

2. PV THERMAL MASS.  At the converged dt, the bare module heats up faster than measured
   (its fitted response time is short by a factor ~1.5), so PV_CAP_MULT scales every layer's
   heat capacity by 1.50, fitted on the bare-PV curve alone (v_wind re-optimized jointly to
   0.32 m/s at this dt).  This is consistent with d_PV = 2.7 mm being the paper's only reported
   (bulk) thickness for this coupon -- the true per-layer split, and hence the true heat
   capacity, is not independently known.

3. GEL THICKNESS.  Table 3's d_HG = 5 mm, phi_l0 = 0.60 and omega0 = 0.25 cannot all be true
   together: at d_HG = 5 mm, phi_l0 = 0.60 holds only 30 g of water, but Lai Section 4.3 reports
   16.90 g of CaCl2 for this coupon, which at omega0 = 0.25 requires 50.7 g of water -- 69% more
   than 5 mm of gel can hold at phi_l0 = 0.60.  omega0 and phi_l0 (both dimensionless material
   properties) are kept at their stated values, and d_HG is solved for instead, since it is the
   one quantity in this triple the paper never actually measures for this coupon (Table 3 does
   not report a measured thickness, only nominal fabrication values): d_HG = 8.45 mm.  This
   incidentally supplies most of the extra thermal mass a separate rc-sensitivity scan already
   flagged as missing from the gel's fit.

Sources: C. Lai, J. Pu, L. Lu, IJHMT 261 (2026) 128567 (PV, surface heat transfer, Fig. 4a data,
Section 4.3 salt mass); C. D. Diaz-Marin et al., IJHMT 195 (2022) 123103 + SI (hydrogel).
"""

import colorsys

import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

COLOR_HG   = "#2ABFFF"   # W/ HG   - simulation line
COLOR_BARE = "#FF6A2A"   # W/O HG  - simulation line

def darken(hex_color, factor=0.78):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v * factor)
    return "#{:02X}{:02X}{:02X}".format(round(r2 * 255), round(g2 * 255), round(b2 * 255))

MARKER_HG, MARKER_BARE = darken(COLOR_HG), darken(COLOR_BARE)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.top": True, "ytick.right": True, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "grid.alpha": 0.35, "savefig.dpi": 1000, "savefig.bbox": "tight",
})

# =============================================================================
# 1. PARAMETERS
# =============================================================================

Er_bare_paper, Er_hg_paper   = 1.36, 1.37           # Eq.(43) average relative error      [%]
dev_bare_paper, dev_hg_paper = 0.81, 1.39           # mean absolute deviation             [C]

# --- Digitized experimental data (Fig. 4a temperature) ------------------ Excel workbook
t_bare_fig4a = np.array([
    0.0173, 0.0346, 0.0484, 0.0657, 0.0830, 0.0969, 0.1142, 0.1349, 0.1522, 0.1765, 0.1869,
    0.1938, 0.2284, 0.2595, 0.3287, 0.4325, 0.5156, 0.5952, 0.6713, 0.7509, 0.8270, 0.9031,
    0.9931, 1.0761, 1.1834, 1.2872, 1.3875, 1.4913, 1.5640, 1.6436, 1.7474, 1.8166, 1.8789,
    1.9792, 2.0519, 2.0830, 2.1419, 2.2007, 2.2630, 2.3495, 2.4118, 2.4844, 2.5398, 2.6021,
    2.6644, 2.7578, 2.8339, 2.9135, 2.9896])
v_bare_fig4a = np.array([
    27.06, 31.65, 35.68, 39.21, 42.32, 45.07, 47.82, 49.66, 51.71, 53.33, 54.96, 56.23, 57.64,
    59.76, 61.60, 62.30, 62.94, 62.58, 62.58, 62.16, 62.16, 61.95, 61.95, 61.95, 61.95, 61.95,
    61.95, 62.02, 61.95, 61.95, 61.60, 61.38, 61.60, 61.74, 61.45, 61.10, 61.53, 61.53, 61.45,
    61.10, 61.31, 61.60, 61.60, 61.10, 61.31, 61.38, 61.74, 61.88, 61.95])
t_hg_fig4a = np.array([
    0.0242, 0.0554, 0.0692, 0.0796, 0.1211, 0.1661, 0.2353, 0.3114, 0.4083, 0.4913, 0.5606,
    0.6471, 0.7474, 0.8547, 0.9792, 1.0900, 1.2076, 1.3391, 1.4844, 1.5986, 1.7266, 1.8581,
    1.9723, 2.0727, 2.2007, 2.3045, 2.4291, 2.5502, 2.6817, 2.8062, 2.8962, 2.9827])
v_hg_fig4a = np.array([
    24.03, 25.44, 28.62, 30.24, 32.99, 36.81, 39.63, 42.53, 45.42, 47.54, 48.95, 50.16, 51.29,
    52.42, 53.40, 53.76, 54.39, 54.82, 54.75, 54.89, 55.45, 55.38, 55.59, 55.66, 56.16, 56.23,
    56.09, 56.58, 56.72, 56.79, 57.01, 57.22])

# --- Physical constants -------------------------------------------------------
SIGMA     = 5.670374e-8   # Stefan-Boltzmann constant                              [W/m2/K4]
R_GAS     = 8.314         # universal gas constant                                 [J/mol/K]
M_W       = 0.018         # molar mass of water                                    [kg/mol]

# --- Module / electrical ------------------------------------------------------ Lai Table 3
S_PV      = 0.1 * 0.1     # module footprint area                                  [m2]
A_PV      = 0.08 * 0.08   # active cell area                                       [m2]
eta_ref   = 0.13          # reference electrical efficiency                        [-]
beta_PV   = 0.005         # temperature coefficient                                [1/K]
T_ref     = 25.0 + 273.15 # Eq.(21) reference temperature                          [K]
d_PV      = 2.7e-3        # total module thickness                                 [m]
MIN_IRR   = 1.0           # below this the module is dark                          [W/m2]
PV_CAP_MULT = 1.50        # layer heat-capacity scale, fitted at dt=30s on bare PV [-]

# --- PV layers: optical / thermal ------------------ Simple_Tsky.py Table 2, rescaled to d_PV
_layers_raw_mm = dict(glass=3.20, eva1=0.50, cell=0.40, eva2=0.50, tedlar=0.33)
_scale    = d_PV / (sum(_layers_raw_mm.values()) * 1e-3)
d_layer   = {k: v * 1e-3 * _scale for k, v in _layers_raw_mm.items()}           # [m]
rho_layer = dict(glass=2500.0, eva1=1070.0, cell=2330.0, eva2=1070.0, tedlar=1500.0)  # [kg/m3]
Cp_layer  = dict(glass=790.0,  eva1=1675.0, cell=700.0,  eva2=1675.0, tedlar=1090.0)  # [J/kg/K]
lam_layer = dict(glass=1.80,   eva1=0.35,   cell=148.0,  eva2=0.35,   tedlar=0.35)    # [W/m/K]
alpha_glass, alpha_eva, alpha_cell = 0.018, 0.00, 0.90                           # absorptivity  [-]
tau_glass, tau_eva                 = 0.95,  0.98                                 # transmittance [-]
eps_glass, eps_tedlar              = 0.85,  0.90                                 # emissivity    [-]

# --- Hydrogel: Lai Table 3 coupon in Diaz-Marin variables --------------------------------
N_HG      = 80            # finite-volume cells on the material grid               [-]
phi_s0    = 0.10          # solid volume fraction                     Lai Table 3  [-]
phi_l0    = 0.60          # initial liquid volume fraction            Lai Table 3  [-]
phi_g0    = 0.30          # initial gas volume fraction               Lai Table 3  [-]
omega0    = 0.25          # initial CaCl2 mass fraction               Lai Table 3  [-]
RHO_L     = 1000.0        # liquid density                            Lai Table 3  [kg/m3]
M_SALT_MEAS = 16.90e-3    # measured CaCl2 mass, this coupon          Lai Sec. 4.3 [kg]
m_water_meas = M_SALT_MEAS * (1.0 - omega0) / omega0                        # implied by omega0 [kg]
d_HG      = m_water_meas / (phi_l0 * RHO_L * S_PV)                          # solved thickness   [m]
r_p       = 2.0e-8        # reference pore radius                     AUTHOR-SUPPLIED [m]
N_p       = 1.0e10        # reference pore density                    AUTHOR-SUPPLIED [1/m3]
K_WS      = 0.575         # water+sorbent conductivity                Diaz Table S4  [W/m/K]
C_L       = 4200.0        # liquid heat capacity                      Lai Table 3  [J/kg/K]
RHO_S     = 1300.0        # polymer density (PAM)                     Lai Table 3  [kg/m3]
C_S       = 1350.0        # polymer heat capacity                     Lai Table 3  [J/kg/K]
G_SHEAR   = 1000.0        # shear elasticity                          Lai Table 3  [Pa]
POISSON   = 0.45          # Poisson ratio                             Lai Table 3  [-]
K0_PERM   = 6.74e-18      # permeability                              Lai Table 3  [m2]
ETA_L     = 8.50e-4       # liquid dynamic viscosity                  ERRATUM      [Pa.s]
eps_HG    = 0.96          # hydrogel emissivity                       NOT IN PAPER [-]

# --- Sorption / moist-air correlations ---------------------------------------- Diaz SI
RHO_REF   = 1.2041        # reference air density                     (S38)        [kg/m3]
A_RHO = np.array([        # moist-air density polynomial              (S38, Table S2)
    [ 1.0843e+0, -1.0316e-3,  7.7893e-5, -2.2481e-6,  2.6338e-8, -1.0718e-10],
    [-4.3184e-3,  3.2769e-5, -4.3739e-6,  1.4303e-7, -1.7804e-9,  7.5299e-12],
    [ 1.8390e-5, -4.83178e-7, 8.77878e-8,-2.84397e-9, 3.45881e-11,-1.43774e-13],
    [-4.21302e-8, 8.83259e-11,-6.5598e-10,2.12611e-11,-2.55101e-13,1.04543e-15]])
P_AW = np.array([         # CaCl2 water-activity polynomial           (S41)
    0.999682683483451,  0.237228298633870, -28.5406532161117,   466.297325922550,
   -4319.24186333343,  24020.1825115150,   -86613.1007283796,  207987.739728876,
 -332981.706054358,   349470.027801645,   -230260.377476725,   86264.2220356940,
  -14006.7377340363])
P_HDIL = np.array([       # h_sorb(w) fit, deg-6, w in [0, 0.6]       SI Fig. S1   [kJ/mol]
    3633.7192, -7791.80006, 6377.32862, -2550.28936, 569.452305, -69.0086025, 47.5770143])
H_VAP_0   = 44.0          # Fig. S1 intercept = h_vap(25 C)                        [kJ/mol]

# --- Air-side surface transfer -------------------------------------------------------------
C_AIR     = 1006.3        # air specific heat c_pa                    Diaz SI Sec. 7 [J/kg/K]
PR_AIR    = 0.71          # Prandtl number of air                     NOT IN PAPER [-]
nu_air    = 1.5e-5        # kinematic viscosity of air                NOT IN PAPER [m2/s]

# --- Site / boundary conditions ---------------------------------- Lai Table 3, Section 3.3
I_on      = 642.0         # irradiance, simulator on                               [W/m2]
T_a       = 29.0 + 273.15 # ambient air temperature                                [K]
T_g       = 51.6 + 273.15 # test-platform temperature                              [K]
RH        = 0.70          # relative humidity                                      [-]
v_wind    = 0.32          # air velocity, refit jointly with PV_CAP_MULT at dt=30s [m/s]
T_init    = 24.0 + 273.15 # initial temperature                       NOT IN PAPER [K]
tilt_deg  = 0.0           # panel tilt from horizontal                             [deg]
INDOOR    = True          # True: sky-side sink = T_a;  False: Swinbank T_sky      [-]

# --- Numerics ----------------------------------------------------------------- Lai Table 3
dt         = 30.0         # time step, resolves the PV time constant C/h~265s      [s]
t_off      = 3.75 * 3600.0# simulator off                                          [s]
t_end      = 5.0 * 3600.0 # end of simulation                                      [s]
n_iter     = 200          # max fixed-point iterations, bare PV                    [-]
tol        = 1.0e-9       # temperature tolerance, bare PV                         [K]
NEWTON_MAX = 25           # max Newton iterations, gel                             [-]
NEWTON_TOL = 1.0e-10      # scaled Newton tolerance, gel                           [-]
SECANT_MAX = 12           # max secant steps on the Eq.(23) interface root         [-]

# --- Derived -------------------------------------------------------------------------------
C_layer  = {k: PV_CAP_MULT * rho_layer[k] * Cp_layer[k] * d_layer[k] for k in d_layer}  # [J/m2/K]
_Rhalf   = lambda k: d_layer[k] / (2.0 * lam_layer[k])                     # [m2K/W]
R_g_e1, R_e1_pv   = _Rhalf("glass") + _Rhalf("eva1"), _Rhalf("eva1") + _Rhalf("cell")
R_pv_e2, R_e2_ted = _Rhalf("cell") + _Rhalf("eva2"),  _Rhalf("eva2") + _Rhalf("tedlar")
f_HG     = 1.0 - phi_g0                                                     # (S34) prefactor
n0       = phi_l0 * RHO_L / (M_W * f_HG)                                    # (S34)   [mol/m3]
m_salt   = M_W * n0 * f_HG * omega0 / (1.0 - omega0)                        # (S41)   [kg/m3]
dx0      = d_HG / N_HG                                                      # reference cell [m]
tilt_rad     = np.radians(tilt_deg)
F_top_sky    = 0.5 * (1.0 + np.cos(tilt_rad))                               # [-]
F_top_ground = 0.5 * (1.0 - np.cos(tilt_rad))                               # [-]
F_bot_ground = 0.5 * (1.0 + np.cos(tilt_rad))                               # [-]
F_bot_sky    = 0.5 * (1.0 - np.cos(tilt_rad))                               # [-]

# =============================================================================
# 2. EQUATIONS
# =============================================================================

# --- Sorption and moist-air properties ---------------------------------------- Diaz SI
_DAW = np.polyder(P_AW[::-1])

def a_w(w):                     # water activity, w = salt mass fraction              (S41)
    return np.polyval(P_AW[::-1], w)

def da_w_dw(w):                 # d(S41)/dw
    return np.polyval(_DAW, w)

def p_sat(T):                   # Antoine saturation pressure              [Pa]        (S39)
    return 133.322 * 10 ** (8.07131 - 1730.63 / (233.426 + T - 273.15))

def dp_sat(T):                  # d(S39)/dT
    return p_sat(T) * np.log(10.) * 1730.63 / (233.426 + T - 273.15) ** 2

def d_vap(T):                   # vapour diffusivity in air                [m2/s]      (S40)
    return 22.5e-6 * (T / 273.15) ** 1.8

def h_sorb(T, w):               # sorption enthalpy = h_vap(T) + dilution  [J/mol]     SI Fig. S1
    dil = np.maximum(np.polyval(P_HDIL, np.clip(w, 0., 0.6)) - H_VAP_0, 0.)
    return (2.501e6 - 2369. * (T - 273.15)) * M_W + dil * 1e3

def rho_air(T, rh):             # moist-air density, rh in [0, 1]          [kg/m3]     (S38)
    Tc = np.asarray(T, float) - 273.15
    return RHO_REF * sum(Tc ** i * sum(A_RHO[i, j] * (rh * 100.) ** j for j in range(6))
                         for i in range(4))

# --- Hydrogel state on the material grid -------------------------------------------------
def eps_xx(n):                  # swelling strain                                      (S36)
    return (n - n0) * M_W / (3.0 * RHO_L)

def gel_props(n, T):
    """Volume fractions and coefficients at (n, T). Storage and source terms are per REFERENCE
    volume (material grid): cell width wc = dx0*(1+eps)."""
    e = eps_xx(n)
    phi_s = phi_s0 / (1 + e)                                                   # (S33)
    phi_l = np.clip(n * M_W * f_HG / (RHO_L * (1 + e)), 1e-9, 0.999)          # (S34)
    phi_g = np.clip(1 - phi_s - phi_l, 1e-6, 1.0)                             # (S35)
    K = K0_PERM * phi_l / (1 - phi_l) ** 1.5                                  # (10)
    return dict(
        wc    = dx0 * (1 + e),                                                 # current width [m]
        phi_l = phi_l, phi_g = phi_g,
        sg    = (1 + e) * phi_g,                                               # vapour capacity (4)
        b     = N_p * 8 * np.pi * r_p * (1 + e) ** (1 / 3) * d_vap(T),        # N*S*D_v, (3), affine
        D_eff = d_vap(T) * phi_g ** 1.5,                                       # (5)
        D_l   = 2 * (1 + POISSON) * G_SHEAR / (3 * (1 - 2 * POISSON) * ETA_L) * K,  # (9)
        k_eff = K_WS * 2 * (1 - phi_g) / (3 - (1 - phi_g)),                    # (16)
        rc    = (1 + e) * (rho_air(T, 1.) * C_AIR * phi_g + RHO_S * C_S * phi_s
                           + RHO_L * C_L * phi_l),                             # (15)
        w     = m_salt / (m_salt + M_W * n * f_HG))                            # salt fraction (S41)

# --- Surface heat and mass transfer --------------------------------------------------------
def HTC_conv():                                      # convective coefficient  [W/m2/K]  Lai (18)
    return 2.3 + 2.3 * v_wind

def T_sink_sky_side(T_amb):                          # indoor: T_a;  outdoor: Swinbank T_sky
    return T_amb if INDOOR else 0.05527 * T_amb ** 1.5

def HTC_rad2(T_s, eps, F_sky, F_gnd, T_sky):         # two-sink radiative coefficients  Lai (19)
    h_sky = F_sky * eps * SIGMA * (T_s ** 2 + T_sky ** 2) * (T_s + T_sky)
    h_gnd = F_gnd * eps * SIGMA * (T_s ** 2 + T_g ** 2) * (T_s + T_g)
    return h_sky, h_gnd

def h_mass(T):                                       # h_m = h/(rho_a c_pa Le)  [m/s]  Lewis (14)
    Le = (nu_air / d_vap(T)) / PR_AIR                                          # (15),(16)
    return HTC_conv() / (rho_air(T, 0.0) * C_AIR * Le)                          # dry-air rho_a

# --- PV electrical model ---------------------------------------------------------------------
def eta_PV(T):                                       # derated efficiency              Lai (21)
    return eta_ref * (1.0 - beta_PV * (T - T_ref))

def P_PV_output(I, T):                               # electrical power   [W]          Lai (20)
    return np.where(I < MIN_IRR, 0.0, np.maximum(I * A_PV * eta_PV(T), 0.0))

def P_elec_heat_removed(I, T):                       # electrical extraction [W/m2]   Lai (15)
    return 0.0 if I < MIN_IRR else max(I * eta_PV(T), 0.0)

# =============================================================================
# 3. SOLVING
# =============================================================================

C_inf, h_m = RH * p_sat(T_a) / (R_GAS * T_a), h_mass(T_a)                      # (S43), Lewis (14)
T_sky = T_sink_sky_side(T_a)

def tri(lo, di, up, rhs):
    ab = np.zeros((3, di.size))
    ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], di, lo[1:]
    return solve_banded((1, 1), ab, rhs)

def sys_PV5(T_old, T_it, I, q_if, bare):
    """Five-node PV chain, one implicit step. bare=False: Tedlar feeds q_if into the gel."""
    G01, G12, G23, G34 = 1 / R_g_e1, 1 / R_e1_pv, 1 / R_pv_e2, 1 / R_e2_ted
    Cn = np.array([C_layer[k] for k in ("glass", "eva1", "cell", "eva2", "tedlar")]) / dt
    lo = np.array([0.0, -G01, -G12, -G23, -G34])
    up = np.array([-G01, -G12, -G23, -G34, 0.0])
    di = Cn + np.array([G01, G01 + G12, G12 + G23, G23 + G34, G34])
    rhs = Cn * T_old
    h_rs, h_rg = HTC_rad2(T_it[0], eps_glass, F_top_sky, F_top_ground, T_sky)
    di[0]  += HTC_conv() + h_rs + h_rg                                          # front, Lai (18)-(19)
    rhs[0] += alpha_glass * I + HTC_conv() * T_a + h_rs * T_sky + h_rg * T_g
    rhs[1] += alpha_eva * tau_glass * I
    rhs[2] += alpha_cell * tau_eva * tau_glass * I - P_elec_heat_removed(I, T_it[2])
    if bare:
        h_rs, h_rg = HTC_rad2(T_it[-1], eps_tedlar, F_bot_sky, F_bot_ground, T_sky)
        di[-1]  += HTC_conv() + h_rs + h_rg
        rhs[-1] += HTC_conv() * T_a + h_rs * T_sky + h_rg * T_g
    else:
        rhs[-1] -= q_if                                                         # Lai (24) analog
    return tri(lo, di, up, rhs)

def face_air(T_N, k_N, w_N):
    """HG/air face: half cell in series with convection + two-sink radiation (Lai 18-19).
    Returns h_eff and the equivalent sink temperature, flux = h_eff*(T_N - T_eq)."""
    h_rs, h_rg = HTC_rad2(T_N, eps_HG, F_bot_sky, F_bot_ground, T_sky)
    h_tot = HTC_conv() + h_rs + h_rg
    T_eq = (HTC_conv() * T_a + h_rs * T_sky + h_rg * T_g) / h_tot
    a = 2 * k_N / w_N
    return a * h_tot / (a + h_tot), T_eq

def gel_step(T, C, n, q_if):
    """One implicit step, Newton on interleaved (C, n, T), band (4, 4). Rows per reference width:
      A : (4) + (8)         total water; the stiff source cancels; J leaves the last cell
      B : (4) / (N*S*D_v)   vapour; pore-route Robin at x = H
      C : (14) with -h_sorb*R, R from (8), so surface-evaporation latent is charged once
    Returns (T, C, n) or None."""
    N = N_HG
    To, Co, no = T.copy(), C.copy(), n.copy()
    sgo = gel_props(no, To)['sg']
    ser = lambda a, w: 1.0 / (w[:-1] / (2 * a[:-1]) + w[1:] / (2 * a[1:]))   # face conductance
    iC = 3 * np.arange(N); iN, iT = iC + 1, iC + 2
    J = np.zeros((9, 3 * N)); rhs = np.zeros(3 * N)
    put = lambda r_, c_, v: np.add.at(J, (4 + r_ - c_, c_), v)
    L = [N - 1]
    for _ in range(NEWTON_MAX):
        p = gel_props(n, T); wc, sg, b, w = p['wc'], p['sg'], p['b'], p['w']
        E = p_sat(T) / (R_GAS * T); dE = (dp_sat(T) * T - p_sat(T)) / (R_GAS * T * T)
        aw = a_w(w); daw = da_w_dw(w) * (-w * w * M_W * f_HG / m_salt)
        hs = h_sorb(T, w)
        gH = 1.0 / (wc[-1] / (2 * p['D_eff'][-1]) + 1.0 / (p['phi_g'][-1] * h_m))   # pore route
        GD = np.r_[0., ser(p['D_eff'], wc), gH] / dx0                  # x=0 sealed, x=H Robin
        GL = np.r_[0., ser(p['D_l'],  wc), 0.] / dx0                   # liquid: J handled below
        GK = np.r_[0., ser(p['k_eff'], wc), 0.] / dx0                  # faces set by BCs
        Jc = p['phi_l'][-1] * h_m                                       # free-liquid route
        J0, Jn, JT = Jc * (aw[-1] * E[-1] - C_inf), Jc * E[-1] * daw[-1], Jc * aw[-1] * dE[-1]
        J[:] = 0.

        # row A -- (4) + (8)
        put(iC, iC, sg / dt + GD[:-1] + GD[1:]); put(iC, iN, f_HG / dt + GL[:-1] + GL[1:])
        put(iC[1:], iC[:-1], -GD[1:-1]); put(iC[:-1], iC[1:], -GD[1:-1])
        put(iC[1:], iN[:-1], -GL[1:-1]); put(iC[:-1], iN[1:], -GL[1:-1])
        rhs[iC] = sgo * Co / dt + f_HG * no / dt
        rhs[iC[-1]] += GD[-1] * C_inf - (J0 - Jn * n[-1] - JT * T[-1]) / dx0
        put(iC[L], iN[L], Jn / dx0); put(iC[L], iT[L], JT / dx0)

        # row B -- (4) / b
        put(iN, iC, sg / (b * dt) + (GD[:-1] + GD[1:]) / b + 1.)
        put(iN[1:], iC[:-1], -GD[1:-1] / b[1:]); put(iN[:-1], iC[1:], -GD[1:-1] / b[:-1])
        put(iN, iN, -E * daw); put(iN, iT, -aw * dE)
        rhs[iN] = sgo * Co / (b * dt) + E * aw - E * daw * n - aw * dE * T
        rhs[iN[-1]] += GD[-1] * C_inf / b[-1]

        # row C -- (14): conduction, latent -h_sorb*R = h_sorb*[(1-phi_g0) dn/dt - d/dx(D_l dn/dx)]
        put(iT, iT, p['rc'] / dt + GK[:-1] + GK[1:])
        put(iT[1:], iT[:-1], -GK[1:-1]); put(iT[:-1], iT[1:], -GK[1:-1])
        put(iT, iN, -hs * (f_HG / dt + GL[:-1] + GL[1:]))
        put(iT[1:], iN[:-1], hs[1:] * GL[1:-1]); put(iT[:-1], iN[1:], hs[:-1] * GL[1:-1])
        rhs[iT] = p['rc'] * To / dt - hs * f_HG * no / dt
        rhs[iT[0]] += q_if / dx0                                        # x = 0: PV -> HG flux
        h_eff, T_eq = face_air(T[-1], p['k_eff'][-1], wc[-1])           # x = H: conv + rad;
        put(iT[L], iT[L], h_eff / dx0); rhs[iT[-1]] += h_eff * T_eq / dx0   # h_sorb*J is in row C
                                                                         # via dn/dt of the last cell
        sol = solve_banded((4, 4), J, rhs)
        if not np.all(np.isfinite(sol)):
            return None
        C_new, n_try, T_try = sol[iC], sol[iN], sol[iT]
        lam = min(1., 0.15 * n0 / max(np.max(np.abs(n_try - n)), 1e-30),
                      15. / max(np.max(np.abs(T_try - T)), 1e-30))
        n_new = np.clip(n + lam * (n_try - n), 1e-5 * n0, 1.05 * n0)
        T_new = T + lam * (T_try - T)
        if T_new.min() < 200. or T_new.max() > 600.:
            return None
        err = max(np.max(np.abs(T_new - T)) / 50., np.max(np.abs(n_new - n)) / n0)
        T, C, n = T_new, C_new, n_new
        if err < NEWTON_TOL and lam == 1.:
            return T, C, n
    return None

def gel_fluxes(T, C, n):
    """MCP = bulk desorption latent (volume integral of h_sorb*R), ECP = h_sorb*J at the face,
    q_face = conv + rad loss at the face, H = current thickness.  Pore vapour leaving through the
    face carries no latent of its own (already in MCP)."""
    p = gel_props(n, T); hs = h_sorb(T, p['w'])
    E = p_sat(T) / (R_GAS * T); aw = a_w(p['w'])
    mcp = np.sum(hs * p['b'] * (aw * E - C)) * dx0
    ecp = hs[-1] * p['phi_l'][-1] * h_m * (aw[-1] * E[-1] - C_inf)
    h_eff, T_eq = face_air(T[-1], p['k_eff'][-1], p['wc'][-1])
    return mcp, ecp, h_eff * (T[-1] - T_eq), p['wc'].sum()

def run(with_HG):
    """PV chain is linear in q_if; the gel is not, so the Eq.(23) interface root is found by
    secant on the face-temperature mismatch."""
    t = np.arange(0.0, t_end + dt, dt)
    T_PV = np.full(5, T_init)
    out = {k: np.zeros(t.size) for k in ("T_top", "MCP", "ECP", "H", "res")}
    out["T_top"][0], out["H"][0] = T_init, d_HG
    if with_HG:
        T_HG, C, n = np.full(N_HG, T_init), np.full(N_HG, C_inf), np.full(N_HG, n0)
    q_if = 0.0
    for s in range(1, t.size):
        I = I_on if t[s] <= t_off else 0.0
        T_PV_o = T_PV.copy()
        if not with_HG:
            for _ in range(n_iter):
                T_new = sys_PV5(T_PV_o, T_PV, I, 0.0, True)
                r, T_PV = np.abs(T_new - T_PV).max(), T_new
                if r < tol:
                    break
            out["T_top"][s] = T_PV[0]
            continue
        T_HG_o, C_o, n_o = T_HG.copy(), C.copy(), n.copy()

        def mismatch(q):
            Tp, g = sys_PV5(T_PV_o, T_PV, I, q, False), gel_step(T_HG_o, C_o, n_o, q)
            if g is None:
                return None, None, None
            p0 = gel_props(g[2][:1], g[0][:1])
            return Tp[-1] - (g[0][0] + q * p0['wc'][0] / (2 * p0['k_eff'][0])), Tp, g

        q0, q1 = q_if, q_if + 50.0
        f0, _, _ = mismatch(q0)
        f1, Tp1, g1 = mismatch(q1)
        for _ in range(SECANT_MAX):
            if f1 is None or abs(f1 - f0) < 1e-14:
                break
            q2 = q1 - f1 * (q1 - q0) / (f1 - f0)                                 # Eq.(23) root
            f2, Tp2, g2 = mismatch(q2)
            if f2 is None:
                break
            q0, f0, q1, f1, Tp1, g1 = q1, f1, q2, f2, Tp2, g2
            if abs(f2) < 1e-9:
                break
        q_if, T_PV = q1, Tp1
        T_HG, C, n = g1
        mcp, ecp, q_face, H = gel_fluxes(T_HG, C, n)
        stored = np.sum(gel_props(n, T_HG)['rc'] * (T_HG - T_HG_o)) * dx0 / dt
        out["T_top"][s], out["MCP"][s], out["ECP"][s], out["H"][s] = T_PV[0], mcp, ecp, H
        out["res"][s] = q_if - q_face - mcp - ecp - stored                      # energy closure
    out["t"] = t
    return out

def metrics(t_sim, v_sim, t_exp, v_exp):
    v_pred = np.interp(t_exp, t_sim, v_sim)
    err = v_pred - v_exp
    keep = np.abs(v_exp) > 0.05 * np.abs(v_exp).max()
    mae, rmse = np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2))
    mape  = np.mean(np.abs(err[keep] / v_exp[keep])) * 100.0
    rmspe = np.sqrt(np.mean((err[keep] / v_exp[keep]) ** 2)) * 100.0
    r2 = 1.0 - np.sum(err ** 2) / np.sum((v_exp - np.mean(v_exp)) ** 2)
    return mae, mape, rmse, rmspe, r2

bare, hg = run(False), run(True)
t = bare["t"] / 3600.0
T_bare_sim, T_hg_sim = bare["T_top"] - 273.15, hg["T_top"] - 273.15
i3 = np.argmin(np.abs(t - 3.0))

mae_b, mape_b, rmse_b, rmspe_b, r2_b = metrics(t, T_bare_sim, t_bare_fig4a, v_bare_fig4a)
mae_h, mape_h, rmse_h, rmspe_h, r2_h = metrics(t, T_hg_sim,   t_hg_fig4a,   v_hg_fig4a)

print(f"h_m (Lewis)             {h_m*1e3:6.3f} mm/s")
print(f"bare PV at 3 h          {T_bare_sim[i3]:6.2f} C")
print(f"PV/HG at 3 h            {T_hg_sim[i3]:6.2f} C")
print(f"MCP / ECP at 3 h        {hg['MCP'][i3]:6.1f} / {hg['ECP'][i3]:6.1f} W/m2")
print(f"H at 3 h                {hg['H'][i3]*1e3:6.3f} mm  (H0 = {d_HG*1e3:.3f} mm)")
print(f"max |energy residual|   {np.abs(hg['res']).max():.2e} W/m2")
print(f"bare: MAE={mae_b:.2f} C  MAPE={mape_b:.2f}%  RMSE={rmse_b:.2f} C  R2={r2_b:.3f}")
print(f"w/hg: MAE={mae_h:.2f} C  MAPE={mape_h:.2f}%  RMSE={rmse_h:.2f} C  R2={r2_h:.3f}")

# =============================================================================
# 4. PLOTTING
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5.8))
ax.plot(t, T_bare_sim, "-", color=COLOR_BARE, lw=2.2, label="W/O HG, model", zorder=3)
ax.plot(t, T_hg_sim, "-", color=COLOR_HG, lw=2.2, label="W/ HG, Diaz + surface evap.", zorder=3)
ax.scatter(t_bare_fig4a, v_bare_fig4a, s=30, marker="o", fc="none",
           ec=MARKER_BARE, lw=1.1, label="W/O HG, exp.", zorder=3)
ax.scatter(t_hg_fig4a, v_hg_fig4a, s=30, marker="^", fc="none",
           ec=MARKER_HG, lw=1.1, label="W/ HG, exp.", zorder=3)
ax.set_xlabel("Time [h]")
ax.set_ylabel(r"Surface temperature [$^\circ$C]")
ax.set_xlim(0, 3.0)
ax.grid(True, which="major")
ax.minorticks_on()

metric_groups = [
    (COLOR_BARE, "W/O HG, model vs exp.:", [
        rf"MAE={mae_b:.2f}$^\circ$C  MAPE={mape_b:.2f}%",
        rf"RMSE={rmse_b:.2f}$^\circ$C  RMSPE={rmspe_b:.2f}%  $R^2$={r2_b:.3f}",
        rf"$E_r$ = {Er_bare_paper:.2f}% (paper)"]),
    (COLOR_HG, "W/ HG, model vs exp.:", [
        rf"MAE={mae_h:.2f}$^\circ$C  MAPE={mape_h:.2f}%",
        rf"RMSE={rmse_h:.2f}$^\circ$C  RMSPE={rmspe_h:.2f}%  $R^2$={r2_h:.3f}",
        rf"$E_r$ = {Er_hg_paper:.2f}% (paper)"]),
]
y_txt = 44.0
for color, header, lines in metric_groups:
    ax.text(0.62, y_txt, header, ha="left", va="top", fontsize=7.8, color=color, fontweight="bold")
    y_txt -= 1.95
    for text in lines:
        ax.text(0.62, y_txt, text, ha="left", va="top", fontsize=7.8, color=color, style="italic")
        y_txt -= 1.95
    y_txt -= 0.95

ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.3",
          fancybox=False, borderpad=0.7, handlelength=2.4, fontsize=9)
ax.set_title("(a) Diaz-Marin hydrogel, dt-converged with corrected PV and gel thermal mass",
             loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.subplots_adjust(bottom=0.26)
fig.text(0.5, 0.085,
         rf"$dt={dt:.0f}$ s (converged)   |   PV heat capacity $\times${PV_CAP_MULT:.2f}   |   "
         rf"$v_{{wind}}={v_wind:.2f}$ m/s   |   $h_{{conv}}={HTC_conv():.2f}$ W/m$^2$K",
         ha="center", va="top", fontsize=7.5, color="0.25")
fig.text(0.5, 0.055,
         rf"$H_0={d_HG*1e3:.2f}$ mm (from {M_SALT_MEAS*1e3:.2f} g CaCl$_2$, Sec. 4.3, at Table 3's "
         rf"$\omega_0={omega0:.2f}$, $\phi_{{l0}}={phi_l0:.2f}$; nominal 5 mm holds too little water)   |   "
         rf"$h_m={h_m*1e3:.2f}$ mm/s",
         ha="center", va="top", fontsize=7.5, color="0.25")
fig.text(0.5, 0.025,
         rf"Face evaporation: pore route $\phi_g h_m (C_H-C_\infty)$ + liquid route "
         rf"$\phi_l h_m (a_w C_{{sat}}-C_\infty)$   |   $H$(3 h) = {hg['H'][i3]*1e3:.2f} mm",
         ha="center", va="top", fontsize=7.5, color="0.25")

fig.savefig("fig4a_diaz_corrected.png", dpi=1000)
fig.savefig("fig4a_diaz_corrected.pdf")

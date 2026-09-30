"""
Five-node PV + Diaz-Marin hydrogel model, refactored from diaz_hg_fig4a_5node_corrected.py so that the
same equations can be driven by time-varying forcing (Lai et al., Appl. Energy 426 (2026) 128583,
Fig. 5) and so that individual physics corrections can be switched on and off.

With the default arguments the model reproduces diaz_hg_fig4a_5node_corrected.py to round-off
(see check_regression() below).  Switches, all defaulting to the ORIGINAL behaviour:

  lewis_exp       h_m = h / (rho_a c_pa Le^lewis_exp).  Original 1.0; Chilton-Colburn gives 2/3.
  elec_area_frac  electrical extraction per m2 of footprint = eta*G*elec_area_frac.  Original 1.0;
                  the 1-D balance is per m2 of footprint S_PV but eta applies to the cell area A_PV,
                  so the consistent value is A_PV/S_PV = 0.64.
  iface_Rted      include the Tedlar half-layer resistance between the Tedlar node and the gel face
                  in the Eq.(23) temperature-continuity mismatch.  Original False.
  relin_pv        re-linearise PV radiation and eta(T) at the new time level in the PV/HG run (the
                  bare run already does).  Original False (lagged one step).
  dilution        add the SI Fig. S1 dilution enthalpy to h_vap(T).  Original True.
  back_sink       'ground': rear face (Tedlar or gel) radiates to T_g (original);
                  'ambient': to T_a, as written in Lai Eq.(23)/(25).
  htc_a, htc_b    HTC_conv = htc_a + htc_b*v.  IJHMT Eq.(18): 2.3 + 2.3v;  Appl. Energy Eq.(24): 2.8 + 3v.
  Dl_mult         diagnostic multiplier on the poroelastic liquid diffusivity D_l (Eq. 9).  Original 1.0.
  vent_pores      DIAGNOSTIC, emulates Lai Eq.(13)/(17): pore vapour pinned at the ambient C_a, i.e. vapour
                  made inside the gel leaves with no pore-diffusion or boundary-layer resistance.
                  Original False (Diaz two-concentration model with a Robin face).
  ecp_area        DIAGNOSTIC, face area fraction of the liquid route J.  None = phi_l (original);
                  1.0 emulates Lai Eq.(26), where ECP acts on the whole face.
"""

import numpy as np
from scipy.linalg import solve_banded

SIGMA, R_GAS, M_W = 5.670374e-8, 8.314, 0.018

# Diaz SI correlations (unchanged from the original script)
RHO_REF = 1.2041
A_RHO = np.array([
    [ 1.0843e+0, -1.0316e-3,  7.7893e-5, -2.2481e-6,  2.6338e-8, -1.0718e-10],
    [-4.3184e-3,  3.2769e-5, -4.3739e-6,  1.4303e-7, -1.7804e-9,  7.5299e-12],
    [ 1.8390e-5, -4.83178e-7, 8.77878e-8,-2.84397e-9, 3.45881e-11,-1.43774e-13],
    [-4.21302e-8, 8.83259e-11,-6.5598e-10,2.12611e-11,-2.55101e-13,1.04543e-15]])
P_AW = np.array([
    0.999682683483451,  0.237228298633870, -28.5406532161117,   466.297325922550,
   -4319.24186333343,  24020.1825115150,   -86613.1007283796,  207987.739728876,
 -332981.706054358,   349470.027801645,   -230260.377476725,   86264.2220356940,
  -14006.7377340363])
P_HDIL = np.array([3633.7192, -7791.80006, 6377.32862, -2550.28936, 569.452305, -69.0086025, 47.5770143])
H_VAP_0 = 44.0
_AW, _DAW = P_AW[::-1], np.polyder(P_AW[::-1])


def a_w(w):
    return np.polyval(_AW, w)


def da_w_dw(w):
    return np.polyval(_DAW, w)


def p_sat(T):
    return 133.322 * 10 ** (8.07131 - 1730.63 / (233.426 + T - 273.15))


def dp_sat(T):
    return p_sat(T) * np.log(10.) * 1730.63 / (233.426 + T - 273.15) ** 2


def d_vap(T):
    return 22.5e-6 * (T / 273.15) ** 1.8


def rho_air(T, rh):
    Tc = np.asarray(T, float) - 273.15
    return RHO_REF * sum(Tc ** i * sum(A_RHO[i, j] * (rh * 100.) ** j for j in range(6)) for i in range(4))


def tri(lo, di, up, rhs):
    ab = np.zeros((3, di.size))
    ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], di, lo[1:]
    return solve_banded((1, 1), ab, rhs)


def const(v):
    return lambda t: v


class PVHG:
    def __init__(self, *,
                 # forcing: callables of t [s]
                 G=lambda t: 642.0 if t <= 3.75 * 3600.0 else 0.0, Ta=const(29.0 + 273.15), Tg=const(51.6 + 273.15), RH=0.70,
                 T_init_bare=24.0 + 273.15, T_init_hg=24.0 + 273.15, t_end=5.0 * 3600.0, dt=30.0,
                 # PV
                 S_PV=0.01, A_PV=0.0064, eta_ref=0.13, beta_PV=0.005, T_ref=298.15, d_PV=2.7e-3,
                 PV_CAP_MULT=1.50, v_wind=0.32, htc_a=2.3, htc_b=2.3,
                 eps_glass=0.85, eps_tedlar=0.90, eps_HG=0.96, MIN_IRR=1.0,
                 # hydrogel
                 N_HG=80, d_HG=None, phi_s0=0.10, phi_l0=0.60, phi_g0=0.30, omega0=0.25,
                 M_SALT_MEAS=16.90e-3, r_p=2.0e-8, N_p=1.0e10, Dl_mult=1.0,
                 # physics switches (defaults = original script)
                 lewis_exp=1.0, elec_area_frac=1.0, iface_Rted=False, relin_pv=False, dilution=True,
                 back_sink="ground", vent_pores=False, ecp_area=None,
                 # numerics
                 n_iter=200, tol=1e-9, NEWTON_MAX=25, NEWTON_TOL=1e-10, SECANT_MAX=12):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.RHO_L, self.K_WS, self.C_L, self.RHO_S, self.C_S = 1000.0, 0.575, 4200.0, 1300.0, 1350.0
        self.G_SHEAR, self.POISSON, self.K0_PERM, self.ETA_L = 1000.0, 0.45, 6.74e-18, 8.5e-4
        self.C_AIR, self.PR_AIR, self.nu_air = 1006.3, 0.71, 1.5e-5
        # original: d_HG solved from the Sec. 4.3 salt mass; pass d_HG to fix it instead
        if d_HG is None:
            d_HG = M_SALT_MEAS * (1 - omega0) / omega0 / (phi_l0 * self.RHO_L * S_PV)
        self.d_HG = d_HG
        # PV layers (Simple_Tsky.py Table 2 rescaled to d_PV)
        raw = dict(glass=3.20, eva1=0.50, cell=0.40, eva2=0.50, tedlar=0.33)
        sc = d_PV / (sum(raw.values()) * 1e-3)
        self.d_layer = {k: v * 1e-3 * sc for k, v in raw.items()}
        rho = dict(glass=2500.0, eva1=1070.0, cell=2330.0, eva2=1070.0, tedlar=1500.0)
        cp = dict(glass=790.0, eva1=1675.0, cell=700.0, eva2=1675.0, tedlar=1090.0)
        self.lam = dict(glass=1.80, eva1=0.35, cell=148.0, eva2=0.35, tedlar=0.35)
        self.alpha_glass, self.alpha_eva, self.alpha_cell = 0.018, 0.00, 0.90
        self.tau_glass, self.tau_eva = 0.95, 0.98
        self.C_layer = np.array([PV_CAP_MULT * rho[k] * cp[k] * self.d_layer[k]
                                 for k in ("glass", "eva1", "cell", "eva2", "tedlar")])
        Rh = lambda k: self.d_layer[k] / (2.0 * self.lam[k])
        self.Rhalf_ted = Rh("tedlar")
        self.Gc = 1.0 / np.array([Rh("glass") + Rh("eva1"), Rh("eva1") + Rh("cell"),
                                  Rh("cell") + Rh("eva2"), Rh("eva2") + Rh("tedlar")])
        # gel
        self.f_HG = 1.0 - phi_g0
        self.n0 = phi_l0 * self.RHO_L / (M_W * self.f_HG)
        self.m_salt = M_W * self.n0 * self.f_HG * omega0 / (1.0 - omega0)
        self.dx0 = d_HG / N_HG
        self.HTC = htc_a + htc_b * v_wind

    # ---------------------------------------------------------------- properties
    def h_sorb(self, T, w):
        dil = np.maximum(np.polyval(P_HDIL, np.clip(w, 0., 0.6)) - H_VAP_0, 0.) if self.dilution else 0.
        return (2.501e6 - 2369. * (T - 273.15)) * M_W + dil * 1e3

    def eps_xx(self, n):
        return (n - self.n0) * M_W / (3.0 * self.RHO_L)

    def gel_props(self, n, T):
        e = self.eps_xx(n)
        phi_s = self.phi_s0 / (1 + e)
        phi_l = np.clip(n * M_W * self.f_HG / (self.RHO_L * (1 + e)), 1e-9, 0.999)
        phi_g = np.clip(1 - phi_s - phi_l, 1e-6, 1.0)
        K = self.K0_PERM * phi_l / (1 - phi_l) ** 1.5
        return dict(
            wc=self.dx0 * (1 + e), phi_l=phi_l, phi_g=phi_g, sg=(1 + e) * phi_g,
            b=self.N_p * 8 * np.pi * self.r_p * (1 + e) ** (1 / 3) * d_vap(T),
            D_eff=d_vap(T) * phi_g ** 1.5,
            D_l=self.Dl_mult * 2 * (1 + self.POISSON) * self.G_SHEAR / (3 * (1 - 2 * self.POISSON) * self.ETA_L) * K,
            k_eff=self.K_WS * 2 * (1 - phi_g) / (3 - (1 - phi_g)),
            rc=(1 + e) * (rho_air(T, 1.) * self.C_AIR * phi_g + self.RHO_S * self.C_S * phi_s
                          + self.RHO_L * self.C_L * phi_l),
            w=self.m_salt / (self.m_salt + M_W * n * self.f_HG))

    # ---------------------------------------------------------------- surfaces
    def h_mass(self, Ta):
        Le = (self.nu_air / d_vap(Ta)) / self.PR_AIR
        return self.HTC / (rho_air(Ta, 0.0) * self.C_AIR * Le ** self.lewis_exp)

    def h_rad(self, Ts, eps, Tsink):
        return eps * SIGMA * (Ts ** 2 + Tsink ** 2) * (Ts + Tsink)

    def eta(self, T):
        return self.eta_ref * (1.0 - self.beta_PV * (T - self.T_ref))

    def power(self, G, T_cell):                                   # Lai IJHMT Eq.(20) [W]
        return np.where(G < self.MIN_IRR, 0.0, np.maximum(G * self.A_PV * self.eta(T_cell), 0.0))

    def env(self, t):
        Ta, Tg = self.Ta(t), self.Tg(t)
        return dict(G=self.G(t), Ta=Ta, Tsky=Ta, Tback=Tg if self.back_sink == "ground" else Ta,
                    C_inf=self.RH * p_sat(Ta) / (R_GAS * Ta), h_m=self.h_mass(Ta))

    # ---------------------------------------------------------------- PV chain
    def sys_PV5(self, T_old, T_it, ev, q_if, bare):
        G01, G12, G23, G34 = self.Gc
        Cn = self.C_layer / self.dt
        lo = np.array([0.0, -G01, -G12, -G23, -G34])
        up = np.array([-G01, -G12, -G23, -G34, 0.0])
        di = Cn + np.array([G01, G01 + G12, G12 + G23, G23 + G34, G34])
        rhs = Cn * T_old
        I, h = ev["G"], self.HTC
        hr = self.h_rad(T_it[0], self.eps_glass, ev["Tsky"])       # tilt 0: front sees sky only
        di[0] += h + hr
        rhs[0] += self.alpha_glass * I + h * ev["Ta"] + hr * ev["Tsky"]
        rhs[1] += self.alpha_eva * self.tau_glass * I
        q_el = 0.0 if I < self.MIN_IRR else max(I * self.eta(T_it[2]), 0.0) * self.elec_area_frac
        rhs[2] += self.alpha_cell * self.tau_eva * self.tau_glass * I - q_el
        if bare:
            hr = self.h_rad(T_it[-1], self.eps_tedlar, ev["Tback"])  # rear sees ground only
            di[-1] += h + hr
            rhs[-1] += h * ev["Ta"] + hr * ev["Tback"]
        else:
            rhs[-1] -= q_if
        return tri(lo, di, up, rhs)

    # ---------------------------------------------------------------- gel
    def face_air(self, T_N, k_N, w_N, ev):
        hr = self.h_rad(T_N, self.eps_HG, ev["Tback"])
        h_tot = self.HTC + hr
        T_eq = (self.HTC * ev["Ta"] + hr * ev["Tback"]) / h_tot
        a = 2 * k_N / w_N
        return a * h_tot / (a + h_tot), T_eq

    def gel_step(self, T, C, n, q_if, ev):
        N, dt, dx0, f_HG = self.N_HG, self.dt, self.dx0, self.f_HG
        C_inf, h_m = ev["C_inf"], ev["h_m"]
        To, Co, no = T.copy(), C.copy(), n.copy()
        sgo = self.gel_props(no, To)['sg']
        ser = lambda a, w: 1.0 / (w[:-1] / (2 * a[:-1]) + w[1:] / (2 * a[1:]))
        iC = 3 * np.arange(N); iN, iT = iC + 1, iC + 2
        J = np.zeros((9, 3 * N)); rhs = np.zeros(3 * N)
        put = lambda r_, c_, v: np.add.at(J, (4 + r_ - c_, c_), v)
        L = [N - 1]
        for _ in range(self.NEWTON_MAX):
            p = self.gel_props(n, T); wc, sg, b, w = p['wc'], p['sg'], p['b'], p['w']
            E = p_sat(T) / (R_GAS * T); dE = (dp_sat(T) * T - p_sat(T)) / (R_GAS * T * T)
            aw = a_w(w); daw = da_w_dw(w) * (-w * w * M_W * f_HG / self.m_salt)
            hs = self.h_sorb(T, w)
            gH = 1.0 / (wc[-1] / (2 * p['D_eff'][-1]) + 1.0 / (p['phi_g'][-1] * h_m))
            GD = np.r_[0., ser(p['D_eff'], wc), gH] / dx0
            GL = np.r_[0., ser(p['D_l'], wc), 0.] / dx0
            GK = np.r_[0., ser(p['k_eff'], wc), 0.] / dx0
            Jc = (p['phi_l'][-1] if self.ecp_area is None else self.ecp_area) * h_m
            J0, Jn, JT = Jc * (aw[-1] * E[-1] - C_inf), Jc * E[-1] * daw[-1], Jc * aw[-1] * dE[-1]
            J[:] = 0.
            if self.vent_pores:                 # Lai Eq.(13): C = C_a, liquid Eq.(8) alone for n
                put(iC, iC, np.ones(N)); rhs[iC] = C_inf
                put(iN, iN, f_HG / dt + GL[:-1] + GL[1:] + b * E * daw); put(iN, iT, b * aw * dE)
                put(iN[1:], iN[:-1], -GL[1:-1]); put(iN[:-1], iN[1:], -GL[1:-1])
                rhs[iN] = f_HG * no / dt - b * (aw * E - C_inf) + b * E * daw * n + b * aw * dE * T
                rhs[iN[-1]] -= (J0 - Jn * n[-1] - JT * T[-1]) / dx0
                put(iN[L], iN[L], Jn / dx0); put(iN[L], iT[L], JT / dx0)
            else:
                put(iC, iC, sg / dt + GD[:-1] + GD[1:]); put(iC, iN, f_HG / dt + GL[:-1] + GL[1:])
                put(iC[1:], iC[:-1], -GD[1:-1]); put(iC[:-1], iC[1:], -GD[1:-1])
                put(iC[1:], iN[:-1], -GL[1:-1]); put(iC[:-1], iN[1:], -GL[1:-1])
                rhs[iC] = sgo * Co / dt + f_HG * no / dt
                rhs[iC[-1]] += GD[-1] * C_inf - (J0 - Jn * n[-1] - JT * T[-1]) / dx0
                put(iC[L], iN[L], Jn / dx0); put(iC[L], iT[L], JT / dx0)
                put(iN, iC, sg / (b * dt) + (GD[:-1] + GD[1:]) / b + 1.)
                put(iN[1:], iC[:-1], -GD[1:-1] / b[1:]); put(iN[:-1], iC[1:], -GD[1:-1] / b[:-1])
                put(iN, iN, -E * daw); put(iN, iT, -aw * dE)
                rhs[iN] = sgo * Co / (b * dt) + E * aw - E * daw * n - aw * dE * T
                rhs[iN[-1]] += GD[-1] * C_inf / b[-1]
            put(iT, iT, p['rc'] / dt + GK[:-1] + GK[1:])
            put(iT[1:], iT[:-1], -GK[1:-1]); put(iT[:-1], iT[1:], -GK[1:-1])
            put(iT, iN, -hs * (f_HG / dt + GL[:-1] + GL[1:]))
            put(iT[1:], iN[:-1], hs[1:] * GL[1:-1]); put(iT[:-1], iN[1:], hs[:-1] * GL[1:-1])
            rhs[iT] = p['rc'] * To / dt - hs * f_HG * no / dt
            rhs[iT[0]] += q_if / dx0
            h_eff, T_eq = self.face_air(T[-1], p['k_eff'][-1], wc[-1], ev)
            put(iT[L], iT[L], h_eff / dx0); rhs[iT[-1]] += h_eff * T_eq / dx0
            sol = solve_banded((4, 4), J, rhs)
            if not np.all(np.isfinite(sol)):
                return None
            C_new, n_try, T_try = sol[iC], sol[iN], sol[iT]
            lam = min(1., 0.15 * self.n0 / max(np.max(np.abs(n_try - n)), 1e-30),
                      15. / max(np.max(np.abs(T_try - T)), 1e-30))
            n_new = np.clip(n + lam * (n_try - n), 1e-5 * self.n0, 1.05 * self.n0)
            T_new = T + lam * (T_try - T)
            if T_new.min() < 200. or T_new.max() > 600.:
                return None
            err = max(np.max(np.abs(T_new - T)) / 50., np.max(np.abs(n_new - n)) / self.n0)
            T, C, n = T_new, C_new, n_new
            if err < self.NEWTON_TOL and lam == 1.:
                return T, C, n
        return None

    def gel_fluxes(self, T, C, n, ev):
        p = self.gel_props(n, T); hs = self.h_sorb(T, p['w'])
        E = p_sat(T) / (R_GAS * T); aw = a_w(p['w'])
        mcp = np.sum(hs * p['b'] * (aw * E - C)) * self.dx0
        area = p['phi_l'][-1] if self.ecp_area is None else self.ecp_area
        ecp = hs[-1] * area * ev["h_m"] * (aw[-1] * E[-1] - ev["C_inf"])
        h_eff, T_eq = self.face_air(T[-1], p['k_eff'][-1], p['wc'][-1], ev)
        water = np.sum(M_W * n * self.f_HG) * self.dx0                    # kg/m2 (liquid only)
        return mcp, ecp, h_eff * (T[-1] - T_eq), p['wc'].sum(), water, p['w'].mean(), T[-1]

    # ---------------------------------------------------------------- driver
    def run(self, with_HG, t0=0.0):
        t = np.arange(t0, self.t_end + self.dt / 2, self.dt)
        keys = ("T_top", "T_cell", "P", "MCP", "ECP", "H", "res", "water", "w_mean", "T_face", "q_if")
        out = {k: np.zeros(t.size) for k in keys}
        T_PV = np.full(5, self.T_init_hg if with_HG else self.T_init_bare)
        out["T_top"][0] = out["T_cell"][0] = T_PV[0]
        out["P"][0] = self.power(self.G(t[0]), T_PV[2])
        out["H"][0] = self.d_HG
        if with_HG:
            T_HG = np.full(self.N_HG, self.T_init_hg)
            C = np.full(self.N_HG, self.env(t[0])["C_inf"])
            n = np.full(self.N_HG, self.n0)
            out["water"][0] = M_W * self.n0 * self.f_HG * self.d_HG
            out["w_mean"][0], out["T_face"][0] = self.omega0, self.T_init_hg
        q_if = 0.0
        for s in range(1, t.size):
            ev = self.env(t[s])
            T_PV_o = T_PV.copy()
            if not with_HG:
                for _ in range(self.n_iter):
                    T_new = self.sys_PV5(T_PV_o, T_PV, ev, 0.0, True)
                    r, T_PV = np.abs(T_new - T_PV).max(), T_new
                    if r < self.tol:
                        break
            else:
                T_HG_o, C_o, n_o = T_HG.copy(), C.copy(), n.copy()
                T_lin = T_PV.copy()
                for _outer in range(4 if self.relin_pv else 1):
                    def mismatch(q):
                        Tp, g = self.sys_PV5(T_PV_o, T_lin, ev, q, False), self.gel_step(T_HG_o, C_o, n_o, q, ev)
                        if g is None:
                            return None, None, None
                        p0 = self.gel_props(g[2][:1], g[0][:1])
                        T_face_pv = Tp[-1] - (q * self.Rhalf_ted if self.iface_Rted else 0.0)
                        return T_face_pv - (g[0][0] + q * p0['wc'][0] / (2 * p0['k_eff'][0])), Tp, g
                    q0, q1 = q_if, q_if + 50.0
                    f0, _, _ = mismatch(q0)
                    f1, Tp1, g1 = mismatch(q1)
                    if f0 is None:          # the original crashed here with a TypeError in abs(f1 - f0)
                        raise RuntimeError(f"gel Newton failed at the secant start, t = {t[s]:.0f} s")
                    for _ in range(self.SECANT_MAX):
                        if f1 is None or abs(f1 - f0) < 1e-14:
                            break
                        q2 = q1 - f1 * (q1 - q0) / (f1 - f0)
                        f2, Tp2, g2 = mismatch(q2)
                        if f2 is None:
                            break
                        q0, f0, q1, f1, Tp1, g1 = q1, f1, q2, f2, Tp2, g2
                        if abs(f2) < 1e-9:
                            break
                    if g1 is None:
                        raise RuntimeError(f"gel step failed at t = {t[s]:.0f} s")
                    q_if = q1
                    if np.abs(Tp1 - T_lin).max() < 1e-6:
                        break
                    T_lin = Tp1
                T_PV = Tp1
                T_HG, C, n = g1
                mcp, ecp, q_face, H, water, w_mean, T_face = self.gel_fluxes(T_HG, C, n, ev)
                stored = np.sum(self.gel_props(n, T_HG)['rc'] * (T_HG - T_HG_o)) * self.dx0 / self.dt
                out["MCP"][s], out["ECP"][s], out["H"][s] = mcp, ecp, H
                out["water"][s], out["w_mean"][s], out["T_face"][s], out["q_if"][s] = water, w_mean, T_face, q_if
                out["res"][s] = q_if - q_face - mcp - ecp - stored
            out["T_top"][s], out["T_cell"][s] = T_PV[0], T_PV[2]
            out["P"][s] = self.power(ev["G"], T_PV[2])
        out["t"] = t
        return out


def metrics(t_sim, v_sim, t_exp, v_exp):
    """MAE, mean relative error Er (Lai Eq. 43, in %), RMSE, R2 -- model interpolated to data times."""
    v_pred = np.interp(t_exp, t_sim, v_sim)
    err = v_pred - v_exp
    mae, rmse = np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2))
    er = np.mean(np.abs(err / v_exp)) * 100.0
    r2 = 1.0 - np.sum(err ** 2) / np.sum((v_exp - np.mean(v_exp)) ** 2)
    return dict(MAE=mae, Er=er, RMSE=rmse, R2=r2)


def check_regression():
    """Default model == original Fig. 4a script (printed values of the original at 3 h)."""
    m = PVHG()
    b, h = m.run(False), m.run(True)
    i3 = np.argmin(np.abs(b["t"] - 3 * 3600))
    got = (b["T_top"][i3] - 273.15, h["T_top"][i3] - 273.15, h["MCP"][i3], h["ECP"][i3], h["H"][i3] * 1e3)
    ref = (61.83, 57.60, 73.2, 54.1, 8.173)
    print("regression (bare, hg, MCP, ECP, H):", np.round(got, 3), " original:", ref,
          " max|res| =", f"{np.abs(h['res']).max():.1e}")
    return b, h


if __name__ == "__main__":
    check_regression()

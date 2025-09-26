import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

st.set_page_config(page_title="Twin Charging Calculator", page_icon="🛠️", layout="wide")
st.title("Twin Charging Calculator — Full Package")

# ---------- Username + Password auth ----------
def auth_gate():
    user_secret = st.secrets.get("APP_USER", None)
    pass_secret = st.secrets.get("APP_PASSWORD", None)
    # If missing, allow access (useful for local dev); set both in Streamlit Secrets to enable lock
    if not user_secret or not pass_secret:
        return True
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True
    with st.form("login"):
        st.caption("This app is protected. Enter credentials to continue.")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
        if ok:
            if u == user_secret and p == pass_secret:
                st.session_state.authed = True
                return True
            else:
                st.error("Incorrect username or password.")
    return False

if not auth_gate():
    st.stop()

# ---------- Units & helpers ----------
ATM_PSI = 14.7
PSI_PER_BAR = 14.5037738
LITERS_TO_CUIN = 61.024
LB_TO_KG = 0.45359237
LPM_PER_GPM = 3.785411784
KPA_PER_PSI = 6.894757293

colU = st.columns(5)
with colU[0]:
    temp_unit = st.selectbox("Temperature unit", ["°C","°F"], index=0)
with colU[1]:
    press_unit = st.selectbox("Manifold pressure unit", ["psi","bar"], index=0)
with colU[2]:
    manifold_ref = st.selectbox("Manifold reference", ["Gauge (above atm)","Absolute"], index=0)
with colU[3]:
    pump_flow_unit = st.selectbox("Pump flow unit", ["L/min","gpm"], index=0)
with colU[4]:
    pump_head_unit = st.selectbox("Pump head unit", ["kPa","psi"], index=0)

def to_psi(x): return x if press_unit == "psi" else x*PSI_PER_BAR
def from_psi(x): return x if press_unit == "psi" else x/PSI_PER_BAR
def flow_to_lpm(x): return x if pump_flow_unit=="L/min" else x*LPM_PER_GPM
def flow_from_lpm(x): return x if pump_flow_unit=="L/min" else x/LPM_PER_GPM
def head_to_kpa(x): return x if pump_head_unit=="kPa" else x*KPA_PER_PSI
def head_from_kpa(x): return x if pump_head_unit=="kPa" else x/KPA_PER_PSI
def C_to_K(c): return c + 273.15
def F_to_K(f): return (f - 32.0)*5.0/9.0 + 273.15
def K_to_C(k): return k - 273.15
def K_to_F(k): return (k - 273.15)*9.0/5.0 + 32.0
def show_temp(k): return f"{K_to_C(k):.1f} °C" if temp_unit=="°C" else f"{K_to_F(k):.1f} °F"

def pr_from_boost_gauge(boost_user_unit: float) -> float:
    boost_psi = to_psi(boost_user_unit)
    return 1.0 + boost_psi / ATM_PSI

def manifold_user_unit(total_pr: float, ref: str) -> float:
    psi_g = (total_pr - 1.0)*ATM_PSI
    if ref.startswith("Absolute"):
        psi = psi_g + ATM_PSI
    else:
        psi = psi_g
    return psi if press_unit=="psi" else psi/PSI_PER_BAR

# ---------- Engine & thermal ----------
def airflow_lb_min(displ_l: float, ve: float, rpm: int, total_pr: float) -> float:
    return (displ_l * LITERS_TO_CUIN * ve * rpm / 3456 / 14.27) * total_pr

def injector_size_cc_min(whp: float, bsfc: float, inj_count: int, max_duty: float) -> float:
    total_lb_hr = whp * bsfc
    per_inj_lb_hr = total_lb_hr / inj_count / max_duty
    return per_inj_lb_hr * 10.5

def sc_drive_ratio(crank_in: float, sc_in: float) -> float:
    return crank_in / sc_in

def sc_rpm(redline_rpm: int, ratio: float) -> float:
    return redline_rpm * ratio

def turbo_outlet_temp_K(T1_K: float, PR_t: float, k: float, eta_c: float) -> float:
    return T1_K * (1.0 + (PR_t ** ((k - 1.0) / k) - 1.0) / max(eta_c, 1e-6))

def simple_ic(T_hot_in_K: float, T_cold_in_K: float, eps: float) -> float:
    return T_cold_in_K + (T_hot_in_K - T_cold_in_K) * (1.0 - max(min(eps, 1.0), 0.0))

def sc_outlet_temp_K(T_in_K: float, PR_sc: float, k: float, eta_sc: float) -> float:
    return T_in_K * (1.0 + (PR_sc ** ((k - 1.0) / k) - 1.0) / max(eta_sc, 1e-6))

def water_flow_L_min(Q_kW: float, cp_w_kJ_per_kgK: float, deltaT_C: float) -> float:
    if cp_w_kJ_per_kgK <= 0 or deltaT_C <= 0: return 0.0
    kg_s = Q_kW / (cp_w_kJ_per_kgK * deltaT_C)
    return kg_s * 60.0

def lmtd(Th_in, Th_out, Tc_in, Tc_out):
    dT1 = Th_in - Tc_out
    dT2 = Th_out - Tc_in
    if dT1 <= 0 or dT2 <= 0 or abs(dT1 - dT2) < 1e-9:
        return max(min(dT1, dT2), 1e-6)
    return (dT1 - dT2) / (np.log(dT1) - np.log(dT2))

def radiator_area_m2(Q_kW: float, U_W_m2K: float, lmtd_K: float):
    if U_W_m2K <= 0 or lmtd_K <= 0: return 0.0
    UA = (Q_kW * 1000.0) / lmtd_K
    return UA / U_W_m2K

# ---------- Pumps & pipes ----------
def darcy_f(Re, rel_rough):
    if Re < 2300: return 64.0 / max(Re,1e-9)
    return 0.25 / (np.log10(rel_rough/3.7 + 5.74/(Re**0.9)))**2

def kpa_from_segment(flow_Lmin, D_mm, L_m, bends, angle_deg, K90, rho, mu, eps_rel):
    if D_mm <= 0 or L_m < 0: return 0.0
    Q = flow_Lmin / 1000.0 / 60.0
    D = D_mm / 1000.0
    A = np.pi * (D**2) / 4.0
    v = Q / A
    Re = (rho * v * D) / mu
    f = darcy_f(Re, eps_rel)
    dp_fric = f * (L_m / D) * 0.5 * rho * v*v
    K_equiv = (angle_deg/90.0) * K90 * bends
    dp_bend = K_equiv * 0.5 * rho * v*v
    return (dp_fric + dp_bend) / 1000.0

def merge_return_id(DA_mm, DB_mm):
    if DA_mm <= 0 and DB_mm <= 0: return 0.0
    return sqrt(max(DA_mm,0.0)**2 + max(DB_mm,0.0)**2)

def pump_presets(name):
    if name=="Built-in: Pump A (generic)":
        return [(0,80),(20,70),(40,58),(60,42),(80,28),(100,12),(120,5)]
    if name=="Built-in: Pump B (generic)":
        return [(0,120),(20,110),(40,95),(60,70),(80,40),(100,20),(120,10)]
    if name=="Built-in: Pump C (generic)":
        return [(0,90),(20,78),(40,60),(60,45),(80,30),(100,18),(120,8)]
    if name=="Preset: Bosch CWA50 (approx)":
        return [(0,70),(20,60),(40,48),(60,34),(80,22),(100,12)]
    if name=="Preset: Bosch CWA100 (approx)":
        return [(0,160),(20,145),(40,125),(60,95),(80,60),(100,30)]
    if name=="Preset: Pierburg CWP35 (approx)":
        return [(0,75),(20,63),(40,50),(60,36),(80,20),(100,8)]
    return []

# ---------- Air System Library ----------
st.sidebar.header("Air System Library")
st.sidebar.subheader("Turbo Database")
turbo_csv = st.sidebar.file_uploader("Upload turbo map CSV", type=["csv"], key="turbo_csv")
turbo_pts = []
if turbo_csv is not None:
    import io, csv as _csv
    rows = list(_csv.reader(io.StringIO(turbo_csv.read().decode("utf-8"))))
    headers = [h.strip().lower() for h in rows[0]]
    idx = {h:i for i,h in enumerate(headers)}
    for r in rows[1:]:
        try:
            man = r[idx["manufacturer"]]; mdl = r[idx["model"]]
            wc = float(r[idx["wc_lb_min"]]); pr = float(r[idx["pr"]])
            eta = float(r[idx["eta_c"]]); n = float(r[idx["n_rpm"]])
            surge = (r[idx.get("is_surge","")].strip().lower()=="true") if "is_surge" in idx else False
            choke = (r[idx.get("is_choke","")].strip().lower()=="true") if "is_choke" in idx else False
            turbo_pts.append((man,mdl,wc,pr,eta,n,surge,choke))
        except: pass

if turbo_pts:
    mans = sorted({p[0] for p in turbo_pts})
    sel_tman = st.sidebar.selectbox("Turbo Manufacturer", mans)
    models = sorted({p[1] for p in turbo_pts if p[0]==sel_tman})
    sel_tmodel = st.sidebar.selectbox("Turbo Model", models)
    map_pts = [p for p in turbo_pts if p[0]==sel_tman and p[1]==sel_tmodel]
    st.sidebar.caption(f"Loaded {len(map_pts)} points for {sel_tman} {sel_tmodel}")
else:
    map_pts = []

interp_mode = st.sidebar.selectbox("Turbo interpolation", ["Bilinear","Nearest"])

st.sidebar.subheader("Supercharger Database")
sc_const_csv = st.sidebar.file_uploader("Upload SC constants CSV", type=["csv"], key="sc_const_csv")
sc_records = []
if sc_const_csv is not None:
    import io, csv as _csv
    rows = list(_csv.reader(io.StringIO(sc_const_csv.read().decode("utf-8"))))
    headers = [h.strip().lower() for h in rows[0]]
    idx = {h:i for i,h in enumerate(headers)}
    for r in rows[1:]:
        try:
            man = r[idx["manufacturer"]]; mdl = r[idx["model"]]
            disp = float(r[idx["displacement_per_rev_l"]])
            maxn = float(r[idx["max_speed_rpm"]])
            dVE  = float(r[idx["default_ve"]]); deta = float(r[idx["default_eta_sc"]])
            sc_records.append((man,mdl,disp,maxn,dVE,deta))
        except: pass

sc_grid_csv = st.sidebar.file_uploader("Upload SC efficiency grid CSV (optional)", type=["csv"], key="sc_grid_csv")
scg_records = []
if sc_grid_csv is not None:
    import io, csv as _csv
    rows = list(_csv.reader(io.StringIO(sc_grid_csv.read().decode("utf-8"))))
    headers = [h.strip().lower() for h in rows[0]]
    idx = {h:i for i,h in enumerate(headers)}
    for r in rows[1:]:
        try:
            man = r[idx["manufacturer"]]; mdl = r[idx["model"]]
            N = float(r[idx["n_rpm"]]); PR = float(r[idx["pr"]])
            VE = float(r[idx["ve"]]); eta = float(r[idx["eta_sc"]])
            scg_records.append((man,mdl,N,PR,VE,eta))
        except: pass

use_grid = st.sidebar.checkbox("Use SC grid if available", value=True)

sel_sc = None
if sc_records:
    mans = sorted({r[0] for r in sc_records})
    sel_sc_man = st.sidebar.selectbox("SC Manufacturer", mans)
    sc_models = sorted({r[1] for r in sc_records if r[0]==sel_sc_man})
    sel_sc_model = st.sidebar.selectbox("SC Model", sc_models)
    sc_const = [r for r in sc_records if r[0]==sel_sc_man and r[1]==sel_sc_model][0]
    sel_sc = {"manufacturer": sc_const[0], "model": sc_const[1],
              "disp_L_rev": sc_const[2], "max_speed": sc_const[3],
              "def_VE": sc_const[4], "def_eta": sc_const[5]}

sc_grid = [r for r in scg_records if sel_sc and r[0]==sel_sc["manufacturer"] and r[1]==sel_sc["model"]] if (use_grid and scg_records and sel_sc) else []

# ---------- Input form ----------
with st.form("inputs"):
    st.subheader("Engine & Boost")
    c1, c2 = st.columns(2)
    with c1:
        displacement_l = st.number_input("Displacement (L)", value=2.4, min_value=0.1, step=0.1, format="%.2f")
        redline_rpm    = st.number_input("Redline RPM", value=8800, min_value=1000, step=100)
        ve             = st.number_input("Volumetric Efficiency (0–1)", value=0.95, min_value=0.1, max_value=1.0, step=0.01)
        target_whp     = st.number_input("Target Power (whp)", value=600, min_value=1, step=1)
        bsfc           = st.number_input("BSFC (lb/hp/hr)", value=0.60, min_value=0.3, step=0.01)
    with c2:
        st.caption("Boost entries are **gauge** (above atmosphere). Manifold display can be Gauge or Absolute.")
        sc_boost_user  = st.number_input(f"Supercharger Boost (gauge, {press_unit})", value=10.0, min_value=0.0, step=0.1)
        turbo_boost_user=st.number_input(f"Turbo Boost (gauge, {press_unit})", value=15.0, min_value=0.0, step=0.1)
        crank_pulley_in= st.number_input("Crank Pulley (in)", value=6.37, min_value=0.1, step=0.01)
        sc_pulley_in   = st.number_input("SC Pulley (in)", value=3.90, min_value=0.1, step=0.01)
        max_inj_duty   = st.number_input("Max Injector Duty (0–1)", value=0.85, min_value=0.1, max_value=0.99, step=0.01)
        injectors      = st.number_input("Number of Injectors", value=4, min_value=1, step=1)

    st.subheader("Cooling (Thermal Model)")
    c3, c4 = st.columns(2)
    with c3:
        amb_user       = st.number_input(f"Ambient Air Temp ({temp_unit})", value=25.0, step=1.0)
        cool_in_user   = st.number_input(f"Coolant Inlet Temp to IC2 ({temp_unit})", value=35.0, step=1.0)
        ic1_eff        = st.number_input("IC1 Effectiveness (0–1)", value=0.65, min_value=0.0, max_value=1.0, step=0.05)
        ic2_eff        = st.number_input("IC2 Effectiveness (0–1)", value=0.70, min_value=0.0, max_value=1.0, step=0.05)
        turbo_eta_user = st.number_input("Turbo ηc override (0→use map)", value=0.00, min_value=0.0, max_value=1.0, step=0.01)
        sc_eta_user    = st.number_input("SC η override (0→use grid/default)", value=0.00, min_value=0.0, max_value=1.0, step=0.01)
    with c4:
        cp_air         = st.number_input("cp_air (kJ/kg-K)", value=1.005, min_value=0.5, max_value=1.2, step=0.001, format="%.3f")
        cp_w           = st.number_input("cp_water (kJ/kg-K)", value=4.186, min_value=3.5, max_value=5.0, step=0.001, format="%.3f")
        deltaT_w_ic1   = st.number_input("Allowed Water ΔT in IC1 Loop (°C)", value=10.0, min_value=2.0, max_value=30.0, step=1.0)
        deltaT_w_ic2   = st.number_input("Allowed Water ΔT in IC2 Loop (°C)", value=10.0, min_value=2.0, max_value=30.0, step=1.0)
        U_radiator     = st.number_input("Radiator U (W/m²-K)", value=250.0, min_value=100.0, max_value=800.0, step=10.0)
        k_air          = st.number_input("Gamma for air (k)", value=1.4, min_value=1.30, max_value=1.40, step=0.01)

    st.subheader("Pump Helper — Presets & Custom Curves")
    known_flow = st.checkbox("I have a required flow target (known)")
    pump_source = st.selectbox("Pump curve source", [
        "Built-in: Pump A (generic)",
        "Built-in: Pump B (generic)",
        "Built-in: Pump C (generic)",
        "Preset: Bosch CWA50 (approx)",
        "Preset: Bosch CWA100 (approx)",
        "Preset: Pierburg CWP35 (approx)",
        "Custom CSV (upload)",
        "Custom CSV (paste)",
    ])
    uploaded = None
    default_csv = "0,80\n20,70\n40,58\n60,42\n80,28\n100,12\n120,5"
    custom_csv = ""
    if pump_source == "Custom CSV (upload)":
        uploaded = st.file_uploader("Upload CSV with two columns: Flow, Head", type=["csv"])
    elif pump_source == "Custom CSV (paste)":
        custom_csv = st.text_area("Paste CSV pairs (Flow,Head) in selected units", value=default_csv, height=120)

    colK1, colK2 = st.columns(2)
    with colK1:
        K_ic1 = st.number_input(f"IC1 system K ({pump_head_unit}/({pump_flow_unit})^2)", value=0.01, step=0.005, format="%.5f")
        S_ic1 = st.number_input(f"IC1 system S ({pump_head_unit})", value=10.0, step=1.0)
        req_ic1 = st.number_input(f"IC1 required flow ({pump_flow_unit})", value=20.0, step=1.0) if known_flow else None
    with colK2:
        K_ic2 = st.number_input(f"IC2 system K ({pump_head_unit}/({pump_flow_unit})^2)", value=0.01, step=0.005, format="%.5f")
        S_ic2 = st.number_input(f"IC2 system S ({pump_head_unit})", value=10.0, step=1.0)
        req_ic2 = st.number_input(f"IC2 required flow ({pump_flow_unit})", value=20.0, step=1.0) if known_flow else None

    st.subheader("Extended Pipe Calculator → derive K & S")
    use_pipe_calc = st.checkbox("Use extended pipe calculator (feeds K & S above)")
    with st.expander("Pipe geometry & components"):
        st.caption("Two supply branches → two IC cores → merged single return. Darcy–Weisbach + bends + elevation + components.")
        cwp1, cwp2, cwp3 = st.columns(3)
        with cwp1:
            rho = st.number_input("Water density ρ (kg/m³)", value=997.0, step=1.0)
        with cwp2:
            mu = st.number_input("Viscosity μ (Pa·s)", value=0.00089, step=0.00001, format="%.5f")
        with cwp3:
            eps_rel = st.number_input("Relative roughness ε/D", value=0.0002, step=0.0001, format="%.5f")

        # Branch A
        st.markdown("**Supply Branch A → IC Core A**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            D_A_mm = st.number_input("ID_A (mm)", value=19.0, step=1.0)
        with a2:
            L_A_m  = st.number_input("Length_A (m)", value=2.0, step=0.1)
        with a3:
            bends_A = st.number_input("Bends_A (count)", value=4, step=1)
        with a4:
            angle_A = st.number_input("Typical bend angle_A (°)", value=90, step=5)
        a5, a6 = st.columns(2)
        with a5:
            K90_A = st.number_input("K per 90° (A)", value=0.9, step=0.1)
        with a6:
            elev_A = st.number_input("Elevation gain A (m)", value=0.0, step=0.1)

        # Branch B
        st.markdown("**Supply Branch B → IC Core B**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            D_B_mm = st.number_input("ID_B (mm)", value=19.0, step=1.0)
        with b2:
            L_B_m  = st.number_input("Length_B (m)", value=2.0, step=0.1)
        with b3:
            bends_B = st.number_input("Bends_B (count)", value=4, step=1)
        with b4:
            angle_B = st.number_input("Typical bend angle_B (°)", value=90, step=5)
        b5, b6 = st.columns(2)
        with b5:
            K90_B = st.number_input("K per 90° (B)", value=0.9, step=0.1)
        with b6:
            elev_B = st.number_input("Elevation gain B (m)", value=0.0, step=0.1)

        # Return
        st.markdown("**Single Return Line (after the 2→1 merge)**")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            D_R_mm = st.number_input("Return ID (mm) (0 = auto area-sum)", value=0.0, step=1.0)
        with r2:
            L_R_m  = st.number_input("Return length (m)", value=2.0, step=0.1)
        with r3:
            bends_R = st.number_input("Return bends (count)", value=2, step=1)
        with r4:
            angle_R = st.number_input("Return bend angle (°)", value=90, step=5)
        r5, r6 = st.columns(2)
        with r5:
            K90_R = st.number_input("K per 90° (Return)", value=0.9, step=0.1)
        with r6:
            elev_R = st.number_input("Return elevation gain (m)", value=0.0, step=0.1)

        # Components
        st.markdown("**Component ΔP @ Flow_ref**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            dp_ic_kpa_at = st.number_input("IC core ΔP (kPa)", value=5.0, step=0.5)
        with cc2:
            dp_hx_kpa_at = st.number_input("Heat Exchanger ΔP (kPa)", value=10.0, step=0.5)
        with cc3:
            flow_ref_Lmin = st.number_input("Reference Flow_ref (L/min)", value=30.0, step=1.0)

    run = st.form_submit_button("Compute")

# ---------- Turbo & SC interpolation helpers ----------
def interp_turbo(map_pts, Wc, PR, mode="Bilinear"):
    if not map_pts:
        return None
    wc_vals = sorted(set([p[2] for p in map_pts]))
    pr_vals = sorted(set([p[3] for p in map_pts]))
    eta_grid = np.full((len(pr_vals), len(wc_vals)), np.nan)
    n_grid   = np.full_like(eta_grid, np.nan, dtype=float)
    for man,mdl,wc,pr,eta,N,surge,choke in map_pts:
        i = pr_vals.index(pr); j = wc_vals.index(wc)
        eta_grid[i,j] = eta; n_grid[i,j] = N
    def find_idx(arr, x):
        if x <= arr[0]: return 0,0,0.0
        if x >= arr[-1]: return len(arr)-1, len(arr)-1, 0.0
        for k in range(len(arr)-1):
            if arr[k] <= x <= arr[k+1]:
                t = (x - arr[k])/(arr[k+1]-arr[k])
                return k, k+1, t
        return len(arr)-1, len(arr)-1, 0.0
    i0,i1,ti = find_idx(pr_vals, PR)
    j0,j1,tj = find_idx(wc_vals, Wc)
    def bilerp(grid):
        q00 = grid[i0,j0]; q01 = grid[i0,j1]
        q10 = grid[i1,j0]; q11 = grid[i1,j1]
        if any(np.isnan([q00,q01,q10,q11])):
            ii = i0 if ti<0.5 else i1
            jj = j0 if tj<0.5 else j1
            return grid[ii,jj]
        return (1-ti)*(1-tj)*q00 + (1-ti)*tj*q01 + ti*(1-tj)*q10 + ti*tj*q11
    def nearest(grid):
        ii = i0 if ti<0.5 else i1
        jj = j0 if tj<0.5 else j1
        return grid[ii,jj]
    if mode=="Bilinear":
        eta = bilerp(eta_grid); N = bilerp(n_grid)
    else:
        eta = nearest(eta_grid); N = nearest(n_grid)
    return {"eta_c": None if np.isnan(eta) else float(eta),
            "N_rpm": None if np.isnan(N) else float(N)}

def surge_choke_status(map_pts, Wc, PR):
    surge_pts = [(p[2],p[3]) for p in map_pts if p[6]]
    choke_pts = [(p[2],p[3]) for p in map_pts if p[7]]
    status = {"surge_margin_pct": None, "in_surge": False, "in_choke": False}
    if surge_pts:
        pr_list = [p[1] for p in surge_pts]; wc_list = [p[0] for p in surge_pts]
        idx = int(np.argmin([abs(PR - pr) for pr in pr_list]))
        wc_at_pr = wc_list[idx]
        status["in_surge"] = Wc < wc_at_pr
        if wc_at_pr>0:
            status["surge_margin_pct"] = max(0.0, (Wc - wc_at_pr)/wc_at_pr*100.0)
    if choke_pts:
        pr_list = [p[1] for p in choke_pts]; wc_list = [p[0] for p in choke_pts]
        idx = int(np.argmin([abs(PR - pr) for pr in pr_list]))
        wc_at_pr = wc_list[idx]
        status["in_choke"] = Wc > wc_at_pr
    return status

def interp_sc(sc_grid, N, PR):
    if not sc_grid:
        return None
    N_vals = sorted(set([r[2] for r in sc_grid]))
    PR_vals= sorted(set([r[3] for r in sc_grid]))
    VE_g = np.full((len(PR_vals), len(N_vals)), np.nan)
    ETA_g= np.full_like(VE_g, np.nan)
    for man,mdl,n,pr,ve,eta in sc_grid:
        i = PR_vals.index(pr); j = N_vals.index(n)
        VE_g[i,j] = ve; ETA_g[i,j]=eta
    def find_idx(arr, x):
        if x <= arr[0]: return 0,0,0.0
        if x >= arr[-1]: return len(arr)-1, len(arr)-1, 0.0
        for k in range(len(arr)-1):
            if arr[k] <= x <= arr[k+1]:
                t = (x - arr[k])/(arr[k+1]-arr[k])
                return k, k+1, t
        return len(arr)-1, len(arr)-1, 0.0
    i0,i1,ti = find_idx(PR_vals, PR)
    j0,j1,tj = find_idx(N_vals, N)
    def bilerp(grid):
        q00 = grid[i0,j0]; q01 = grid[i0,j1]
        q10 = grid[i1,j0]; q11 = grid[i1,j1]
        if any(np.isnan([q00,q01,q10,q11])):
            cands=[]
            for ii in [i0,i1]:
                for jj in [j0,j1]:
                    v = grid[ii,jj]
                    if not np.isnan(v): cands.append(v)
            return np.mean(cands) if cands else np.nan
        return (1-ti)*(1-tj)*q00 + (1-ti)*tj*q01 + ti*(1-tj)*q10 + ti*tj*q11
    VE  = bilerp(VE_g)
    ETA = bilerp(ETA_g)
    return {"VE": None if np.isnan(VE) else float(VE),
            "eta_sc": None if np.isnan(ETA) else float(ETA)}

# ---------- Compute ----------
if run:
    T_amb_K  = C_to_K(amb_user) if temp_unit=="°C" else F_to_K(amb_user)
    T_cw_K   = C_to_K(cool_in_user) if temp_unit=="°C" else F_to_K(cool_in_user)

    sc_pr    = pr_from_boost_gauge(sc_boost_user)
    turbo_pr = pr_from_boost_gauge(turbo_boost_user)
    total_pr = sc_pr * turbo_pr
    mani_user= manifold_user_unit(total_pr, manifold_ref)

    air_lb_min = airflow_lb_min(displacement_l, ve, redline_rpm, total_pr)
    air_kg_s   = air_lb_min * LB_TO_KG / 60.0
    inj_cc     = injector_size_cc_min(target_whp, bsfc, injectors, max_inj_duty)

    ratio    = sc_drive_ratio(crank_pulley_in, sc_pulley_in)
    scspeed  = sc_rpm(redline_rpm, ratio)

    turbo_eta = turbo_eta_user if turbo_eta_user>0 else None
    turbo_speed = None
    wc = None; status = {}
    if map_pts:
        W = air_lb_min
        T_in = T_amb_K; P_in_kpa = 101.325
        T_ref = 288.15; P_ref = 101.325
        Wc = W * np.sqrt(T_in/T_ref) / (P_in_kpa/P_ref)
        wc = Wc
        interp = interp_turbo(map_pts, Wc, turbo_pr, mode=interp_mode)
        if interp and turbo_eta is None:
            turbo_eta = interp["eta_c"]
        turbo_speed = interp["N_rpm"] if interp else None
        status = surge_choke_status(map_pts, Wc, turbo_pr)

    eta_c_use = turbo_eta if (turbo_eta is not None and turbo_eta>0) else 0.72

    T1_K     = T_amb_K
    T2_K     = turbo_outlet_temp_K(T1_K, turbo_pr, 1.4, eta_c_use)
    T2i_K    = simple_ic(T2_K, T1_K, ic1_eff)

    sc_VE = None; sc_eta_val = None
    if sc_grid:
        gi = interp_sc(sc_grid, scspeed, sc_pr)
        if gi:
            sc_VE = gi["VE"]
            sc_eta_val = gi["eta_sc"]
    if sel_sc and sc_VE is None: sc_VE = sel_sc["def_VE"]
    if sc_eta_user>0: sc_eta_val = sc_eta_user
    elif sel_sc and sc_eta_val is None: sc_eta_val = sel_sc["def_eta"]
    if sc_eta_val is None: sc_eta_val = 0.65
    if sc_VE is None: sc_VE = 0.93

    T3_K     = sc_outlet_temp_K(T2i_K, sc_pr, 1.4, sc_eta_val)
    T4_K     = simple_ic(T3_K, T_cw_K, ic2_eff)

    Q_ic1_kW = air_kg_s * 1.005 * max(T2_K - T2i_K, 0.0)
    Q_sc_kW  = air_kg_s * 1.005 * max(T3_K - T2i_K, 0.0)
    Q_ic2_kW = air_kg_s * 1.005 * max(T3_K - T4_K, 0.0)

    flow_ic1_Lmin = water_flow_L_min(Q_ic1_kW, 4.186, deltaT_w_ic1)
    flow_ic2_Lmin = water_flow_L_min(Q_ic2_kW, 4.186, deltaT_w_ic2)

    LMTD_ic2  = lmtd(cool_in_user + deltaT_w_ic2, cool_in_user, amb_user, amb_user)
    area_ic2  = radiator_area_m2(Q_ic2_kW, 250.0, LMTD_ic2)
    LMTD_ic1  = lmtd(amb_user + deltaT_w_ic1, amb_user, amb_user, amb_user)
    area_ic1  = radiator_area_m2(Q_ic1_kW, 250.0, LMTD_ic1)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("SC PR", f"{sc_pr:.3f}")
        st.metric("Turbo PR", f"{turbo_pr:.3f}")
        st.metric(f"Manifold ({manifold_ref}, {press_unit})", f"{mani_user:.2f}")
    with c2:
        st.metric("Airflow (lb/min)", f"{air_lb_min:.1f}")
        st.metric("SC RPM @ redline", f"{scspeed:.0f}")
        st.metric("Injector size (cc/min each)", f"{inj_cc:.0f}")
    with c3:
        st.metric("Q_IC1 (kW)", f"{Q_ic1_kW:.1f}")
        st.metric("Q_SC added (kW)", f"{Q_sc_kW:.1f}")
        st.metric("Q_IC2 (kW)", f"{Q_ic2_kW:.1f}")

    st.subheader("Temperatures")
    st.write({
        "T2 Turbo Outlet": show_temp(T2_K),
        "T2i After IC1": show_temp(T2i_K),
        "T3 SC Outlet": show_temp(T3_K),
        "T4 After IC2 (Plenum Inlet)": show_temp(T4_K),
    })

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Turbo Map (Wc vs PR)**")
        if map_pts:
            xs = [p[2] for p in map_pts]; ys = [p[3] for p in map_pts]; cs = [p[4] for p in map_pts]
            fig, ax = plt.subplots()
            scat = ax.scatter(xs, ys, c=cs)
            cb = fig.colorbar(scat, ax=ax); cb.set_label("ηc")
            if wc is not None:
                ax.scatter([wc],[turbo_pr], marker="x")
            ax.set_xlabel("Corrected Flow Wc (lb/min)")
            ax.set_ylabel("PR")
            ax.set_title("Turbo Map (colored by ηc)")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Upload a turbo CSV to see the map.")

    with g2:
        st.markdown("**Supercharger Operating Line**")
        fig2, ax2 = plt.subplots()
        ax2.scatter([scspeed],[sc_pr])
        ax2.set_xlabel("SC Speed (rpm)")
        ax2.set_ylabel("PR")
        ax2.set_title("SC PR vs Speed (current point)")
        ax2.grid(True)
        st.pyplot(fig2)

    st.subheader("Cooling Results")
    st.write("- **IC1 water flow target**:", f"{flow_from_lpm(flow_ic1_Lmin):.1f} {('gpm' if pump_flow_unit=='gpm' else 'L/min')}")
    st.write("- **IC2 water flow target**:", f"{flow_from_lpm(flow_ic2_Lmin):.1f} {('gpm' if pump_flow_unit=='gpm' else 'L/min')}")
    st.write("- **Estimated radiator area IC1** (m²): ", f"{area_ic1:.3f}")
    st.write("- **Estimated radiator area IC2** (m²): ", f"{area_ic2:.3f}")

    # Minimal CSV export
    import io, csv as _csv
    export = io.StringIO()
    w = _csv.writer(export)
    w.writerow(["Metric","Value","Unit"])
    for k,v,u in [
        ("SC_PR", sc_pr,""),("Turbo_PR", turbo_pr,""),("Total_PR", total_pr,""),
        (f"Manifold({manifold_ref},{press_unit})", mani_user, press_unit),
        ("Airflow_lb_min", air_lb_min,"lb/min"),("Q_IC1_kW", Q_ic1_kW,"kW"),("Q_IC2_kW", Q_ic2_kW,"kW")
    ]: w.writerow([k,v,u])
    st.download_button("Download results as CSV", data=export.getvalue(), file_name="twincharge_results.csv", mime="text/csv")
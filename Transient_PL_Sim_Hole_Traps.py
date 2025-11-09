import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd

# ----------------------------- CONSTANTS --------------------------------
k_B = 8.617333e-5  # eV/K
m_e0 = 9.11e-28
m_eff = 0.2 * m_e0
nu0 = 1e13  # s^-1

def v_thermal(T):
    k_B_erg = 1.380649e-16
    return np.sqrt(3 * k_B_erg * T / m_eff) * 1e2  # cm/s

# -------------------------- APP TITLE -----------------------------------
st.title("📈 Transient PL Simulator (Extended SRH & Interface Model)")

# -------------------------- SIDEBAR SETUP -------------------------------
st.sidebar.header("Model setup")

with st.sidebar.expander("🌡️ Global parameters"):
    use_physical = st.checkbox("Compute trap rates from σ and E_t", value=True)
    T = st.number_input("Temperature (K)", 50.0, 1000.0, 300.0, step=10.0)
    R_eh = st.number_input("Radiative recombination R_eh (cm³/s)", 1e-30, 1e30, 1e-10, format="%.1e")
    L = st.number_input("Film thickness L (cm)", 1e-9, 1e2, 5e-5, format="%.1e")

vth = v_thermal(T)

# ---------- Electron traps ----------
# ---------- Electron traps ----------
with st.sidebar.expander("🧲 Electron traps"):
    N_t_e = st.number_input("Electron trap density N_t_e (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")

    if use_physical:
        sigma_e = st.number_input("Electron capture σ_e (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")
        E_t_e   = st.number_input("Trap depth E_t_e (eV below CB)", 0.0, 2.0, 0.4, step=0.05)
        sigma_h_trap = st.number_input("Hole capture σ_h^(trap) (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")

        R_pop_e    = sigma_e * vth
        R_depop_e  = sigma_h_trap * vth
        k_detrap_e = nu0 * np.exp(-E_t_e / (k_B * T))

        st.write(f"→ R_pop_e = {R_pop_e:.2e} cm³/s")
        st.write(f"→ R_depop_e = {R_depop_e:.2e} cm³/s  (hole capture)")
        st.write(f"→ k_detrap_e = {k_detrap_e:.2e} s⁻¹")

    else:
        R_pop_e    = st.number_input("R_pop_e (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
        R_depop_e  = st.number_input("R_depop_e (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
        k_detrap_e = st.number_input("k_detrap_e (s⁻¹)", 1e0, 1e30, 1e5, format="%.1e")


# ---------- Hole traps ----------
with st.sidebar.expander("⚡ Hole traps (optional)"):
    use_hole_traps = st.checkbox("Enable hole traps", value=False)
    N_t_h = R_pop_h = R_depop_h = k_detrap_h = 0.0
    if use_hole_traps:
        N_t_h = st.number_input("Hole trap density N_t_h (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
        if use_physical:
            sigma_h = st.number_input("σ_h (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")
            E_t_h = st.number_input("Trap depth E_t_h (eV above VB)", 0.0, 2.0, 0.4, step=0.05)
            R_pop_h = sigma_h * vth
            k_detrap_h = nu0 * np.exp(-E_t_h / (k_B * T))
            st.write(f"→ R_pop_h = {R_pop_h:.2e} cm³/s,  k_detrap_h = {k_detrap_h:.2e} s⁻¹")
        else:
            R_pop_h = st.number_input("R_pop_h (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
            k_detrap_h = st.number_input("k_detrap_h (s⁻¹)", 1e0, 1e30, 1e5, format="%.1e")
        R_depop_h = st.number_input("R_depop_h (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")

# ---------- Presets (must be before we define interface parameters) ----------
st.sidebar.subheader("Presets")

preset = st.sidebar.selectbox(
    "Parameter preset",
    [
        "Custom / keep sliders",
        "Good interface (fast extraction, low R_AI)",
        "Bad interface (fast extraction, high R_AI)",
        "Bulk only (no extraction, no interface recomb)"
    ]
)

# initialize session_state defaults if not present
if "k_ht" not in st.session_state:
    st.session_state.k_ht = 1e8
if "k_hbt" not in st.session_state:
    st.session_state.k_hbt = 1e6
if "R_AI" not in st.session_state:
    st.session_state.R_AI = 1e-8

if "current_preset" not in st.session_state:
    st.session_state.current_preset = "Custom / keep sliders"

# apply preset if changed
if preset != st.session_state.current_preset:
    st.session_state.current_preset = preset

    if preset == "Good interface (fast extraction, low R_AI)":
        st.session_state.k_ht  = 1e9
        st.session_state.k_hbt = 1e5
        st.session_state.R_AI  = 1e-10

    elif preset == "Bad interface (fast extraction, high R_AI)":
        st.session_state.k_ht  = 1e9
        st.session_state.k_hbt = 1e6
        st.session_state.R_AI  = 1e-7

    elif preset == "Bulk only (no extraction, no interface recomb)":
        # your requested preset: negligible extraction and interface loss
        st.session_state.k_ht  = 1.0    # ~ no extraction
        st.session_state.k_hbt = 1.0    # ~ no back-transfer dynamics
        st.session_state.R_AI  = 0.0    # no interface recombination

    # "Custom / keep sliders" leaves current values as they are

# ---------- Extraction & interface block ----------
with st.sidebar.expander("🔄 Extraction & interface"):
    k_ht = st.number_input(
        "Hole transfer k_ht (s⁻¹)",
        1e0, 1e30,
        value=st.session_state.k_ht,
        key="k_ht",
        format="%.1e"
    )
    k_hbt = st.number_input(
        "Back transfer k_hbt (s⁻¹)",
        1e0, 1e30,
        value=st.session_state.k_hbt,
        key="k_hbt",
        format="%.1e"
    )
    R_AI = st.number_input(
        "Interface recombination R_AI (cm³/s)",
        1e-20, 1e-3,
        value=st.session_state.R_AI,
        key="R_AI",
        format="%.1e"
    )
    st.markdown("*(Effective interface recombination constant used in ODEs.)*")



# ---------- Initial conditions ----------
with st.sidebar.expander("🧮 Initial conditions"):
    n0 = st.number_input("Initial carrier density n0 (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
    f_nt0 = st.number_input("Initial electron trap occupancy f_nt0 (0–1)", 0.0, 1.0, 0.0, step=0.05)
    f_pt0 = st.number_input("Initial hole trap occupancy f_pt0 (0–1)", 0.0, 1.0, 0.0, step=0.05)

# ---------- Simulation options ----------
with st.sidebar.expander("⚙️ Simulation options"):
    do_fit = st.checkbox("Fit double exponential", value=True)
    normalize_pl = st.checkbox("Normalize PL", value=True)
    log_xaxis = st.checkbox("Logarithmic x-axis", value=True)

# -------------------------- KINETIC MODEL -------------------------------
t_eval = np.logspace(-9, -5, 400)
t_span = (t_eval[0], t_eval[-1])

def deriv(t, y):
    if use_hole_traps:
        ne, nh, nt, pt, nhPTAA = y
    else:
        ne, nh, nt, nhPTAA = y
        pt = 0.0

    dne = (-R_eh*ne*nh
           - R_pop_e*(N_t_e-nt)*ne
           - R_depop_h*pt*ne
           - R_AI*ne*nhPTAA
           + k_detrap_e*nt)

    dnh = (-R_eh*ne*nh
           - R_depop_e*nt*nh
           - R_pop_h*(N_t_h-pt)*nh
           - k_ht*nh
           + k_hbt*nhPTAA
           + k_detrap_h*pt)

    dnt = (R_pop_e*(N_t_e-nt)*ne
           - R_depop_e*nt*nh
           - k_detrap_e*nt)

    dpt = (R_pop_h*(N_t_h-pt)*nh
           - R_depop_h*pt*ne
           - k_detrap_h*pt)

    dnhPTAA = (k_ht*nh
               - k_hbt*nhPTAA
               - R_AI*ne*nhPTAA)

    if use_hole_traps:
        return [dne, dnh, dnt, dpt, dnhPTAA]
    else:
        return [dne, dnh, dnt, dnhPTAA]

def simulate():
    """Return both absolute and normalized PL for consistency."""
    if use_hole_traps:
        y0 = [n0, n0, f_nt0*N_t_e, f_pt0*N_t_h, 0.0]
    else:
        y0 = [n0, n0, f_nt0*N_t_e, 0.0]

    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, method="LSODA")
    if use_hole_traps:
        ne, nh, nt, pt, nhPTAA = sol.y
    else:
        ne, nh, nt, nhPTAA = sol.y
        pt = np.zeros_like(ne)

    PL_abs = R_eh * ne * nh
    PL_norm = PL_abs / PL_abs[0]
    if normalize_pl:
        PL = PL_norm
    else:
        PL = PL_abs
    return sol.t, ne, nh, nt, pt, nhPTAA, PL, PL_abs, PL_norm


def compute_Seff():
    t, ne, nh, nt, pt, nhPTAA, PL, PL_abs, PL_norm = simulate()
    idx = np.argmin(np.abs(PL_norm - 1/np.e))
    nhPTAA_char = nhPTAA[idx]
    Seff = R_AI * nhPTAA_char * L
    return Seff, t[idx], nhPTAA_char, (t, PL_norm)


def double_exp(t, A1, tau1, A2, tau2):
    return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)

def fit_double_exp(t, pl):
    mask = pl > 1e-6
    t_fit, pl_fit = t[mask], pl[mask]
    p0 = [0.7, 5e-9, 0.3, 1e-7]
    bounds = ([-np.inf, 1e-10, -np.inf, 1e-10],
              [np.inf, 1e-3, np.inf, 1e-3])
    popt, _ = curve_fit(double_exp, t_fit, pl_fit, p0=p0,
                        bounds=bounds, maxfev=20000)
    return popt, (t_fit, pl_fit)

# ------------------------- MAIN SIMULATION ------------------------------
Seff, t_char, nhPTAA_char, (t, PL) = compute_Seff()

with st.expander("Plot Simulated PL"):
    fig, ax = plt.subplots()
    ax.plot(t, PL, 'k.', markersize=3, label="Simulated PL")
    if log_xaxis:
        ax.set_xscale('log'); ax.set_yscale('log')
    else:
        ax.set_xscale('linear'); ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized PL")
    ax.set_ylim(1e-6, 1)
    ax.legend()
    st.pyplot(fig)

# ----------------------- DIFFERENTIAL LIFETIME --------------------------
t, ne, nh, nt, pt, nhPTAA, PL, PL_abs, PL_norm = simulate()
mask = (PL > 1e-10)
t_valid = t[mask]
PL_valid = PL[mask]

dlnPL_dt = np.gradient(np.log(PL_valid), t_valid)
tau_diff = -2.0 / dlnPL_dt # need to multiply by 2 for high injection
tau_diff = np.clip(tau_diff, 1e-12, 1e-3)

# Fit
tau1 = tau2 = np.nan
if do_fit:
    try:
        popt, (t_fit, pl_fit) = fit_double_exp(t, PL)
        A1, tau1, A2, tau2 = popt
        PL_model = double_exp(t_fit, *popt)
    except Exception:
        tau1 = tau2 = np.nan
        PL_model = None

# SRH lifetimes (bulk)
Cn = R_pop_e
Cp = R_depop_e
tau_n0 = 1.0 / (Cn * N_t_e)
tau_p0 = 1.0 / (Cp * N_t_e)
tau_SRH_eff = tau_n0 + tau_p0

# Interface lifetime
tau_interface = L / Seff if Seff > 0 else np.nan

# Plot PL + τ_diff
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))

# Left: PL
axL.plot(t, PL, 'k-', lw=1.5, label="PL")
if do_fit and PL_model is not None:
    axL.plot(t_fit, PL_model, 'r--', lw=1, label="fit")
axL.set_xscale('log'); axL.set_yscale('log')
axL.set_xlabel("Time (s)")
axL.set_ylabel("Normalized PL")
axL.set_ylim(1e-6, 1)
axL.legend()
axL.grid(True, which="both", ls="--", lw=0.5)

# Right: τ_diff
axR.plot(t_valid, tau_diff, 'b-', lw=1.2)
axR.set_xscale('log'); axR.set_yscale('log')
axR.set_xlabel("Time (s)")
axR.set_ylabel("τ_diff (s)")
axR.grid(True, which="both", ls="--", lw=0.5)

# Dynamic label helper
# Compute and overlay reference lifetimes
# Overlays with dynamic labels
def label_line(y, color, text):
    if not np.isfinite(y): return
    axR.axhline(y, color=color, ls='--', lw=1.2)
    x_pos = t_valid[int(len(t_valid) * 0.7)]
    axR.text(x_pos, y * 1.1, text, color=color, fontsize=8)

# Compute and overlay reference lifetimes
tau_rad = 1.0 / (R_eh * n0)
label_line(tau_rad, 'gray', 'τ_rad')

tau_ext = 1.0 / k_ht
label_line(tau_ext, 'brown', 'τ_ext = 1/k_ht')

label_line(tau_SRH_eff, 'orange', 'τ_SRH,eff')
label_line(tau_interface, 'purple', 'τ_interface = L/S_eff')
if do_fit and not np.isnan(tau1): label_line(tau1, 'green', 'τ1 (fit)')
if do_fit and not np.isnan(tau2): label_line(tau2, 'red', 'τ2 (fit)')


label_line(tau1, 'green', 'τ1 (fit)')
label_line(tau2, 'red', 'τ2 (fit)')

st.pyplot(fig2)

# Export τ_diff data
df_tau = pd.DataFrame({
    "time_s": t_valid,
    "PL_norm": PL_valid,
    "tau_diff_s": tau_diff
})
st.download_button(
    "💾 Download τ_diff and PL",
    df_tau.to_csv(index=False).encode("utf-8"),
    file_name="TRPL_tau_diff.csv",
    mime="text/csv"
)

# ------------------------ DERIVED QUANTITIES ----------------------------
with st.expander("Derived results and steady-state estimates"):
    st.write(f"τ_n0 = {tau_n0:.2e} s (electron)")
    st.write(f"τ_p0 = {tau_p0:.2e} s (hole)")
    st.write(f"τ_SRH,eff ≈ {tau_SRH_eff:.2e} s")
    st.write(f"τ_interface = {tau_interface:.2e} s")
    st.write(f"S_eff = {Seff:.2e} cm/s")
    st.write(f"t_char (PL=1/e) = {t_char:.2e} s")
    st.write(f"n_h,PTAA(char) = {nhPTAA_char:.2e} cm⁻³")

# -------------------- TRAP/CARRIER DYNAMICS EXPANDER -------------------
with st.expander("📉 Trap, carrier, and extraction dynamics"):
    fig3, ax3 = plt.subplots()

    # Left axis: perovskite carrier and trap densities (log)
    ax3.plot(t, ne/ne[0], label="ne/ne0")
    ax3.plot(t, nh/nh[0], label="nh/nh0")
    ax3.plot(t, nt/(N_t_e if N_t_e > 0 else 1), label="nt/N_t_e")
    if use_hole_traps and N_t_h > 0:
        ax3.plot(t, pt/(N_t_h if N_t_h > 0 else 1), label="pt/N_t_h")

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-6, 1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalized perovskite quantities")
    ax3.grid(True, which="both", ls="--", lw=0.5)

    # Right axis: PTAA holes and extraction fraction (linear)
    ax4 = ax3.twinx()
    f_ext = nhPTAA / (nh + nhPTAA + 1e-30)  # fraction of holes in PTAA
    ax4.plot(t, nhPTAA/n0, color='purple', lw=1.4, label="n_h,PTAA / n0")
    ax4.plot(t, f_ext, color='orange', lw=1.2, ls='--',
             label="f_ext = n_h,PTAA / (n_h + n_h,PTAA)")
    ax4.set_ylabel("PTAA-related quantities (linear)")
    ax4.set_ylim(0, 1.05)

    # Combined legend
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    st.pyplot(fig3)



# ---------------------- INFO / REFERENCE PANEL --------------------------
with st.sidebar.expander("📘 Physical meaning, formulas, and reference values"):
    st.markdown(r"""
### Fundamental relations

- Thermal velocity  
  \[
  v_{th} = \sqrt{\frac{3 k_B T}{m^*}}
  \]

- Trapping (capture) coefficient  
  \[
  R_{pop} = \sigma \, v_{th}
  \]

- Detrapping (thermal emission) rate  
  \[
  k_{detrap} = \nu_0 \, e^{-E_t / (k_B T)}
  \]

- SRH hole capture on filled electron traps  
  \[
  R_{depop,e} = \sigma_h^{(\mathrm{trap})} \, v_{th}
  \]

### Typical values

| Quantity | Typical value | Units |
|---------|---------------|-------|
| \(v_{th}\) | 1–2×10⁷ | cm/s |
| σ | 10⁻¹⁶–10⁻¹⁸ | cm² |
| \(E_t\) | 0.3–0.6 | eV |
| \(\nu_0\) | 10¹³ | s⁻¹ |
| \(R_{pop}\) | 10⁻⁹–10⁻¹¹ | cm³/s |
| \(k_{detrap}\) | 10⁰–10⁶ | s⁻¹ |

Smaller σ → slower trapping  
Deeper \(E_t\) → slower detrapping
""")
    st.markdown("---")
    st.markdown("### 🔬 Detrapping-rate explorer")

    T_calc = st.number_input("Temperature for k_detrap plot (K)", 100, 800, int(T), step=50)
    E_vals = np.linspace(0.1, 0.8, 200)
    k_vals = nu0 * np.exp(-E_vals / (k_B * T_calc))

    fig_calc, ax_calc = plt.subplots()
    ax_calc.semilogy(E_vals, k_vals)
    ax_calc.set_xlabel("Trap depth E_t (eV)")
    ax_calc.set_ylabel("k_detrap (s⁻¹)")
    ax_calc.set_title(f"k_detrap vs E_t at T = {T_calc} K")
    ax_calc.grid(True, which="both", ls="--", lw=0.5)
    st.pyplot(fig_calc)

    if use_physical and "E_t_e" in locals():
        k_example = nu0 * np.exp(-E_t_e / (k_B * T))
        st.markdown(f"**Example:** For E_t = {E_t_e:.2f} eV at {T:.0f} K → k_detrap ≈ {k_example:.2e} s⁻¹")
    else:
        st.markdown("Enable physical mode and set E_t_e to see a numerical k_detrap example.")

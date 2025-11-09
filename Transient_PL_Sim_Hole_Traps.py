import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd

st.title("📈 Perovskite Interface Kinetics Visualizer (Extended Model)")

st.markdown("""
Interactive simulation of transient photoluminescence (PL) for a perovskite–PTAA interface,  
based on the kinetic model of **Lee et al. (2024)**, extended with optional **hole traps**  
and physically computed trapping/detrapping rates.

Features:
- Grouped parameter expanders for clarity  
- Compute rates from cross-sections and trap depths  
- Fit PL decays, show trap dynamics  
- Sweep parameters, analyze τ₁, τ₂, and Sₑff trends  
- Export results as CSV
""")

# ---------- Physical constants ----------
k_B = 8.617333e-5  # eV/K
m_e0 = 9.11e-28    # g
m_eff = 0.2 * m_e0
nu0 = 1e13         # s^-1 attempt frequency

def v_thermal(T):
    k_B_erg = 1.380649e-16
    return np.sqrt(3*k_B_erg*T/m_eff)*1e2  # cm/s

# ---------- Sidebar ----------
st.sidebar.header("Model Setup")

with st.sidebar.expander("🌡️ Global parameters"):
    use_physical = st.checkbox("Compute trap rates from cross section & depth", value=True)
    T = st.number_input("Temperature (K)", 50.0, 1000.0, 300.0, step=10.0)
    R_eh = st.number_input("Radiative recombination R_eh (cm³/s)", 1e-30, 1e30, 1e-10, format="%.1e")
    L = st.number_input("Film thickness L (cm)", 1e-9, 1e2, 5e-5, format="%.1e")

# ---- Electron traps ----
with st.sidebar.expander("🧲 Electron traps"):
    N_t_e = st.number_input("Electron trap density N_t_e (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
    if use_physical:
        sigma_e = st.number_input("Capture cross section σ_e (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")
        E_t_e = st.number_input("Trap depth E_t_e (eV below CB)", 0.0, 2.0, 0.4, step=0.05)
        vth = v_thermal(T)
        R_pop_e = sigma_e * vth
        k_detrap_e = nu0 * np.exp(-E_t_e / (k_B*T))
        st.write(f"→ R_pop_e = {R_pop_e:.2e} cm³/s,  k_detrap_e = {k_detrap_e:.2e} s⁻¹")
    else:
        R_pop_e = st.number_input("Electron trapping R_pop_e (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
        k_detrap_e = st.number_input("Electron detrapping k_detrap_e (s⁻¹)", 1e0, 1e30, 1e5, format="%.1e")
    R_depop_e = st.number_input("SRH (hole capture) R_depop_e (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")

# ---- SRH hole capture coefficient (R_depop_e) ----
with st.sidebar.expander("🌀 SRH recombination (hole capture on electron traps)"):
    sigma_h_trap = st.number_input("Hole capture cross section σ_h^(trap) (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")
    R_depop_e = sigma_h_trap * v_thermal(T)
    st.write(f"→ R_depop_e = {R_depop_e:.2e} cm³/s  (from σ_h^(trap) × v_th)")


# ---- Hole traps ----
with st.sidebar.expander("⚡ Hole traps (optional)"):
    use_hole_traps = st.checkbox("Enable hole traps", value=False)
    N_t_h = 0.0; R_pop_h = R_depop_h = k_detrap_h = 0.0
    if use_hole_traps:
        N_t_h = st.number_input("Hole trap density N_t_h (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
        if use_physical:
            sigma_h = st.number_input("Capture cross section σ_h (cm²)", 1e-22, 1e-14, 1e-17, format="%.1e")
            E_t_h = st.number_input("Trap depth E_t_h (eV above VB)", 0.0, 2.0, 0.4, step=0.05)
            R_pop_h = sigma_h * v_thermal(T)
            k_detrap_h = nu0 * np.exp(-E_t_h / (k_B*T))
            st.write(f"→ R_pop_h = {R_pop_h:.2e} cm³/s,  k_detrap_h = {k_detrap_h:.2e} s⁻¹")
        else:
            R_pop_h = st.number_input("Hole trapping R_pop_h (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
            k_detrap_h = st.number_input("Hole detrapping k_detrap_h (s⁻¹)", 1e0, 1e30, 1e5, format="%.1e")
        R_depop_h = st.number_input("SRH (electron capture) R_depop_h (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")

# ---- Extraction & interface ----
with st.sidebar.expander("🔄 Extraction & interface"):
    k_ht = st.number_input("Hole transfer k_ht (s⁻¹)", 1e0, 1e30, 1e8, format="%.1e")
    k_hbt = st.number_input("Back transfer k_hbt (s⁻¹)", 1e0, 1e30, 1e6, format="%.1e")
    R_AI = st.number_input("Interface recombination R_AI (cm³/s)", 1e-30, 1e30, 1e-8, format="%.1e")

# ---- Initial conditions ----
with st.sidebar.expander("🧮 Initial conditions"):
    n0 = st.number_input("Initial carrier density n0 (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
    f_nt0 = st.number_input("Initial electron trap occupancy f_nt0 (0–1)", 0.0, 1.0, 0.0, step=0.05)
    f_pt0 = st.number_input("Initial hole trap occupancy f_pt0 (0–1)", 0.0, 1.0, 0.0, step=0.05)

# ---- Simulation settings ----
with st.sidebar.expander("⚙️ Simulation options"):
    do_fit = st.checkbox("Fit double exponential", value=True)
    normalize_pl = st.checkbox("Normalize PL to unity", value=True)
    log_xaxis = st.checkbox("Use logarithmic time axis", value=True)
    show_Seff = st.checkbox("Show Seff vs parameter (right axis)", value=True)

with st.sidebar.expander("📘 Physical meaning & formulas"):
    st.markdown(r"""
**Thermal velocity**
\[
v_{th} = \sqrt{\frac{3 k_B T}{m^*}}
\]
**Trapping (capture) rate constant**
\[
R_\text{pop} = \sigma \, v_{th}
\]
**Detrapping (thermal emission) rate**
\[
k_\text{detrap} = \nu_0 \, e^{-E_t / k_B T}
\]
where  
- \( \sigma \) – capture cross section (cm²)  
- \( m^* \) – effective mass (~0.2 mₑ for perovskites)  
- \( \nu_0 \) – attempt frequency (~10¹³ s⁻¹)  
- \( E_t \) – trap depth (eV)  
- \( T \) – temperature (K)
""")
    st.info("Tip: smaller σ → slower trapping; deeper E_t → slower detrapping.")

with st.sidebar.expander("📘 Physical meaning, formulas, and reference values"):
    st.markdown(r"""
### Fundamental relations
**Thermal velocity**
\[
v_{th} = \sqrt{\frac{3 k_B T}{m^*}}
\]

**Trapping (capture) coefficient**
\[
R_{pop} = \sigma \, v_{th}
\]

**Detrapping (thermal emission) rate**
\[
k_{detrap} = \nu_0 \, e^{-E_t / (k_B T)}
\]

**Shockley–Read–Hall recombination step**
\[
R_{depop,e} = \sigma_h^{(trap)} \, v_{th,h}
\]
*(hole capture by filled electron traps)*

---

### Typical constants and values
| Quantity | Symbol | Typical value | Units | Notes |
|-----------|---------|----------------|--------|-------|
| Boltzmann constant | k_B | 8.617×10⁻⁵ | eV/K | |
| Effective mass | m* | 0.2 mₑ | – | halide perovskite |
| Attempt frequency | ν₀ | 10¹³ | s⁻¹ | phonon frequency |
| Thermal velocity | v_th | 1–2×10⁷ | cm/s | at 300 K |
| Capture cross section | σ | 10⁻¹⁶–10⁻¹⁸ | cm² | defect-dependent |
| Trap depth | E_t | 0.3–0.6 | eV | below CB or above VB |
| R_pop (σv_th) | – | 10⁻⁹–10⁻¹¹ | cm³/s | trapping coefficient |
| k_detrap (ν₀e⁻ᴱ/ᵏᵀ) | – | 10⁰–10⁶ | s⁻¹ | depends exponentially on E_t |

---

💡 **Intuition:**  
- Large σ → fast trapping  
- Deep E_t → slow detrapping  
- Shallow traps (E_t ≈ 0.3 eV) empty quickly (k_detrap ≈ 10⁶ s⁻¹)  
- Deep traps (E_t ≈ 0.6 eV) are long-lived (k_detrap ≈ 10² s⁻¹)
""")

    # --- Interactive detrapping-rate calculator ---
    st.markdown("---")
    st.markdown("### 🔬 Detrapping rate explorer")
    st.markdown("Compute and visualize how k_detrap varies with trap depth and temperature.")

    T_calc = st.number_input("Temperature for plot (K)", 100, 800, int(T), step=50)
    E_min, E_max = 0.1, 0.8
    E_vals = np.linspace(E_min, E_max, 200)
    k_vals = nu0 * np.exp(-E_vals / (k_B * T_calc))

    fig_calc, ax_calc = plt.subplots()
    ax_calc.semilogy(E_vals, k_vals)
    ax_calc.set_xlabel("Trap depth E_t (eV)")
    ax_calc.set_ylabel("k_detrap (s⁻¹)")
    ax_calc.set_title(f"Thermal emission rate vs. trap depth (T = {T_calc} K)")
    ax_calc.grid(True, which="both", ls="--", lw=0.5)
    st.pyplot(fig_calc)

    k_example = nu0 * np.exp(-E_t_e / (k_B * T))
    st.markdown(f"**Example:** For Eₜ = {E_t_e:.2f} eV at {T:.0f} K → k_detrap = {k_example:.2e} s⁻¹")


# ---------- Time grid ----------
t_eval = np.logspace(-9, -5, 400)
t_span = (t_eval[0], t_eval[-1])

# ---------- Differential equations ----------
def deriv(t, y):
    if use_hole_traps:
        ne, nh, nt, pt, nhPTAA = y
    else:
        ne, nh, nt, nhPTAA = y
        pt = 0.0

    dne = (
        - R_eh * ne * nh
        - R_pop_e * (N_t_e - nt) * ne
        - R_depop_h * pt * ne
        - R_AI * ne * nhPTAA
        + k_detrap_e * nt
    )

    dnh = (
        - R_eh * ne * nh
        - R_depop_e * nt * nh
        - R_pop_h * (N_t_h - pt) * nh
        - k_ht * nh
        + k_hbt * nhPTAA
        + k_detrap_h * pt
    )

    dnt = (
        R_pop_e * (N_t_e - nt) * ne
        - R_depop_e * nt * nh
        - k_detrap_e * nt
    )

    dpt = (
        R_pop_h * (N_t_h - pt) * nh
        - R_depop_h * pt * ne
        - k_detrap_h * pt
    )

    dnhPTAA = k_ht * nh - k_hbt * nhPTAA - R_AI * ne * nhPTAA

    return [dne, dnh, dnt, dpt, dnhPTAA] if use_hole_traps else [dne, dnh, dnt, dnhPTAA]

# ---------- Simulation ----------
def simulate():
    if use_hole_traps:
        y0 = [n0, n0, f_nt0*N_t_e, f_pt0*N_t_h, 0.0]
    else:
        y0 = [n0, n0, f_nt0*N_t_e, 0.0]
    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, method='LSODA')
    if use_hole_traps:
        ne, nh, nt, pt, nhPTAA = sol.y
    else:
        ne, nh, nt, nhPTAA = sol.y
        pt = np.zeros_like(ne)
    PL = R_eh * ne * nh
    if normalize_pl: PL /= PL[0]
    return sol.t, ne, nh, nt, pt, nhPTAA, PL

def compute_Seff():
    t, ne, nh, nt, pt, nhPTAA, PL = simulate()
    idx = np.argmin(np.abs(PL - 1/np.e))
    nhPTAA_char = nhPTAA[idx]
    Seff = R_AI * nhPTAA_char * L
    return Seff, t[idx], nhPTAA_char, (t, PL)

# ---------- Fit ----------
def double_exp(t, A1, tau1, A2, tau2):
    return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)

def fit_double_exp(t, pl):
    mask = pl > 1e-6
    t_fit, pl_fit = t[mask], pl[mask]
    p0 = [0.7, 5e-9, 0.3, 1e-7]
    bounds = ([-np.inf, 1e-10, -np.inf, 1e-10], [np.inf, 1e-3, np.inf, 1e-3])
    popt, _ = curve_fit(double_exp, t_fit, pl_fit, p0=p0, bounds=bounds, maxfev=20000)
    return popt, (t_fit, pl_fit)

# ---------- Main simulation and plot ----------
Seff, t_char, nhPTAA_char, (t, PL) = compute_Seff()

fig, ax = plt.subplots()
ax.plot(t, PL, 'k.', markersize=3, label="Simulated PL")
if log_xaxis: ax.set_xscale('log'); ax.set_yscale('log')
else:          ax.set_xscale('linear'); ax.set_yscale('log')

if do_fit:
    try:
        popt, (t_fit, pl_fit) = fit_double_exp(t, PL)
        A1, tau1, A2, tau2 = popt
        PL_model = double_exp(t_fit, *popt)
        ax.plot(t_fit, PL_model, 'r-', label="Double-exp fit")
        st.subheader("Fitted parameters")
        st.write(f"tau1 = {tau1:.3e} s (fast ≈ extraction)")
        st.write(f"tau2 = {tau2:.3e} s (slow ≈ bulk/interface)")
        st.write(f"A1 = {A1:.2g}, A2 = {A2:.2g}")
    except Exception as e:
        st.warning(f"Fit failed: {e}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized PL")
ax.set_ylim(1e-6, 1)
ax.legend()
st.pyplot(fig)

#----------------- calc and plot diff lifetime
# ---------- Differential lifetime calculation ----------
# Compute d(ln(PL))/dt and tau_diff = -1 / (dlnPL/dt)
t, ne, nh, nt, pt, nhPTAA, PL = simulate()
mask = (PL > 1e-10)  # avoid numerical noise at tail
t_valid = t[mask]
PL_valid = PL[mask]

# Numerical derivative using np.gradient on log(t) grid
dlnPL_dt = np.gradient(np.log(PL_valid), t_valid)
tau_diff = -1.0 / dlnPL_dt

# Smooth out numerical noise (optional)
tau_diff = np.clip(tau_diff, 1e-12, 1e-3)

# Create side-by-side plots
fig3, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))

# Left: PL
axL.plot(t, PL, 'k-', lw=1.5, label="Simulated PL")
if do_fit:
    axL.plot(t_fit, PL_model, 'r--', lw=1, label="Double-exp fit")
axL.set_xscale('log'); axL.set_yscale('log')
axL.set_xlabel("Time (s)")
axL.set_ylabel("Normalized PL")
axL.set_ylim(1e-6, 1)
axL.legend()
axL.grid(True, which="both", ls="--", lw=0.5)

# Right: differential lifetime
axR.plot(t_valid, tau_diff, 'b-')
axR.set_xscale('log')
axR.set_yscale('log')
axR.set_xlabel("Time (s)")
axR.set_ylabel("τ_diff (s)")
axR.set_title("Differential lifetime τ_diff(t)")
axR.grid(True, which="both", ls="--", lw=0.5)

#st.pyplot(fig3)

with st.expander("📈 Show differential lifetime τ_diff(t)"):
    st.pyplot(fig3)



# ---------- Trap dynamics expander ----------
with st.expander("📉 Show trap and carrier dynamics over time"):
    t, ne, nh, nt, pt, nhPTAA, PL = simulate()
    fig2, ax2 = plt.subplots()
    ax2.plot(t, ne / ne[0], label="ne / ne0")
    ax2.plot(t, nh / nh[0], label="nh / nh0")
    ax2.plot(t, nt / (N_t_e if N_t_e > 0 else 1), label="nt / N_t_e")
    if use_hole_traps:
        ax2.plot(t, pt / (N_t_h if N_t_h > 0 else 1), label="pt / N_t_h")
    ax2.set_xscale('log')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized density")
    ax2.set_ylim(1e-6, 1.1)
    ax2.legend()
    st.pyplot(fig2)

st.subheader("Derived quantities")
st.write(f"Seff = {Seff:.2e} cm/s")
st.write(f"Characteristic time (PL = 1/e): {t_char:.2e} s")
st.write(f"n_h,PTAA(char) = {nhPTAA_char:.2e} cm⁻³")

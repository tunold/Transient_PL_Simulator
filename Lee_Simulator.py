import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ---------- Title ----------
st.title("📈 Perovskite Interface Kinetics Visualizer")

st.markdown("""
Simulate transient photoluminescence (PL) for a perovskite–PTAA interface using
the kinetic model of **Lee et al. (2024)**.  
You can adjust kinetic parameters, fit a double-exponential, and even perform parameter sweeps.
""")

# ---------- Sidebar ----------
st.sidebar.header("Model Parameters")

R_eh     = st.sidebar.number_input("Radiative recombination R_eh (cm³/s)", 1e-12, 1e-8, 1e-10, format="%.1e")
R_pop    = st.sidebar.number_input("Trapping rate R_pop (cm³/s)", 1e-10, 1e-6, 5e-9, format="%.1e")
R_depop  = st.sidebar.number_input("Trap-assisted recomb. R_depop (cm³/s)", 1e-10, 1e-6, 5e-9, format="%.1e")
k_detrap = st.sidebar.number_input("Detrapping rate k_detrap (s⁻¹)", 1e3, 1e7, 1e5, format="%.1e")
N_t      = st.sidebar.number_input("Trap density N_t (cm⁻³)", 1e13, 1e17, 1e15, format="%.1e")

k_ht     = st.sidebar.number_input("Hole transfer rate k_ht (s⁻¹)", 1e6, 1e10, 1e8, format="%.1e")
k_hbt    = st.sidebar.number_input("Back-transfer rate k_hbt (s⁻¹)", 1e5, 1e9, 1e6, format="%.1e")
R_AI     = st.sidebar.number_input("Across-interface recomb. R_AI (cm³/s)", 1e-12, 1e-4, 1e-8, format="%.1e")

n0       = st.sidebar.number_input("Initial carrier density n0 (cm⁻³)", 1e13, 1e17, 1e15, format="%.1e")
L        = st.sidebar.number_input("Film thickness L (cm)", 1e-6, 1e-4, 5e-5, format="%.1e")

do_fit   = st.sidebar.checkbox("Fit double exponential", value=True)
show_res = st.sidebar.checkbox("Show residuals", value=False)
do_sweep = st.sidebar.checkbox("Enable parameter sweep", value=False)

# ---------- Simulation time grid ----------
t_eval = np.logspace(-9, -5, 400)
t_span = (t_eval[0], t_eval[-1])

# ---------- ODE system ----------
def deriv(t, y, params):
    ne, nh, nt, nhPTAA = y
    k_ht, k_hbt, R_AI = params
    dne = -R_eh*ne*nh - R_pop*(N_t-nt)*ne - R_AI*ne*nhPTAA + k_detrap*nt
    dnh = -R_eh*ne*nh - R_depop*nt*nh - k_ht*nh + k_hbt*nhPTAA
    dnt =  R_pop*(N_t-nt)*ne - R_depop*nt*nh - k_detrap*nt
    dnhPTAA = k_ht*nh - k_hbt*nhPTAA - R_AI*ne*nhPTAA
    return [dne, dnh, dnt, dnhPTAA]

def simulate_populations(k_ht, k_hbt, R_AI):
    y0 = [n0, n0, 0.0, 0.0]
    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval,
                    args=([k_ht, k_hbt, R_AI],), method='LSODA')
    ne, nh, nt, nhPTAA = sol.y
    PL = R_eh * ne * nh
    PL /= PL[0]
    return sol.t, ne, nh, nt, nhPTAA, PL

def compute_S_eff(k_ht, k_hbt, R_AI):
    t, ne, nh, nt, nhPTAA, PL = simulate_populations(k_ht, k_hbt, R_AI)
    idx = np.argmin(np.abs(PL - 1/np.e))
    nhPTAA_char = nhPTAA[idx]
    S_eff = R_AI * nhPTAA_char * L
    return S_eff, t[idx], nhPTAA_char, (t, PL)

# ---------- Double exponential fitting ----------
def double_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2)

def fit_double_exp(t, pl, t_min=None, t_max=None, min_rel=1e-6):
    if t_min is None: t_min = t[0]
    if t_max is None: t_max = t[-1]
    mask = (t >= t_min) & (t <= t_max) & (pl > min_rel)
    t_fit, pl_fit = t[mask], pl[mask]
    A1_0, A2_0 = 0.7, 0.3
    tau1_0, tau2_0 = 5e-9, 1e-7
    p0 = [A1_0, tau1_0, A2_0, tau2_0]
    bounds = ([-np.inf, 1e-10, -np.inf, 1e-10],
              [np.inf, 1e-3, np.inf, 1e-3])
    popt, _ = curve_fit(double_exp, t_fit, pl_fit, p0=p0,
                        bounds=bounds, maxfev=20000)
    return popt, (t_fit, pl_fit)

# ---------- Main simulation ----------
def plot_pl_curve(k_ht, k_hbt, R_AI, label=None, color=None):
    S_eff, t_char, nhPTAA_char, (t, PL) = compute_S_eff(k_ht, k_hbt, R_AI)
    plt.loglog(t, PL, '.', markersize=3, label=label or f"k_ht={k_ht:.1e}", color=color)
    return S_eff, t_char, nhPTAA_char, (t, PL)

# ---------- Plot section ----------
fig, ax = plt.subplots()

if not do_sweep:
    # --- single simulation ---
    S_eff, t_char, nhPTAA_char, (t, PL) = compute_S_eff(k_ht, k_hbt, R_AI)
    ax.loglog(t, PL, 'k.', markersize=3, label='Simulated PL')
    if do_fit:
        try:
            popt, (t_fit, pl_fit) = fit_double_exp(t, PL)
            A1, tau1, A2, tau2 = popt
            PL_model = double_exp(t_fit, *popt)
            ax.loglog(t_fit, PL_model, 'r-', label='Double-exp fit')
            st.subheader("Fitted parameters")
            st.write(f"**τ₁** = {tau1:.3e} s (fast) — ≈ extraction / k_ht")
            st.write(f"**τ₂** = {tau2:.3e} s (slow) — ≈ bulk/interface recomb")
            st.write(f"A₁ = {A1:.2g}, A₂ = {A2:.2g}")
            if show_res:
                res = pl_fit - PL_model
                fig2, ax2 = plt.subplots()
                ax2.semilogx(t_fit, res)
                ax2.axhline(0, color='k', lw=0.8)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Residual (sim - fit)")
                ax2.set_title("Fit residuals")
                st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Fit failed: {e}")

    ax.legend()
    st.pyplot(fig)
    st.subheader("Derived quantities")
    st.write(f"**S_eff** = {S_eff:.2e} cm/s")
    st.write(f"Characteristic time (PL=1/e): {t_char:.2e} s")
    st.write(f"nₕ,PTAA(char) = {nhPTAA_char:.2e} cm⁻³")

else:
    # --- sweep mode ---
    sweep_param = st.sidebar.selectbox("Sweep parameter", ["k_ht", "k_hbt", "R_AI"])
    sweep_min = st.sidebar.number_input("Sweep min (decade)", 1e2, 1e17, 1e6, format="%.1e")
    sweep_max = st.sidebar.number_input("Sweep max (decade)", 1e2, 1e17, 1e9, format="%.1e")
    n_points   = st.sidebar.slider("Number of sweep points", 2, 10, 5)

    values = np.logspace(np.log10(sweep_min), np.log10(sweep_max), n_points)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    S_list = []

    for val, c in zip(values, colors):
        if sweep_param == "k_ht":
            S_eff, *_ = plot_pl_curve(val, k_hbt, R_AI, label=f"k_ht={val:.0e}", color=c)
        elif sweep_param == "k_hbt":
            S_eff, *_ = plot_pl_curve(k_ht, val, R_AI, label=f"k_hbt={val:.0e}", color=c)
        else:
            S_eff, *_ = plot_pl_curve(k_ht, k_hbt, val, label=f"R_AI={val:.0e}", color=c)
        S_list.append(S_eff)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized PL")
    ax.set_ylim(1e-6, 1)
    ax.legend()
    st.pyplot(fig)

    st.subheader(f"S_eff values for swept {sweep_param}")
    for val, S in zip(values, S_list):
        st.write(f"{sweep_param} = {val:.1e} → S_eff = {S:.2e} cm/s")

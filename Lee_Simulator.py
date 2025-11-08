import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd

# ---------- Title ----------
st.title("📈 Perovskite Interface Kinetics Visualizer")

st.markdown("""
Interactive simulation of transient photoluminescence (PL) for a perovskite–PTAA interface  
based on the kinetic model of **Lee et al. (2024)**.

You can:
- Adjust kinetic & defect parameters over very wide ranges  
- Fit a double-exponential decay  
- Sweep any parameter (including *n₀*) and view tau₁, tau₂, and Seff trends  
- Normalize PL curves for comparison  
- Download sweep results as CSV (ASCII headers)
""")

# ---------- Sidebar ----------
st.sidebar.header("Model Parameters")

R_eh     = st.sidebar.number_input("Radiative recombination R_eh (cm³/s)", 1e-30, 1e30, 1e-10, format="%.1e")
R_pop    = st.sidebar.number_input("Trapping rate R_pop (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
R_depop  = st.sidebar.number_input("Trap-assisted recomb. R_depop (cm³/s)", 1e-30, 1e30, 5e-9, format="%.1e")
k_detrap = st.sidebar.number_input("Detrapping rate k_detrap (s⁻¹)", 1e0, 1e30, 1e5, format="%.1e")
N_t      = st.sidebar.number_input("Trap density N_t (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")

k_ht     = st.sidebar.number_input("Hole transfer k_ht (s⁻¹)", 1e0, 1e30, 1e8, format="%.1e")
k_hbt    = st.sidebar.number_input("Back transfer k_hbt (s⁻¹)", 1e0, 1e30, 1e6, format="%.1e")
R_AI     = st.sidebar.number_input("Interface recomb. R_AI (cm³/s)", 1e-30, 1e30, 1e-8, format="%.1e")

n0       = st.sidebar.number_input("Initial carrier density n0 (cm⁻³)", 1e0, 1e30, 1e15, format="%.1e")
L        = st.sidebar.number_input("Film thickness L (cm)", 1e-9, 1e2, 5e-5, format="%.1e")

# Toggles
do_fit       = st.sidebar.checkbox("Fit double exponential", value=True)
show_res     = st.sidebar.checkbox("Show residuals", value=False)
do_sweep     = st.sidebar.checkbox("Enable parameter sweep", value=False)
log_xaxis    = st.sidebar.checkbox("Use logarithmic time axis", value=True)
normalize_pl = st.sidebar.checkbox("Normalize PL curves to same max", value=True)
show_Seff    = st.sidebar.checkbox("Show Seff vs parameter (right axis)", value=True)

# ---------- Time grid ----------
t_eval = np.logspace(-9, -5, 400)
t_span = (t_eval[0], t_eval[-1])

# ---------- Kinetic model ----------
def deriv(t, y, params):
    ne, nh, nt, nhPTAA = y
    k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, R_eh = params
    dne = -R_eh*ne*nh - R_pop*(N_t-nt)*ne - R_AI*ne*nhPTAA + k_detrap*nt
    dnh = -R_eh*ne*nh - R_depop*nt*nh - k_ht*nh + k_hbt*nhPTAA
    dnt =  R_pop*(N_t-nt)*ne - R_depop*nt*nh - k_detrap*nt
    dnhPTAA = k_ht*nh - k_hbt*nhPTAA - R_AI*ne*nhPTAA
    return [dne, dnh, dnt, dnhPTAA]

def simulate_populations(k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, n0, R_eh):
    y0 = [n0, n0, 0.0, 0.0]
    sol = solve_ivp(
        deriv, t_span, y0, t_eval=t_eval,
        args=([k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, R_eh],),
        method='LSODA'
    )
    ne, nh, nt, nhPTAA = sol.y
    PL = R_eh * ne * nh
    PL /= PL[0]
    return sol.t, ne, nh, nt, nhPTAA, PL

def compute_Seff(k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, n0, R_eh):
    t, ne, nh, nt, nhPTAA, PL = simulate_populations(k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, n0, R_eh)
    idx = np.argmin(np.abs(PL - 1/np.e))
    nhPTAA_char = nhPTAA[idx]
    Seff = R_AI * nhPTAA_char * L
    return Seff, t[idx], nhPTAA_char, (t, PL)

# ---------- Fitting ----------
def double_exp(t, A1, tau1, A2, tau2):
    return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)

def fit_double_exp(t, pl, min_rel=1e-6):
    mask = pl > min_rel
    t_fit, pl_fit = t[mask], pl[mask]
    A1_0, A2_0 = 0.7, 0.3
    tau1_0, tau2_0 = 5e-9, 1e-7
    p0 = [A1_0, tau1_0, A2_0, tau2_0]
    bounds = ([-np.inf, 1e-10, -np.inf, 1e-10],
              [np.inf, 1e-3, np.inf, 1e-3])
    popt, _ = curve_fit(double_exp, t_fit, pl_fit, p0=p0,
                        bounds=bounds, maxfev=20000)
    return popt, (t_fit, pl_fit)

# ---------- Plot ----------
if not do_sweep:
    fig, ax = plt.subplots()
    Seff, t_char, nhPTAA_char, (t, PL) = compute_Seff(k_ht, k_hbt, R_AI, R_pop, R_depop, k_detrap, N_t, n0, R_eh)
    ax.plot(t, PL, 'k.', markersize=3, label="Simulated PL")

    if log_xaxis:
        ax.set_xscale('log'); ax.set_yscale('log')
    else:
        ax.set_xscale('linear'); ax.set_yscale('log')

    if do_fit:
        try:
            popt, (t_fit, pl_fit) = fit_double_exp(t, PL)
            A1, tau1, A2, tau2 = popt
            PL_model = double_exp(t_fit, *popt)
            ax.plot(t_fit, PL_model, 'r-', label="Double-exp fit")
            st.subheader("Fitted parameters")
            st.write(f"**tau1** = {tau1:.3e} s (fast ≈ 1/k_ht)")
            st.write(f"**tau2** = {tau2:.3e} s (slow ≈ bulk/interface)")
            st.write(f"A1 = {A1:.2g}, A2 = {A2:.2g}")
        except Exception as e:
            st.warning(f"Fit failed: {e}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized PL")
    ax.set_ylim(1e-6, 1)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Derived quantities")
    st.write(f"**Seff** = {Seff:.2e} cm/s")
    st.write(f"Characteristic time (PL = 1/e): {t_char:.2e} s")
    st.write(f"n_h,PTAA(char) = {nhPTAA_char:.2e} cm⁻³")

else:
    # ---- Sweep mode ----
    sweep_param = st.sidebar.selectbox(
        "Sweep parameter", ["k_ht", "k_hbt", "R_AI", "R_pop", "R_depop", "N_t", "n0"])
    sweep_min = st.sidebar.number_input("Sweep min value", min_value=1e-30, max_value=1e30, value=1e6, format="%.1e")
    sweep_max = st.sidebar.number_input("Sweep max value", min_value=1e-30, max_value=1e30, value=1e9, format="%.1e")
    n_points  = st.sidebar.slider("Number of sweep points", 3, 15, 6)

    values = np.logspace(np.log10(sweep_min), np.log10(sweep_max), n_points)
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 1, n_points))
    results = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for val, c in zip(values, colors):
        params = dict(k_ht=k_ht, k_hbt=k_hbt, R_AI=R_AI,
                      R_pop=R_pop, R_depop=R_depop, N_t=N_t, n0=n0)
        params[sweep_param] = val
        Seff, t_char, nhPTAA_char, (t, PL) = compute_Seff(
            params["k_ht"], params["k_hbt"], params["R_AI"],
            params["R_pop"], params["R_depop"], k_detrap,
            params["N_t"], params["n0"], R_eh)
        if normalize_pl: PL /= np.max(PL)
        ax1.plot(t, PL, '.', color=c, markersize=3, label=f"{sweep_param}={val:.1e}")

        if log_xaxis:
            ax1.set_xscale('log'); ax1.set_yscale('log')
        else:
            ax1.set_xscale('linear'); ax1.set_yscale('log')

        tau1 = tau2 = np.nan
        if do_fit:
            try:
                popt, _ = fit_double_exp(t, PL)
                _, tau1, _, tau2 = popt
            except Exception:
                pass
        results.append({
            "parameter": sweep_param,
            "value": val,
            "Seff_cm_per_s": Seff,
            "tau1_s": tau1,
            "tau2_s": tau2,
            "color": c
        })

    # Left: PL curves
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalized PL")
    ax1.set_ylim(1e-6, 1)
    ax1.legend(fontsize=7)

    # Right: tau1, tau2, Seff
    df = pd.DataFrame(results)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    for i, row in df.iterrows():
        ax2.plot(row["value"], row["tau1_s"], 'o', color=row["color"])
        ax2.plot(row["value"], row["tau2_s"], 's', color=row["color"])
    ax2.plot(df["value"], df["tau1_s"], 'k--', alpha=0.4)
    ax2.plot(df["value"], df["tau2_s"], 'k--', alpha=0.4)
    if show_Seff:
        ax2b = ax2.twinx()
        ax2b.set_yscale('log')
        ax2b.plot(df["value"], df["Seff_cm_per_s"], '^-', color='gray', alpha=0.6, label="Seff")
        ax2b.set_ylabel("Seff (cm/s)", color='gray')
    ax2.set_xlabel(sweep_param)
    ax2.set_ylabel("Lifetime (s)")
    ax2.grid(True, which="both", ls="--", lw=0.5)
    st.pyplot(fig)

    # Table + download (ASCII headers)
    st.subheader("Sweep results")
    st.dataframe(df[["value","Seff_cm_per_s","tau1_s","tau2_s"]]
        .style.format({"value":"{:.1e}","Seff_cm_per_s":"{:.2e}","tau1_s":"{:.2e}","tau2_s":"{:.2e}"}))

    csv = df[["parameter","value","Seff_cm_per_s","tau1_s","tau2_s"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download results as CSV",
        csv,
        file_name=f"sweep_{sweep_param}.csv",
        mime="text/csv"
    )

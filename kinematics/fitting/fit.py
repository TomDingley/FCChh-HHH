import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uproot
from tools import numeric, build_mask_from_selection
from config import SIGNAL, BACKGROUNDS, SELECTION, N_BINS_1D, XLIM_MAP, LUMINOSITY_PB
from scipy.optimize import minimize_scalar
from aesthetics import process_labels, process_colours, LABEL_MAP, banner, banner_heatmaps
from tools import get_weights

def compute_asimov_significance(
    var: str,
    files: dict[str, Path],
    k3: float = 1.0,
    k4: float = 1.0,
    outdir: Path | None = None,
    channel: str = "Combined"
) -> float:
    """Compute the Asimov discovery significance for given k3, k4 using Poisson LLR and optionally plot the distributions."""

    # --- Load signal ---
    with uproot.open(files[SIGNAL]) as f_sig:
        t_sig = f_sig["events"]
        selection = SELECTION[channel]
        m_sig = build_mask_from_selection(t_sig, selection)
        sig = numeric(t_sig[var].array(library="ak")[m_sig])
        if sig.size == 0:
            print(f"[!] No signal entries for {var}")
            return np.nan

        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
        else:
            xmin, xmax = sig.min(), sig.max()
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)

        # Weights
        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {
            key: numeric(t_sig[key].array(library="ak")[m_sig])
            for key in basis_keys
        }
        weights = get_weights(k3, k4, weight_dict)
        xsec_weight = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig]) * LUMINOSITY_PB
        weights *= xsec_weight
        h_sig, _ = np.histogram(sig, bins=edges, weights=weights)

    # --- Load background ---
    bkg_total = np.zeros_like(h_sig)
    for proc in BACKGROUNDS:
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            t_bkg = f_bkg["events"]
            arr_raw = t_bkg[var].array(library="ak")
            m_bkg = build_mask_from_selection(t_bkg, selection)
            arr = numeric(arr_raw[m_bkg])
            weight_bkg = numeric(t_bkg["weight_xsec"].array(library="ak")[m_bkg]) * LUMINOSITY_PB
            if arr.size == 0:
                continue
            h_bkg, _ = np.histogram(arr, bins=edges, weights=weight_bkg)
            bkg_total += h_bkg

    # --- Compute Asimov significance ---
    s = h_sig
    b = bkg_total
    print(f"[✓] Signal histogram sum: {s.sum():.2f}, Background histogram sum: {b.sum():.2f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = b > 0
        q0 = np.zeros_like(b)
        q0[mask] = 2 * ((s[mask] + b[mask]) * np.log1p(s[mask] / b[mask]) - s[mask])
        Z = np.sqrt(np.sum(q0))

    print(f"[✓] Asimov significance = {Z:.2f}σ (k3={k3}, k4={k4})")

    # --- Plot signal + background stack ---
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        ax.bar(bin_centers, b, width=np.diff(edges), align="center", color="grey", alpha=0.6, label="Background")
        ax.step(bin_centers, s + b, where="mid", color="black", lw=1.2, label="Signal + Background")
        ax.step(bin_centers, s, where="mid", color=process_colours[SIGNAL], lw=2, linestyle="--",
                label=f"Signal (k3={k3:.1f}, k4={k4:.1f})")

        ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
        ax.set_ylabel("Events / bin", loc="top")
        if (s + b).ptp() > 1e3:
            ax.set_yscale("log")
        ax.legend(frameon=False, loc="upper right")
        banner(ax, fr"Asimov $Z$ = {Z:.2f}$\sigma$")

        outpath = outdir / f"asimov_dist_{var}_k3_{k3:.1f}_k4_{k4:.1f}.pdf"
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Distribution plot saved to {outpath}")

    return Z

def compute_exclusion_significance_from_SM(
    var: str,                         # kept for API compatibility; not used
    files: dict[str, Path],
    k3: float,
    k4: float,
    outdir: Path | None = None,       # ignored for yield-only; kept for API compatibility
    channel: str = "Combined",
) -> float:
    """
    Yield-only Asimov exclusion significance for (k3,k4), assuming SM is true.

    Definition:
      - Asimov "data" (n) = s_SM + b  (SM signal + total background)
      - Prediction under NP hypothesis (nu) = s_NP + b
      - Z = sqrt( 2 * [ n*ln(n/nu) - (n - nu) ] ), with 0 if n==0 or nu==0

    Notes:
      - Uses the same event selection SELECTION[channel].
      - Uses get_weights(k3,k4, weight_dict) to morph NP signal weights.
      - Multiplies by weight_xsec * LUMINOSITY_PB.
      - No histogramming; pure counting.
    """
    import numpy as np
    import uproot
    import awkward as ak

    selection = SELECTION[channel]

    # --- Signal (read once, reuse arrays for all weight combinations) ---
    with uproot.open(files[SIGNAL]) as f_sig:
        t_sig = f_sig["events"]
        m_sig = build_mask_from_selection(t_sig, selection)

        # SM-basis weights available in the tree (same keys as your original)
        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1",
        ]
        weight_dict = {
            key: numeric(t_sig[key].array(library="ak")[m_sig]) for key in basis_keys
        }

        # base xsec weight
        w_xsec_sig = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig]) * LUMINOSITY_PB

        # SM and NP total signal yields
        w_SM = get_weights(1.0, 1.0, weight_dict) * w_xsec_sig
        w_NP = get_weights(k3,  k4,  weight_dict) * w_xsec_sig
        w_point = get_weights(1,  -2.441,  weight_dict) 
        w_checkSMpoint = get_weights(1,  1,  weight_dict) 

        ratio = w_point / w_checkSMpoint

        # arithmetic mean over events
        avg_ratio = np.mean(ratio)

        print("Extracting xSM weight for (1, -2.44): ", avg_ratio)
        s_SM = float(np.sum(w_SM)) if w_SM.size else 0.0
        s_NP = float(np.sum(w_NP)) if w_NP.size else 0.0

    # --- Backgrounds (sum over processes) ---
    B = 0.0
    for proc in BACKGROUNDS:
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg:
                continue
            t_bkg = f_bkg["events"]
            m_bkg = build_mask_from_selection(t_bkg, selection)
            w_bkg = numeric(t_bkg["weight_xsec"].array(library="ak")[m_bkg]) * LUMINOSITY_PB
            if w_bkg.size:
                B += float(np.sum(w_bkg))

    # --- Yield-only Asimov significance (SM Asimov data vs NP prediction) ---
    n  = s_SM + B            # Asimov data under SM
    nu = s_NP + B            # expectation under NP hypothesis

    if n <= 0 or nu <= 0:
        Z = 0.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            Z2 = 2.0 * (n * np.log(n / nu) - (n - nu))
            Z = float(np.sqrt(max(Z2, 0.0)))

    print(f"[✓] Yield-only exclusion vs SM: Z = {Z:.2f}σ  (k3={k3}, k4={k4})")
    return Z


def scan_k3k4_limits(var: str, files: dict[str, Path], outdir: Path, k3_range=(0.0, 2.0), k4_range=(0.0, 2.0), nsteps=5, channel="Combined"):
    k3_vals = np.linspace(*k3_range, nsteps)
    k4_vals = np.linspace(*k4_range, nsteps)

    mu_bestfit = np.zeros((nsteps, nsteps))
    mu_95cl = np.zeros((nsteps, nsteps))
    sig_asimov = np.zeros((nsteps, nsteps))
    sig_basimov = np.zeros((nsteps, nsteps))
    
    for i, k3 in enumerate(k3_vals):
        for j, k4 in enumerate(k4_vals):
            print(f"[ ] Scanning (k3={k3:.2f}, k4={k4:.2f})...")
            #mu_hat, mu_95 = fit_signal_normalisation(var, files, outdir, k3=k3, k4=k4)
            #Z = compute_asimov_significance(var, files, k3=k3, k4=k4, outdir=outdir)
            #Z_bkg = compute_bonly_asimov_significance(var, files, k3=k3, k4=k4, outdir=outdir)
            Z_excl = compute_exclusion_significance_from_SM(var, files, k3=k3, k4=k4, outdir=outdir, channel=channel)
            #mu_bestfit[i, j] = mu_hat
            #mu_95cl[i, j] = mu_95
            sig_asimov[i, j] = Z_excl
            
            #sig_basimov[i,j] = Z_bkg

    outdir.mkdir(parents=True, exist_ok=True)

    # Save μ grid
    #out_limit = outdir / f"fit/limit_scan_{var}.npz"
    #np.savez(out_limit, mu=mu_bestfit, limit=mu_95cl, k3=k3_vals, k4=k4_vals)
    #print(f"[✓] Saved 95% CL limit grid: {out_limit}")

    # Save significance grid separately
    #outdir.mkdir("fit", exist_ok=True) 
    out_signif = outdir / f"fit/{channel}_significance_scan_{var}.npz"
    out_signif_bonly = outdir / f"fit/{channel}_significance_scan_bonly_{var}.npz"

    np.savez(out_signif, Z=sig_asimov, k3=k3_vals, k4=k4_vals)

    #print(f"[✓] Saved Asimov significance grid: {out_signif}")
    #np.savez(out_signif_bonly, Z=sig_basimov, k3=k3_vals, k4=k4_vals)
    print(f"[✓] Saved B-only Asimov significance grid: {out_signif_bonly}")



def plot_k3k4_limit_contours(npz_file: Path, outdir: Path, channel: str):
    sig_file = outdir / f"fit/{channel}_significance_scan_bonly_m_hhh_vis.npz"
    if not sig_file.exists():
        print(f"[!] Skipping significance plot: {sig_file} not found.")
        return

    sig_data = np.load(sig_file)
    k3_vals = sig_data["k3"]
    k4_vals = sig_data["k4"]
    Z_vals  = sig_data["Z"]
    K3, K4  = np.meshgrid(k3_vals, k4_vals, indexing="ij")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Fixed color limits (clip above 10)
    vmin, vmax = 0, 20

    # Filled contours (values > vmax get clipped but will be indicated by the colorbar arrow)
    levels = np.linspace(vmin, vmax, 50)
    c = ax.contourf(
        K3, K4, Z_vals,
        levels=levels, cmap='coolwarm_r',
        origin="lower", vmin=vmin, vmax=vmax, extend="max"
    )

    # Colorbar with an arrow above vmax, unconditionally
    cbar = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label(r"Asimov significance $Z$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)


    # Contour lines
    levels = [2, 5]
    contours = ax.contour(K3, K4, Z_vals, levels=levels, colors="white", linewidths=1.8)
    ax.clabel(contours, fmt={l: f"{l}$\\sigma$" for l in levels}, fontsize=11, inline=True)

    # Axes & cosmetics
    ax.set_xlabel(r"$k_3$", fontsize=13)
    ax.set_ylabel(r"$k_4$", fontsize=13)
    ax.set_xlim(k3_vals[0], k3_vals[-1])
    ax.set_ylim(k4_vals[0], k4_vals[-1])
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(r"B-only Asimov Significance in $k_3$-$k_4$ Plane", fontsize=14, weight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.5)

    outpath_sig = outdir / f"fit/{channel}_significance_contour_bonly.pdf"
    fig.tight_layout()
    fig.savefig(outpath_sig, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Significance plot saved to {outpath_sig}")
    
    # --- Now plot significance heatmap if available ---
    sig_file = outdir / f"fit/significance_scan_m_hhh_vis.npz"
    if not sig_file.exists():
        print(f"[!] Skipping significance plot: {sig_file} not found.")
        return

    sig_data = np.load(sig_file)
    k3_vals = sig_data["k3"]
    k4_vals = sig_data["k4"]
    Z_vals = sig_data["Z"]
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")


    fig, ax = plt.subplots(figsize=(7, 6))

    c = ax.contourf(K3, K4, Z_vals, levels=50, cmap="coolwarm_r")
    #fig.colorbar(c, ax=ax, label="Asimov significance $Z$")

    levels = np.linspace(vmin, vmax, 50)
    c = ax.contourf(
        K3, K4, Z_vals,
        levels=levels, cmap='coolwarm',
        origin="lower", vmin=vmin, vmax=vmax, extend="max"
    )

    # Colorbar with an arrow above vmax, unconditionally
    cbar = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label(r"Asimov significance $Z$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)


    # Contour lines
    levels = [2, 5]
    contours = ax.contour(K3, K4, Z_vals, levels=levels, colors="white", linewidths=1.8)
    ax.clabel(contours, fmt={l: f"{l}$\\sigma$" for l in levels}, fontsize=11, inline=True)

    ax.set_xlabel("$k_3$")
    ax.set_ylabel("$k_4$")
    ax.set_xlim(k3_vals[0], k3_vals[-1])
    ax.set_ylim(k4_vals[0], k4_vals[-1])
    ax.set_title("Asimov Significance in $k_3$-$k_4$ Plane")
    ax.grid(True, linestyle=":")

    outpath_sig = outdir / f"fit/{channel}_significance_contour.pdf"
    fig.savefig(outpath_sig, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Significance plot saved to {outpath_sig}")
    
def plot_k3k4_limit_contours_comparison(
    outdir: Path,
    channels=("LepHad", "LepHad_resolved", "LepHad_1BB", "LepHad_2BB"),
    npz_pattern="fit/{channel}_significance_scan_m_hhh_vis_LR.npz",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {
        "LepHad":   "tab:blue",
        "HadHad":   "tab:green",
        "Combined": "black",
        "LepHad_1BB":   "tab:blue",
        "HadHad_1BB":   "pink",
        "LepHad_2BB":   "tab:red",
        "HadHad_2BB":   "tab:red",

    }
    
    
    linestyles = {
        "HadHad":   "--",
        "HadHad_1BB":   "--",
        "HadHad_2BB":   "--",
        "HadHad_resolved":   "--",
        
        "LepHad":   "-.",
        "LepHad_resolved":   "-.",
        "LepHad_1BB":   "-.",
        "LepHad_2BB":   "-.",

        "Combined": "-"
    }

    legend_handles = []

    # Store loaded data for reuse in heatmaps
    all_data = {}

    for chan in channels:
        sig_file = outdir / npz_pattern.format(channel=chan)
        if not sig_file.exists():
            print(f"[!] Missing: {sig_file}")
            continue

        data = np.load(sig_file)
        k3_vals, k4_vals, Z_vals = data["k3"], data["k4"], data["Z"]
        K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
        all_data[chan] = (K3, K4, Z_vals)

        # Plot only the 2σ contour
        cs = ax.contour(
            K3, K4, Z_vals,
            #levels=[1.52,2.45],
            levels=[1.52],
            colors=colors.get(chan, "black"),
            linestyles=linestyles.get(chan, "-"),
            linewidths=2,
        )

        # Add custom handle for legend
        legend_handles.append(
            Line2D([0], [0],
                   color=colors.get(chan, "black"),
                   linestyle=linestyles.get(chan, "-"),
                   lw=2,
                   label=chan)
        )

    # === Shared comparison plot ===
    ax.set_xlabel(r"$k_3$", loc='right', fontsize=13)
    ax.set_ylabel(r"$k_4$", loc='top', fontsize=13)
    #ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(-40, 60)
    ax.set_xlim(-5, 10)

    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    banner_heatmaps(ax)

    outpath = outdir / "fit/comparison_contours_nllComp.pdf"
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Comparison plot saved to {outpath}")
    vmin, vmax = 0, 20

    # === Individual heatmaps ===
    for chan, (K3, K4, Z_vals) in all_data.items():
        fig, ax = plt.subplots(figsize=(7, 6))

        # Cap the colormap range at [0, 20]
        extent = [K3.min(), K3.max(), K4.min(), K4.max()]
        im = ax.imshow(
            Z_vals.T,               # transpose so axes match meshgrid
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            aspect="auto",
            interpolation="bilinear"  # try "bicubic" or "gaussian" too
        )

        # Contours computed on full data (not clipped)
        cs = ax.contour(
            K3, K4, Z_vals,
            levels=[2],
            colors="black",
            linewidths=2
        )

        from matplotlib.lines import Line2D
        legend_handle = Line2D([0], [0], color="black", linewidth=2, label=r"2$\sigma$ contour")

        cbar = fig.colorbar(im, ax=ax, label=r"Significance $Z$", extend="max")

        ax.legend(
            handles=[legend_handle],
            frameon=False,
            loc="right",          # anchor corner of the legend box
            bbox_to_anchor=(1.0, 1.04), # position above top-right of axes
            fontsize=14
        )

        ax.set_xlabel(r"$k_3$", loc='right', fontsize=13)
        ax.set_ylabel(r"$k_4$", loc='top', fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(-50, 60)
        ax.set_xlim(-5, 10)
        banner_heatmaps(ax)

        outpath = outdir / f"fit/{chan}_heatmap.pdf"
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Heatmap for {chan} saved to {outpath}")
        
def compute_bonly_asimov_significance(
    var: str,
    files: dict[str, Path],
    k3: float = 1.0,
    k4: float = 1.0,
    outdir: Path | None = None,
    channel: str = "Combined",
) -> float:
    """Compute Asimov discovery significance under the B-only hypothesis: data = background, test = signal + background."""
    selection = SELECTION[channel]
    # --- Load signal ---
    with uproot.open(files[SIGNAL]) as f_sig:
        t_sig = f_sig["events"]
        m_sig = build_mask_from_selection(t_sig, selection)
        sig = numeric(t_sig[var].array(library="ak")[m_sig])
        if sig.size == 0:
            print(f"[!] No signal entries for {var}")
            return np.nan

        # Histogram binning
        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
        else:
            xmin, xmax = sig.min(), sig.max()
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)

        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {
            key: numeric(t_sig[key].array(library="ak")[m_sig])
            for key in basis_keys
        }
        weights = get_weights(k3, k4, weight_dict)
        xsec_weight = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig]) * LUMINOSITY_PB
        weights *= xsec_weight
        h_sig, _ = np.histogram(sig, bins=edges, weights=weights)

    # --- Load backgrounds ---
    bkg_total = np.zeros_like(h_sig)
    for proc in BACKGROUNDS:
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            t_bkg = f_bkg["events"]
            arr_raw = t_bkg[var].array(library="ak")
            m_bkg = build_mask_from_selection(t_bkg, selection)
            arr = numeric(arr_raw[m_bkg])
            weight_bkg = numeric(t_bkg["weight_xsec"].array(library="ak")[m_bkg]) * LUMINOSITY_PB
            if arr.size == 0:
                continue
            h_bkg, _ = np.histogram(arr, bins=edges, weights=weight_bkg)
            bkg_total += h_bkg

    # --- Compute significance with Asimov data = background ---
    s = h_sig
    b = bkg_total
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = (b > 0) & (s > 0)
        q0 = np.zeros_like(b)
        q0[mask] = 2 * ((s[mask] + b[mask]) * np.log1p(s[mask] / b[mask]) - s[mask])
        Z = np.sqrt(np.sum(q0))

    print(f"[✓] Discovery significance (Asimov, B-only) = {Z:.2f}σ (k3={k3}, k4={k4})")

    # --- Optional plot ---
    plot = False
    if plot:
        outdir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        ax.bar(bin_centers, b, width=np.diff(edges), align="center", color="grey", alpha=0.6, label="Background")
        ax.step(bin_centers, b + s, where="mid", color="black", lw=1.2, label="Signal + Background")
        ax.step(bin_centers, s, where="mid", color=process_colours[SIGNAL], lw=2, linestyle="--",
                label=f"Signal (k3={k3:.1f}, k4={k4:.1f})")
        ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
        ax.set_ylabel("Events / bin", loc="top")
        if (b + s).ptp() > 1e3:
            ax.set_yscale("log")
        ax.legend(frameon=False, loc="upper right")
        banner(ax, f"Z = {Z:.2f}σ")

        outpath = outdir / f"bonly_asimov_dist_{var}_k3_{k3:.1f}_k4_{k4:.1f}.pdf"
        #fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] B-only Asimov distribution plot saved to {outpath}")

    return Z

def plot_signal_yield_morphing_grid(
    files: dict[str, Path],
    outdir: Path,
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    channel: str = "Combined"
):
    """Evaluate and plot signal yield via moment morphing over (k3, k4) grid."""
    import numpy as np
    import matplotlib.pyplot as plt

    from tools import get_weights, build_masks, parse_selection, numeric
    from config import SIGNAL, SELECTION
    selection = SELECTION[channel]
    # --- Load signal basis weights ---
    with uproot.open(files[SIGNAL]) as f_sig:
        t_sig = f_sig["events"]
        m_sig = build_mask_from_selection(t_sig, selection)

        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {
            key: numeric(t_sig[key].array(library="ak")[m_sig])
            for key in basis_keys
        }
        xsec_weight = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig])
        base_weights = {k: v for k, v in weight_dict.items()}

    # --- Prepare grid ---
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    Z_yield = np.zeros_like(K3)

    for i in range(K3.shape[0]):
        for j in range(K3.shape[1]):
            k3, k4 = K3[i, j], K4[i, j]
            weights = get_weights(k3, k4, base_weights)
            Z_yield[i, j] = np.sum(weights * xsec_weight) * LUMINOSITY_PB
            print(f"[ ] Evaluating yield at (k3={k3:.2f}, k4={k4:.2f}): {Z_yield[i, j]:.2f}")
            
    

    
    # --- Save and plot ---
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "fit/signal_yield_grid.npz"
    np.savez(npz_path, yield_=Z_yield, k3=k3_vals, k4=k4_vals)
    print(f"[✓] Saved signal yield grid to {npz_path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.contourf(K3, K4, Z_yield, levels=50, cmap="coolwarm_r")
    fig.colorbar(c, ax=ax, label="Signal yield (pb x eff)")
    ax.set_xlabel("$k_3$")
    ax.set_ylabel("$k_4$")
    ax.set_title("Signal Yield from Morphing over $k_3$-$k_4$")
    ax.grid(True, linestyle=":")
    fig.savefig(outdir / "fit/signal_yield_contour.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Signal yield contour plot saved to: {outdir/'fit/signal_yield_contour.pdf'}")
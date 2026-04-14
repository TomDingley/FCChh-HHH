import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import laplace, norm
import uproot
from pathlib import Path

from config import SIGNAL, XLIM_MAP, N_BINS_1D, SELECTION
from aesthetics import LABEL_MAP, banner
from tools import get_weights,numeric, build_mask_from_selection


from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np

def gaussian_resolution_pm_rms(data, nsig_rms=1.5, bins=40):
    """
    Resolution := fitted Gaussian sigma in the window [mean ± nsig_rms * RMS].

    Returns:
      mu_fit, sigma_fit, (lo, hi), n_used
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 10:
        return np.nan, np.nan, (np.nan, np.nan), 0

    mu0 = float(np.mean(data))
    rms = float(np.std(data, ddof=1))
    if not np.isfinite(rms) or rms <= 0:
        return np.nan, np.nan, (np.nan, np.nan), 0

    lo = mu0 - nsig_rms * rms
    hi = mu0 + nsig_rms * rms
    sub = data[(data >= lo) & (data <= hi)]
    n_used = int(sub.size)
    if n_used < 10:
        return np.nan, np.nan, (lo, hi), n_used

    # Option A (recommended): unbinned MLE for Gaussian
    mu_fit, sigma_fit = norm.fit(sub)

    # Option B (if you prefer binned fit): uncomment this block and comment out norm.fit
    # counts, edges = np.histogram(sub, bins=bins, density=True)
    # centers = 0.5 * (edges[:-1] + edges[1:])
    # popt, _ = curve_fit(lambda x, mu, sig: norm.pdf(x, mu, sig),
    #                     centers, counts, p0=[mu0, rms], maxfev=20000)
    # mu_fit, sigma_fit = popt

    return float(mu_fit), float(sigma_fit), (float(lo), float(hi)), n_used

def dscb(x, mu, sigma, alpha_low, n_low, alpha_high, n_high):
    """Double-sided Crystal Ball function."""
    x = np.array(x)
    t = (x - mu) / sigma

    # left tail
    mask_low = t < -alpha_low
    # right tail
    mask_high = t > alpha_high
    # gaussian core
    mask_core = (~mask_low) & (~mask_high)

    A_low = (n_low / abs(alpha_low)) ** n_low * np.exp(-alpha_low**2 / 2)
    B_low = n_low / abs(alpha_low) - abs(alpha_low)

    A_high = (n_high / abs(alpha_high)) ** n_high * np.exp(-alpha_high**2 / 2)
    B_high = n_high / abs(alpha_high) - abs(alpha_high)

    result = np.empty_like(t)
    result[mask_core] = np.exp(-0.5 * t[mask_core] ** 2)
    result[mask_low] = A_low * (B_low - t[mask_low]) ** -n_low
    result[mask_high] = A_high * (B_high + t[mask_high]) ** -n_high

    return result / (sigma * np.sqrt(2 * np.pi))  

def scb_low(x, mu, sigma, alpha_L, n_L):
    """
    Single-sided Crystal Ball with tail on the low-mass side.
    Normalised as a PDF.
    """
    x = np.array(x, dtype=float)
    t = (x - mu) / sigma

    # left tail region
    mask_low = t < -alpha_L
    mask_core = ~mask_low

    A_L = (n_L / abs(alpha_L))**n_L * np.exp(-0.5 * alpha_L**2)
    B_L = n_L / abs(alpha_L) - abs(alpha_L)

    res = np.empty_like(t)
    res[mask_core] = np.exp(-0.5 * t[mask_core]**2)
    res[mask_low]  = A_L * (B_L - t[mask_low])**(-n_L)

    # normalisation factor (approximate, or you can integrate numerically if pedantic)
    norm = sigma * np.sqrt(2*np.pi)
    return res / norm

def fit_signal_mass_shape(files: dict[str, Path], outdir: Path, var: str, comment: str, channel: str):
    fp = files.get(SIGNAL)
    if not fp:
        print(f"[!] No signal file found for {SIGNAL}")
        return

    with uproot.open(fp) as f:
        tree = f["events"]
        selection = SELECTION[channel]
        print("Applying selection prior to fit: ", selection)
        mask = build_mask_from_selection(tree, selection)
        data = numeric(tree[var].array(library="ak")[mask])
        if data.size == 0:
            print(f"[!] No entries for {var}")
            return

        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
            data = data[(data >= xmin) & (data <= xmax)]
        else:
            xmin, xmax = data.min(), data.max()
        
        # --- Resolution: Gaussian sigma in ±1.5 RMS window around mean ---
        res_mu, res_sigma, (res_lo, res_hi), res_n = gaussian_resolution_pm_rms(
            data, nsig_rms=1.5, bins=N_BINS_1D
        )

        if data.size == 0:
            print(f"[!] All data outside range for {var}")
            return

        counts, edges = np.histogram(data, bins=N_BINS_1D, density=True)
        centers = (edges[:-1] + edges[1:]) / 2

        popt_gauss, _ = curve_fit(lambda x, mu, sigma: norm.pdf(x, mu, sigma),
                                  centers, counts, p0=[data.mean(), data.std()])
        popt_laplace, _ = curve_fit(lambda x, mu, b: laplace.pdf(x, mu, b),
                                    centers, counts, p0=[data.mean(), data.std() / np.sqrt(2)])
        
        # Initial guess: [mu, sigma, alpha_low, n_low, alpha_high, n_high]
        mu_init, sigma_init = data.mean(), data.std()
        p0_dscb = [mu_init, sigma_init, 1.5, 2, 1.5, 2]

        # Fix: make bounds more flexible
        bounds = (
            [mu_init - 10 * sigma_init, 1e-3, 0.1, 0.1, 0.1, 0.1],  # lower bounds
            [mu_init + 10 * sigma_init, 100.0, 10.0, 20.0, 10.0, 20.0]  # upper bounds (note: sigma upper = 100.0)
        )
        try:
            popt_dscb, _ = curve_fit(dscb, centers, counts, p0=p0_dscb, bounds=bounds)
            
        except (RuntimeError, ValueError) as e:
            print(f"[!] DSCB fit failed: {e}")
            popt_dscb = None
        print("[debug] mu_init =", mu_init)
        print("[debug] sigma_init =", sigma_init)
        print("[debug] p0_dscb =", p0_dscb)
        print("[debug] bounds lower =", bounds[0])
        print("[debug] bounds upper =", bounds[1])

        # Check element-wise feasibility
        for i, (p, low, high) in enumerate(zip(p0_dscb, bounds[0], bounds[1])):
            if not (low <= p <= high):
                print(f"[!] Parameter {i} = {p} out of bounds: [{low}, {high}]")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(data, bins=edges, density=True, histtype='step', label="Signal (normalised)", color="black")
        xvals = np.linspace(xmin, xmax, 500)
        ax.plot(xvals, norm.pdf(xvals, *popt_gauss), label="Gaussian fit", color="blue")
        ax.plot(xvals, laplace.pdf(xvals, *popt_laplace), label="Laplacian fit", color="red", linestyle="--")
        #if popt_dscb is not None:
        #    ax.plot(xvals, dscb(xvals, *popt_dscb), label="DSCB fit", color="green", linestyle=":")
        ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
        ax.set_ylabel("Normalised density", loc="top")
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        banner(ax, comment)
        # --- Add Gaussian & Laplace fit parameters to the plot ---
        gauss_mu, gauss_sigma = popt_gauss
        lap_mu, lap_b = popt_laplace

        stats_text = (
            r"$\bf{Gaussian}$" + "\n"
            rf"$\mu = {gauss_mu:.2f}$" + "\n"
            rf"$\sigma = {gauss_sigma:.2f}$" + "\n\n"
            r"$\bf{Laplace}$" + "\n"
            rf"$\mu = {lap_mu:.2f}$" + "\n"
            rf"$b = {lap_b:.2f}$" + "\n\n"
            r"$\bf{Resolution}$" + "\n"
            rf"$\sigma_{{\pm 1.5\,RMS}} = {res_sigma:.2f}$" + "\n"
            rf"$N={res_n}$"
        )

        ax.text(
            0.05, 0.73,
            stats_text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            family="monospace",
        )

        out = outdir / f"{channel}" / f"fit_{var}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓] Fit saved to", out)
        
                # (after the Gaussian/Laplace figure is saved)
        out = outdir / f"{channel}" / f"fit_{var}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓] Fit saved to", out)

        # --- New: separate DSCB plot ---
        if popt_dscb is not None:
            fig_d, ax_d = plt.subplots(figsize=(6, 5))
            # same histogram as before
            ax_d.hist(
                data, bins=edges, density=True,
                histtype="step", label="Signal (normalised)", color="black"
            )

            xvals = np.linspace(xmin, xmax, 500)
            ax_d.plot(
                xvals,
                dscb(xvals, *popt_dscb),
                label="DSCB fit",
                color="green",
                linestyle=":"
            )

            ax_d.set_xlabel(LABEL_MAP.get(var, var), loc="right")
            ax_d.set_ylabel("Normalised density", loc="top")
            ax_d.legend(frameon=False, fontsize=10, loc="upper right")
            banner(ax_d, comment)

            # Add DSCB parameters
            mu, sigma, aL, nL, aR, nR = popt_dscb
            stats_text_d = (
                r"$\bf{DSCB}$" + "\n"
                rf"$\mu = {mu:.2f}$" + "\n"
                rf"$\sigma = {sigma:.2f}$" + "\n"
                rf"$\alpha_L = {aL:.2f}$, $n_L = {nL:.1f}$" + "\n"
                rf"$\alpha_R = {aR:.2f}$, $n_R = {nR:.1f}$"
            )

            ax_d.text(
                0.05, 0.65,
                stats_text_d,
                transform=ax_d.transAxes,
                va="top", ha="left",
                fontsize=9,
                family="monospace",
            )

            out_d = outdir / f"{channel}" / f"fit_{var}_dscb.pdf"
            fig_d.savefig(out_d, bbox_inches="tight")
            plt.close(fig_d)
            print("[✓] DSCB fit plot saved to", out_d)
            
                    # --- Single-sided CB fit and plot ---
            p0_scb = [data.mean(), data.std(), 1.5, 2.0]  # mu, sigma, alpha_L, n_L
            bounds_scb = (
                [data.mean() - 10*data.std(), 1e-3, 0.1, 0.1],
                [data.mean() + 10*data.std(), 100.0, 10.0, 20.0],
            )

            try:
                popt_scb, _ = curve_fit(
                    scb_low,
                    centers,
                    counts,
                    p0=p0_scb,
                    bounds=bounds_scb,
                )
            except (RuntimeError, ValueError) as e:
                print(f"[!] SCB fit failed: {e}")
                popt_scb = None

            if popt_scb is not None:
                fig_scb, ax_scb = plt.subplots(figsize=(6, 5))
                ax_scb.hist(
                    data, bins=edges, density=True,
                    histtype='step', label="Signal (normalised)", color="black"
                )
                xvals = np.linspace(xmin, xmax, 500)

                ax_scb.plot(
                    xvals, norm.pdf(xvals, *popt_gauss),
                    label="Gaussian fit", color="blue"
                )
                ax_scb.plot(
                    xvals, laplace.pdf(xvals, *popt_laplace),
                    label="Laplacian fit", color="red", linestyle="--"
                )

                ax_scb.plot(
                    xvals,
                    scb_low(xvals, *popt_scb),
                    label="Single-sided CB fit",
                    color="green",
                    linestyle=":",
                )

                scb_mu, scb_sigma, scb_aL, scb_nL = popt_scb
                stats_text = (
                    r"$\bf{SCB}$" + "\n"
                    rf"$\mu = {scb_mu:.2f}$" + "\n"
                    rf"$\sigma = {scb_sigma:.2f}$" + "\n"
                    rf"$\alpha_L = {scb_aL:.2f}$, $n_L = {scb_nL:.1f}$"
                )
                ax_scb.text(
                    0.05, 0.65,
                    stats_text,
                    transform=ax_scb.transAxes,
                    va="top", ha="left",
                    fontsize=9,
                    family="monospace",
                )

                ax_scb.set_xlabel(LABEL_MAP.get(var, var), loc="right")
                ax_scb.set_ylabel("Normalised density", loc="top")
                ax_scb.legend(frameon=False, fontsize=10, loc="upper right")
                banner(ax_scb, comment)

                out_d = outdir / f"{channel}" / f"fit_{var}_scb.pdf"
                fig_scb.savefig(out_d, bbox_inches="tight")
                plt.close(fig_scb)
                print("[✓] SCB fit plot saved to", out_d)

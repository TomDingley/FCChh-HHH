#!/usr/bin/env python3
"""Minimal standalone log-normal fit for delta-theta vs pT(reco).

This script keeps the branch conventions of ``scipy_fit_pTReco.py`` but strips
away the Gaussian+Moyal workflow. The global fit is unbinned: every selected tau
leg contributes directly to the likelihood. Histograms are only used for
per-pT diagnostic plots and summary tables.
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import uproot
from scipy.special import erf


mpl.rcParams.update(
    {
        "figure.figsize": (7, 5),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "figure.autolayout": True,
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class TauSample:
    pt: np.ndarray
    theta: np.ndarray
    nprongs: np.ndarray


@dataclass(frozen=True)
class LognormalFitResult:
    params: np.ndarray
    nll: float
    success: bool
    message: str
    seed_params: np.ndarray


def find_tree(root_file, preferred=("events", "Events"), requested: Optional[str] = None):
    """Return the requested tree, or the first common default tree name."""
    if requested is not None:
        return root_file[requested]

    keys = list(root_file.keys())
    keynames = [key.split(";")[0] for key in keys]
    for name in preferred:
        if name in keynames:
            return root_file[keys[keynames.index(name)]]
    raise KeyError(
        f"Could not find a TTree named any of {preferred}. Available keys: {keynames}"
    )


def parse_bins(spec: str) -> np.ndarray:
    """Parse a comma-separated bin-edge string into a validated array."""
    parts = [float(token.strip()) for token in spec.split(",") if token.strip()]
    bins = np.asarray(parts, dtype=float)
    if bins.size < 2 or not np.all(np.diff(bins) > 0.0):
        raise ValueError("pT bin edges must be increasing and contain at least two values")
    return bins


def branch_name(tau_index: int, suffix: str) -> str:
    return f"tau{tau_index}_{suffix}"


def add_fcc_banner(ax: plt.Axes) -> None:
    ax.text(
        0.0,
        1.02,
        r"$\mathbf{FCC\text{-}hh}$ $\mathit{Delphes}$ Simulation" "\n"
        r"$\sqrt{s}=84\,\mathrm{TeV}$, 30 ab$^{-1}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )


def category_display_name(category_name: str) -> str:
    labels = {
        "nprongs_0_lep": "Leptonic",
        "nprongs_1_had1p": "1-prong",
        "nprongs_3_had3p": "3-prong",
    }
    return labels.get(category_name, category_name)


def lognormal_params_from_lnp(lnp: np.ndarray, params: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m0, m1, s0, s1, c0, c1 = np.asarray(params, dtype=float).tolist()
    mu = m0 + m1 * lnp
    sigma = np.exp(np.clip(s0 + s1 * lnp, -10.0, 5.0))
    c = np.exp(np.clip(c0 + c1 * lnp, -20.0, 5.0))
    return mu, sigma, c


def lognormal_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    x_shift = x + c
    out = np.zeros_like(x_shift, dtype=float)
    ok = x_shift > 0.0
    if np.any(ok):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(x_shift) - mu) / (sigma * np.sqrt(2.0))
        out = np.where(ok, 0.5 * (1.0 + erf(z)), 0.0)
    return out


def lognormal_logpdf_trunc(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    c: np.ndarray,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Log-pdf of a shifted log-normal truncated to [x_min, x_max]."""
    x_shift = x + c
    logpdf = np.full_like(x_shift, -np.inf, dtype=float)
    ok = (x_shift > 0.0) & (x >= x_min) & (x <= x_max)
    if not np.any(ok):
        return logpdf

    z = (np.log(x_shift[ok]) - mu[ok]) / sigma[ok]
    logpdf[ok] = -np.log(x_shift[ok] * sigma[ok] * np.sqrt(2.0 * np.pi)) - 0.5 * z * z

    cdf_hi = lognormal_cdf(np.full_like(x[ok], x_max), mu[ok], sigma[ok], c[ok])
    cdf_lo = lognormal_cdf(np.full_like(x[ok], x_min), mu[ok], sigma[ok], c[ok])
    norm = np.clip(cdf_hi - cdf_lo, 1e-12, None)
    logpdf[ok] -= np.log(norm)
    return logpdf


def lognormal_probs_for_all_p_common(
    pt_centers: np.ndarray,
    theta_edges: np.ndarray,
    params: Sequence[float],
    theta_max: float,
) -> np.ndarray:
    """Bin-integrated probabilities for a common theta binning."""
    lnp = np.log(np.clip(pt_centers, 1e-6, None))
    mu, sigma, c = lognormal_params_from_lnp(lnp, params)

    cdf_lo = lognormal_cdf(theta_edges[:-1][None, :], mu[:, None], sigma[:, None], c[:, None])
    cdf_hi = lognormal_cdf(theta_edges[1:][None, :], mu[:, None], sigma[:, None], c[:, None])
    cdf_min = lognormal_cdf(np.full((mu.size, 1), 0.0), mu[:, None], sigma[:, None], c[:, None])
    cdf_max = lognormal_cdf(np.full((mu.size, 1), theta_max), mu[:, None], sigma[:, None], c[:, None])

    norm = np.clip(cdf_max - cdf_min, 1e-12, None)
    probs = np.clip((cdf_hi - cdf_lo) / norm, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.where(row_sum > 0.0, probs / row_sum, probs)


def collect_perbin_theta(
    pt: np.ndarray,
    theta: np.ndarray,
    pt_bins: np.ndarray,
    theta_max: float,
) -> List[np.ndarray]:
    samples = []
    for pt_lo, pt_hi in zip(pt_bins[:-1], pt_bins[1:]):
        mask = (
            (pt >= pt_lo)
            & (pt < pt_hi)
            & (theta >= 0.0)
            & (theta <= theta_max)
            & np.isfinite(pt)
            & np.isfinite(theta)
        )
        samples.append(theta[mask])
    return samples


def make_common_theta_histograms(
    pt: np.ndarray,
    theta: np.ndarray,
    pt_bins: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    counts = []
    theta_min = float(theta_edges[0])
    theta_max = float(theta_edges[-1])
    for pt_lo, pt_hi in zip(pt_bins[:-1], pt_bins[1:]):
        mask = (
            (pt >= pt_lo)
            & (pt < pt_hi)
            & (theta >= theta_min)
            & (theta <= theta_max)
            & np.isfinite(pt)
            & np.isfinite(theta)
        )
        hist, _ = np.histogram(theta[mask], bins=theta_edges)
        counts.append(hist.astype(float))
    return np.asarray(counts, dtype=float)


def fit_line(x: np.ndarray, y: np.ndarray, fallback: float) -> Tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) >= 2:
        design = np.vstack([np.ones_like(x[ok]), x[ok]]).T
        coef, _, _, _ = np.linalg.lstsq(design, y[ok], rcond=None)
        return float(coef[0]), float(coef[1])
    if np.count_nonzero(ok) == 1:
        return float(y[ok][0]), 0.0
    return float(fallback), 0.0


def lognormal_seed_from_perbin(perbin_theta: List[np.ndarray], pt_centers: np.ndarray) -> np.ndarray:
    """Seed the global fit from per-bin log-space moments."""
    mu_seed = np.full_like(pt_centers, np.nan, dtype=float)
    sigma_seed = np.full_like(pt_centers, np.nan, dtype=float)

    for index, values in enumerate(perbin_theta):
        values = values[(values > 0.0) & np.isfinite(values)]
        if values.size < 5:
            continue
        log_values = np.log(values)
        mu_seed[index] = float(np.mean(log_values))
        sigma_seed[index] = float(np.std(log_values, ddof=1)) if values.size > 1 else 0.3

    lnp = np.log(np.clip(pt_centers, 1e-6, None))
    m0, m1 = fit_line(lnp, mu_seed, fallback=np.log(0.05))
    s0, s1 = fit_line(lnp, np.log(np.clip(sigma_seed, 1e-6, None)), fallback=np.log(0.3))
    c0, c1 = np.log(1e-6), 0.0
    return np.array([m0, m1, s0, s1, c0, c1], dtype=float)


def lognormal_global_fit(
    pt_vals: np.ndarray,
    theta_vals: np.ndarray,
    theta_max: float,
    params0: np.ndarray,
) -> LognormalFitResult:
    lnp = np.log(np.clip(pt_vals, 1e-6, None))
    x = theta_vals

    def nll(params: np.ndarray) -> float:
        mu, sigma, c = lognormal_params_from_lnp(lnp, params)
        logpdf = lognormal_logpdf_trunc(x, mu, sigma, c, 0.0, theta_max)
        if not np.isfinite(logpdf).all():
            return 1e12
        return float(-np.sum(logpdf))

    result = opt.minimize(
        nll,
        params0,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-7},
    )
    return LognormalFitResult(
        params=np.asarray(result.x, dtype=float),
        nll=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        seed_params=np.asarray(params0, dtype=float),
    )


def compute_support_window(
    hist: np.ndarray,
    theta_edges: np.ndarray,
    q_low: float = 0.001,
    q_high: float = 0.999,
) -> Tuple[float, float]:
    total = float(np.sum(hist))
    if total <= 0.0:
        return float(theta_edges[0]), float(theta_edges[-1])

    cdf = np.cumsum(hist) / total
    valid_bins = np.where((cdf >= q_low) & (cdf <= q_high))[0]
    if valid_bins.size == 0:
        return float(theta_edges[0]), float(theta_edges[-1])

    first = max(int(valid_bins[0]) - 1, 0)
    last = min(int(valid_bins[-1]) + 1, len(theta_edges) - 2)
    return float(theta_edges[first]), float(theta_edges[last + 1])


def plot_lognormal_overlays(
    counts: np.ndarray,
    pt_edges: np.ndarray,
    theta_edges: np.ndarray,
    params: np.ndarray,
    theta_max: float,
    outdir: Path,
    theta_unit: str,
    category_label: str,
    mc_plot_bins: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_widths = np.diff(theta_edges)

    probs_fine = lognormal_probs_for_all_p_common(pt_centers, theta_edges, params, theta_max)
    density_fine = probs_fine / theta_widths[None, :]

    for index, hist in enumerate(counts):
        total = float(np.sum(hist))
        if total <= 0.0:
            continue

        pt_lo = pt_edges[index]
        pt_hi = pt_edges[index + 1]
        theta_min_plot, theta_max_plot = compute_support_window(hist, theta_edges)

        n_plot_bins = max(1, min(int(mc_plot_bins), len(theta_edges) - 1))
        plot_edges = np.linspace(theta_edges[0], theta_edges[-1], n_plot_bins + 1)

        # Rebin only the displayed MC points; the fit itself stays unbinned.
        cumulative_counts = np.concatenate(([0.0], np.cumsum(hist.astype(float))))
        counts_plot = np.diff(np.interp(plot_edges, theta_edges, cumulative_counts))
        widths_plot = np.diff(plot_edges)
        centers_plot = 0.5 * (plot_edges[:-1] + plot_edges[1:])
        probabilities_plot = counts_plot / total
        density_plot = probabilities_plot / widths_plot
        density_err_plot = np.sqrt(
            np.clip(probabilities_plot * (1.0 - probabilities_plot) / max(total, 1.0), 0.0, None)
        ) / widths_plot

        probs_plot = lognormal_probs_for_all_p_common(
            np.array([pt_centers[index]], dtype=float),
            plot_edges,
            params,
            theta_max,
        )[0]
        model_density_plot = probs_plot / widths_plot

        ratio = np.full_like(density_plot, np.nan, dtype=float)
        ratio_err = np.full_like(density_plot, np.nan, dtype=float)
        valid = (model_density_plot > 0.0) & np.isfinite(model_density_plot)
        ratio[valid] = density_plot[valid] / model_density_plot[valid]
        ratio_err[valid] = density_err_plot[valid] / model_density_plot[valid]

        fig, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            figsize=(7, 5),
            dpi=140,
        )

        ax_main.grid(False)
        ax_ratio.grid(False)
        ax_main.errorbar(
            centers_plot,
            density_plot,
            yerr=density_err_plot,
            fmt="o",
            capsize=1.5,
            color="black",
            label=f"MC",
        )
        ax_main.plot(
            theta_centers,
            density_fine[index],
            color="black",
            linestyle="--",
            label="Log-normal fit",
        )
        ax_main.set_ylim(bottom=0)
        ax_main.set_ylabel("Density", loc="top")
        ax_main.legend(frameon=False, loc='upper right')
        ax_main.text(
            0.97,
            0.78,
            rf"{int(pt_lo)} $\leq p_{{T}}^{{\mathrm{{reco}}}}$ < {int(pt_hi)} GeV" "\n",
            transform=ax_main.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        add_fcc_banner(ax_main)
        ax_main.text(
            1,
            1.02,
            rf"{category_label}",
            transform=ax_main.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )

        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.errorbar(
            centers_plot,
            ratio,
            yerr=ratio_err,
            fmt="o",
            capsize=1.5,
            color="black",
        )
        ax_ratio.set_xlabel(rf"$\Delta\theta$ [{theta_unit}]", loc="right")
        ax_ratio.set_ylabel("MC/fit")
        ax_ratio.set_yticks([0.75, 1.25])
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_xlim(left=0.001)
        ax_main.set_xlim(left=0.001)

        ax_main.set_xlim(theta_min_plot, theta_max_plot)
        fig.tight_layout()
        fig.savefig(outdir / f"lognormal_ptbin_{int(pt_lo)}_{int(pt_hi)}.pdf")
        plt.close(fig)


def load_tau_sample(args: argparse.Namespace) -> Tuple[TauSample, float, str]:
    theta0_branch = branch_name(0, args.theta_suffix)
    theta1_branch = branch_name(1, args.theta_suffix)
    pt0_branch = branch_name(0, args.pt_suffix)
    pt1_branch = branch_name(1, args.pt_suffix)
    npr0_branch = branch_name(0, args.nprongs_suffix)
    npr1_branch = branch_name(1, args.nprongs_suffix)
    matched0_branch = "tau0_matched"
    matched1_branch = "tau1_matched"

    with uproot.open(args.infile) as root_file:
        tree = find_tree(root_file, requested=args.tree)
        wanted = [theta0_branch, theta1_branch, pt0_branch, pt1_branch, npr0_branch, npr1_branch]

        has_matched0 = matched0_branch in tree.keys()
        has_matched1 = matched1_branch in tree.keys()
        if args.require_matched:
            if has_matched0:
                wanted.append(matched0_branch)
            if has_matched1:
                wanted.append(matched1_branch)

        missing = [branch for branch in wanted if branch not in tree.keys()]
        if missing:
            raise KeyError(f"Missing branches: {missing}")
        arrays = tree.arrays(wanted, library="np")

    theta0 = arrays[theta0_branch].astype(float)
    theta1 = arrays[theta1_branch].astype(float)
    pt0 = arrays[pt0_branch].astype(float)
    pt1 = arrays[pt1_branch].astype(float)
    npr0 = arrays[npr0_branch].astype(int)
    npr1 = arrays[npr1_branch].astype(int)

    mask0 = np.isfinite(theta0) & np.isfinite(pt0) & (pt0 > 0.0) & (theta0 >= 0.0)
    mask1 = np.isfinite(theta1) & np.isfinite(pt1) & (pt1 > 0.0) & (theta1 >= 0.0)

    if args.require_matched:
        if has_matched0:
            mask0 &= arrays[matched0_branch].astype(int) == 1
        else:
            warnings.warn("--require-matched set but tau0_matched was not found", stacklevel=2)
        if has_matched1:
            mask1 &= arrays[matched1_branch].astype(int) == 1
        else:
            warnings.warn("--require-matched set but tau1_matched was not found", stacklevel=2)

    theta0 = theta0[mask0]
    theta1 = theta1[mask1]
    pt0 = pt0[mask0]
    pt1 = pt1[mask1]
    npr0 = npr0[mask0]
    npr1 = npr1[mask1]

    if args.degrees:
        theta0 = np.degrees(theta0)
        theta1 = np.degrees(theta1)
        theta_max = float(np.degrees(args.theta_max))
        theta_unit = "deg"
    else:
        theta_max = float(args.theta_max)
        theta_unit = "rad"

    sample = TauSample(
        pt=np.concatenate([pt0, pt1]),
        theta=np.concatenate([theta0, theta1]),
        nprongs=np.concatenate([npr0, npr1]),
    )
    return sample, theta_max, theta_unit


def iter_categories(sample: TauSample, requested_nprongs: Optional[int]):
    
    categories = [
        ("nprongs_0_lep", 0),
        ("nprongs_1_had1p", 1),
        ("nprongs_3_had3p", 3),
    ]

    for category_name, nprongs_value in categories:
        if nprongs_value is None:
            mask = np.ones_like(sample.nprongs, dtype=bool)
        else:
            mask = sample.nprongs == nprongs_value
        yield category_name, TauSample(
            pt=sample.pt[mask],
            theta=sample.theta[mask],
            nprongs=sample.nprongs[mask],
        )


def run_category_fit(
    sample: TauSample,
    pt_bins: np.ndarray,
    theta_max: float,
    theta_unit: str,
    args: argparse.Namespace,
    category_name: str,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    valid = (
        np.isfinite(sample.pt)
        & np.isfinite(sample.theta)
        & (sample.pt > 0.0)
        & (sample.theta >= 0.0)
        & (sample.theta <= theta_max)
    )
    if np.count_nonzero(valid) == 0:
        raise ValueError(f"No valid entries left for category {category_name}")

    pt = sample.pt[valid]
    theta = sample.theta[valid]
    theta_edges = np.linspace(0.0, theta_max, args.theta_bins + 1)
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    perbin_theta = collect_perbin_theta(pt, theta, pt_bins, theta_max)
    counts = make_common_theta_histograms(pt, theta, pt_bins, theta_edges)
    params0 = lognormal_seed_from_perbin(perbin_theta, pt_centers)
    fit_result = lognormal_global_fit(pt_vals=pt, theta_vals=theta, theta_max=theta_max, params0=params0)

    category_label = category_display_name(category_name)

    summary = {
        "fit_model": "unbinned_truncated_shifted_lognormal",
        "params": fit_result.params.astype(float).tolist(),
        "nll": float(fit_result.nll),
        "success": bool(fit_result.success),
        "message": str(fit_result.message),
        "seed_params": fit_result.seed_params.astype(float).tolist(),
        "entries": int(pt.size),
        "pt_bins": pt_bins.astype(float).tolist(),
        "theta_max": float(theta_max),
        "theta_bins_for_plots": int(args.theta_bins),
        "theta_unit": theta_unit,
        "category": category_name,
        "category_label": category_label,
        "theta_suffix": str(args.theta_suffix),
        "pt_suffix": str(args.pt_suffix),
        "nprongs_suffix": str(args.nprongs_suffix),
        "require_matched": bool(args.require_matched),
        "degrees": bool(args.degrees),
        "mc_plot_bins": int(args.mc_plot_bins),
    }
    
    (outdir / "global_fit_lognormal.json").write_text(json.dumps(summary, indent=2))

    plot_lognormal_overlays(
        counts=counts,
        pt_edges=pt_bins,
        theta_edges=theta_edges,
        params=fit_result.params,
        theta_max=theta_max,
        outdir=outdir / "perbin_overlays_lognormal",
        theta_unit=theta_unit,
        category_label=category_label,
        mc_plot_bins=args.mc_plot_bins,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone log-normal fit for delta-theta vs pT(reco). "
            "The fit is unbinned; theta histograms are only used for diagnostics."
        )
    )
    parser.add_argument("--infile", default="/data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_visangleCalib/merged/mgp8_pp_z_4f_84TeV.root", help="Path to the ROOT file")
    parser.add_argument("--outdir", default="./fit_ptreco_lognormal", help="Directory for outputs")
    parser.add_argument("--tree", default=None, help="Optional explicit TTree name")
    parser.add_argument(
        "--pt-bins",
        default="25,27.5,30,35,40,45,50,60,70,85,150",
        help="Comma-separated pT(reco) bin edges in GeV",
    )
    parser.add_argument("--theta-max", type=float, default=0.13, help="Upper theta bound")
    parser.add_argument(
        "--theta-bins",
        type=int,
        default=100,
        help="Number of theta bins used for diagnostic histograms and overlays",
    )
    parser.add_argument(
        "--theta-suffix",
        default="theta_rt",
        help="Suffix after tau{0,1}_ for the theta branch, e.g. theta_rt",
    )
    parser.add_argument("--pt-suffix", default="pt_reco", help="Suffix after tau{0,1}_ for pT(reco)")
    parser.add_argument("--nprongs-suffix", default="nprongs", help="Suffix after tau{0,1}_ for nprongs")
    parser.add_argument("--require-matched", action="store_true", help="Require tau{0,1}_matched == 1 if present")
    parser.add_argument("--degrees", action="store_true", help="Convert theta inputs and outputs from rad to deg")
    parser.add_argument("--nprongs", type=int, default=None, help="Restrict to a single nprongs category")
    parser.add_argument(
        "--mc-plot-bins",
        type=int,
        default=20,
        help="Displayed MC bins in each overlay plot",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pt_bins = parse_bins(args.pt_bins)

    sample, theta_max, theta_unit = load_tau_sample(args)
    for category_name, category_sample in iter_categories(sample, args.nprongs):
        if category_sample.pt.size == 0:
            continue
        category_outdir = outdir / "tau01_concat" / category_name
        run_category_fit(
            sample=category_sample,
            pt_bins=pt_bins,
            theta_max=theta_max,
            theta_unit=theta_unit,
            args=args,
            category_name=category_name,
            outdir=category_outdir,
        )


if __name__ == "__main__":
    main()

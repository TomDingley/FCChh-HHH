#!/usr/bin/env python3
"""Visible-fraction fits with overlay and mode-separated ratio plots.

This is a cleaned-up standalone alternative to ``fit_visfrac.py``. It keeps the
existing overlay outputs, including the combined prong-mode plot, and adds
fit-vs-MC ratio plots separated by prong mode.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

try:
    from scipy.stats import beta as scibeta
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


mpl.rcParams.update(
    {
        "figure.figsize": (7, 5),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "figure.autolayout": True,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


MODE_LABELS = {
    0: "Leptonic",
    1: "1-prong",
    3: "3-prong",
}
MODE_COLORS = {
    0: "red",
    1: "green",
    3: "blue",
}
MODE_LINESTYLES = {
    0: "--",
    1: "-",
    3: "-.",
}
SAMPLE_LABELS = {
    "tau1": "Tau leg 1",
    "tau2": "Tau leg 2",
    "combined": "Tau legs 1+2 combined",
}
OVERLAY_TITLES = {
    "tau1": "Tau leg 1",
    "tau2": "Tau leg 2",
    "combined": "Visible Fraction Calibration",
}


@dataclass(frozen=True)
class InputData:
    n_charged_1: ak.Array
    n_charged_2: ak.Array
    x_misP_1: ak.Array
    x_misP_2: ak.Array
    branch_names: Mapping[str, str]


@dataclass(frozen=True)
class BetaFitResult:
    alpha: float
    beta: float
    fit_lo: float
    fit_hi: float
    n_total: int
    n_fit: int


@dataclass(frozen=True)
class HistogramView:
    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    raw_counts: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    n_total: int


def add_fcc_banner(ax: plt.Axes, subtitle: str = "") -> None:
    text = (
        r"$\mathbf{FCC\text{-}hh}$ $\mathit{Delphes}$ Simulation" "\n"
        r"$\sqrt{s}=84\,\mathrm{TeV}$, 30 ab$^{-1}$"
    )
    if subtitle:
        text += "\n" + subtitle
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )


def mode_label(mode: int) -> str:
    return MODE_LABELS.get(mode, f"{mode}-prong")


def mode_slug(mode: int) -> str:
    return mode_label(mode).lower().replace(" ", "_").replace("-", "")


def parse_range(spec: str) -> Tuple[float, float]:
    lo, hi = [float(token.strip()) for token in spec.split(",")]
    if hi <= lo:
        raise ValueError(f"Invalid range '{spec}': expected hi > lo")
    return lo, hi


def parse_modes(spec: str) -> List[int]:
    modes = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not modes:
        raise ValueError("At least one mode must be provided")
    return modes


def find_tree(root_file, requested: Optional[str] = None, preferred=("events", "Events")):
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


def first_existing_branch(tree, candidates: Sequence[str]) -> str:
    available = set(tree.keys())
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(
        "None of the candidate branches exist:\n"
        + "\n".join([f"  - {candidate}" for candidate in candidates])
    )


def to_numpy_1d(arr: ak.Array) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.flatten(arr, axis=None)), dtype=float)


def load_input_data(file_path: str, tree_name: Optional[str]) -> InputData:
    with uproot.open(file_path) as root_file:
        tree = find_tree(root_file, requested=tree_name)

        branch_n1 = first_existing_branch(tree, ["n_charged_truthHadronicTaus_1"])
        branch_n2 = first_existing_branch(tree, ["n_charged_truthHadronicTaus_2"])
        branch_x1 = first_existing_branch(
            tree,
            [
                "x_misP_1",
                "x_misP_tau1",
                "x_misP_truthHadronicTaus_1",
                "x_misPTruth_1",
            ],
        )
        branch_x2 = first_existing_branch(
            tree,
            [
                "x_misP_2",
                "x_misP_tau2",
                "x_misP_truthHadronicTaus_2",
                "x_misPTruth_2",
            ],
        )

        arrays = {
            branch_n1: tree[branch_n1].array(library="ak"),
            branch_n2: tree[branch_n2].array(library="ak"),
            branch_x1: tree[branch_x1].array(library="ak"),
            branch_x2: tree[branch_x2].array(library="ak"),
        }

    return InputData(
        n_charged_1=arrays[branch_n1],
        n_charged_2=arrays[branch_n2],
        x_misP_1=arrays[branch_x1],
        x_misP_2=arrays[branch_x2],
        branch_names={
            "n1": branch_n1,
            "n2": branch_n2,
            "x1": branch_x1,
            "x2": branch_x2,
        },
    )


def select_mode_values(
    data: InputData,
    leg: int,
    mode: int,
    x_range: Tuple[float, float],
) -> np.ndarray:
    if leg == 1:
        n_values = data.n_charged_1
        x_values = data.x_misP_1
    elif leg == 2:
        n_values = data.n_charged_2
        x_values = data.x_misP_2
    else:
        raise ValueError(f"Unsupported leg: {leg}")

    mask = n_values == mode
    x_selected = to_numpy_1d(x_values[mask])
    x_lo, x_hi = x_range
    valid = np.isfinite(x_selected) & (x_selected >= x_lo) & (x_selected <= x_hi)
    return x_selected[valid]


def sample_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "N": 0,
            "mean": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q50": np.nan,
            "q95": np.nan,
        }
    return {
        "N": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def fit_beta_distribution(
    values: np.ndarray,
    fit_range: Tuple[float, float],
    min_count: int,
) -> Optional[BetaFitResult]:
    if not SCIPY_AVAILABLE or values.size < min_count:
        return None

    fit_lo, fit_hi = fit_range
    mask = np.isfinite(values) & (values > fit_lo) & (values < fit_hi)
    fit_values = values[mask]
    if fit_values.size < min_count:
        return None

    scaled = (fit_values - fit_lo) / (fit_hi - fit_lo)
    try:
        alpha, beta, _, _ = scibeta.fit(scaled, floc=0, fscale=1)
    except Exception:
        return None

    return BetaFitResult(
        alpha=float(alpha),
        beta=float(beta),
        fit_lo=float(fit_lo),
        fit_hi=float(fit_hi),
        n_total=int(values.size),
        n_fit=int(fit_values.size),
    )


def beta_cdf_on_edges(edges: np.ndarray, fit: BetaFitResult) -> np.ndarray:
    cdf = np.zeros_like(edges, dtype=float)
    inside = (edges > fit.fit_lo) & (edges < fit.fit_hi)
    above = edges >= fit.fit_hi
    if np.any(inside):
        scaled = (edges[inside] - fit.fit_lo) / (fit.fit_hi - fit.fit_lo)
        cdf[inside] = scibeta.cdf(scaled, fit.alpha, fit.beta)
    cdf[above] = 1.0
    return cdf


def beta_bin_heights(edges: np.ndarray, fit: BetaFitResult, density: bool) -> np.ndarray:
    probabilities = np.diff(beta_cdf_on_edges(edges, fit))
    widths = np.diff(edges)
    if density:
        return (fit.n_fit / max(fit.n_total, 1)) * probabilities / widths
    return fit.n_fit * probabilities


def beta_curve(x_grid: np.ndarray, fit: BetaFitResult, density: bool, bin_width: float) -> np.ndarray:
    curve = np.zeros_like(x_grid, dtype=float)
    inside = (x_grid >= fit.fit_lo) & (x_grid <= fit.fit_hi)
    if not np.any(inside):
        return curve

    scaled = (x_grid[inside] - fit.fit_lo) / (fit.fit_hi - fit.fit_lo)
    pdf = scibeta.pdf(scaled, fit.alpha, fit.beta) / (fit.fit_hi - fit.fit_lo)
    if density:
        curve[inside] = pdf * fit.n_fit / max(fit.n_total, 1)
    else:
        curve[inside] = pdf * fit.n_fit * bin_width
    return curve


def build_histogram_view(values: np.ndarray, edges: np.ndarray, density: bool) -> HistogramView:
    raw_counts, _ = np.histogram(values, bins=edges)
    raw_counts = raw_counts.astype(float)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = int(np.sum(raw_counts))

    if density and total > 0:
        probabilities = raw_counts / total
        hist_values = probabilities / widths
        hist_errors = np.sqrt(np.clip(probabilities * (1.0 - probabilities) / total, 0.0, None)) / widths
    elif density:
        hist_values = np.zeros_like(raw_counts)
        hist_errors = np.zeros_like(raw_counts)
    else:
        hist_values = raw_counts
        hist_errors = np.sqrt(raw_counts)

    return HistogramView(
        edges=np.asarray(edges, dtype=float),
        centers=centers,
        widths=widths,
        raw_counts=raw_counts,
        values=hist_values,
        errors=hist_errors,
        n_total=total,
    )


def step_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    edges: np.ndarray,
    density: bool,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    if values.size == 0:
        return
    counts, _ = np.histogram(values, bins=edges, density=density)
    step_x = np.append(edges[:-1], edges[-1])
    step_y = np.append(counts, counts[-1])
    ax.step(step_x, step_y, where="post", label=label, color=color, linestyle=linestyle)


def plot_overlay(
    outpath: Path,
    samples: Mapping[int, np.ndarray],
    bins: np.ndarray,
    title_label: str,
    density: bool,
    x_label: str,
    x_range: Tuple[float, float],
    fit_results: Optional[Mapping[int, BetaFitResult]] = None,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    x_grid = np.linspace(x_range[0], x_range[1], 500)
    bin_width = bins[1] - bins[0]

    for mode, values in samples.items():
        step_histogram(
            ax=ax,
            values=values,
            edges=bins,
            density=density,
            label=mode_label(mode),
            color=MODE_COLORS.get(mode, "tab:gray"),
            linestyle=MODE_LINESTYLES.get(mode, "-"),
        )
        if fit_results is not None and mode in fit_results:
            fit = fit_results[mode]
            ax.plot(
                x_grid,
                beta_curve(x_grid, fit, density=density, bin_width=bin_width),
                color=MODE_COLORS.get(mode, "tab:gray"),
                linestyle=":",
                linewidth=2.0,
                label=f"{mode_label(mode)} fit",
            )

    if logy:
        ax.set_yscale("log")
    ax.set_xlim(*x_range)
    ax.set_xlabel(x_label, loc="right")
    ax.set_ylabel("Density" if density else "Events", loc="top")
    ax.legend(loc="upper left")
    add_fcc_banner(ax, title_label)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_ratio_panel(
    outpath: Path,
    values: np.ndarray,
    fit: BetaFitResult,
    bins: np.ndarray,
    density: bool,
    x_label: str,
    x_range: Tuple[float, float],
    sample_label: str,
    mode: int,
    ratio_ylim: Tuple[float, float],
) -> None:
    hist = build_histogram_view(values, bins, density=density)
    model_heights = beta_bin_heights(hist.edges, fit, density=density)
    valid = (model_heights > 0.0) & np.isfinite(model_heights)

    ratio = np.full_like(hist.values, np.nan, dtype=float)
    ratio_err = np.full_like(hist.errors, np.nan, dtype=float)
    ratio[valid] = hist.values[valid] / model_heights[valid]
    ratio_err[valid] = hist.errors[valid] / model_heights[valid]

    x_grid = np.linspace(x_range[0], x_range[1], 500)
    bin_width = bins[1] - bins[0]
    fit_curve = beta_curve(x_grid, fit, density=density, bin_width=bin_width)

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
    ax_ratio.set_yticks([0.75,1.25])
    ax_main.errorbar(
        hist.centers,
        hist.values,
        yerr=hist.errors,
        fmt="o",
        capsize=1,
        color="black",
        label="MC",
    )
    ax_main.plot(
        x_grid,
        fit_curve,
        color=MODE_COLORS.get(mode, "tab:gray"),
        linestyle="--",
        label="Beta fit",
    )
    ax_main.set_xlim(*x_range)
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("Density" if density else "Events", loc="top")
    ax_main.legend(loc="upper left")
    ax_main.text(
        1,
        1.02,
        f"{mode_label(mode)}",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    add_fcc_banner(ax_main)

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_ratio.errorbar(
        hist.centers,
        ratio,
        yerr=ratio_err,
        fmt="o",
        capsize=1.5,
        color="black",
    )
    ax_ratio.set_xlim(*x_range)
    ax_ratio.set_xlim(left=0.01)
    ax_ratio.set_xlabel(x_label, loc="right")
    ax_ratio.set_ylabel("MC/fit")
    ax_ratio.set_ylim(*ratio_ylim)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay x_misP distributions by prong mode and optionally fit Beta PDFs. "
            "When fits are enabled, per-mode ratio plots are also produced."
        )
    )
    parser.add_argument("--file", required=True, help="Path to the ROOT file")
    parser.add_argument("--outdir", default="./xmisP_plots_ratio", help="Output directory")
    parser.add_argument("--tree", default=None, help="Optional explicit TTree name")
    parser.add_argument("--bins", type=int, default=30, help="Number of x bins")
    parser.add_argument("--x-range", default="0,1", help="x range 'lo,hi'")
    parser.add_argument("--modes", default="0,1,3", help="Comma-separated prong modes")
    parser.add_argument("--density", action="store_true", help="Plot as density instead of counts")
    parser.add_argument("--logy", action="store_true", help="Also write log-y overlay plots")
    parser.add_argument("--fit", action="store_true", help="Fit Beta PDFs and write mode-separated ratio plots")
    parser.add_argument(
        "--fit-range",
        default=None,
        help="Optional fit range 'lo,hi' (defaults to x-range)",
    )
    parser.add_argument(
        "--min-fit-count",
        type=int,
        default=10,
        help="Minimum number of entries required to attempt a Beta fit",
    )
    parser.add_argument(
        "--ratio-range",
        default="0.6,1.4",
        help="Ratio-panel y-range 'lo,hi'",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fit and not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for --fit and the ratio plots")

    x_range = parse_range(args.x_range)
    fit_range = parse_range(args.fit_range) if args.fit_range else x_range
    ratio_ylim = parse_range(args.ratio_range)
    modes = parse_modes(args.modes)
    bins = np.linspace(x_range[0], x_range[1], args.bins + 1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_input_data(args.file, args.tree)

    samples_by_label: Dict[str, Dict[int, np.ndarray]] = {
        "tau1": {},
        "tau2": {},
        "combined": {},
    }
    fit_results_by_label: Dict[str, Dict[int, BetaFitResult]] = {
        "tau1": {},
        "tau2": {},
        "combined": {},
    }
    summary_rows = []
    fit_rows = []

    for mode in modes:
        tau1_values = select_mode_values(data, leg=1, mode=mode, x_range=x_range)
        tau2_values = select_mode_values(data, leg=2, mode=mode, x_range=x_range)
        combined_values = (
            np.concatenate([tau1_values, tau2_values])
            if (tau1_values.size + tau2_values.size) > 0
            else np.array([], dtype=float)
        )

        samples_by_label["tau1"][mode] = tau1_values
        samples_by_label["tau2"][mode] = tau2_values
        samples_by_label["combined"][mode] = combined_values

        for sample_key, sample_values in (
            ("tau1", tau1_values),
            ("tau2", tau2_values),
            ("combined", combined_values),
        ):
            summary_rows.append(
                {
                    "mode": mode,
                    "mode_label": mode_label(mode),
                    "sample": sample_key,
                    **sample_stats(sample_values),
                }
            )
            if args.fit:
                fit_result = fit_beta_distribution(sample_values, fit_range, min_count=args.min_fit_count)
                if fit_result is None:
                    continue
                fit_results_by_label[sample_key][mode] = fit_result
                fit_rows.append(
                    {
                        "mode": mode,
                        "mode_label": mode_label(mode),
                        "sample": sample_key,
                        "alpha": fit_result.alpha,
                        "beta": fit_result.beta,
                        "fit_lo": fit_result.fit_lo,
                        "fit_hi": fit_result.fit_hi,
                        "n_total": fit_result.n_total,
                        "n_fit": fit_result.n_fit,
                    }
                )

    x_label_tau = r"$x_{\mathrm{mis}}^{P}$ (missing momentum fraction)"
    x_label_combined = "Missing momentum fraction"

    plot_overlay(
        outpath=outdir / "x_misP_tau1_overlay.pdf",
        samples=samples_by_label["tau1"],
        bins=bins,
        title_label=OVERLAY_TITLES["tau1"],
        density=args.density,
        x_label=x_label_tau,
        x_range=x_range,
        fit_results=fit_results_by_label["tau1"] if args.fit else None,
        logy=False,
    )
    plot_overlay(
        outpath=outdir / "x_misP_tau2_overlay.pdf",
        samples=samples_by_label["tau2"],
        bins=bins,
        title_label=OVERLAY_TITLES["tau2"],
        density=args.density,
        x_label=x_label_tau,
        x_range=x_range,
        fit_results=fit_results_by_label["tau2"] if args.fit else None,
        logy=False,
    )
    plot_overlay(
        outpath=outdir / "x_misP_combined_overlay.pdf",
        samples=samples_by_label["combined"],
        bins=bins,
        title_label=OVERLAY_TITLES["combined"],
        density=args.density,
        x_label=x_label_combined,
        x_range=x_range,
        fit_results=fit_results_by_label["combined"] if args.fit else None,
        logy=False,
    )

    if args.logy:
        plot_overlay(
            outpath=outdir / "x_misP_tau1_overlay_logy.pdf",
            samples=samples_by_label["tau1"],
            bins=bins,
            title_label=OVERLAY_TITLES["tau1"],
            density=args.density,
            x_label=x_label_tau,
            x_range=x_range,
            fit_results=fit_results_by_label["tau1"] if args.fit else None,
            logy=True,
        )
        plot_overlay(
            outpath=outdir / "x_misP_tau2_overlay_logy.pdf",
            samples=samples_by_label["tau2"],
            bins=bins,
            title_label=OVERLAY_TITLES["tau2"],
            density=args.density,
            x_label=x_label_tau,
            x_range=x_range,
            fit_results=fit_results_by_label["tau2"] if args.fit else None,
            logy=True,
        )
        plot_overlay(
            outpath=outdir / "x_misP_combined_overlay_logy.pdf",
            samples=samples_by_label["combined"],
            bins=bins,
            title_label=OVERLAY_TITLES["combined"],
            density=args.density,
            x_label=x_label_combined,
            x_range=x_range,
            fit_results=fit_results_by_label["combined"] if args.fit else None,
            logy=True,
        )

    if args.fit:
        ratio_root = outdir / "ratio_by_mode"
        for mode in modes:
            mode_dir = ratio_root / f"mode_{mode}_{mode_slug(mode)}"
            for sample_key, x_label in (
                ("tau1", x_label_tau),
                ("tau2", x_label_tau),
                ("combined", x_label_combined),
            ):
                if mode not in fit_results_by_label[sample_key]:
                    continue
                plot_ratio_panel(
                    outpath=mode_dir / f"x_misP_{sample_key}_ratio.pdf",
                    values=samples_by_label[sample_key][mode],
                    fit=fit_results_by_label[sample_key][mode],
                    bins=bins,
                    density=args.density,
                    x_label=x_label,
                    x_range=x_range,
                    sample_label=SAMPLE_LABELS[sample_key],
                    mode=mode,
                    ratio_ylim=ratio_ylim,
                )

    pd.DataFrame(summary_rows).to_csv(outdir / "x_misP_summary.csv", index=False)
    if args.fit and fit_rows:
        pd.DataFrame(fit_rows).to_csv(outdir / "x_misP_fits_beta.csv", index=False)


if __name__ == "__main__":
    main()

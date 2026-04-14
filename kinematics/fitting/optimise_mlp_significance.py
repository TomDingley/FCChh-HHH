from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from aesthetics import LABEL_MAP, process_colours, process_labels, banner_heatmaps
from config import BACKGROUNDS, LUMINOSITY_PB, N_BINS_1D, SELECTION, SIGNAL, XLIM_MAP
from tools import BASIS_KEYS, build_mask_from_selection, build_moments, evaluate_weights_from_moments


@dataclass(frozen=True)
class ScanRange:
    start: float
    stop: float
    steps: int

    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.steps)

# lots of defaults
DEFAULT_OUTDIR = "fit"
DEFAULT_CHANNEL = "HadHad"
DEFAULT_SIGNAL = SIGNAL
DEFAULT_VAR = "m_hhh_vis"
DEFAULT_MLP_VAR = "mlp_score"
DEFAULT_MLP_RANGE = ScanRange(0.5, 0.99, 51)

DEFAULT_K3 = 1.0
DEFAULT_K4 = 1.0
DEFAULT_K3_TRUE = 1.0
DEFAULT_K4_TRUE = 1.0
DEFAULT_K3K4_TRUE = (1.0, 1.0)

DEFAULT_K3_RANGE = ScanRange(-2.5, 5.0, 101)
DEFAULT_K4_RANGE = ScanRange(-10.0, 30.0, 101)
DEFAULT_K3_2D_RANGE = ScanRange(-3.0, 6.0, 41)
DEFAULT_K4_2D_RANGE = ScanRange(-15.0, 35.0, 51)

RUN_K3_SCAN = True
RUN_K4_SCAN = True
DIAGNOSTICS_N = 20
DIAGNOSTICS_K3 = True
DIAGNOSTICS_K4 = True
DIAGNOSTICS_K3K4_CONTOUR = True
DIAGNOSTICS_SHAPE_K4 = True
SHAPE_K4_POINTS = [-20.0, 1.0, 20.0]
MLP_K4_LIMIT_CURVE = True


@dataclass
class SignalCache:
    var: np.ndarray
    mlp: np.ndarray
    w_xsec: np.ndarray
    moments: np.ndarray


@dataclass
class BackgroundCache:
    proc: str
    var: np.ndarray
    mlp: np.ndarray
    w_xsec: np.ndarray


def _strip_outer_parens(expr: str) -> str:
    out = expr.strip()
    while out.startswith("(") and out.endswith(")"):
        inner = out[1:-1].strip()
        if not inner:
            return ""
        out = inner
    return out


def _is_trivial_bool(expr: str) -> bool | None:
    if not expr:
        return True
    out = _strip_outer_parens(expr)
    if out == "True":
        return True
    if out == "False":
        return False
    return None


def _strip_mlp_cut(selection: str, mlp_var: str) -> str:
    if not selection:
        return ""
    num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = [
        rf"{mlp_var}\s*[<>]=?\s*{num}",
        rf"{num}\s*[<>]=?\s*{mlp_var}",
    ]
    out = selection
    for pat in patterns:
        out = re.sub(pat, "True", out)
    out = re.sub(r"\s+", " ", out).strip()
    trivial = _is_trivial_bool(out)
    if trivial is True:
        return ""
    if trivial is False:
        return "False"
    return out


def _build_mask(tree, selection: str) -> np.ndarray:
    if not selection:
        return np.ones(tree.num_entries, dtype=bool)
    sel = selection.strip()
    trivial = _is_trivial_bool(sel)
    if trivial is True:
        return np.ones(tree.num_entries, dtype=bool)
    if trivial is False:
        return np.zeros(tree.num_entries, dtype=bool)
    return np.asarray(build_mask_from_selection(tree, selection))


def _load_signal_cache(
    files: dict[str, Path],
    signal_name: str,
    selection: str,
    var: str,
    mlp_var: str,
) -> SignalCache:
    fp = files.get(signal_name)
    if not fp:
        raise FileNotFoundError(f"Signal file for {signal_name} not found.")

    with uproot.open(fp) as f_sig:
        tree = f_sig["events"]
        mask = _build_mask(tree, selection)

        var_arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
        mlp_arr = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
        w_xsec = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB

        weight_dict = {
            key: np.asarray(tree[key].array(library="np")[mask], dtype=float)
            for key in BASIS_KEYS
        }

    finite = np.isfinite(var_arr) & np.isfinite(mlp_arr) & np.isfinite(w_xsec)
    for arr in weight_dict.values():
        finite &= np.isfinite(arr)

    var_arr = var_arr[finite]
    mlp_arr = mlp_arr[finite]
    w_xsec = w_xsec[finite]
    weight_dict = {k: v[finite] for k, v in weight_dict.items()}

    moments = build_moments(weight_dict)
    return SignalCache(var=var_arr, mlp=mlp_arr, w_xsec=w_xsec, moments=moments)


def _load_background_caches(files: dict[str, Path], selection: str, var: str, mlp_var: str) -> list[BackgroundCache]:
    caches: list[BackgroundCache] = []
    for proc in BACKGROUNDS:
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            tree = f_bkg["events"]
            mask = _build_mask(tree, selection)

            var_arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
            mlp_arr = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
            w_xsec = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB

        finite = np.isfinite(var_arr) & np.isfinite(mlp_arr) & np.isfinite(w_xsec)
        var_arr = var_arr[finite]
        mlp_arr = mlp_arr[finite]
        w_xsec = w_xsec[finite]

        if var_arr.size:
            caches.append(BackgroundCache(proc=proc, var=var_arr, mlp=mlp_arr, w_xsec=w_xsec))
    return caches


def _hist_for_cut(var_arr: np.ndarray, mlp_arr: np.ndarray, weights: np.ndarray, cut: float, edges: np.ndarray) -> np.ndarray:
    mask = mlp_arr > cut
    if not np.any(mask):
        return np.zeros(len(edges) - 1, dtype=float)
    return np.histogram(var_arr[mask], bins=edges, weights=weights[mask])[0]


def _asimov_z_from_hist(s: np.ndarray, b: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = b > 0
        q0 = np.zeros_like(b, dtype=float)
        q0[mask] = 2.0 * ((s[mask] + b[mask]) * np.log1p(s[mask] / b[mask]) - s[mask])
        z2 = np.sum(q0)
    return float(np.sqrt(max(z2, 0.0)))


def _asimov_z_single(s: float, b: float) -> float:
    if b <= 0 or s <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(max(z2, 0.0)))


def _nll_poisson(n: np.ndarray, mu: np.ndarray) -> float:
    term = np.zeros_like(n, dtype=float)
    mask_mu = mu > 0
    mask_n = (n > 0) & mask_mu
    term[mask_n] = mu[mask_n] - n[mask_n] + n[mask_n] * np.log(n[mask_n] / mu[mask_n])
    term[mask_mu & ~mask_n] = mu[mask_mu & ~mask_n]
    if np.any(~mask_mu & (n > 0)):
        return float("inf")
    return float(2.0 * np.sum(term))


def _compute_k4_nll_arrays(
    *,
    sig_cache: SignalCache,
    bkg_caches: list[BackgroundCache],
    edges: np.ndarray,
    mlp_cut: float,
    k3: float,
    k4_vals: np.ndarray,
    k4_true: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    sig_sel = sig_cache.mlp > mlp_cut
    if not np.any(sig_sel):
        return None

    var_sel = sig_cache.var[sig_sel]
    w_xsec_sel = sig_cache.w_xsec[sig_sel]
    moments_sel = sig_cache.moments[sig_sel]

    h_bkg = np.zeros(len(edges) - 1, dtype=float)
    for bkg in bkg_caches:
        h_bkg += _hist_for_cut(bkg.var, bkg.mlp, bkg.w_xsec, mlp_cut, edges)

    w_true = evaluate_weights_from_moments(k3, k4_true, moments_sel) * w_xsec_sel
    h_sig_true = np.histogram(var_sel, bins=edges, weights=w_true)[0]
    n_hist = h_sig_true + h_bkg

    n_total = float(np.sum(h_sig_true) + np.sum(h_bkg))
    b_total = float(np.sum(h_bkg))

    nll_binned = np.zeros_like(k4_vals, dtype=float)
    nll_single = np.zeros_like(k4_vals, dtype=float)

    for i, k4 in enumerate(k4_vals):
        w = evaluate_weights_from_moments(k3, k4, moments_sel) * w_xsec_sel
        h_sig = np.histogram(var_sel, bins=edges, weights=w)[0]
        mu_hist = h_sig + h_bkg
        nll_binned[i] = _nll_poisson(n_hist, mu_hist)

        s_total = float(np.sum(w))
        mu_total = s_total + b_total
        nll_single[i] = _nll_poisson(np.array([n_total]), np.array([mu_total]))

    nll_binned -= np.nanmin(nll_binned)
    nll_single -= np.nanmin(nll_single)

    return nll_binned, nll_single


def _compute_k3_nll_arrays(
    *,
    sig_cache: SignalCache,
    bkg_caches: list[BackgroundCache],
    edges: np.ndarray,
    mlp_cut: float,
    k3_vals: np.ndarray,
    k4: float,
    k3_true: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    sig_sel = sig_cache.mlp > mlp_cut
    if not np.any(sig_sel):
        return None

    var_sel = sig_cache.var[sig_sel]
    w_xsec_sel = sig_cache.w_xsec[sig_sel]
    moments_sel = sig_cache.moments[sig_sel]

    h_bkg = np.zeros(len(edges) - 1, dtype=float)
    for bkg in bkg_caches:
        h_bkg += _hist_for_cut(bkg.var, bkg.mlp, bkg.w_xsec, mlp_cut, edges)

    w_true = evaluate_weights_from_moments(k3_true, k4, moments_sel) * w_xsec_sel
    h_sig_true = np.histogram(var_sel, bins=edges, weights=w_true)[0]
    n_hist = h_sig_true + h_bkg

    n_total = float(np.sum(h_sig_true) + np.sum(h_bkg))
    b_total = float(np.sum(h_bkg))

    nll_binned = np.zeros_like(k3_vals, dtype=float)
    nll_single = np.zeros_like(k3_vals, dtype=float)

    for i, k3 in enumerate(k3_vals):
        w = evaluate_weights_from_moments(k3, k4, moments_sel) * w_xsec_sel
        h_sig = np.histogram(var_sel, bins=edges, weights=w)[0]
        mu_hist = h_sig + h_bkg
        nll_binned[i] = _nll_poisson(n_hist, mu_hist)

        s_total = float(np.sum(w))
        mu_total = s_total + b_total
        nll_single[i] = _nll_poisson(np.array([n_total]), np.array([mu_total]))

    nll_binned -= np.nanmin(nll_binned)
    nll_single -= np.nanmin(nll_single)

    return nll_binned, nll_single


def _compute_k3k4_nll_grid(
    *,
    sig_cache: SignalCache,
    bkg_caches: list[BackgroundCache],
    edges: np.ndarray,
    mlp_cut: float,
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    k3_true: float,
    k4_true: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    sig_sel = sig_cache.mlp > mlp_cut
    if not np.any(sig_sel):
        return None

    var_sel = sig_cache.var[sig_sel]
    w_xsec_sel = sig_cache.w_xsec[sig_sel]
    moments_sel = sig_cache.moments[sig_sel]

    h_bkg = np.zeros(len(edges) - 1, dtype=float)
    for bkg in bkg_caches:
        h_bkg += _hist_for_cut(bkg.var, bkg.mlp, bkg.w_xsec, mlp_cut, edges)

    w_true = evaluate_weights_from_moments(k3_true, k4_true, moments_sel) * w_xsec_sel
    h_sig_true = np.histogram(var_sel, bins=edges, weights=w_true)[0]
    n_hist = h_sig_true + h_bkg

    nll_true = _nll_poisson(n_hist, h_sig_true + h_bkg)
    b_total = float(np.sum(h_bkg))
    n_total = float(np.sum(n_hist))
    s_true_total = float(np.sum(w_true))
    mu_true_total = s_true_total + b_total
    nll_true_single = _nll_poisson(np.array([n_total]), np.array([mu_true_total]))

    nll_grid = np.zeros((len(k3_vals), len(k4_vals)), dtype=float)
    nll_single_grid = np.zeros_like(nll_grid)
    for i, k3 in enumerate(k3_vals):
        for j, k4 in enumerate(k4_vals):
            w = evaluate_weights_from_moments(k3, k4, moments_sel) * w_xsec_sel
            h_sig = np.histogram(var_sel, bins=edges, weights=w)[0]
            mu_hist = h_sig + h_bkg
            nll_grid[i, j] = _nll_poisson(n_hist, mu_hist)

            s_total = float(np.sum(w))
            mu_total = s_total + b_total
            nll_single_grid[i, j] = _nll_poisson(np.array([n_total]), np.array([mu_total]))

    nll_grid -= nll_true
    nll_single_grid -= nll_true_single
    return nll_grid, nll_single_grid


def _edges_for_var(var: str, sig_cache: SignalCache) -> np.ndarray:
    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
    else:
        xmin, xmax = float(np.min(sig_cache.var)), float(np.max(sig_cache.var))
    return np.linspace(xmin, xmax, N_BINS_1D + 1)


def _plot_binned_distribution(
    *,
    edges: np.ndarray,
    h_sig: np.ndarray,
    bkg_hists: dict[str, np.ndarray],
    signal_name: str,
    var_label: str,
    banner_text: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    x = np.append(edges[:-1], edges[-1])
    bottom = np.zeros_like(h_sig)

    # Stack backgrounds
    for proc, hist in bkg_hists.items():
        if not np.any(hist):
            continue
        step_bottom = np.append(bottom, bottom[-1])
        step_y = np.append(hist, hist[-1])
        color = process_colours.get(proc, "tab:gray")
        label = process_labels.get(proc, proc)
        ax.fill_between(x, step_bottom, step_bottom + step_y, step="post", color=color, alpha=0.9, label=label)
        bottom += hist

    # Total background outline
    if np.any(bottom):
        ax.step(x, np.append(bottom, bottom[-1]), where="post", color="black", lw=1.2, label="Total background")

    # Signal overlay
    sig_color = process_colours.get(signal_name, "tab:red")
    sig_label = process_labels.get(signal_name, signal_name)
    ax.step(x, np.append(h_sig, h_sig[-1]), where="post", color=sig_color, lw=2.0, linestyle="--", label=sig_label)

    ax.set_xlabel(var_label, loc="right")
    ax.set_ylabel("Events / bin", loc="top")
    ax.set_xlim(edges[0], edges[-1])
    if (bottom + h_sig).ptp() > 1e3:
        ax.set_yscale("log")

    ax.legend(frameon=False, fontsize=9)
    banner_heatmaps(ax, banner_text)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_single_bin_stack(
    *,
    s_total: float,
    bkg_totals: dict[str, float],
    signal_name: str,
    banner_text: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    x = np.array([0.0])
    width = 0.6
    bottom = 0.0

    for proc, total in bkg_totals.items():
        if total <= 0:
            continue
        color = process_colours.get(proc, "tab:gray")
        label = process_labels.get(proc, proc)
        ax.bar(x, total, bottom=bottom, width=width, color=color, alpha=0.9, label=label)
        bottom += total

    sig_color = process_colours.get(signal_name, "tab:red")
    sig_label = process_labels.get(signal_name, signal_name)
    if s_total > 0:
        ax.bar(x, s_total, bottom=bottom, width=width, color=sig_color, alpha=0.7, label=sig_label)

    ax.set_xticks([0.0])
    ax.set_xticklabels(["Total"])
    ax.set_ylabel("Events", loc="top")
    if (bottom + s_total) > 0 and (bottom + s_total) > 1e3:
        ax.set_yscale("log")

    ax.legend(frameon=False, fontsize=9)
    banner_heatmaps(ax, banner_text)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_signal_k4_variations(
    *,
    var_arr: np.ndarray,
    moments: np.ndarray,
    w_xsec: np.ndarray,
    edges: np.ndarray,
    k3: float,
    k4_points: list[float],
    k4_ref: float,
    var_label: str,
    banner_text: str,
    outpath: Path,
) -> None:
    if var_arr.size == 0:
        return

    # ensure ref is plotted first and ratios exist for all other points
    unique_points = []
    for k4 in k4_points:
        if k4 not in unique_points:
            unique_points.append(k4)
    if k4_ref in unique_points:
        unique_points = [k4_ref] + [k4 for k4 in unique_points if k4 != k4_ref]
    else:
        unique_points = [k4_ref] + unique_points

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(6.4, 6.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    step_x = np.append(edges[:-1], edges[-1])
    # reference histogram
    weights_ref = evaluate_weights_from_moments(k3, k4_ref, moments) * w_xsec
    h_ref = np.histogram(var_arr, bins=edges, weights=weights_ref)[0]
    if h_ref.sum() == 0:
        return
    h_ref_norm = h_ref / h_ref.sum()
    ref_hist = np.append(h_ref_norm, h_ref_norm[-1])
    ax.step(step_x, ref_hist, where="post", lw=2.2, color="black", label=f"$k_4={k4_ref:g}$")

    for k4 in unique_points:
        if k4 == k4_ref:
            continue
        weights = evaluate_weights_from_moments(k3, k4, moments) * w_xsec
        h = np.histogram(var_arr, bins=edges, weights=weights)[0]
        if h.sum() == 0:
            continue
        h_norm = h / h.sum()
        step_y = np.append(h_norm, h_norm[-1])

        label = f"$k_4={k4:g}$"
        ax.step(step_x, step_y, where="post", lw=1.6, label=label)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(step_y, ref_hist, out=np.ones_like(step_y), where=ref_hist != 0)
        ax_ratio.step(step_x, ratio, where="post", lw=1.4)

    ax.set_ylabel("Normalised events", loc="top")
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(frameon=False, fontsize=9)
    banner_heatmaps(ax, banner_text)
    ax.tick_params(labelbottom=False)

    ax_ratio.set_xlabel(var_label, loc="right")
    ax_ratio.set_ylabel(f"Ratio to SM", loc="center")
    ax_ratio.axhline(1.0, color="black", linestyle="--", lw=1)
    ax_ratio.set_ylim(0.5, 1.5)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_k3k4_contour(
    *,
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    nll_binned: np.ndarray,
    nll_single: np.ndarray,
    level: float,
    k3_true: float,
    k4_true: float,
    outpath: Path,
    comment: str,
) -> None:
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    ax.contour(K3, K4, nll_binned, levels=[level], colors="black", linewidths=1.8)
    ax.contour(K3, K4, nll_single, levels=[level], colors="gray", linewidths=1.6, linestyles="--")
    ax.scatter([k3_true], [k4_true], color="red", marker="*", s=60, zorder=5)

    ax.set_xlabel(r"$k_3$", loc="right")
    ax.set_ylabel(r"$k_4$", loc="top")
    ax.set_xlim(k3_vals.min(), k3_vals.max())
    ax.set_ylim(k4_vals.min(), k4_vals.max())
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.8, label="Binned"),
            Line2D([0], [0], color="gray", lw=1.6, ls="--", label="Single-bin"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    banner_heatmaps(ax, comment=comment)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_k4_limit_widths(
    *,
    cuts: np.ndarray,
    widths_binned: np.ndarray,
    widths_single: np.ndarray,
    outpath: Path,
    mlp_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    ax.plot(cuts, widths_binned, lw=2.0, label="Binned")
    ax.plot(cuts, widths_single, lw=2.0, ls="--", label="Single-bin")

    ax.set_ylabel("k4 range", loc="top")
    ax.set_xlabel(mlp_label, loc="right")
    ax.legend(frameon=False, fontsize=9)
    banner_heatmaps(ax, comment=title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _select_diagnostic_indices(
    n_cuts: int,
    n_diag: int,
    best_idxs: list[int],
    *,
    force_include_best: bool = True,
) -> list[int]:
    if n_cuts <= 0 or n_diag <= 0:
        return []
    n_diag = min(n_diag, n_cuts)
    base = list(np.linspace(0, n_cuts - 1, n_diag, dtype=int))

    if force_include_best:
        base = base + [idx for idx in best_idxs if 0 <= idx < n_cuts]

    seen = set()
    uniq: list[int] = []
    for idx in base:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)

    if len(uniq) < n_diag:
        for idx in range(n_cuts):
            if idx not in seen:
                uniq.append(idx)
                seen.add(idx)
            if len(uniq) >= n_diag:
                break

    return uniq


def scan_mlp_significance(
    *,
    files: dict[str, Path],
    signal_name: str,
    var: str,
    channel: str,
    mlp_var: str,
    cuts: np.ndarray,
    k3: float,
    k4: float,
    outdir: Path,
    diagnostics_n: int = 0,
    diagnostics_k4: bool = False,
    diagnostics_k4_vals: np.ndarray | None = None,
    diagnostics_k4_true: float | None = None,
    diagnostics_k3: bool = False,
    diagnostics_k3_vals: np.ndarray | None = None,
    diagnostics_k3_true: float | None = None,
    diagnostics_shape_k4: bool = False,
    diagnostics_shape_k4_points: list[float] | None = None,
    diagnostics_k3k4_contour: bool = False,
    diagnostics_k3_vals_2d: np.ndarray | None = None,
    diagnostics_k4_vals_2d: np.ndarray | None = None,
    diagnostics_k3k4_true: tuple[float, float] | None = None,
    limit_curve: bool = False,
    limit_k4_vals: np.ndarray | None = None,
    limit_k4_true: float | None = None,
    limit_thresholds: list[float] | None = None,
) -> dict[str, np.ndarray | float]:
    base_sel = _strip_mlp_cut(SELECTION[channel], mlp_var)
    sig_cache = _load_signal_cache(files, signal_name, base_sel, var, mlp_var)
    bkg_caches = _load_background_caches(files, base_sel, var, mlp_var)

    edges = _edges_for_var(var, sig_cache)
    sig_weights = evaluate_weights_from_moments(k3, k4, sig_cache.moments) * sig_cache.w_xsec

    z_binned = np.zeros_like(cuts, dtype=float)
    z_single = np.zeros_like(cuts, dtype=float)
    s_totals = np.zeros_like(cuts, dtype=float)
    b_totals = np.zeros_like(cuts, dtype=float)

    for i, cut in enumerate(cuts):
        h_sig = _hist_for_cut(sig_cache.var, sig_cache.mlp, sig_weights, cut, edges)
        h_bkg = np.zeros_like(h_sig)
        for bkg in bkg_caches:
            h_bkg += _hist_for_cut(bkg.var, bkg.mlp, bkg.w_xsec, cut, edges)

        s = float(np.sum(h_sig))
        b = float(np.sum(h_bkg))
        s_totals[i] = s
        b_totals[i] = b
        z_binned[i] = _asimov_z_from_hist(h_sig, h_bkg)
        z_single[i] = _asimov_z_single(s, b)

    best_binned_idx = int(np.nanargmax(z_binned)) if np.any(np.isfinite(z_binned)) else 0
    best_single_idx = int(np.nanargmax(z_single)) if np.any(np.isfinite(z_single)) else 0

    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / f"mlp_significance_scan_{channel}_{var}.npz"
    np.savez(
        out_npz,
        cuts=cuts,
        z_binned=z_binned,
        z_single=z_single,
        s_total=s_totals,
        b_total=b_totals,
        k3=k3,
        k4=k4,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(cuts, z_binned, lw=2.0, label="Binned m_hhh_vis")
    ax.plot(cuts, z_single, lw=2.0, ls="--", label="Single-bin")
    ax.axvline(cuts[best_binned_idx], color="tab:blue", ls=":", lw=1.6)
    ax.axvline(cuts[best_single_idx], color="tab:orange", ls=":", lw=1.6)
    ax.set_xlabel(f"{LABEL_MAP.get(mlp_var, mlp_var)} cut", loc="right")
    ax.set_ylabel("Asimov Z", loc="top")
    ax.legend(frameon=False, loc="best")
    banner_heatmaps(ax, fr"{channel} | ($\kappa_3$={k3:g}, $\kappa_4$={k4:g})")
    out_pdf = outdir / f"mlp_significance_scan_{channel}_{var}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    if diagnostics_n > 0:
        diag_dir = outdir / "diagnostics"
        best_idxs = [best_binned_idx, best_single_idx]
        diag_idxs = _select_diagnostic_indices(
            len(cuts),
            diagnostics_n,
            best_idxs,
            force_include_best=True,
        )

        var_label = LABEL_MAP.get(var, var)
        for idx in diag_idxs:
            cut = float(cuts[idx])
            h_sig = _hist_for_cut(sig_cache.var, sig_cache.mlp, sig_weights, cut, edges)
            bkg_hists = {
                bkg.proc: _hist_for_cut(bkg.var, bkg.mlp, bkg.w_xsec, cut, edges)
                for bkg in bkg_caches
            }

            banner_text = f"{channel} | {mlp_var}>{cut:.3f} | k3={k3:g}, k4={k4:g}"
            tag = f"cut_{idx:03d}_mlp_{cut:.3f}"

            _plot_binned_distribution(
                edges=edges,
                h_sig=h_sig,
                bkg_hists=bkg_hists,
                signal_name=signal_name,
                var_label=var_label,
                banner_text=banner_text,
                outpath=diag_dir / f"{var}_{tag}.pdf",
            )

            bkg_totals = {proc: float(np.sum(hist)) for proc, hist in bkg_hists.items()}
            _plot_single_bin_stack(
                s_total=float(np.sum(h_sig)),
                bkg_totals=bkg_totals,
                signal_name=signal_name,
                banner_text=banner_text,
                outpath=diag_dir / f"singlebin_{tag}.pdf",
            )

            if diagnostics_k4:
                if diagnostics_k4_vals is None:
                    raise ValueError("diagnostics_k4_vals must be provided when diagnostics_k4=True.")
                k4_true = float(k4 if diagnostics_k4_true is None else diagnostics_k4_true)
                diag_k4_dir = diag_dir / "k4_scans"
                scan_k4_likelihood(
                    sig_cache=sig_cache,
                    bkg_caches=bkg_caches,
                    edges=edges,
                    mlp_cut=cut,
                    k3=k3,
                    k4_vals=diagnostics_k4_vals,
                    k4_true=k4_true,
                    outdir=diag_k4_dir,
                    channel=channel,
                    var=var,
                    tag=tag,
                )

            if diagnostics_k3:
                if diagnostics_k3_vals is None:
                    raise ValueError("diagnostics_k3_vals must be provided when diagnostics_k3=True.")
                k3_true = float(k3 if diagnostics_k3_true is None else diagnostics_k3_true)
                diag_k3_dir = diag_dir / "k3_scans"
                scan_k3_likelihood(
                    sig_cache=sig_cache,
                    bkg_caches=bkg_caches,
                    edges=edges,
                    mlp_cut=cut,
                    k3_vals=diagnostics_k3_vals,
                    k4=k4,
                    k3_true=k3_true,
                    outdir=diag_k3_dir,
                    channel=channel,
                    var=var,
                    tag=tag,
                )

            if diagnostics_shape_k4:
                if diagnostics_shape_k4_points is None:
                    raise ValueError("diagnostics_shape_k4_points must be provided when diagnostics_shape_k4=True.")
                sig_sel = sig_cache.mlp > cut
                _plot_signal_k4_variations(
                    var_arr=sig_cache.var[sig_sel],
                    moments=sig_cache.moments[sig_sel],
                    w_xsec=sig_cache.w_xsec[sig_sel],
                    edges=edges,
                    k3=k3,
                    k4_points=diagnostics_shape_k4_points,
                    k4_ref=k4,
                    var_label=var_label,
                    banner_text=banner_text,
                    outpath=diag_dir / "k4_shapes" / f"{var}_k4shape_{tag}.pdf",
                )

            if diagnostics_k3k4_contour:
                if diagnostics_k3_vals_2d is None or diagnostics_k4_vals_2d is None:
                    raise ValueError("diagnostics_k3_vals_2d and diagnostics_k4_vals_2d must be provided when diagnostics_k3k4_contour=True.")
                k3_true, k4_true = diagnostics_k3k4_true or (1.0, 1.0)
                nll = _compute_k3k4_nll_grid(
                    sig_cache=sig_cache,
                    bkg_caches=bkg_caches,
                    edges=edges,
                    mlp_cut=cut,
                    k3_vals=diagnostics_k3_vals_2d,
                    k4_vals=diagnostics_k4_vals_2d,
                    k3_true=k3_true,
                    k4_true=k4_true,
                )
                if nll is None:
                    continue
                nll_binned, nll_single = nll
                contour_dir = diag_dir / "k3k4_contours"
                comment = f"{channel} | mlp>{cut:.3f} | 1$\\sigma$ ($\\Delta$NLL=2.30)"
                out_pdf = contour_dir / f"k3k4_1sigma_{tag}.pdf"
                _plot_k3k4_contour(
                    k3_vals=diagnostics_k3_vals_2d,
                    k4_vals=diagnostics_k4_vals_2d,
                    nll_binned=nll_binned,
                    nll_single=nll_single,
                    level=2.30,
                    k3_true=k3_true,
                    k4_true=k4_true,
                    outpath=out_pdf,
                    comment=comment,
                )

                out_npz = contour_dir / f"k3k4_nll_grid_{tag}.npz"
                np.savez(out_npz, k3=diagnostics_k3_vals_2d, k4=diagnostics_k4_vals_2d,
                         nll_binned=nll_binned, nll_single=nll_single,
                         mlp_cut=cut, k3_true=k3_true, k4_true=k4_true)

    if limit_curve:
        if limit_k4_vals is None:
            raise ValueError("limit_k4_vals must be provided when limit_curve=True.")
        k4_true = float(k4 if limit_k4_true is None else limit_k4_true)
        thresholds = limit_thresholds or [1.0, 3.84]

        widths_binned = {thr: np.full_like(cuts, np.nan, dtype=float) for thr in thresholds}
        widths_single = {thr: np.full_like(cuts, np.nan, dtype=float) for thr in thresholds}

        for i, cut in enumerate(cuts):
            nll = _compute_k4_nll_arrays(
                sig_cache=sig_cache,
                bkg_caches=bkg_caches,
                edges=edges,
                mlp_cut=float(cut),
                k3=k3,
                k4_vals=limit_k4_vals,
                k4_true=k4_true,
            )
            if nll is None:
                continue
            nll_binned, nll_single = nll

            for thr in thresholds:
                mask_b = nll_binned <= thr
                mask_s = nll_single <= thr
                if np.any(mask_b):
                    widths_binned[thr][i] = float(limit_k4_vals[mask_b].max() - limit_k4_vals[mask_b].min())
                if np.any(mask_s):
                    widths_single[thr][i] = float(limit_k4_vals[mask_s].max() - limit_k4_vals[mask_s].min())

        mlp_label = f"{LABEL_MAP.get(mlp_var, mlp_var)} cut"
        for thr in thresholds:
            if abs(thr - 1.0) < 1e-6:
                title = "68% CL"
                tag = "68"
            elif abs(thr - 3.84) < 1e-6:
                title = "95% CL"
                tag = "95"
            else:
                title = f"ΔNLL={thr:g}"
                tag = f"thr_{thr:g}"

            out_limit = outdir / f"k4_limit_range_vs_mlp_{channel}_{var}_{tag}.pdf"
            _plot_k4_limit_widths(
                cuts=cuts,
                widths_binned=widths_binned[thr],
                widths_single=widths_single[thr],
                outpath=out_limit,
                mlp_label=mlp_label,
                title=title,
            )

        out_npz = outdir / f"k4_limit_range_vs_mlp_{channel}_{var}.npz"
        np.savez(out_npz, cuts=cuts, **{f"binned_thr_{thr:g}": arr for thr, arr in widths_binned.items()},
                 **{f"single_thr_{thr:g}": arr for thr, arr in widths_single.items()})

    return {
        "cuts": cuts,
        "z_binned": z_binned,
        "z_single": z_single,
        "best_cut_binned": float(cuts[best_binned_idx]),
        "best_cut_single": float(cuts[best_single_idx]),
        "sig_cache": sig_cache,
        "bkg_caches": bkg_caches,
        "edges": edges,
    }


def scan_k4_likelihood(
    *,
    sig_cache: SignalCache,
    bkg_caches: list[BackgroundCache],
    edges: np.ndarray,
    mlp_cut: float,
    k3: float,
    k4_vals: np.ndarray,
    k4_true: float,
    outdir: Path,
    channel: str,
    var: str,
    tag: str | None = None,
) -> dict[str, np.ndarray]:
    nll = _compute_k4_nll_arrays(
        sig_cache=sig_cache,
        bkg_caches=bkg_caches,
        edges=edges,
        mlp_cut=mlp_cut,
        k3=k3,
        k4_vals=k4_vals,
        k4_true=k4_true,
    )
    if nll is None:
        raise RuntimeError("No signal entries after mlp cut.")
    nll_binned, nll_single = nll

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_npz = outdir / f"k4_nll_scan_{channel}_{var}{suffix}.npz"
    np.savez(
        out_npz,
        k4=k4_vals,
        nll_binned=nll_binned,
        nll_single=nll_single,
        mlp_cut=mlp_cut,
        k3=k3,
        k4_true=k4_true,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(k4_vals, nll_binned, lw=2.0, label="Binned m_hhh_vis")
    ax.plot(k4_vals, nll_single, lw=2.0, ls="--", label="Single-bin")
    ax.axvline(k4_true, color="black", ls=":", lw=1.4)
    ax.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax.axhline(3.84, color="gray", ls="--", lw=1.0)
    ax.set_ylim(0.0, 5)
    ax.set_xlabel(r"$k_4$", loc="right")
    ax.set_ylabel(r"$-2\Delta\ln L$", loc="top")
    ax.legend(frameon=False, loc="best")
    banner_heatmaps(ax, f"{channel} | mlp>{mlp_cut:.3f} | k3={k3:g}")
    out_pdf = outdir / f"k4_nll_scan_{channel}_{var}{suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "k4": k4_vals,
        "nll_binned": nll_binned,
        "nll_single": nll_single,
    }


def scan_k3_likelihood(
    *,
    sig_cache: SignalCache,
    bkg_caches: list[BackgroundCache],
    edges: np.ndarray,
    mlp_cut: float,
    k3_vals: np.ndarray,
    k4: float,
    k3_true: float,
    outdir: Path,
    channel: str,
    var: str,
    tag: str | None = None,
) -> dict[str, np.ndarray]:
    nll = _compute_k3_nll_arrays(
        sig_cache=sig_cache,
        bkg_caches=bkg_caches,
        edges=edges,
        mlp_cut=mlp_cut,
        k3_vals=k3_vals,
        k4=k4,
        k3_true=k3_true,
    )
    if nll is None:
        raise RuntimeError("No signal entries after mlp cut.")
    nll_binned, nll_single = nll

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_npz = outdir / f"k3_nll_scan_{channel}_{var}{suffix}.npz"
    np.savez(
        out_npz,
        k3=k3_vals,
        nll_binned=nll_binned,
        nll_single=nll_single,
        mlp_cut=mlp_cut,
        k4=k4,
        k3_true=k3_true,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(k3_vals, nll_binned, lw=2.0, label="Binned m_hhh_vis")
    ax.plot(k3_vals, nll_single, lw=2.0, ls="--", label="Single-bin")
    ax.axvline(k3_true, color="black", ls=":", lw=1.4)
    ax.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax.axhline(3.84, color="gray", ls="--", lw=1.0)
    ax.set_ylim(0.0, 5)
    ax.set_xlabel(r"$k_3$", loc="right")
    ax.set_ylabel(r"$-2\Delta\ln L$", loc="top")
    ax.legend(frameon=False, loc="best")
    banner_heatmaps(ax, f"{channel} | mlp>{mlp_cut:.3f} | k4={k4:g}")
    out_pdf = outdir / f"k3_nll_scan_{channel}_{var}{suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "k3": k3_vals,
        "nll_binned": nll_binned,
        "nll_single": nll_single,
    }


def cli() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize the standard MLP cut and make the standard k3/k4 diagnostics."
    )
    ap.add_argument("--root-dir", required=True, help="Directory containing ROOT files")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"Output directory (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL, choices=list(SELECTION.keys()))
    ap.add_argument("--mlp-cut", type=float, default=None, help="Override mlp cut for k4 scan (default: best binned)")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = {
        proc: root / f"{proc}.root"
        for proc in [DEFAULT_SIGNAL] + [p for p in BACKGROUNDS if (root / f"{p}.root").is_file()]
    }
    if not files.get(DEFAULT_SIGNAL) or not files[DEFAULT_SIGNAL].is_file():
        raise FileNotFoundError(f"Signal file '{DEFAULT_SIGNAL}.root' not found in {root}")

    cuts = DEFAULT_MLP_RANGE.values()
    k3_vals = DEFAULT_K3_RANGE.values()
    k4_vals = DEFAULT_K4_RANGE.values()
    k3_vals_2d = DEFAULT_K3_2D_RANGE.values()
    k4_vals_2d = DEFAULT_K4_2D_RANGE.values()

    scan = scan_mlp_significance(
        files=files,
        signal_name=DEFAULT_SIGNAL,
        var=DEFAULT_VAR,
        channel=args.channel,
        mlp_var=DEFAULT_MLP_VAR,
        cuts=cuts,
        k3=DEFAULT_K3,
        k4=DEFAULT_K4,
        outdir=outdir,
        diagnostics_n=DIAGNOSTICS_N,
        diagnostics_k4=DIAGNOSTICS_K4,
        diagnostics_k4_vals=k4_vals,
        diagnostics_k4_true=DEFAULT_K4_TRUE,
        diagnostics_k3=DIAGNOSTICS_K3,
        diagnostics_k3_vals=k3_vals,
        diagnostics_k3_true=DEFAULT_K3_TRUE,
        diagnostics_shape_k4=DIAGNOSTICS_SHAPE_K4,
        diagnostics_shape_k4_points=SHAPE_K4_POINTS,
        diagnostics_k3k4_contour=DIAGNOSTICS_K3K4_CONTOUR,
        diagnostics_k3_vals_2d=k3_vals_2d,
        diagnostics_k4_vals_2d=k4_vals_2d,
        diagnostics_k3k4_true=DEFAULT_K3K4_TRUE,
        limit_curve=MLP_K4_LIMIT_CURVE,
        limit_k4_vals=k4_vals,
        limit_k4_true=DEFAULT_K4_TRUE,
    )

    mlp_cut = args.mlp_cut if args.mlp_cut is not None else scan["best_cut_binned"]
    if RUN_K4_SCAN:
        scan_k4_likelihood(
            sig_cache=scan["sig_cache"],
            bkg_caches=scan["bkg_caches"],
            edges=scan["edges"],
            mlp_cut=mlp_cut,
            k3=DEFAULT_K3,
            k4_vals=k4_vals,
            k4_true=DEFAULT_K4_TRUE,
            outdir=outdir,
            channel=args.channel,
            var=DEFAULT_VAR,
        )

    if RUN_K3_SCAN:
        scan_k3_likelihood(
            sig_cache=scan["sig_cache"],
            bkg_caches=scan["bkg_caches"],
            edges=scan["edges"],
            mlp_cut=mlp_cut,
            k3_vals=k3_vals,
            k4=DEFAULT_K4,
            k3_true=DEFAULT_K3_TRUE,
            outdir=outdir,
            channel=args.channel,
            var=DEFAULT_VAR,
        )


if __name__ == "__main__":
    cli()

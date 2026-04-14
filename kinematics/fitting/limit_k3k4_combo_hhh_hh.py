from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import uproot

from aesthetics import banner_heatmaps
from config import BACKGROUNDS, LUMINOSITY_PB, N_BINS_1D, SELECTION, SIGNAL, XLIM_MAP
from tools import BASIS_KEYS, build_mask_from_selection, build_moments, evaluate_weights_from_moments
import matplotlib.pyplot as plt

def _nll_poisson(n: np.ndarray, mu: np.ndarray) -> float:
    term = np.zeros_like(n, dtype=float)
    mask_mu = mu > 0
    mask_n = (n > 0) & mask_mu
    term[mask_n] = mu[mask_n] - n[mask_n] + n[mask_n] * np.log(n[mask_n] / mu[mask_n])
    term[mask_mu & ~mask_n] = mu[mask_mu & ~mask_n]
    if np.any(~mask_mu & (n > 0)):
        return float("inf")
    return float(2.0 * np.sum(term))


def _load_fit_coeffs(path: Path) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split(",", 1)
            coeffs[key.strip()] = float(val)
    return coeffs


def _xsec_from_fit(coeffs: dict[str, float], k3: float, k4: float) -> float:
    terms = {
        "const": 1.0,
        "k3": k3,
        "k4": k4,
        "k3^2": k3**2,
        "k3*k4": k3 * k4,
        "k4^2": k4**2,
        "k3^2*k4": (k3**2) * k4,
        "k3*k4^2": k3 * (k4**2),
        "k3^2*k4^2": (k3**2) * (k4**2),
    }
    xsec = 0.0
    for key, val in coeffs.items():
        if key not in terms:
            continue
        xsec += val * terms[key]
    return float(xsec)


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k3_vals = []
    k4_vals = []
    xsec_vals = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            k3_vals.append(float(row["chhh"]))
            k4_vals.append(float(row["ch4"]))
            xsec_vals.append(float(row["xsec_pb"]))
    return np.array(k3_vals), np.array(k4_vals), np.array(xsec_vals)


def _build_xsec_model(
    *,
    fit_path: Path,
    csv_path: Path,
    use_fit: bool,
) -> Callable[[float, float], float]:
    if use_fit and fit_path.is_file():
        coeffs = _load_fit_coeffs(fit_path)
        return lambda k3, k4: _xsec_from_fit(coeffs, k3, k4)

    k3_vals, k4_vals, xsec_vals = _load_grid_csv(csv_path)
    points = np.column_stack([k3_vals, k4_vals])

    try:
        from scipy.interpolate import LinearNDInterpolator

        interp = LinearNDInterpolator(points, xsec_vals, fill_value=np.nan)

        def _eval(k3: float, k4: float) -> float:
            val = float(interp(k3, k4))
            if np.isfinite(val):
                return val
            d2 = np.sum((points - np.array([k3, k4])) ** 2, axis=1)
            return float(xsec_vals[np.argmin(d2)])

        return _eval
    except Exception:
        def _eval_nn(k3: float, k4: float) -> float:
            d2 = np.sum((points - np.array([k3, k4])) ** 2, axis=1)
            return float(xsec_vals[np.argmin(d2)])

        return _eval_nn


def _edges_for_var(var: str, signal_arr: np.ndarray) -> np.ndarray:
    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
    else:
        xmin, xmax = float(np.min(signal_arr)), float(np.max(signal_arr))
    return np.linspace(xmin, xmax, N_BINS_1D + 1)


def _load_signal_with_weights(
    *,
    root: Path,
    signal_name: str,
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fp = root / f"{signal_name}.root"
    if not fp.is_file():
        raise FileNotFoundError(f"Signal file '{signal_name}.root' not found in {root}")

    with uproot.open(fp) as f_sig:
        tree = f_sig["events"]
        mask = build_mask_from_selection(tree, SELECTION[channel])

        var_arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
        mlp_arr = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
        w_xsec = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB
        weight_dict = {k: np.asarray(tree[k].array(library="np")[mask], dtype=float) for k in BASIS_KEYS}

    finite = np.isfinite(var_arr) & np.isfinite(mlp_arr) & np.isfinite(w_xsec)
    for arr in weight_dict.values():
        finite &= np.isfinite(arr)

    var_arr = var_arr[finite]
    mlp_arr = mlp_arr[finite]
    w_xsec = w_xsec[finite]
    weight_dict = {k: v[finite] for k, v in weight_dict.items()}

    if mlp_cut is not None:
        keep = mlp_arr > mlp_cut
        var_arr = var_arr[keep]
        mlp_arr = mlp_arr[keep]
        w_xsec = w_xsec[keep]
        weight_dict = {k: v[keep] for k, v in weight_dict.items()}

    return var_arr, w_xsec, mlp_arr, weight_dict


def _load_background_hist(
    *,
    root: Path,
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
    edges: np.ndarray,
    backgrounds: list[str],
) -> np.ndarray:
    h_bkg = np.zeros(len(edges) - 1, dtype=float)
    for proc in backgrounds:
        fp = root / f"{proc}.root"
        if not fp.is_file():
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            tree = f_bkg["events"]
            mask = build_mask_from_selection(tree, SELECTION[channel])
            arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
            mlp = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
            w_bkg = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB

        finite = np.isfinite(arr) & np.isfinite(mlp) & np.isfinite(w_bkg)
        arr = arr[finite]
        mlp = mlp[finite]
        w_bkg = w_bkg[finite]

        if mlp_cut is not None:
            keep = mlp > mlp_cut
            arr = arr[keep]
            w_bkg = w_bkg[keep]

        if arr.size:
            h_bkg += np.histogram(arr, bins=edges, weights=w_bkg)[0]
    return h_bkg


def _load_background_hist_parts(
    *,
    root: Path,
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
    edges: np.ndarray,
    backgrounds: list[str],
    reweightable: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    h_fixed = np.zeros(len(edges) - 1, dtype=float)
    h_rew = np.zeros_like(h_fixed)
    for proc in backgrounds:
        fp = root / f"{proc}.root"
        if not fp.is_file():
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            tree = f_bkg["events"]
            mask = build_mask_from_selection(tree, SELECTION[channel])
            arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
            mlp = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
            w_bkg = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB

        finite = np.isfinite(arr) & np.isfinite(mlp) & np.isfinite(w_bkg)
        arr = arr[finite]
        mlp = mlp[finite]
        w_bkg = w_bkg[finite]

        if mlp_cut is not None:
            keep = mlp > mlp_cut
            arr = arr[keep]
            w_bkg = w_bkg[keep]

        if arr.size:
            hist = np.histogram(arr, bins=edges, weights=w_bkg)[0]
            if proc in reweightable:
                h_rew += hist
            else:
                h_fixed += hist
    return h_fixed, h_rew


def _compute_hhh_nll_grid(
    *,
    root: Path,
    signal_name: str,
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    k3_true: float,
    k4_true: float,
    backgrounds: list[str],
    reweightable_bkgs: set[str],
    xsec_ratio: Callable[[float, float], float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    var_arr, w_xsec, _, weight_dict = _load_signal_with_weights(
        root=root,
        signal_name=signal_name,
        channel=channel,
        var=var,
        mlp_var=mlp_var,
        mlp_cut=mlp_cut,
    )
    edges = _edges_for_var(var, var_arr)
    if reweightable_bkgs and xsec_ratio is None:
        raise ValueError("Reweightable backgrounds requested, but xsec_ratio is None.")
    h_fixed, h_rew = _load_background_hist_parts(
        root=root,
        channel=channel,
        var=var,
        mlp_var=mlp_var,
        mlp_cut=mlp_cut,
        edges=edges,
        backgrounds=backgrounds,
        reweightable=reweightable_bkgs,
    )
    ratio_true = xsec_ratio(k3_true, k4_true) if reweightable_bkgs else 0.0
    h_bkg_true = h_fixed + (h_rew * ratio_true if reweightable_bkgs else 0.0)

    moments = build_moments(weight_dict)
    w_true = evaluate_weights_from_moments(k3_true, k4_true, moments) * w_xsec
    h_sig_true = np.histogram(var_arr, bins=edges, weights=w_true)[0]
    n_hist = h_sig_true + h_bkg_true

    b_total = float(np.sum(h_bkg_true))
    n_total = float(np.sum(n_hist))
    s_true_total = float(np.sum(w_true))
    mu_true_total = s_true_total + b_total

    nll_true_binned = _nll_poisson(n_hist, h_sig_true + h_bkg_true)
    nll_true_single = _nll_poisson(np.array([n_total]), np.array([mu_true_total]))

    nll_binned = np.zeros((len(k3_vals), len(k4_vals)), dtype=float)
    nll_single = np.zeros_like(nll_binned)

    for i, k3 in enumerate(k3_vals):
        for j, k4 in enumerate(k4_vals):
            w = evaluate_weights_from_moments(k3, k4, moments) * w_xsec
            h_sig = np.histogram(var_arr, bins=edges, weights=w)[0]
            if reweightable_bkgs:
                ratio = xsec_ratio(k3, k4)
                h_bkg = h_fixed + h_rew * ratio
            else:
                h_bkg = h_fixed
            mu = h_sig + h_bkg
            nll_binned[i, j] = _nll_poisson(n_hist, mu)

            s_total = float(np.sum(w))
            mu_total = s_total + float(np.sum(h_bkg))
            nll_single[i, j] = _nll_poisson(np.array([n_total]), np.array([mu_total]))

    nll_binned -= nll_true_binned
    nll_single -= nll_true_single
    return nll_binned, nll_single


def _compute_hh_nll_grid(
    *,
    root: Path,
    signal_name: str,
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
    xsec_ratio: Callable[[float, float], float],
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    k3_true: float,
    k4_true: float,
    backgrounds: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    fp = root / f"{signal_name}.root"
    if not fp.is_file():
        raise FileNotFoundError(f"Signal file '{signal_name}.root' not found in {root}")

    with uproot.open(fp) as f_sig:
        tree = f_sig["events"]
        mask = build_mask_from_selection(tree, SELECTION[channel])
        arr = np.asarray(tree[var].array(library="np")[mask], dtype=float)
        mlp = np.asarray(tree[mlp_var].array(library="np")[mask], dtype=float)
        w_sig = np.asarray(tree["weight_xsec"].array(library="np")[mask], dtype=float) * LUMINOSITY_PB

    finite = np.isfinite(arr) & np.isfinite(mlp) & np.isfinite(w_sig)
    arr = arr[finite]
    mlp = mlp[finite]
    w_sig = w_sig[finite]

    if mlp_cut is not None:
        keep = mlp > mlp_cut
        arr = arr[keep]
        w_sig = w_sig[keep]

    edges = _edges_for_var(var, arr)
    h_sig_sm = np.histogram(arr, bins=edges, weights=w_sig)[0]
    h_bkg = _load_background_hist(
        root=root,
        channel=channel,
        var=var,
        mlp_var=mlp_var,
        mlp_cut=mlp_cut,
        edges=edges,
        backgrounds=backgrounds,
    )

    ratio_true = xsec_ratio(k3_true, k4_true)
    s_true = h_sig_sm * ratio_true
    n_hist = s_true + h_bkg
    n_total = float(np.sum(n_hist))
    b_total = float(np.sum(h_bkg))
    s_true_total = float(np.sum(s_true))
    mu_true_total = s_true_total + b_total

    nll_true_binned = _nll_poisson(n_hist, s_true + h_bkg)
    nll_true_single = _nll_poisson(np.array([n_total]), np.array([mu_true_total]))

    nll_binned = np.zeros((len(k3_vals), len(k4_vals)), dtype=float)
    nll_single = np.zeros_like(nll_binned)

    for i, k3 in enumerate(k3_vals):
        for j, k4 in enumerate(k4_vals):
            ratio = xsec_ratio(k3, k4)
            s = h_sig_sm * ratio
            mu = s + h_bkg
            nll_binned[i, j] = _nll_poisson(n_hist, mu)

            s_total = float(np.sum(s))
            mu_total = s_total + b_total
            nll_single[i, j] = _nll_poisson(np.array([n_total]), np.array([mu_total]))

    nll_binned -= nll_true_binned
    nll_single -= nll_true_single
    return nll_binned, nll_single


def _plot_contours(
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
    shade_excluded: bool = False,
) -> None:
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    if shade_excluded:
        ax.contourf(
            K3,
            K4,
            nll_binned,
            levels=[-np.inf, level],
            colors=["#c0c0c0"],
            alpha=0.25,
        )
    ax.contour(K3, K4, nll_binned, levels=[level], colors="black", linewidths=1.8)
    ax.scatter([k3_true], [k4_true], color="red", marker="*", s=60, zorder=5)

    ax.set_xlabel(r"$k_3$", loc="right")
    ax.set_ylabel(r"$k_4$", loc="top")
    ax.set_xlim(k3_vals.min(), k3_vals.max())
    ax.set_ylim(k4_vals.min(), k4_vals.max())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.8, label="Stat-only HH + HHH"),
            Patch(facecolor="#c0c0c0", edgecolor="none", alpha=0.25, label="Allowed region"),
            Line2D([0], [0], marker="*", color="red", lw=0, markersize=10, label="SM point"),
        ],
        frameon=False,
        loc="center",
        fontsize=9,
    )
    banner_heatmaps(ax, comment=comment)


    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_contours(
    *,
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    nll_hhh: np.ndarray,
    nll_hh: np.ndarray,
    nll_combo: np.ndarray | None,
    level: float,
    k3_true: float,
    k4_true: float,
    outpath: Path,
    comment: str,
    title: str,
    labels: tuple[str, str, str] = ("HHH", "HH", "Combined"),
    include_combo: bool = True,
) -> None:
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    cs_hhh = ax.contour(
        K3, K4, nll_hhh,
        levels=[level],
        colors=["tab:blue"],
        linestyles=["dashdot"],
        linewidths=[1.8],
    )

    cs_hh = ax.contour(
        K3, K4, nll_hh,
        levels=[level],
        colors=["tab:green"],
        linestyles=["dashed"],
        linewidths=[1.8],
    )

    if include_combo and nll_combo is not None:
        cs_combo = ax.contour(
            K3, K4, nll_combo,
            levels=[level],
            colors=["black"],
            linestyles=["-"],
            linewidths=[2.5],
        )

    ax.scatter([k3_true], [k4_true], color="red", marker="*", s=60, zorder=5)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linestyle="dashdot", lw=1, label=labels[0]),
        Line2D([0], [0], color="tab:green",     linestyle="dashed", lw=1, label=labels[1]),
    ]
    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\kappa_4$", loc='top')

    if include_combo and nll_combo is not None:
        legend_handles.append(Line2D([0], [0], color="black", linestyle="-", lw=2.5, label=labels[2]))

    ax.legend(handles=legend_handles, frameon=False, loc="center", fontsize=9)
    banner_heatmaps(ax, comment=f"{comment} | {title}")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_1d_limit(
    *,
    x_vals: np.ndarray,
    nll_binned: np.ndarray,
    nll_single: np.ndarray,
    x_true: float,
    xlabel: str,
    outpath: Path,
    comment: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(x_vals, nll_binned, lw=2.0, label="Stat-only", color="black")
    #ax.plot(x_vals, nll_single, lw=2.0, ls="--", label="Stat-only")
    #ax.axvline(x_true, color="black", ls=":", lw=1.4)
    ax.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax.axhline(3.84, color="gray", ls="--", lw=1.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(r"$-2\Delta\ln L$", loc="top")
    ax.legend(frameon=False, loc="center")
    banner_heatmaps(ax, comment=f"{comment} | {title}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def cli() -> None:
    ap = argparse.ArgumentParser(description="Combine HHH (per-event weights) with HH (xsec map) in k3/k4.")
    ap.add_argument("--root-hhh", required=True)
    ap.add_argument("--root-hh", required=True)
    ap.add_argument("--outdir", default="fit_combo")

    ap.add_argument("--channel-hhh", default="HadHad")
    ap.add_argument("--channel-hh", default="HadHad")
    ap.add_argument("--signal-hhh", default=SIGNAL)
    ap.add_argument("--signal-hh", required=True)
    ap.add_argument("--var-hhh", default="m_hhh_vis")
    ap.add_argument("--var-hh", default="m_hh")
    ap.add_argument("--mlp-var", default="mlp_score")
    ap.add_argument("--mlp-cut", type=float, nargs=2, default=None, metavar=("HH", "HHH"),
                    help="MLP cuts (HH then HHH). Overrides --mlp-cut-hh/--mlp-cut-hhh if set.")
    ap.add_argument("--mlp-cut-hhh", type=float, default=None)
    ap.add_argument("--mlp-cut-hh", type=float, default=None)
    ap.add_argument("--bkg-hhh", nargs="+", default=None,
                    help="Background processes for HHH (space-separated). Defaults to config BACKGROUNDS.")
    ap.add_argument("--bkg-hh", nargs="+", default=None,
                    help="Background processes for HH (space-separated). Defaults to config BACKGROUNDS.")
    ap.add_argument("--bkg-hhh-reweight", nargs="+", default=[],
                    help="HHH background processes to scale with xsec map (e.g. HH sample).")

    ap.add_argument("--xsec-csv", default="hhh/kinematics/xsec_grid/xsec_grid.csv")
    ap.add_argument("--xsec-fit", default="hhh/kinematics/xsec_grid/xsec_fit_coeffs.txt")
    ap.add_argument("--use-fit", action="store_true", default=True)
    ap.add_argument("--no-use-fit", action="store_false", dest="use_fit")

    ap.add_argument("--k3-min", type=float, default=-3.0)
    ap.add_argument("--k3-max", type=float, default=6.0)
    ap.add_argument("--k3-steps", type=int, default=101)
    ap.add_argument("--k4-min", type=float, default=-15.0)
    ap.add_argument("--k4-max", type=float, default=35.0)
    ap.add_argument("--k4-steps", type=int, default=101)
    ap.add_argument("--k3-true", type=float, default=1.0)
    ap.add_argument("--k4-true", type=float, default=1.0)
    ap.add_argument("--level", type=float, default=2.30, help="ΔNLL level for 2D contour (2.30=1σ)")
    args = ap.parse_args()

    root_hhh = Path(args.root_hhh).expanduser().resolve()
    root_hh = Path(args.root_hh).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mlp_cut is not None:
        mlp_cut_hh = float(args.mlp_cut[0])
        mlp_cut_hhh = float(args.mlp_cut[1])
    else:
        mlp_cut_hhh = args.mlp_cut_hhh
        mlp_cut_hh = args.mlp_cut_hh

    def _ensure_value_in_grid(vals: np.ndarray, target: float) -> np.ndarray:
        if np.any(np.isclose(vals, target, atol=1e-12)):
            return vals
        return np.sort(np.unique(np.concatenate([vals, np.array([target])])))

    k3_vals = np.linspace(args.k3_min, args.k3_max, args.k3_steps)
    k4_vals = np.linspace(args.k4_min, args.k4_max, args.k4_steps)
    # Ensure SM point is exactly on the grid for 1D slices/labels
    k3_vals = _ensure_value_in_grid(k3_vals, 1.0)
    k4_vals = _ensure_value_in_grid(k4_vals, 1.0)

    xsec_model = _build_xsec_model(
        fit_path=Path(args.xsec_fit),
        csv_path=Path(args.xsec_csv),
        use_fit=args.use_fit,
    )
    xsec_sm = xsec_model(args.k3_true, args.k4_true)
    if xsec_sm == 0:
        raise ValueError("SM cross-section is zero in the xsec model.")
    xsec_ratio = lambda k3, k4: xsec_model(k3, k4) / xsec_sm

    bkg_hhh = args.bkg_hhh if args.bkg_hhh is not None else list(BACKGROUNDS)
    bkg_hh = args.bkg_hh if args.bkg_hh is not None else list(BACKGROUNDS)
    rew_bkgs = set(args.bkg_hhh_reweight)

    nll_hhh_binned, nll_hhh_single = _compute_hhh_nll_grid(
        root=root_hhh,
        signal_name=args.signal_hhh,
        channel=args.channel_hhh,
        var=args.var_hhh,
        mlp_var=args.mlp_var,
        mlp_cut=mlp_cut_hhh,
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        backgrounds=bkg_hhh,
        reweightable_bkgs=rew_bkgs,
        xsec_ratio=xsec_ratio,
    )

    # HHH without reweighted background (for comparison)
    nll_hhh_norew_binned, nll_hhh_norew_single = _compute_hhh_nll_grid(
        root=root_hhh,
        signal_name=args.signal_hhh,
        channel=args.channel_hhh,
        var=args.var_hhh,
        mlp_var=args.mlp_var,
        mlp_cut=mlp_cut_hhh,
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        backgrounds=bkg_hhh,
        reweightable_bkgs=set(),
        xsec_ratio=None,
    )

    nll_hh_binned, nll_hh_single = _compute_hh_nll_grid(
        root=root_hh,
        signal_name=args.signal_hh,
        channel=args.channel_hh,
        var=args.var_hh,
        mlp_var=args.mlp_var,
        mlp_cut=mlp_cut_hh,
        xsec_ratio=xsec_ratio,
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        backgrounds=bkg_hh,
    )

    nll_combo_binned = nll_hhh_binned + nll_hh_binned
    nll_combo_single = nll_hhh_single + nll_hh_single

    out_npz = outdir / "k3k4_combo_nll.npz"
    np.savez(
        out_npz,
        k3=k3_vals,
        k4=k4_vals,
        nll_hhh_binned=nll_hhh_binned,
        nll_hhh_single=nll_hhh_single,
        nll_hhh_norew_binned=nll_hhh_norew_binned,
        nll_hhh_norew_single=nll_hhh_norew_single,
        nll_hh_binned=nll_hh_binned,
        nll_hh_single=nll_hh_single,
        nll_combo_binned=nll_combo_binned,
        nll_combo_single=nll_combo_single,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        mlp_cut_hhh=mlp_cut_hhh,
        mlp_cut_hh=mlp_cut_hh,
    )

    comment = f"Combined | HHH+HH | ΔNLL={args.level:g}"
    out_pdf = outdir / "k3k4_combo_contour.pdf"
    _plot_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_binned=nll_combo_binned,
        nll_single=nll_combo_single,
        level=args.level,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=out_pdf,
        comment=comment,
    )

    # HH-only contour
    _plot_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_binned=nll_hh_binned,
        nll_single=nll_hh_single,
        level=args.level,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=outdir / "k3k4_hh_only_contour.pdf",
        comment=f"HH only | ΔNLL={args.level:g}",
        shade_excluded=True,
    )

    overlay_comment = f"ΔNLL={args.level:g}"
    _plot_overlay_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_hhh=nll_hhh_binned,
        nll_hh=nll_hh_binned,
        nll_combo=nll_combo_binned,
        level=args.level,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=outdir / "k3k4_overlay_binned.pdf",
        comment=overlay_comment,
        title="Stat-only",
    )
    _plot_overlay_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_hhh=nll_hhh_single,
        nll_hh=nll_hh_single,
        nll_combo=nll_combo_single,
        level=args.level,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=outdir / "k3k4_overlay_singlebin.pdf",
        comment=overlay_comment,
        title="Single-bin",
    )

    _plot_overlay_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_hhh=nll_hhh_norew_binned,
        nll_hh=nll_hhh_binned,
        nll_combo=None,
        level=args.level,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=outdir / "k3k4_hhh_with_without_rew_binned.pdf",
        comment=overlay_comment,
        title="HHH reweight check",
        labels=(r"$HHH$ + $HH$(SM)", r"$HHH$ + $HH(\kappa_3,\kappa_4)$", "Combined"),
        include_combo=False,
    )

    # --- 1D limits ---
    # 1D slices are defined at k3=1 and k4=1 (SM point)
    k3_slice = 1.0
    k4_slice = 1.0
    k3_idx = int(np.argmin(np.abs(k3_vals - k3_slice)))
    k4_idx = int(np.argmin(np.abs(k4_vals - k4_slice)))

    # HH: 1D k3 at k4=1
    _plot_1d_limit(
        x_vals=k3_vals,
        nll_binned=nll_hh_binned[:, k4_idx],
        nll_single=nll_hh_single[:, k4_idx],
        x_true=k3_slice,
        xlabel=r"$k_3$",
        outpath=outdir / "k3_1d_hh.pdf",
        comment="HH",
        title=f"k4={k4_vals[k4_idx]:.3g}",
    )

    # HHH: 1D k4 at k3=1
    _plot_1d_limit(
        x_vals=k4_vals,
        nll_binned=nll_hhh_binned[k3_idx, :],
        nll_single=nll_hhh_single[k3_idx, :],
        x_true=k4_slice,
        xlabel=r"$k_4$",
        outpath=outdir / "k4_1d_hhh.pdf",
        comment="HHH",
        title=f"k3={k3_vals[k3_idx]:.3g}",
    )
    # HHH (no reweight): 1D k4 at k3=1
    _plot_1d_limit(
        x_vals=k4_vals,
        nll_binned=nll_hhh_norew_binned[k3_idx, :],
        nll_single=nll_hhh_norew_single[k3_idx, :],
        x_true=k4_slice,
        xlabel=r"$k_4$",
        outpath=outdir / "k4_1d_hhh_norew.pdf",
        comment="HHH (no rew)",
        title=f"k3={k3_vals[k3_idx]:.3g}",
    )

    # Combined: 1D k3 at k4=1
    _plot_1d_limit(
        x_vals=k3_vals,
        nll_binned=nll_combo_binned[:, k4_idx],
        nll_single=nll_combo_single[:, k4_idx],
        x_true=k3_slice,
        xlabel=r"$k_3$",
        outpath=outdir / "k3_1d_combo.pdf",
        comment="Combined",
        title=f"k4={k4_vals[k4_idx]:.3g}",
    )

    # Combined: 1D k4 at k3=1
    _plot_1d_limit(
        x_vals=k4_vals,
        nll_binned=nll_combo_binned[k3_idx, :],
        nll_single=nll_combo_single[k3_idx, :],
        x_true=k4_slice,
        xlabel=r"$k_4$",
        outpath=outdir / "k4_1d_combo.pdf",
        comment="Combined",
        title=f"k3={k3_vals[k3_idx]:.3g}",
    )


if __name__ == "__main__":
    cli()

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import uproot

from aesthetics import banner_heatmaps
from config import BACKGROUNDS, LUMINOSITY_PB, N_BINS_1D, SELECTION, SIGNAL, XLIM_MAP
from tools2 import build_mask_from_selection, numeric
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
            # nearest-neighbour fallback
            d2 = np.sum((points - np.array([k3, k4])) ** 2, axis=1)
            return float(xsec_vals[np.argmin(d2)])

        return _eval
    except Exception:
        def _eval_nn(k3: float, k4: float) -> float:
            d2 = np.sum((points - np.array([k3, k4])) ** 2, axis=1)
            return float(xsec_vals[np.argmin(d2)])

        return _eval_nn


def _load_histograms(
    *,
    files: dict[str, Path],
    channel: str,
    var: str,
    mlp_var: str,
    mlp_cut: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection = SELECTION[channel]

    fp_sig = files[SIGNAL]
    with uproot.open(fp_sig) as f_sig:
        tree = f_sig["events"]
        mask = build_mask_from_selection(tree, selection)
        arr = numeric(tree[var].array(library="ak")[mask])
        mlp = numeric(tree[mlp_var].array(library="ak")[mask])
        w_sig = numeric(tree["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB

    if mlp_cut is not None:
        keep = mlp > mlp_cut
        arr = arr[keep]
        w_sig = w_sig[keep]

    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
    else:
        xmin, xmax = float(np.min(arr)), float(np.max(arr))
    edges = np.linspace(xmin, xmax, N_BINS_1D + 1)

    h_sig = np.histogram(arr, bins=edges, weights=w_sig)[0]

    h_bkg = np.zeros_like(h_sig)
    for proc in BACKGROUNDS:
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            if "events" not in f_bkg or var not in f_bkg["events"]:
                continue
            tree = f_bkg["events"]
            mask = build_mask_from_selection(tree, selection)
            arr = numeric(tree[var].array(library="ak")[mask])
            mlp = numeric(tree[mlp_var].array(library="ak")[mask])
            w_bkg = numeric(tree["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB

        if mlp_cut is not None:
            keep = mlp > mlp_cut
            arr = arr[keep]
            w_bkg = w_bkg[keep]

        h_bkg += np.histogram(arr, bins=edges, weights=w_bkg)[0]

    return edges, h_sig, h_bkg


def _compute_k3k4_nll_grids(
    *,
    h_sig_sm: np.ndarray,
    h_bkg: np.ndarray,
    xsec_ratio: Callable[[float, float], float],
    k3_vals: np.ndarray,
    k4_vals: np.ndarray,
    k3_true: float,
    k4_true: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratio_true = xsec_ratio(k3_true, k4_true)
    if ratio_true == 0:
        raise ValueError("xsec ratio at truth is zero.")

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
    levels: list[float],
    k3_true: float,
    k4_true: float,
    outpath: Path,
    comment: str,
) -> None:
    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    colors = ["black", "tab:blue", "tab:green", "tab:red"]
    for idx, lvl in enumerate(levels):
        color = colors[idx % len(colors)]
        ax.contour(K3, K4, nll_binned, levels=[lvl], colors=color, linewidths=1.8)
        ax.contour(K3, K4, nll_single, levels=[lvl], colors=color, linewidths=1.6, linestyles="--")

    ax.scatter([k3_true], [k4_true], color="red", marker="*", s=60, zorder=5)
    ax.set_xlabel(r"$k_3$", loc="right")
    ax.set_ylabel(r"$k_4$", loc="top")
    ax.set_xlim(k3_vals.min(), k3_vals.max())
    ax.set_ylim(k4_vals.min(), k4_vals.max())
    banner_heatmaps(ax, comment=comment)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def cli() -> None:
    ap = argparse.ArgumentParser(description="Limits on k3/k4 using cross-section map for signal scaling.")
    ap.add_argument("--root-dir", required=True, help="Directory containing ROOT files")
    ap.add_argument("--outdir", default="fit_xsecmap", help="Output directory")
    ap.add_argument("--channel", default="HadHad", choices=list(SELECTION.keys()))
    ap.add_argument("--signal", default=SIGNAL)
    ap.add_argument("--var", default="m_hhh_vis")
    ap.add_argument("--mlp-var", default="mlp_score")
    ap.add_argument("--mlp-cut", type=float, default=None)

    ap.add_argument("--xsec-csv", default="hhh/kinematics/xsec_grid/xsec_grid.csv")
    ap.add_argument("--xsec-fit", default="hhh/kinematics/xsec_grid/xsec_fit_coeffs.txt")
    ap.add_argument("--use-fit", action="store_true", default=True)
    ap.add_argument("--no-use-fit", action="store_false", dest="use_fit")

    ap.add_argument("--k3-min", type=float, default=-3.0)
    ap.add_argument("--k3-max", type=float, default=6.0)
    ap.add_argument("--k3-steps", type=int, default=41)
    ap.add_argument("--k4-min", type=float, default=-15.0)
    ap.add_argument("--k4-max", type=float, default=35.0)
    ap.add_argument("--k4-steps", type=int, default=51)
    ap.add_argument("--k3-true", type=float, default=1.0)
    ap.add_argument("--k4-true", type=float, default=1.0)
    ap.add_argument("--levels", type=float, nargs="+", default=[2.30],
                    help="ΔNLL contour levels (2D: 2.30=1σ, 5.99=2σ)")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = {p: root / f"{p}.root" for p in [args.signal] + BACKGROUNDS if (root / f"{p}.root").is_file()}
    if args.signal not in files:
        raise FileNotFoundError(f"Signal file '{args.signal}.root' not found in {root}")

    xsec_model = _build_xsec_model(
        fit_path=Path(args.xsec_fit),
        csv_path=Path(args.xsec_csv),
        use_fit=args.use_fit,
    )
    xsec_sm = xsec_model(args.k3_true, args.k4_true)
    if xsec_sm == 0:
        raise ValueError("SM cross-section is zero in the xsec model.")
    xsec_ratio = lambda k3, k4: xsec_model(k3, k4) / xsec_sm

    edges, h_sig_sm, h_bkg = _load_histograms(
        files=files,
        channel=args.channel,
        var=args.var,
        mlp_var=args.mlp_var,
        mlp_cut=args.mlp_cut,
    )

    k3_vals = np.linspace(args.k3_min, args.k3_max, args.k3_steps)
    k4_vals = np.linspace(args.k4_min, args.k4_max, args.k4_steps)

    nll_binned, nll_single = _compute_k3k4_nll_grids(
        h_sig_sm=h_sig_sm,
        h_bkg=h_bkg,
        xsec_ratio=xsec_ratio,
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        k3_true=args.k3_true,
        k4_true=args.k4_true,
    )

    out_npz = outdir / f"k3k4_nll_xsecmap_{args.channel}_{args.var}.npz"
    np.savez(out_npz, k3=k3_vals, k4=k4_vals, nll_binned=nll_binned, nll_single=nll_single,
             k3_true=args.k3_true, k4_true=args.k4_true, mlp_cut=args.mlp_cut)

    mlp_tag = f"mlp>{args.mlp_cut:.3f}" if args.mlp_cut is not None else "mlp: none"
    comment = f"{args.channel} | {mlp_tag}"
    out_pdf = outdir / f"k3k4_contours_xsecmap_{args.channel}_{args.var}.pdf"
    _plot_contours(
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        nll_binned=nll_binned,
        nll_single=nll_single,
        levels=list(args.levels),
        k3_true=args.k3_true,
        k4_true=args.k4_true,
        outpath=out_pdf,
        comment=comment,
    )


if __name__ == "__main__":
    cli()

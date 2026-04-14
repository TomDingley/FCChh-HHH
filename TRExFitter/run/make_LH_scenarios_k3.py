#!/usr/bin/env python3
"""
Compare multiple TRExFitter 1D likelihood scans (dNLL vs parameter) in one PDF.

- Reads an object named "LHscan" (TGraph/TGraphErrors/RooCurve) from each ROOT file.
- Normalizes each curve to min(dNLL)=0 (i.e. y -> y - min(y) per curve).
- Overlays curves, adds 68%/95% CL reference lines, FCC-hh banner.
- Prints best-fit x̂ and [left,right] crossings for dNLL = 0.5 (68% CL) and 1.92 (95% CL).

Usage examples:
  # Use defaults (your three k4 files + labels)
  python compare_k4_scans.py

  # Custom files with labels and output
  python compare_k4_scans.py \
    --labels "No syst" "Scenario I" "Scenario II" \
    --out k4_compare.pdf \
    /path/to/NLLscan_k3_curve.root \
    /path/to/NLLscan_k3_curve_scenI.root \
    /path/to/NLLscan_k3_curve_scenII.root
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import uproot
import matplotlib.pyplot as plt
import argparse

DEFAULT_XLIM = (-1.1, 4.0)
DEFAULT_YLIM = (0.001, 2.5)


# ---------------------------
# Plot cosmetics
# ---------------------------
def banner(ax: plt.Axes, comment: str = "") -> None:
    ax.text(
        0.0, 1.02,
        r"$\mathbf{FCC\text{-}hh}$ Scenario II" "\n"
        r"$\mathit{Delphes}$ Simulation, $\sqrt{s}=84\,\mathrm{TeV}$, $30\,\mathrm{ab}^{-1}$",
        transform=ax.transAxes,
        va="bottom", ha="left", fontsize=11,
    )
    ax.text(1.0, 1.02,
            comment,
            transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11)


# ---------------------------
# I/O helpers
# ---------------------------
def _extract_lhscan_arrays(file: uproot.ReadOnlyDirectory) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find an object whose name contains 'LHscan' and return x, y arrays.
    Supports TGraph/TGraphErrors/RooCurve-like objects (as TGraph via uproot).
    """
    candidates: List[str] = []
    for key in file.keys():
        base = key.split(";")[0]
        if base.lower() == "lhscan":
            candidates.append(key)
    if not candidates:
        for key in file.keys():
            base = key.split(";")[0]
            if "lhscan" in base.lower():
                candidates.append(key)
    if not candidates:
        raise RuntimeError("No object with name containing 'LHscan' found.")

    # Prefer exact 'LHscan' if present
    candidates.sort(key=lambda k: (0 if k.split(";")[0] == "LHscan" else 1, k))
    obj = file[candidates[0]]

    # Try direct member access (TGraph)
    try:
        x = np.asarray(obj.member("fX"))
        y = np.asarray(obj.member("fY"))
    except Exception:
        # Fall back to uproot arrays
        try:
            arrs = obj.arrays(library="np")
            if "fX" in arrs and "fY" in arrs:
                x = np.asarray(arrs["fX"])
                y = np.asarray(arrs["fY"])
            elif "x" in arrs and "y" in arrs:
                x = np.asarray(arrs["x"])
                y = np.asarray(arrs["y"])
            else:
                raise KeyError("Could not find x/y arrays in object.")
        except Exception as e:
            raise RuntimeError(f"Found LHscan object but couldn't extract points: {e}")

    order = np.argsort(x)
    return x[order], y[order]


# ---------------------------
# Math helpers
# ---------------------------
def _delta_nll_intervals(x: np.ndarray, y: np.ndarray, levels=(0.5, 1.92)):
    """
    Return best fit (xhat) and nearest left/right crossings for each dNLL level.
    Assumes x is sorted. Linear interpolation between points. y will be
    normalized internally to y - min(y).
    """
    y0 = y - np.min(y)
    imin = int(np.argmin(y0))
    xhat = float(x[imin])

    def crossings(level: float):
        xs = []
        for i in range(len(x) - 1):
            y1, y2 = y0[i], y0[i + 1]
            if (y1 - level) * (y2 - level) <= 0 and y2 != y1:
                t = (level - y1) / (y2 - y1)
                if 0.0 <= t <= 1.0:
                    xs.append(x[i] + t * (x[i + 1] - x[i]))
        xs = np.array(sorted(xs))
        left  = xs[xs < xhat].max() if np.any(xs < xhat) else None
        right = xs[xs > xhat].min() if np.any(xs > xhat) else None
        return left, right, xs

    out = {"xhat": xhat, "levels": {}}
    for L in levels:
        Lleft, Lright, allx = crossings(L)
        out["levels"][L] = {"left": Lleft, "right": Lright, "all": allx}
    return out


def resolve_axis_limits(
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    default_xlim: Tuple[float, float],
    default_ylim: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xlo = xmin if xmin is not None else default_xlim[0]
    xhi = xmax if xmax is not None else default_xlim[1]
    ylo = ymin if ymin is not None else default_ylim[0]
    yhi = ymax if ymax is not None else default_ylim[1]

    if xlo >= xhi:
        raise ValueError(f"Invalid x-axis limits: xmin={xlo} must be smaller than xmax={xhi}.")
    if ylo >= yhi:
        raise ValueError(f"Invalid y-axis limits: ymin={ylo} must be smaller than ymax={yhi}.")

    return (xlo, xhi), (ylo, yhi)


# ---------------------------
# Main plotting routine
# ---------------------------
def plot_compare_k3(files: List[Path],
                    labels: List[str],
                    out_path: Path,
                    comment: str = "",
                    xmin: float | None = DEFAULT_XLIM[0],
                    xmax: float | None = DEFAULT_XLIM[1],
                    ymin: float | None = DEFAULT_YLIM[0],
                    ymax: float | None = DEFAULT_YLIM[1]) -> None:
    if len(labels) != len(files):
        raise ValueError("Number of labels must match number of files.")

    # Read curves
    curves = []
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        with uproot.open(p) as f:
            x, y = _extract_lhscan_arrays(f)
        curves.append((p, x, y))

    # Prepare plot
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = ["black","blue","green"]
    linestyles = ["--", "-", "-."]
    # Plot each curve normalized to min(dNLL)=0
    for (p, x, y), lab, color, ls in zip(curves, labels, colors, linestyles):
        y0 = y - np.min(y)
        
        ax.plot(x, y0, lw=1.5, label=lab, color=color, linestyle=ls)

    # Reference lines: 68% / 95% CL for 1D (dNLL ≈ 0.5, 1.92)
    ref_lines = [(0.5, "68% CL"), (1.92, "95% CL")]
    for yline, lab in ref_lines:
        ax.axhline(yline, linestyle='--', linewidth=1, alpha=0.9, color="gray")
        ax.annotate(lab, xy=(0.1, yline), xycoords=('axes fraction', 'data'),
                    ha='right', va='bottom', fontsize=10)
    (p_ref, x_ref, y_ref) = curves[1]

    res_ref = _delta_nll_intervals(x_ref, y_ref, levels=(0.5, 1.92))
    L68 = res_ref["levels"][0.5]["left"]
    R68 = res_ref["levels"][0.5]["right"]

    if (L68 is not None) and (R68 is not None):
        xmid = 0.6
        xmid = (xmin + xmax)/2
        ytxt = 0.5 + 0.02  # slightly above 68% line
        if "omparison" in comment:
            txt = ""
        else:
            txt = rf"$\kappa_3 \in [{L68:.2f},\,{R68:.2f}]$"
        ax.text(
            xmid, ytxt, txt,
            ha="center", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0),
            zorder=10,
        )

    # Axes labels, ranges, legend
    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\Delta(-\log L)$", loc='top')
    xlim, ylim = resolve_axis_limits(xmin, xmax, ymin, ymax, default_xlim=DEFAULT_XLIM, default_ylim=DEFAULT_YLIM)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if "omparison" in comment:
        ax.legend(loc="upper left", bbox_to_anchor=(0.15, 1), frameon=False)
    else:
        ax.legend(loc="upper center", frameon=False)

    banner(ax, comment)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[✓] Wrote: {out_path}")

    # Print intervals per curve
    print("\nIntervals (per curve):")
    for (p, x, y), lab in zip(curves, labels):
        res = _delta_nll_intervals(x, y, levels=(0.5, 1.92))
        xhat = res["xhat"]
        print(f"  [{lab}] {p.name}: best fit x̂ = {xhat:.3f}")
        for L, txt in [(0.5, "68% CL"), (1.92, "95% CL (two-sided)")]:
            Lres = res["levels"][L]
            Lleft, Lright = Lres["left"], Lres["right"]
            if Lleft is not None and Lright is not None:
                width = (Lright - Lleft) / 2.0
                print(f"      dNLL={L:.2f} → {txt}: [{Lleft:.3f}, {Lright:.3f}]  (±{width:.3f} about {xhat:.3f})")
            else:
                side = "left" if Lleft is None else "right"
                print(f"      dNLL={L:.2f} → {txt}: {side} side not reached in scan range")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay κ4 likelihood scans and print CL intervals.")
    ap.add_argument("files", nargs="*", help="Input ROOT files with LHscan objects.")
    ap.add_argument("--labels", nargs="*", help="Legend labels for the curves, same order as files.")
    ap.add_argument("--out", default="NLLscan_k3_compare_newscen.pdf", help="Output PDF path.")
    ap.add_argument("--xmin", type=float, default=DEFAULT_XLIM[0], help="x-axis minimum.")
    ap.add_argument("--xmax", type=float, default=DEFAULT_XLIM[1], help="x-axis maximum.")
    ap.add_argument("--ymin", type=float, default=DEFAULT_YLIM[0], help="y-axis minimum.")
    ap.add_argument("--ymax", type=float, default=DEFAULT_YLIM[1], help="y-axis maximum.")
    ap.add_argument(
        "--hh",
        action="store_true",
        help=r"Use $HH\rightarrow bb\gamma\gamma$ in the banner comment instead of the default",
    )
    ap.add_argument(
        "--comb",
        action="store_true",
        help=r"Use combination for the banner comment instead of the default",
    )
    ap.add_argument(
        "--comp",
        action="store_true",
        help=r"Use combination for the banner comment instead of the default",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Defaults (your three scenarios)
    default_files = [
        Path("/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3_NoSyst_syst/LHoodPlots/NLLscan_k3_curve.root"),
        Path("/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3_ScenI_syst/LHoodPlots/NLLscan_k3_curve.root"),
        Path("/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3_ScenII_syst/LHoodPlots/NLLscan_k3_curve.root"),
    ]
    default_labels = ["No syst", "Uncertainty Scenario I", "Uncertainty Scenario II"]

    files = [Path(f) for f in (args.files if args.files else default_files)]
    labels = args.labels if args.labels else default_labels
    out_path = Path(args.out)
    if args.hh:
        comment = r"$HH\rightarrow bb\gamma\gamma$"
    elif args.comb:
        comment = r"$HHH\rightarrow 4b2\tau$ + $HH\rightarrow bb\gamma\gamma$"
    elif args.comp:
        comment = r"$\kappa_3$ comparisons"
    else:
        comment = r"$HHH\rightarrow 4b2\tau$"
    try:
        plot_compare_k3(
            files,
            labels,
            out_path,
            comment,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

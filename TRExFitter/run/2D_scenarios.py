#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

LEVEL_1SIGMA_2D = 2.30  # dNLL for 68% CL in 2D


def read_nll_hist_2d(root_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read TH2 'NLL' from ROOT and return (X, Y, Z) with Z normalized to dNLL (min=0).
    X, Y are meshgrid of bin centers, Z is shape (ny, nx).
    """
    with uproot.open(root_path) as f:
        if "NLL" not in f:
            raise RuntimeError(f"'NLL' object not found in {root_path}")
        h = f["NLL"]
        x_edges = h.axes[0].edges()
        y_edges = h.axes[1].edges()
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)
        Z = h.values().T
        Z = Z - np.nanmin(Z)  # dNLL
    return X, Y, Z


def contour_paths_from_grid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, level: float) -> List[np.ndarray]:
    """
    Returns a list of arrays shaped (Ni, 2) with (k3, k4) points.
    """
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[level])
    paths: List[np.ndarray] = []
    # robust across matplotlib versions via allsegs
    if hasattr(cs, "allsegs") and cs.allsegs and cs.allsegs[0]:
        for seg in cs.allsegs[0]:
            if seg is not None and len(seg) >= 2:
                paths.append(np.asarray(seg))
    plt.close(fig)
    return paths


def path_arc_length(points: np.ndarray) -> float:
    d = np.diff(points, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

def banner_heatmaps(ax: plt.Axes, comment: str = ""):
    ax.text(
        0.02, 1.02,  # position just above the axes
        r"$\mathbf{FCC\text{-}hh}$ Scenario II"
        "\n"
        r"$\mathit{Delphes}$ Simulation, "
        r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=12
    )
    ax.text(1.0, 1.02, 
        rf"{comment}",
        transform=ax.transAxes,
        va="bottom", ha="right",
        fontsize=12
    )
    
def banner(ax: plt.Axes, comment: str = "", mode: str = "hhh") -> None:
    if mode == "hh":
        ax.text(
                0.45, 0.7,  # position just above the axes
                r"$\mathbf{FCC\text{-}hh}$ Scenario II"
                "\n"
                r"$\mathit{Delphes}$ Simulation"
                "\n"
                r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
                transform=ax.transAxes,
                va="bottom", ha="left",
                fontsize=14
            )
        if comment:
            ax.text(0.45, 0.55, comment, transform=ax.transAxes, va="top", ha="left", fontsize=12)
    elif mode == "comb":
        ax.text(
            0.27, 0.54,  # position just above the axes
            r"$\mathbf{FCC\text{-}hh}$ Scenario II"
            "\n"
            r"$\mathit{Delphes}$ Simulation"
            "\n"
            r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=14
        )
        if comment:
            ax.text(0.27, 0.52, comment, transform=ax.transAxes, va="top", ha="left", fontsize=12)
    elif mode == "comp":
        ax.text(
            0.4, 0.54,  # position just above the axes
            r"$\mathbf{FCC\text{-}hh}$ Scenario II"
            "\n"
            r"$\mathit{Delphes}$ Simulation"
            "\n"
            r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=14
        )
        if comment:
            ax.text(0.4, 0.52, comment, transform=ax.transAxes, va="top", ha="left", fontsize=12)


    else:
        ax.text(
            0.35, 0.52,  # position just above the axes
            r"$\mathbf{FCC\text{-}hh}$ Scenario II"
            "\n"
            r"$\mathit{Delphes}$ Simulation"
            "\n"
            r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=14
        )
        if comment:
            ax.text(0.35, 0.5, comment, transform=ax.transAxes, va="top", ha="left", fontsize=12)


def save_csv(paths: List[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("k3,k4\n")
        for i, points in enumerate(paths):
            np.savetxt(handle, points, delimiter=",")
            if i != len(paths) - 1:
                handle.write("\n")


def resolve_axis_limits(xmin, xmax, ymin, ymax, xlim, ylim):
    xlo = xmin if xmin is not None else xlim[0]
    xhi = xmax if xmax is not None else xlim[1]
    ylo = ymin if ymin is not None else ylim[0]
    yhi = ymax if ymax is not None else ylim[1]

    if xlo >= xhi:
        raise ValueError(f"Invalid x-axis limits: xmin={xlo} must be smaller than xmax={xhi}.")
    if ylo >= yhi:
        raise ValueError(f"Invalid y-axis limits: ymin={ylo} must be smaller than ymax={yhi}.")

    return (xlo, xhi), (ylo, yhi)


def main():
    ap = argparse.ArgumentParser(description="Overlay 1sigma (dNLL=2.30) contours from TH2 NLL histograms.")
    ap.add_argument("--stat", required=False,
                    default="/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3k4_syst_StatOnly_2bin/LHoodPlots/NLLscan_k3_k4_histo.root")
    ap.add_argument("--scenI", required=False,
                    default="/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3k4_syst_ScenI_2bin/LHoodPlots/NLLscan_k3_k4_histo.root")
    ap.add_argument("--scenII", required=False,
                    default="/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3k4_syst_ScenII_2bin/LHoodPlots/NLLscan_k3_k4_histo.root")
    ap.add_argument("--scenIII", required=False,
                    default="/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/hhh_SR_asimov_Hist_220925_k3k4_syst_ScenII_2bin/LHoodPlots/NLLscan_k3_k4_histo.root")
    ap.add_argument("--labels", nargs=3, default=["No Syst", "Uncertainty Scenario I", "Uncertainty Scenario II"])
    ap.add_argument("--colors", nargs=3, default=["tab:blue", "tab:green", "tab:red"])
    ap.add_argument("--linestyles", nargs=3, default=["-", "--", "-."])
    ap.add_argument("--sigma", type=float, default=LEVEL_1SIGMA_2D, help="dNLL level (default 2.30 ≡ 68%% CL)")
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian sig for smoothing dNLL grid (bins)")
    ap.add_argument("--out", default="k3k4_contours_compare_combination.pdf")
    ap.add_argument("--write-csv", action="store_true", help="Write extracted contours to CSV next to the PDF")
    ap.add_argument("--xlim", nargs=2, type=float, default=(-1, 4))
    ap.add_argument("--ylim", nargs=2, type=float, default=(-10, 25))
    ap.add_argument("--xmin", type=float, default=None, help="Optional x-axis minimum override")
    ap.add_argument("--xmax", type=float, default=None, help="Optional x-axis maximum override")
    ap.add_argument("--ymin", type=float, default=None, help="Optional y-axis minimum override")
    ap.add_argument("--ymax", type=float, default=None, help="Optional y-axis maximum override")
    ap.add_argument(
        "--hh",
        action="store_true",
        help=r"Use $HH\rightarrow bb\gamma\gamma$ in the banner comment instead of the default $HHH\rightarrow 4b2\tau$",
    )
    ap.add_argument(
        "--comb",
        action="store_true",
        help=r"Use combination banner comment instead of the default $HHH\rightarrow 4b2\tau$",
    )
    ap.add_argument(
        "--comp",
        action="store_true",
        help=r"Plot stylised for comparing the three modes, HHH, HH and then finally the combination.",
    )
    args = ap.parse_args()

    inputs = [Path(args.stat), Path(args.scenI), Path(args.scenII)]
    inputs = inputs
    labels = args.labels
    colors = args.colors
    linestyles = args.linestyles
    xlim, ylim = resolve_axis_limits(args.xmin, args.xmax, args.ymin, args.ymax, args.xlim, args.ylim)
    if args.hh:
        banner_comment = r"$HH\rightarrow bb\gamma\gamma$"
    elif args.comb:
        banner_comment = r"$HHH\rightarrow 4b2\tau$ + $HH\rightarrow bb\gamma\gamma$"
    elif args.comp:
        banner_comment = r"$(\kappa_3,\kappa_4)$ comparison"
    else:
        banner_comment = r"$HHH\rightarrow 4b2\tau$"
    contours = []
    i=0
    for path, lab in zip(inputs, labels):
        if i > 2:
            continue
        i += 1
        X, Y, Z = read_nll_hist_2d(path)
        if args.smooth_sigma and args.smooth_sigma > 0:
            Z = gaussian_filter(Z, sigma=args.smooth_sigma)
        paths = contour_paths_from_grid(X, Y, Z, args.sigma)
        if not paths:
            print(f"[WARN] No contour at dNLL={args.sigma:.2f} in {path}")
            contours.append((lab, []))
            continue

        k3min = min(float(np.min(points[:, 0])) for points in paths)
        k3max = max(float(np.max(points[:, 0])) for points in paths)
        k4min = min(float(np.min(points[:, 1])) for points in paths)
        k4max = max(float(np.max(points[:, 1])) for points in paths)
        total_points = sum(points.shape[0] for points in paths)
        print(
            f"[{lab}] dNLL={args.sigma:.2f} contour: "
            f"{len(paths)} component(s), "
            rf"$\kappa_3 \in$  [{k3min:.3f}, {k3max:.3f}], "
            rf"$\kappa_4 \in$ [{k4min:.3f}, {k4max:.3f}] "
            f"(~{total_points} total points)"
        )
        for idx, points in enumerate(paths, start=1):
            part_k3min, part_k3max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
            part_k4min, part_k4max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
            print(
                f"    component {idx}: "
                rf"$\kappa_3 \in$ [{part_k3min:.3f}, {part_k3max:.3f}], "
                rf"$\kappa_3 \in$ [{part_k4min:.3f}, {part_k4max:.3f}] "
                f"(~{points.shape[0]} points, arc length {path_arc_length(points):.3f})"
            )

        contours.append((lab, paths))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for (lab, paths), col, ls in zip(contours, colors, linestyles):
        if not paths:
            continue
        for points in paths:
            ax.plot(points[:, 0], points[:, 1], color=col, linestyle=ls, linewidth=2)
        handles.append(Line2D([0], [0], color=col, linestyle=ls, linewidth=2, label=lab))

    ax.scatter([1.0], [1.0], marker="*", s=140, color="black", zorder=5, label="SM")
    if args.comp:
        color_star = "black"
    else:
        color_star = "None"
    handles.append(Line2D([0], [0], marker="*", color=color_star, linestyle="None", markersize=12, label="SM"))

    ax.set_xlabel(r"$\kappa_3$", loc="right", fontsize=14)
    ax.set_ylabel(r"$\kappa_4$", loc="top", fontsize=14)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if args.hh:
        mode = "hh"
        banner_heatmaps(ax, banner_comment)
        ax.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.45, 0.5),
            frameon=False,           # keep the opaque box background if you like
            facecolor="white",      # background color
            edgecolor="black",      # border around legend
            fontsize=14
        )
    elif args.comb:
        mode = "comb"
        banner(ax, banner_comment)
        ax.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.465, 0.37),
            frameon=False,           # keep the opaque box background if you like
            facecolor="white",      # background color
            edgecolor="black",      # border around legend
            fontsize=14
        )
    elif args.comp:
        mode = "comp"
        banner(ax, banner_comment, mode)
        ax.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.53, 0.37),
            frameon=False,           # keep the opaque box background if you like
            facecolor="white",      # background color
            edgecolor="black",      # border around legend
            fontsize=14
        )
    else:
        banner(ax, banner_comment)
        ax.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.545, 0.35),
            frameon=False,           # keep the opaque box background if you like
            facecolor="white",      # background color
            edgecolor="black",      # border around legend
            fontsize=14
        )
        
    fig.tight_layout()
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[✓] Wrote {outpath}")

    # optional CSV dumps
    if args.write_csv:
        for (lab, paths), path in zip(contours, inputs):
            if not paths:
                continue
            csv_path = outpath.with_suffix("").with_name(outpath.stem + f"_{Path(lab).stem.replace(' ','_')}.csv")
            save_csv(paths, csv_path)
            print(f"[i] Saved CSV for '{lab}' → {csv_path}")


if __name__ == "__main__":
    main()

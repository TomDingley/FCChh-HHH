import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

ap = argparse.ArgumentParser(description="Make pretty contours within TRExFitter output directory.")
ap.add_argument("--in-dir",  default="/data/atlaasimov_Hist_220925_k3k4_syst_ScenII_2bin_reopt_nnscore",  help="Input directory with ROOT ntuples")
ap.add_argument("--xmin", type=float, default=None, help="Optional x-axis minimum override")
ap.add_argument("--xmax", type=float, default=None, help="Optional x-axis maximum override")
ap.add_argument("--ymin", type=float, default=None, help="Optional y-axis minimum override")
ap.add_argument("--ymax", type=float, default=None, help="Optional y-axis maximum override")
ap.add_argument("--vmax", type=float, default=75, help="Optional cbar maximum override")

ap.add_argument(
    "--hh",
    action="store_true",
    help=r"Use $HH\rightarrow bb\gamma\gamma$ in the banner comment instead of the default",
)
ap.add_argument(
    "--comb",
    action="store_true",
    help=r"Use combination comment instead of the default $HHH\rightarrow 4b2\tau$",
)
args = ap.parse_args()
basedir = Path(args.in_dir)
xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax
if args.hh:
    banner_comment = r"$HH\rightarrow bb\gamma\gamma$"
elif args.comb:
    banner_comment = r"$HHH\rightarrow 4b2\tau$ + $HH\rightarrow bb\gamma\gamma$"
else:
    banner_comment = r"$HHH\rightarrow 4b2\tau$"
# Load the NLL histogram from ROOT file
with uproot.open(basedir / "LHoodPlots" / "NLLscan_k3_k4_histo.root") as f:
    hist = f["NLL"]
import csv


# --- point this to your CSV file ---
atlas_csv_path = "/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/ATLAS_HepDATA_k3k4_95CL.csv"  

def load_k3k4_contour_csv(path):
    """Reads a 2-column CSV (k3,k4), skipping headers/comments, returns (N,2) float array."""
    pts = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            # skip empty / one-column rows
            if len(row) < 2:
                continue
            try:
                k3 = float(row[0].strip())
                k4 = float(row[1].strip())
                pts.append((k3, k4))
            except ValueError:
                # skip header/comment/latex lines
                continue
    return np.asarray(pts)
# banner function

def banner(ax: plt.Axes, comment: str = ""):
    ax.text(
        0.00, 1.01,
         r"$\mathbf{FCC\text{-}hh}$ Scenario II" "\n"
        r"$\mathit{Delphes}$ Simulation, $\sqrt{s}=84\,\mathrm{TeV}$, $30\,\mathrm{ab}^{-1}$",
        transform=ax.transAxes,
        va="bottom", ha="left", fontsize=11
    )
    ax.text(0.04, 0.80, 
        rf"{comment}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=14
    )
# Extract bin centers and values
x_edges = hist.axes[0].edges()
y_edges = hist.axes[1].edges()
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
X, Y = np.meshgrid(x_centers, y_centers)
Z = hist.values().T

# Normalise: shift to ΔNLL (min = 0)
Z -= np.min(Z)

# Smooth the NLL for cleaner contours
Z_smooth = gaussian_filter(Z, sigma=0)

# Unitarity polygon from 2312.04646
k3_k4_points = np.array([
    [-8.838028169014084, 66.24],
    [-8.873239436619718, 65.28],
    [-8.838028169014084, 64.08],
    [-8.661971830985918, 57.12],
    [-8.02816901408451, 36.48],
    [-6.901408450704225, 2.4],
    [-6.091549295774648, -18.72],
    [-4.612676056338029, -50.64],
    [-4.471830985915494, -53.28],
    [-4.15492957746479, -57.12],
    [-3.732394366197185, -60.96],
    [-3.239436619718309, -63.84],
    [-2.676056338028168, -65.28],
    [-2.183098591549296, -65.76],
    [-0.10563380281690193, -65.76],
    [1.9014084507042242, -65.76],
    [2.5704225352112644, -65.52],
    [2.88732394366197, -64.8],
    [3.239436619718308, -63.6],
    [4.295774647887324, -55.44],
    [4.859154929577462, -46.08],
    [5.528169014084504, -31.92],
    [6.373239436619713, -12.24],
    [6.866197183098588, 1.44],
    [7.676056338028172, 24.72],
    [8.521126760563376, 52.8],
    [8.802816901408448, 62.64],
    [8.838028169014084, 64.08],
    [8.83802816901408, 65.28],
    [8.802816901408452, 66.24],
    [7.112676056338028, 66.24],
    [1.5140845070422486, 65.76],
    [-5.669014084507044, 66.0],
    [-8.802816901408454, 66.24]
])
def banner_heatmaps(ax: plt.Axes, comment: str = ""):
    ax.text(
        0.02, 0.85,  # position just above the axes
        r"$\mathbf{FCC\text{-}hh}$ Scenario II"
        "\n"
        r"$\mathit{Delphes}$ Simulation"
        "\n"
        r"$\sqrt{s}=84\,\mathrm{TeV},\;30\,\mathrm{ab}^{-1}$",
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=12
    )
    ax.text(0.02, 0.84, 
        rf"{comment}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=12
    )


def apply_axis_limits(ax: plt.Axes, default_xlim, default_ylim):
    xlo = xmin if xmin is not None else default_xlim[0]
    xhi = xmax if xmax is not None else default_xlim[1]
    ylo = ymin if ymin is not None else default_ylim[0]
    yhi = ymax if ymax is not None else default_ylim[1]

    if xlo >= xhi:
        raise ValueError(f"Invalid x-axis limits: xmin={xlo} must be smaller than xmax={xhi}.")
    if ylo >= yhi:
        raise ValueError(f"Invalid y-axis limits: ymin={ylo} must be smaller than ymax={yhi}.")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

# ========== Plot 1: Contours only ========== #
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Filled contour bands
contour_filled = ax1.contourf(X, Y, Z_smooth, levels=[0, 2.30, 5.99, Z.max()],
                               colors=["#a6cee3", "#1f78b4", "#eeeeee"], alpha=0)

# Line contours
import matplotlib.lines as mlines

levels = [2.30, 5.99]
linestyles = ['-', '--']
labels = ['68% CL', '95% CL']
colours = ["blue", "green"]
contour_handles = []
for lev, ls, lab, col in zip(levels, linestyles, labels, colours):
    cs = ax1.contour(
        X, Y, Z_smooth,
        levels=[lev],
        colors=col,
        linewidths=1.5,
        linestyles=ls
    )
    #ax1.clabel(cs, fmt={lev: lab}, fontsize=11)

    # proxy handle for legend
    proxy = mlines.Line2D([], [], color=col, linestyle=ls, linewidth=1.5, label=lab)
    contour_handles.append(proxy)

# Draw the SM star directly on the plot
ax1.scatter(1, 1, marker='*', color='black', s=200, zorder=5)

# Custom legend handle for SM
sm_handle = Line2D([0], [0], marker='*', color='black', markersize=15,
                   linestyle='None', label='SM')

# Unitarity polygon
unitarity_poly = Polygon(k3_k4_points, closed=True, facecolor='none',
                         edgecolor='darkgreen', linestyle='--', linewidth=2,
                         label='Perturbative Unitarity')
#ax1.add_patch(unitarity_poly)

# Aesthetics
apply_axis_limits(ax1, default_xlim=(-1, 4), default_ylim=(-10, 25))
ax1.set_xlabel(r"$\kappa_3$", loc='right', fontsize=14)
ax1.set_ylabel(r"$\kappa_4$", loc='top', fontsize=14)

# Add banner
banner_heatmaps(ax1, banner_comment)
# Now add to legend
ax1.legend(handles=contour_handles + [sm_handle],
           loc='upper right', fontsize=12, frameon=False)
fig1.tight_layout()
fig1.savefig(basedir / "nll_contours_new_poi1new.pdf")

# ========== Plot 2: Heatmap with contours ========== #
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Heatmap
# Use bin edges and rasterize the mesh to avoid visible seam lines in the PDF.
pcm = ax2.pcolormesh(
    x_edges,
    y_edges,
    Z_smooth,
    shading='flat',
    cmap='viridis',
    vmin=0,
    vmax=args.vmax,
    edgecolors='none',
    linewidth=0,
    antialiased=False,
    rasterized=True,
)
cbar = fig2.colorbar(pcm, ax=ax2)
cbar.set_label(r"$-\Delta\log L$", loc='top', fontsize=12)

# Styled contour lines (match other function)
levels      = [2.30, 5.99]
linestyles  = ['-', '--']
labels      = ['68% CL', '95% CL']
colours     = ['gray', 'green']

contour_handles = []
for lev, ls, lab, col in zip(levels, linestyles, labels, colours):
    cs = ax2.contour(
        X, Y, Z_smooth,
        levels=[lev],
        colors=col,
        linewidths=2,
        linestyles=ls
    )
    # optional inline labels per-level (uncomment if you like text on the lines)
    # ax2.clabel(cs, fmt={lev: lab}, fontsize=11)

    # legend proxy
    contour_handles.append(
        mlines.Line2D([], [], color=col, linestyle=ls, linewidth=1.5, label=lab)
    )

# Axes styling
apply_axis_limits(ax2, default_xlim=(-2.5, 6), default_ylim=(-25, 40))

# put x-label at right and y-label at top (as you used elsewhere)
ax2.set_xlabel(r"$\kappa_3$", loc="right", fontsize=14)
ax2.set_ylabel(r"$\kappa_4$", loc="top", fontsize=14)

# legend 
ax2.legend(
    handles=contour_handles,
    loc='right',
    bbox_to_anchor=(1.0, 1.05),
    ncol=1,
    fontsize=12,
    frameon=False
)

# Banner
banner(ax2)

fig2.tight_layout()
fig2.savefig(basedir / "nll_heatmap_with_contours_new_poi1new.pdf")

# --- load and plot the ATLAS contour ---
atlas_pts = load_k3k4_contour_csv(atlas_csv_path)
if atlas_pts.size == 0:
    raise RuntimeError("No (k3,k4) points parsed from the CSV — check the file path/format.")

k3_atlas, k4_atlas = atlas_pts[:,0], atlas_pts[:,1]
ATLAS_COMPARISON_LABEL = "ATLAS 13 TeV, 126 fb$^{-1}$\n" + r"arXiv:2411.02040"
FCC_COMPARISON_LABEL = "FCC-hh 84 TeV, 30 ab$^{-1}$\n" + r"This work"
UNITARITY_LABEL = "Perturbative Unitarity"

pad_y = 0.05 * (k4_atlas.max() - k4_atlas.min())
comparison_xlim = (-25, 30)
comparison_ylim = (k4_atlas.min() - pad_y, k4_atlas.max() + pad_y)

def atlas_fcc_legend_handles():
    return [
        Line2D([0], [0], color="black", linewidth=2, label=ATLAS_COMPARISON_LABEL),
        Line2D([0], [0], color="green", linewidth=2, label=FCC_COMPARISON_LABEL),
        Line2D([0], [0], color="red", linewidth=2, linestyle='--', label=UNITARITY_LABEL),
    ]

def draw_atlas_fcc_comparison(ax: plt.Axes, add_banner: bool = True):
    ax.plot(k3_atlas, k4_atlas, lw=2, color="black")

    ax.contourf(
        X, Y, Z_smooth,
        levels=[0, 5.99],
        colors=["#1f78b4"],
        alpha=0.25,
    )

    unitarity_poly = Polygon(
        k3_k4_points, closed=True, facecolor='white',
        edgecolor='red', linestyle='--', linewidth=2,
        label=UNITARITY_LABEL
    )
    ax.add_patch(unitarity_poly)

    ax.contour(
        X, Y, Z_smooth,
        levels=[5.99],
        colors='green',
        linewidths=2
    )

    apply_axis_limits(
        ax,
        default_xlim=comparison_xlim,
        default_ylim=comparison_ylim,
    )
    ax.set_xlim(*comparison_xlim)
    ax.set_ylim(*comparison_ylim)
    ax.set_xlabel(r"$\kappa_3$", loc='right', fontsize=14)
    ax.set_ylabel(r"$\kappa_4$", loc='top', fontsize=14)

    if add_banner:
        banner_heatmaps(ax)

fig, ax = plt.subplots(figsize=(9, 6))
draw_atlas_fcc_comparison(ax, add_banner=True)
ax.legend(
    handles=atlas_fcc_legend_handles(),
    loc="right",
    bbox_to_anchor=(1.005, 0.91),
    ncol=1,
    fontsize=12,
    frameon=False,
    borderaxespad=0.0
)
fig.tight_layout()
fig.savefig(basedir / "atlas_fcc_k3k4_comparisonnew.pdf")

fig, ax = plt.subplots(figsize=(10.5, 6))
draw_atlas_fcc_comparison(ax, add_banner=False)
ax.legend(
    handles=atlas_fcc_legend_handles(),
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    fontsize=12,
    frameon=False,
    borderaxespad=0.0
)
fig.tight_layout(rect=(0, 0, 0.80, 1))
fig.savefig(basedir / "atlas_fcc_k3k4_comparison_legend.pdf", bbox_inches="tight")

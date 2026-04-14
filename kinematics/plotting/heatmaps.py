import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection

from config import XLIM_MAP, HEATMAP_PAIRS, LUMINOSITY_PB
from aesthetics import LABEL_MAP, banner_heatmaps, process_labels

from matplotlib.colors import LogNorm

# choose your default colormap here
colourmap = "coolwarm"


# ------------------------------ utilities ------------------------------------
def _extract_weights(tree, mask, *, use_weights=True, weight_field="weight", lumi_scale=LUMINOSITY_PB):
    """
    Returns a numpy array of weights aligned with the masked entries.
    If use_weights=False, returns ones (raw unweighted counts).
    """
    if not use_weights:
        # We'll size this later to match x/y after cropping
        return None

    if weight_field not in tree.keys():
        raise KeyError(f"Weight field '{weight_field}' not found in tree keys: {list(tree.keys())[:10]} ...")

    w = ak.to_numpy(tree[weight_field].array(library="ak")[mask])
    if w is None:
        raise RuntimeError("Failed to extract weights from the tree.")
    return w * float(lumi_scale)


def plot_hist2d_heatmaps(
    tree, mask, proc, outdir, comment, channel,
    *,
    # weighting
    use_weights=False,
    weight_field="weight",
    lumi_scale=LUMINOSITY_PB,
    # viz / binning
    bins=(50, 50),                    # int or (nx, ny)
    cmap=colourmap,
    color_by="weight",           # "weight" or "density"
    norm="linear",               # "linear" or "log"
    vmin=None, vmax=None,        # manual color limits (optional)
    sf=1.0,                      # global scale factor applied to contents
):
    """
    Simple 2D heatmaps (numpy.histogram2d + pcolormesh) for all (vx, vy) in HEATMAP_PAIRS.

    - If use_weights=False  -> raw counts.
    - If use_weights=True   -> uses `weight_field` * `lumi_scale`.
    - color_by="weight"     -> shows sum of (weights or counts) in each bin.
    - color_by="density"    -> divides by bin area to show per-area density.
    - norm="log"            -> logarithmic color scale via LogNorm.
    """
    outdir = Path(outdir / channel)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Pull weights (or None)
    w_all = _extract_weights(
        tree, mask,
        use_weights=use_weights,
        weight_field=weight_field,
        lumi_scale=lumi_scale
    )

    for vx, vy in HEATMAP_PAIRS:
        if vx not in tree.keys() or vy not in tree.keys():
            continue

        # 2) Load arrays and crop to plotting limits
        x = ak.to_numpy(tree[vx].array(library="ak")[mask], allow_missing=False)
        y = ak.to_numpy(tree[vy].array(library="ak")[mask], allow_missing=False)
        if x.size == 0 or y.size == 0:
            continue

        xlim = XLIM_MAP.get(vx, (np.nanmin(x), np.nanmax(x)))
        ylim = XLIM_MAP.get(vy, (np.nanmin(y), np.nanmax(y)))

        keep = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if not np.any(keep):
            continue

        xk = x[keep]; yk = y[keep]
        wk = w_all[keep] if w_all is not None else None

        # 3) Histogram
        H, xedges, yedges = np.histogram2d(
            xk, yk,
            bins=bins,
            range=[xlim, ylim],
            weights=wk
        )

        # Convert to density (per unit area) if requested
        if color_by.lower() == "density":
            dx = np.diff(xedges)[:, None]     # (Nx,1)
            dy = np.diff(yedges)[None, :]     # (1,Ny)
            area = dx * dy                    # (Nx,Ny)
            # avoid /0
            area = np.where(area > 0.0, area, 1.0)
            H = H / area

        H = H * float(sf)

        # 4) Plot
        fig, ax = plt.subplots(figsize=(6, 5))

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        if norm.lower() == "log":
            Hplot = np.where(H > 0, H.T, np.nan)
            im = ax.imshow(
                Hplot,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=cmap,
                norm=LogNorm(
                    vmin=vmin if vmin else max(np.nanmin(Hplot), 1e-12),
                    vmax=vmax if vmax else np.nanmax(Hplot),
                ),
                interpolation="nearest",
            )
        else:
            im = ax.imshow(
                H.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

        cbar_label = "Events / bin" if color_by.lower() == "weight" else "Density (per unit area)"
        cbar = fig.colorbar(im, ax=ax)

        cbar.set_label(cbar_label, rotation=90)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.label.set_verticalalignment("top")
        cbar.ax.yaxis.set_label_coords(3.25, 0.89)


        ax.text(0.97, 1.05, process_labels.get(proc, proc),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(LABEL_MAP.get(vx, vx), loc="right")
        ax.set_ylabel(LABEL_MAP.get(vy, vy), loc="top")
        banner_heatmaps(ax)

        # 5) Save
        out = outdir / f"heatmap_{vx}_vs_{vy}_hist2d.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓]", out)

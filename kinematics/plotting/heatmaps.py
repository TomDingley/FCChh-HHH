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


from matplotlib.colors import LogNorm

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
        
# ----------------------------- quadtree core ---------------------------------
def _quadtree_bins(
    x, y, w, xlim, ylim,
    *,
    max_leaf_weight=50,
    min_points=50,
    min_size=(None, None),
    max_depth=10,
    split="median",
):
    """
    Build quadtree bins. Returns list of (x0, x1, y0, y1, wsum, npts).
    """
    x0, x1 = xlim
    y0, y1 = ylim
    min_dx, min_dy = (None, None) if (min_size is None or len(min_size) != 2) else min_size

    N = x.size
    if N == 0:
        return []

    full_mask = np.ones(N, dtype=bool)
    stack = [(x0, x1, y0, y1, full_mask, 0)]
    leaves = []

    while stack:
        bx0, bx1, by0, by1, mask, depth = stack.pop()
        if not np.any(mask):
            leaves.append((bx0, bx1, by0, by1, 0.0, 0))
            continue

        xi = x[mask]; yi = y[mask]
        wi = w[mask] if w is not None else np.ones_like(xi, dtype=float)
        n = xi.size
        wsum = float(wi.sum())
        dx = bx1 - bx0
        dy = by1 - by0

        stop = (
            wsum <= max_leaf_weight or
            n    <= min_points or
            (min_dx is not None and dx <= min_dx) or
            (min_dy is not None and dy <= min_dy) or
            depth >= max_depth
        )
        if stop:
            leaves.append((bx0, bx1, by0, by1, wsum, int(n)))
            continue

        if split == "median":
            mx = np.median(xi); my = np.median(yi)
            if not (bx0 < mx < bx1): mx = 0.5 * (bx0 + bx1)
            if not (by0 < my < by1): my = 0.5 * (by0 + by1)
        else:
            mx = 0.5 * (bx0 + bx1)
            my = 0.5 * (by0 + by1)

        q1 = mask & (x >= mx) & (y >= my)
        q2 = mask & (x <  mx) & (y >= my)
        q3 = mask & (x <  mx) & (y <  my)
        q4 = mask & (x >= mx) & (y <  my)

        stack.append((mx,  bx1, my,  by1, q1, depth + 1))
        stack.append((bx0, mx,  my,  by1, q2, depth + 1))
        stack.append((bx0, mx,  by0, my,  q3, depth + 1))
        stack.append((mx,  bx1, by0, my,  q4, depth + 1))

    return leaves


def _draw_quadtree(ax, bins, *, cmap="Blues", color_by="density", vmin=None, vmax=None, scale=1.0):
    """
    Draw quadtree bins. color_by in {"weight","density"}.
    """
    patches, vals = [], []
    for (x0, x1, y0, y1, wsum, _npts) in bins:
        if x1 <= x0 or y1 <= y0:
            continue
        area = (x1 - x0) * (y1 - y0)
        val = (wsum / area) if (color_by == "density" and area > 0) else wsum
        patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
        vals.append(val * scale)

    vals = np.asarray(vals, dtype=float)
    pc = PatchCollection(patches, cmap=cmap, linewidth=0.0)
    pc.set_array(vals)
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(pc)
    cbar = ax.figure.colorbar(
        pc, ax=ax,
        label=("Events / bin" if color_by == "weight" else "Density (per unit area)")
    )
    return pc, cbar


def plot_heatmaps(
    tree, mask, proc, outdir, comment, channel,  # comment kept for API parity, not used in banner
    *,
    # weighting controls
    use_weights=False,
    weight_field="weight",
    lumi_scale=LUMINOSITY_PB,
    # visualization controls
    color_by="density",        # "weight" or "density"
    max_leaf_weight=20,
    min_points=5,
    min_bin_size=None,         # (dx_min, dy_min) or None
    max_depth=10,
    cmap=colourmap,
    sf=1.0,
    also_voronoi=True,
    voronoi_kwargs=None,
):
    """
    Adaptive (quadtree) heatmaps for pairs in HEATMAP_PAIRS.

    If use_weights=False => raw unweighted yields (counts).
    If use_weights=True  => use `weight_field` and multiply by `lumi_scale`.

    color_by:
      - "weight"  : sum of weights (or counts) per leaf
      - "density" : (sum of weights or counts) / area
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract weights (or None, for unweighted)
    w_all = _extract_weights(tree, mask, use_weights=use_weights,
                             weight_field=weight_field, lumi_scale=lumi_scale)
    
    for vx, vy in HEATMAP_PAIRS:
        if vx not in tree.keys() or vy not in tree.keys():
            continue

        x = ak.to_numpy(tree[vx].array(library="ak")[mask], allow_missing=False)
        y = ak.to_numpy(tree[vy].array(library="ak")[mask], allow_missing=False)
        if x.size == 0 or y.size == 0:
            continue

        # robust limits
        xlim = XLIM_MAP.get(vx, (np.nanmin(x), np.nanmax(x)))
        ylim = XLIM_MAP.get(vy, (np.nanmin(y), np.nanmax(y)))

        keep = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if not np.any(keep):
            continue
        xk, yk = x[keep], y[keep]
        wk = w_all[keep] if w_all is not None else None  # ones handled inside _quadtree_bins

        bins = _quadtree_bins(
            xk, yk, wk,
            xlim=xlim, ylim=ylim,
            max_leaf_weight=max_leaf_weight,
            min_points=min_points,
            min_size=min_bin_size,
            max_depth=max_depth,
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        _draw_quadtree(ax, bins, cmap=cmap, color_by=color_by, scale=sf)

        ax.text(0.97, 1.05, process_labels.get(proc, proc),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(LABEL_MAP.get(vx, vx), loc="right")
        ax.set_ylabel(LABEL_MAP.get(vy, vy), loc="top")
        banner_heatmaps(ax)

        out = outdir / f"heatmap_{vx}_vs_{vy}_quadtree.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓]", out)
    

    if also_voronoi:
        kwargs = dict(
            use_weights=use_weights,
            weight_field=weight_field,
            lumi_scale=lumi_scale,
            color_by="weight",
            target_cells=800,
            cmap=cmap,
            sf=sf,
        )
        if voronoi_kwargs:
            kwargs.update(voronoi_kwargs)
        plot_voronoi_heatmaps(tree, mask, proc, outdir, comment, channel, **kwargs)



# ------------------------------- voronoi utils -------------------------------
def _clip_poly_to_box(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of polygon to axis-aligned box."""
    def clip_edge(verts, edge_fn):
        if not verts:
            return []
        out = []
        A = verts[-1]
        Ain = edge_fn(A)
        for B in verts:
            Bin = edge_fn(B)
            if Ain and Bin:
                out.append(B)
            elif Ain and not Bin:
                out.append(_segment_box_intersection(A, B, x0, x1, y0, y1))
            elif (not Ain) and Bin:
                out.append(_segment_box_intersection(A, B, x0, x1, y0, y1))
                out.append(B)
            A, Ain = B, Bin
        return out

    def inside_left(p):   return p[0] >= x0
    def inside_right(p):  return p[0] <= x1
    def inside_bottom(p): return p[1] >= y0
    def inside_top(p):    return p[1] <= y1

    v = list(poly)
    for fn in (inside_left, inside_right, inside_bottom, inside_top):
        v = clip_edge(v, fn)
        if not v:
            break
    return v

def _segment_box_intersection(a, b, x0, x1, y0, y1):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    ts = []
    if dx != 0: ts += [(x0 - ax)/dx, (x1 - ax)/dx]
    if dy != 0: ts += [(y0 - ay)/dy, (y1 - ay)/dy]
    cand = []
    for t in ts:
        if 0 <= t <= 1:
            x = ax + t*dx; y = ay + t*dy
            if (abs(x - x0) < 1e-12 or abs(x - x1) < 1e-12 or
                abs(y - y0) < 1e-12 or abs(y - y1) < 1e-12):
                cand.append((x, y))
    if not cand:
        return (np.clip(ax, x0, x1), np.clip(ay, y0, y1))
    d = [ (bx-cx)**2 + (by-cy)**2 for (cx,cy) in cand ]
    return cand[int(np.argmin(d))]

def _bounded_voronoi(points, xlim, ylim):
    """Compute bounded Voronoi regions clipped to a rectangle."""
    from scipy.spatial import Voronoi
    x0, x1 = xlim; y0, y1 = ylim
    padding = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
    P = np.vstack([points, padding])

    V = Voronoi(P)
    regions = []
    for i in range(len(points)):
        region_idx = V.point_region[i]
        verts_idx = V.regions[region_idx]
        if -1 in verts_idx or len(verts_idx) == 0:
            continue
        poly = V.vertices[verts_idx]
        clipped = _clip_poly_to_box(poly, x0, x1, y0, y1)
        if len(clipped) >= 3:
            regions.append(np.asarray(clipped))
    return regions

def _draw_voronoi(ax, regions, values, color_by="weight", cmap=colourmap, vmin=None, vmax=None, scale=1.0):
    patches, vals = [], []
    for poly, val in zip(regions, values):
        if poly is None or len(poly) < 3:
            continue
        patches.append(Polygon(poly, closed=True))
        vals.append(float(val) * scale)

    pc = PatchCollection(patches, cmap=cmap, linewidth=0.0)
    pc.set_array(np.asarray(vals))
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(pc)
    cbar = ax.figure.colorbar(pc, ax=ax,
                              label="Events / cell" if color_by == "weight" else "Density (per unit area)")
    return pc, cbar


# ------------------------------ public: voronoi ------------------------------
def plot_voronoi_heatmaps(
    tree, mask, proc, outdir, comment, channel,
    *,
    # weighting controls
    use_weights=False,
    weight_field="weight",
    lumi_scale=LUMINOSITY_PB,
    # viz controls
    color_by="weight",               # "weight" or "density"
    target_cells=None,               # e.g. 800 seeds; None -> all points
    random_seeds=None,               # fallback seed count if no sklearn
    cmap=colourmap,
    sf=1.0,
    max_points_warn=200_000,
):
    """
    Voronoi-binned heatmaps for each (vx, vy) in HEATMAP_PAIRS.
    """
    outdir = Path(outdir)
    # weights (or None)
    w_all = _extract_weights(tree, mask, use_weights=use_weights,
                             weight_field=weight_field, lumi_scale=lumi_scale)

    for vx, vy in HEATMAP_PAIRS:
        if vx not in tree.keys() or vy not in tree.keys():
            continue

        x = np.asarray(tree[vx].array(library="ak")[mask], dtype=float)
        y = np.asarray(tree[vy].array(library="ak")[mask], dtype=float)
        if x.size == 0 or y.size == 0:
            continue

        xlim = XLIM_MAP.get(vx, (np.nanmin(x), np.nanmax(x)))
        ylim = XLIM_MAP.get(vy, (np.nanmin(y), np.nanmax(y)))

        keep = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if not np.any(keep):
            continue
        x = x[keep]; y = y[keep]
        w = w_all[keep] if w_all is not None else np.ones_like(x, dtype=float)

        pts = np.column_stack([x, y])

        # Optional clustering to reduce seeds
        seeds, assign = None, None
        values = None
        if target_cells is not None and x.size > target_cells:
            try:
                from sklearn.cluster import MiniBatchKMeans
                mbk = MiniBatchKMeans(n_clusters=int(target_cells), batch_size=10_000, n_init="auto", random_state=42)
                labels = mbk.fit_predict(pts, sample_weight=w)
                seeds = mbk.cluster_centers_
                values = np.bincount(labels, weights=w, minlength=seeds.shape[0])
                assign = labels
            except Exception:
                n = int(random_seeds or min(500, max(50, target_cells//2)))
                rng = np.random.default_rng(42)
                # weight-proportional subsample if weighted
                probs = (w / w.sum()) if use_weights else None
                idx = rng.choice(pts.shape[0], size=n, replace=False, p=probs)
                seeds = pts[idx]
                values = w[idx]
        else:
            seeds = pts.copy()
            values = w.copy()

        if seeds.shape[0] > max_points_warn:
            print(f"[Voronoi] Warning: {seeds.shape[0]} seeds; tessellation may be slow.")

        regions = _bounded_voronoi(seeds, xlim, ylim)
        if not regions:
            continue

        # If we didn't already aggregate exactly, assign each point to nearest seed
        if assign is None or assign.shape[0] != pts.shape[0]:
            d2 = ((pts[:,None,:] - seeds[None,:,:])**2).sum(-1)
            assign = np.argmin(d2, axis=1)
            values = np.bincount(assign, weights=w, minlength=seeds.shape[0])

        # Map each seed to a region index (nearest centroid heuristic)
        reg_centroids = np.array([np.mean(poly, axis=0) for poly in regions])
        d2_sr = ((seeds[:,None,:] - reg_centroids[None,:,:])**2).sum(-1)
        nearest_reg = np.argmin(d2_sr, axis=1)

        val_per_region = np.bincount(nearest_reg, weights=values, minlength=len(regions))
        if color_by == "density":
            # divide by polygon area
            areas = np.array([
                0.5*np.abs(np.dot(p[:,0], np.roll(p[:,1], -1)) - np.dot(p[:,1], np.roll(p[:,0], -1)))
                for p in regions
            ])
            areas = np.clip(areas, 1e-16, None)
            val_per_region = val_per_region / areas

        fig, ax = plt.subplots(figsize=(6, 5))
        _draw_voronoi(ax, regions, val_per_region, color_by=color_by, cmap=cmap, scale=sf)

        ax.text(0.97, 1.05, process_labels.get(proc, proc),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(LABEL_MAP.get(vx, vx), loc="right")
        ax.set_ylabel(LABEL_MAP.get(vy, vy), loc="top")
        banner_heatmaps(ax)

        out = Path(outdir) / f"heatmap_{vx}_vs_{vy}_voronoi_{channel}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓]", out)


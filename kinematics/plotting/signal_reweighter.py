#from tools import get_weights
from pathlib import Path
import matplotlib.pyplot as plt
import uproot
from pathlib import Path
import argparse
import uproot
import numpy as np
from config import XLIM_MAP, N_BINS_1D, OVERLAY_PAIRS
from aesthetics import LABEL_MAP, process_labels, process_colours, banner,  banner_heatmaps
from typing import Dict
from config import processes, SIGNAL, BACKGROUNDS, SELECTION, SKIP_VARS
from tools import get_weights, numeric, build_mask_from_selection, BASIS_KEYS, basis_funcs, build_moments
from aesthetics import LABEL_MAP

from matplotlib.colors import Normalize, TwoSlopeNorm

def compare_signal_reweight_points(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    k3k4_points: list[tuple[float, float]],
    comment: str,
    channel: str
):
    from config import SIGNAL, XLIM_MAP
    from matplotlib.gridspec import GridSpec

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")

    with uproot.open(fp) as f:
        tree = f["events"]
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)
        print(files)
        
        #mask = build_masks(tree, parse_selection(SELECTION))[1][-1]

        # --- Collect all named weight branches ---
        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]

        # Load masked arrays
        arr_all_raw = tree[var].array(library="ak")[mask]
        arr_all = numeric(arr_all_raw)

        # Load each weight branch with mask
        weight_dict = {}
        for key in basis_keys:
            weight_arr = tree[key].array(library="np")[mask]
            if len(weight_arr) != len(arr_all):
                print(f"[!] Skipping {key}: mismatch with {var} length ({len(weight_arr)} vs {len(arr_all)})")
                return
            weight_dict[key] = weight_arr

        # Ensure all weight arrays have same length as observable
        for w in weight_dict.values():
        
            if len(arr_all) != len(w):
                return
        min_len = min(len(arr_all), *(len(w) for w in weight_dict.values()))
        arr_all = arr_all[:min_len]
        for key in weight_dict:
            weight_dict[key] = weight_dict[key][:min_len]

        # Apply range cut
        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
        else:
            xmin, xmax = arr_all.min(), arr_all.max()

        range_mask = (arr_all >= xmin) & (arr_all <= xmax)
        arr_all = arr_all[range_mask]
        for key in weight_dict:
            weight_dict[key] = weight_dict[key][range_mask]
            
    # Restrict variable and weights to plotting range
    if arr_all.size == 0:
        print(f"[!] No valid entries for {var}")
        return

    edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    step_x = np.append(edges[:-1], edges[-1])  # Used for all step plots

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    sm_hist = None

    for k3, k4 in k3k4_points:
        weights = get_weights(k3, k4, weight_dict)
        if len(weights) != len(arr_all):
            print(f"[!] Mismatch: len(weights) = {len(weights)} vs len(data) = {len(arr_all)} for {var}")
            continue
        counts, _ = np.histogram(arr_all, bins=edges, weights=weights)
        if counts.sum() == 0:
            continue

        norm_counts = counts / counts.sum()
        step_y = np.append(norm_counts, norm_counts[-1])

        if k3 == 1 and k4 == 1:
            label = "SM"
            sm_hist = step_y.copy()
            ax_main.step(step_x, step_y, where="post", label=label, lw=2, color="black")
        else:
            label = f"$k_3={k3:.1f},\\ k_4={k4:.1f}$"
            ax_main.step(step_x, step_y, where="post", label=label, lw=1.8)

            if sm_hist is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.divide(step_y, sm_hist, out=np.ones_like(step_y), where=sm_hist != 0)
                ax_ratio.step(step_x, ratio, where="post", label=label)
    ymax = np.max(step_y)
    
    ax_main.set_ylabel("Normalised events", loc="top")
    ylim_max = max(0.25, ymax) * 1.25
    ax_main.set_ylim(bottom=0.0001, top=ylim_max)
    ax_main.set_xlim(xmin, xmax)
    ax_main.legend(frameon=False, fontsize=10)
    ax_main.tick_params(labelbottom=False)
    banner(ax_main, comment)

    ax_ratio.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax_ratio.set_ylabel("Ratio to SM", loc="center")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.axhline(1.0, color="black", linestyle="--", lw=1)
    #ax_ratio.grid(True, linestyle=":", linewidth=0.5)

    out = outdir / f"signal_reweight_variation/{var}_ratio_{channel}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)
    
    
def heatmap_signal_reweight_efficiency(
    files: dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str
):
    """
    Make a 2D heatmap of the efficiency for events with isRecoMatched==6
    across (k3, k4) points, normalised to total signal yield.
    """

    from config import SIGNAL
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    import uproot

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")

    with uproot.open(fp) as f:
        tree = f["events"]
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)

        # Load isRecoMatched and weights
        isRecoMatched = tree["isRecoMatchedHHH"].array(library="np")[mask]

        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {k: tree[k].array(library="np")[mask] for k in basis_keys}

    # --- Compute efficiency heatmap ---
    heatmap = np.zeros((len(k3_grid), len(k4_grid)))

    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = get_weights(k3, k4, weight_dict)
            total_w = np.sum(w)
            matched_w = np.sum(w[isRecoMatched == 6])
            eff = matched_w / total_w if total_w > 0 else 0.0
            heatmap[i, j] = eff

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
    )
    banner_heatmaps(ax)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reco-matched efficiency", loc='top')

    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\kappa_4$", loc='top')

    out = outdir / f"signal_reweight_efficiency/isRecoMatched_efficiency_{channel}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)



def heatmap_signal_reweight_efficiency_presel(
    files: Dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str,
) -> None:
    """
    Make a 2D heatmap of the selection efficiency across (k3, k4):

      eff(k3, k4) = sum_w[passes selection] / sum_w[all events],

    where weights w(k3,k4) are obtained by morphing from the basis weights.
    The *same* selection SELECTION[channel] you already use is applied in the numerator.
    """

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found in files dict.")

    basis_keys = [
        "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
        "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
        "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1",
    ]

    with uproot.open(fp) as f:
        tree = f["events"]

        # Mask for "pass" events = your usual analysis selection
        selection = SELECTION[channel]
        mask_pass = build_mask_from_selection(tree, selection)  # shape: (N_events,)

        # Basis weights for ALL events (no mask here; we want N_total over full sample)
        weight_dict = {k: tree[k].array(library="np") for k in basis_keys}

    # --- Compute efficiency heatmap: eff = N_pass / N_total (weighted) ---
    n_k3 = len(k3_grid)
    n_k4 = len(k4_grid)
    heatmap = np.full((n_k3, n_k4), np.nan, dtype=float)

    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = get_weights(k3, k4, weight_dict)  # shape: (N_events,)

            total_w = np.sum(w)                 # N_total(k3,k4)
            pass_w  = np.sum(w[mask_pass])      # N_pass(k3,k4)

            eff = pass_w / total_w if total_w > 0.0 else np.nan
            heatmap[i, j] = eff

    # --- Plot ---
    outdir = Path(outdir)
    out = outdir / f"signal_reweight_efficiency/selection_efficiency_{channel}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        heatmap.T,  # transpose: x = k3, y = k4
        origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap)),
    )

    banner_heatmaps(ax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selection efficiency", loc="top")

    ax.set_xlabel(r"$\kappa_3$", loc="right")
    ax.set_ylabel(r"$\kappa_4$", loc="top")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Reweighted selection-efficiency heatmap saved to {out}")


def heatmap_signal_reweight_xsm_after_selection(
    files: Dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str,
    *,
    signal_key: str | None = None,
    selection_override: str | None = None,
    tree_name: str = "events",
    weight_branch: str = "weight_xsec",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    contour_levels: list[float] | None = None,
    make_before_after_ratio: bool = True,
) -> None:
    """
    Plot xSM (selected) across the (k3, k4) plane:

      xSM(k3, k4) = sigma_sel(k3, k4) / sigma_sel(SM)

    using the full SELECTION[channel] unless overridden.
    Optionally also write a ratio heatmap:

      ratio(k3, k4) = xSM(selected) / xSM(no selection)
    """
    signal_key = signal_key or SIGNAL
    fp = files.get(signal_key)
    if not fp:
        raise FileNotFoundError(f"Signal file for {signal_key} not found.")

    selection = SELECTION[channel] if selection_override is None else selection_override

    sm_vec = basis_funcs(1.0, 1.0)

    def _build_moments_scaled_sum(tree, mask, tag: str) -> np.ndarray | None:
        mask = np.asarray(mask, dtype=bool)
        weight_dict = {k: tree[k].array(library="np")[mask] for k in BASIS_KEYS}
        w_xsec = tree[weight_branch].array(library="np")[mask]

        lengths = [len(w_xsec)] + [len(arr) for arr in weight_dict.values()]
        min_len = min(lengths) if lengths else 0
        if min_len == 0:
            print(f"[!] No events found for {tag} in {fp}.")
            return None
        if any(l != min_len for l in lengths):
            print(f"[!] Length mismatch in {tag} weights; trimming to {min_len} entries.")
            w_xsec = w_xsec[:min_len]
            for key in weight_dict:
                weight_dict[key] = weight_dict[key][:min_len]

        finite = np.isfinite(w_xsec)
        for arr in weight_dict.values():
            finite &= np.isfinite(arr)
        if not np.any(finite):
            print(f"[!] No finite-weight events for {tag} in {fp}.")
            return None

        w_xsec = w_xsec[finite]
        for key in weight_dict:
            weight_dict[key] = weight_dict[key][finite]

        moments = build_moments(weight_dict)
        denom = moments @ sm_vec
        scale = np.divide(
            w_xsec,
            denom,
            out=np.zeros_like(w_xsec, dtype=float),
            where=denom != 0,
        )
        return np.sum(moments * scale[:, None], axis=0)

    with uproot.open(fp) as f:
        if tree_name not in f:
            raise KeyError(f"Tree '{tree_name}' not found in {fp}")
        tree = f[tree_name]

        n_entries = tree.num_entries
        mask_all = np.ones(n_entries, dtype=bool)
        if selection:
            mask_sel = build_mask_from_selection(tree, selection)
        else:
            mask_sel = mask_all

        moments_scaled_sum_sel = _build_moments_scaled_sum(
            tree, mask_sel, f"{channel} selection"
        )
        if moments_scaled_sum_sel is None:
            return

        moments_scaled_sum_all = None
        if make_before_after_ratio:
            moments_scaled_sum_all = _build_moments_scaled_sum(tree, mask_all, "no selection")
            if moments_scaled_sum_all is None:
                return

    sm_yield_sel = float(moments_scaled_sum_sel @ sm_vec)
    if sm_yield_sel <= 0.0:
        print(f"[!] SM selected yield is zero for {channel}; cannot form xSM.")
        return

    xsm_selected = np.full((len(k3_grid), len(k4_grid)), np.nan, dtype=float)
    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            target_vec = basis_funcs(float(k3), float(k4))
            yield_k_sel = float(moments_scaled_sum_sel @ target_vec)
            xsm_selected[i, j] = yield_k_sel / sm_yield_sel

    finite_vals = xsm_selected[np.isfinite(xsm_selected)]
    if finite_vals.size:
        auto_vmin = float(np.nanmin(finite_vals))
        auto_vmax = float(np.nanmax(finite_vals))
        use_vmin = auto_vmin if vmin is None else float(vmin)
        use_vmax = auto_vmax if vmax is None else float(vmax)
        norm = Normalize(vmin=use_vmin, vmax=use_vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(
        xsm_selected.T,
        origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    if contour_levels:
        K3, K4 = np.meshgrid(k3_grid, k4_grid, indexing="ij")
        cs = ax.contour(
            K3,
            K4,
            xsm_selected,
            levels=contour_levels,
            colors="white",
            linewidths=1.0,
        )
        ax.clabel(cs, fmt="%g", fontsize=8, inline=True)
    banner_heatmaps(ax, comment)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\frac{\sigma\left(\kappa_3,\kappa_4\right)}{\sigma\left(\text{SM}\right)}$", loc="top")

    ax.set_xlabel(r"$\kappa_3$", loc="right")
    ax.set_ylabel(r"$\kappa_4$", loc="top")

    outdir = Path(outdir)
    out = outdir / f"signal_reweight_xsm/xsm_k3k4_{channel}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    
    fig.savefig(out, bbox_inches="tight")
    out = outdir / f"signal_reweight_xsm/xsm_k3k4_{channel}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] xSM heatmap saved to {out}")

    if not make_before_after_ratio or moments_scaled_sum_all is None:
        return

    sm_yield_all = float(moments_scaled_sum_all @ sm_vec)
    if sm_yield_all <= 0.0:
        print("[!] SM no-selection yield is zero; cannot form before/after xSM ratio.")
        return

    xsm_nosel = np.full((len(k3_grid), len(k4_grid)), np.nan, dtype=float)
    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            target_vec = basis_funcs(float(k3), float(k4))
            yield_k_all = float(moments_scaled_sum_all @ target_vec)
            xsm_nosel[i, j] = yield_k_all / sm_yield_all

    ratio_heatmap = np.divide(
        xsm_selected,
        xsm_nosel,
        out=np.full_like(xsm_selected, np.nan),
        where=xsm_nosel != 0.0,
    )

    ratio_vals = ratio_heatmap[np.isfinite(ratio_heatmap)]
    if ratio_vals.size == 0:
        print("[!] No finite entries for before/after xSM ratio heatmap.")
        return

    ratio_min = float(np.nanmin(ratio_vals))
    ratio_max = float(np.nanmax(ratio_vals))
    if ratio_min < 1.0 < ratio_max and ratio_min != ratio_max:
        ratio_norm = TwoSlopeNorm(vmin=ratio_min, vcenter=1.0, vmax=ratio_max)
        ratio_cmap = "RdBu_r"
    elif ratio_min == ratio_max:
        ratio_norm = None
        ratio_cmap = "magma"
    else:
        ratio_norm = Normalize(vmin=ratio_min, vmax=ratio_max)
        ratio_cmap = "magma"

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(
        ratio_heatmap.T,
        origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto",
        cmap=ratio_cmap,
        norm=ratio_norm,
    )
    ratio_levels = [0.8, 0.9, 1.0, 1.1]
    levels_in_range = [lvl for lvl in ratio_levels if ratio_min <= lvl <= ratio_max]
    if levels_in_range:
        K3, K4 = np.meshgrid(k3_grid, k4_grid, indexing="ij")
        ratio_masked = np.ma.masked_invalid(ratio_heatmap)
        cs = ax.contour(
            K3,
            K4,
            ratio_masked,
            levels=levels_in_range,
            colors="black",
            linewidths=1.0,
        )
        ax.clabel(cs, fmt="%g", fontsize=8, inline=True, colors="black")

    banner_heatmaps(ax, comment)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\frac{N(\text{SR})}{N(\text{Tight Presel})}$", loc="top")

    ax.set_xlabel(r"$\kappa_3$", loc="right")
    ax.set_ylabel(r"$\kappa_4$", loc="top")

    ratio_pdf = outdir / f"signal_reweight_xsm/xsm_k3k4_before_after_ratio_{channel}.pdf"
    ratio_png = outdir / f"signal_reweight_xsm/xsm_k3k4_before_after_ratio_{channel}.png"
    ratio_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(ratio_pdf, bbox_inches="tight")
    fig.savefig(ratio_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] xSM before/after ratio heatmap saved to {ratio_png}")
    
def heatmap_signal_reweight_efficiency_LR(
    files: dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str
):
    """
    2D heatmap of efficiency over (k3, k4), normalized to total signal yield
    after `SELECTION[channel]`, for events satisfying:
        ((isRecoMatched & 0x6) == 0x6) OR ((isRecoMatched_LR & 0x7) == 0x7)

    Meaning:
      - 0x2 (2) = 2 truth Bs → distinct reco b-jets (R=0.4)
      - 0x4 (4) = 2 truth hadronic taus → distinct reco tau jets
      - 0x1 (1) = ≥2 truth Bs inside at least one large-R jet
    """
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    import uproot

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")

    with uproot.open(fp) as f:
        tree = f["events"]

        # Build selection mask once
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)

        # Load flags (masked)
        isRecoMatched     = tree["isRecoMatchedHHH"].array(library="np")[mask]
        isRecoMatched_LR  = tree["isRecoMatchedHHH_LR"].array(library="np")[mask]

        # Basis weights (masked)
        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {
            k: tree[k].array(library="np")[mask].astype(np.float64)
            for k in basis_keys
        }

    # Matching masks (bitwise)
    need_b_and_tau   = (isRecoMatched    & 0x6) == 0x6  # 2 + 4
    need_LR_both_tau = (isRecoMatched_LR & 0x7) == 0x7  # 1 + 2 + 4
    match_mask = need_b_and_tau | need_LR_both_tau      # union, no double count

    # --- Compute efficiency heatmap ---
    heatmap = np.zeros((len(k3_grid), len(k4_grid)), dtype=np.float64)

    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = np.asarray(get_weights(k3, k4, weight_dict), dtype=np.float64)
            total_w   = float(w.sum())
            matched_w = float(w[match_mask].sum())
            heatmap[i, j] = (matched_w / total_w) if total_w > 0.0 else 0.0

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap.T,  # transpose so x=k3, y=k4
        origin="lower",
        extent=(float(k3_grid.min()), float(k3_grid.max()),
                float(k4_grid.min()), float(k4_grid.max())),
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
    )
    banner_heatmaps(ax)  # your styling hook

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reco-matched efficiency", loc='top')

    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\kappa_4$", loc='top')
    #if comment:
    #    ax.set_title(comment, fontsize=10)

    out = outdir / f"signal_reweight_efficiency/isRecoMatched_efficiency_{channel}_LR.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)

    return out, heatmap

def heatmap_signal_pairing_mean(
    files: dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str,
    type: str
):
    """
    Function to compute the pairing efficiency for the signal HH->4b set
    Then perform moment morphing across the plane.
    """
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    import uproot

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")
    
    reco_dict = {
        "dRminmax": ["n_reco_matched_h1", "n_reco_matched_h2"],
        "absmass": ["n_reco_matched_h1_absmass", "n_reco_matched_h2_absmass"],
        "squaremass": ["n_reco_matched_h1_squaremass", "n_reco_matched_h2_squaremass"]
    }
    
    
    vars = reco_dict[type]
    
    with uproot.open(fp) as f:
        tree = f["events"]

        # Build selection mask once
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)

        # load reco matched sets
        nreco1 = tree[vars[0]].array(library="np")[mask]
        nreco2 = tree[vars[1]].array(library="np")[mask]

        # Basis weights (masked)
        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        
        weight_dict = {
            k: tree[k].array(library="np")[mask].astype(np.float64)
            for k in basis_keys
        }
    matched_mask = (nreco1 == 2) & (nreco2 == 2)
    
    # --- Compute efficiency heatmap ---
    heatmap = np.zeros((len(k3_grid), len(k4_grid)), dtype=np.float64)

    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = np.asarray(get_weights(k3, k4, weight_dict), dtype=np.float64)
            total_w   = float(w.sum())
            matched_w = float(w[matched_mask].sum())
            heatmap[i, j] = (matched_w / total_w) if total_w > 0.0 else 0.0

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap.T,  # transpose so x=k3, y=k4
        origin="lower",
        extent=(float(k3_grid.min()), float(k3_grid.max()),
                float(k4_grid.min()), float(k4_grid.max())),
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
    )
    banner_heatmaps(ax)  # your styling hook

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"4$b$ pairing efficiency", loc='top')

    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\kappa_4$", loc='top')
    #if comment:
    #    ax.set_title(comment, fontsize=10)

    out = outdir / f"signal_reweight_efficiency/4bMatching_{type}_{channel}_LR.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)

    return out, heatmap
    

def heatmap_signal_reweight_mean(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str,
):
    """
    Make a 2D heatmap showing how the mean value of a given observable (e.g. Higgs_HT)
    varies across (k3, k4) Reweighted points.
    """

    from config import SIGNAL, XLIM_MAP
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    import uproot

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")

    with uproot.open(fp) as f:
        tree = f["events"]
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)

        # Load observable and EFT basis weights
        arr_all = numeric(tree[var].array(library="ak")[mask])

        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {k: tree[k].array(library="np")[mask] for k in basis_keys}

        # Apply range cut
        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
        else:
            xmin, xmax = arr_all.min(), arr_all.max()

        range_mask = (arr_all >= xmin) & (arr_all <= xmax)
        arr_all = arr_all[range_mask]
        for k in weight_dict:
            weight_dict[k] = weight_dict[k][range_mask]

    # Allocate 2D mean map
    mean_map = np.zeros((len(k3_grid), len(k4_grid)))

    # Loop over k3k4 grid
    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = get_weights(k3, k4, weight_dict)
            total_w = np.sum(w)
            if total_w <= 0:
                mean_map[i, j] = np.nan
                continue
            mean_map[i, j] = np.sum(w * arr_all) / total_w

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        mean_map.T,
        origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=np.nanmin(mean_map), vmax=np.nanmax(mean_map))
    )
    cbar = fig.colorbar(im, ax=ax)
    pretty_labels = LABEL_MAP[var].replace(" [GeV]", "")
    cbar.set_label(rf"Mean {pretty_labels}", loc='top')

    ax.set_xlabel(r"$\kappa_3$", loc='right')
    ax.set_ylabel(r"$\kappa_4$", loc='top')
    banner_heatmaps(ax)

    out = outdir / f"signal_reweight_heatmap/{var}_mean_{channel}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)
    
    
def heatmap_signal_reweight_mean_with_slices(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    k3_grid: np.ndarray,
    k4_grid: np.ndarray,
    comment: str,
    channel: str,
):
    """
    2D heatmap of mean(var) across (k3,k4) + 1D slices at k4=1 and k3=1.
    For the k4-slice (at k3=1) we fit f(k4)=(a+b k4 + c k4^2)/(1 + d k4 + e k4^2).
    """

    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import uproot

    fp = files.get(SIGNAL)
    if not fp:
        raise FileNotFoundError(f"Signal file for {SIGNAL} not found.")

    with uproot.open(fp) as f:
        tree = f["events"]
        selection = SELECTION[channel]
        mask = build_mask_from_selection(tree, selection)

        # Load observable and Reweighted basis weights
        arr_all = numeric(tree[var].array(library="ak")[mask])

        basis_keys = [
            "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40",
            "weight_k30_k4m1", "weight_k3m1_k40", "weight_k3m2_k4m1",
            "weight_k3m1_k4m2", "weight_k3m0p5_k4m1", "weight_k3m1p5_k4m1"
        ]
        weight_dict = {k: tree[k].array(library="np")[mask] for k in basis_keys}

        # Apply plotting range
        if var in XLIM_MAP:
            xmin, xmax = XLIM_MAP[var]
        else:
            xmin, xmax = arr_all.min(), arr_all.max()

        sel = (arr_all >= xmin) & (arr_all <= xmax)
        arr_all = arr_all[sel]
        for k in weight_dict:
            weight_dict[k] = weight_dict[k][sel]

    # --- Compute 2D mean map ---
    mean_map = np.zeros((len(k3_grid), len(k4_grid)))
    for i, k3 in enumerate(k3_grid):
        for j, k4 in enumerate(k4_grid):
            w = get_weights(k3, k4, weight_dict)
            tw = np.sum(w)
            mean_map[i, j] = np.sum(w * arr_all) / tw if tw > 0 else np.nan

    # Helpers
    def nearest_index(grid, value): return int(np.argmin(np.abs(grid - value)))
    pretty_label = LABEL_MAP.get(var, var).replace(" [GeV]", "")

    # === (1) 2D heatmap ===
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        mean_map.T, origin="lower",
        extent=(k3_grid.min(), k3_grid.max(), k4_grid.min(), k4_grid.max()),
        aspect="auto", cmap="viridis",
        norm=Normalize(vmin=np.nanmin(mean_map), vmax=np.nanmax(mean_map))
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Mean {pretty_label}", loc="top")

    ax.set_xlabel(r"$\kappa_3$", loc="right")
    ax.set_ylabel(r"$\kappa_4$", loc="top")
    banner_heatmaps(ax)

    out2d = outdir / f"signal_reweight_heatmap/{var}_mean.pdf"
    out2d.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out2d, bbox_inches="tight")
    plt.close(fig)

    # === (2) 1D slice: mean vs k3 (at k4=1) ===
    # === (2) 1D slice: mean vs k3 (at k4=1) + XS ratio ===
    j_k4_1 = nearest_index(k4_grid, 1.0)
    mean_vs_k3 = mean_map[:, j_k4_1]

    # cross-section proxy (total weight) along k3 at k4=1
    tot_w_k3 = np.array([np.sum(get_weights(k3, 1.0, weight_dict)) for k3 in k3_grid], dtype=float)
    tot_w_sm = float(np.sum(get_weights(1.0, 1.0, weight_dict)))  # SM normalization
    ratio_k3 = np.divide(tot_w_k3, tot_w_sm, out=np.full_like(tot_w_k3, np.nan), where=(tot_w_sm != 0.0))

    # plot with a 3:1 main:ratio layout
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(6, 5),
        sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05),
    )
    ax.plot(k3_grid, mean_vs_k3, color='black', lw=2)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_ylabel(f"Mean {pretty_label}")
    banner_heatmaps(ax)

    # ratio panel
    rax.plot(k3_grid, ratio_k3, lw=1.8)
    rax.axhline(1.0, color="grey", ls="--", lw=1)
    rax.set_xlabel(r"$\kappa_3$")
    rax.set_ylabel(r"$\sigma/\sigma_{\rm SM}$")

    # optional: tidy y-range if finite
    finite = np.isfinite(ratio_k3)
    if finite.any():
        rmin, rmax = np.nanmin(ratio_k3[finite]), np.nanmax(ratio_k3[finite])
        pad = 0.05 * max(1e-12, rmax - rmin)
        rax.set_ylim(rmin - pad, rmax + pad)

    out_k3 = outdir / f"signal_reweight_heatmap/{var}_mean_slice_k3.pdf"
    fig.savefig(out_k3, bbox_inches="tight")
    plt.close(fig)

    # === (3) 1D slice: mean vs k4 (at k3=1) + FIT + XS ratio ===
    i_k3_1 = nearest_index(k3_grid, 1.0)
    k4_vals = k4_grid.copy()
    mu = mean_map[i_k3_1, :].copy()

    # total weights per k4 (already used for WLS)
    tot_w = np.array([np.sum(get_weights(1.0, k4, weight_dict)) for k4 in k4_vals], dtype=float)
    tot_w_sm = float(np.sum(get_weights(1.0, 1.0, weight_dict)))
    ratio_k4 = np.divide(tot_w, tot_w_sm, out=np.full_like(tot_w, np.nan), where=(tot_w_sm != 0.0))

    # Mask any nans/infs
    m = np.isfinite(mu) & np.isfinite(k4_vals) & (tot_w > 0)
    x = k4_vals[m]
    y = mu[m]
    w = tot_w[m]  # weights ~ total yield at each Reweighted point

    # --- Linear least-squares for rational form ---
    # y = (a + b x + c x^2) / (1 + d x + e x^2)
    # => a*1 + b*x + c*x^2 + d*(-y*x) + e*(-y*x^2) = y
    X = np.column_stack([np.ones_like(x), x, x**2, -y*x, -y*x**2])
    # Weighted LS: multiply rows by sqrt(w)
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Xw = X * sw[:, None]
    yw = y * sw
    theta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    a, b, c, d, e = theta.tolist()

    # Goodness-of-fit (χ² with weights w)
    yfit = (a + b*x + c*x**2) / (1 + d*x + e*x**2)
    chi2 = np.sum(w * (y - yfit)**2)
    ndof = max(len(y) - 5, 1)

    # Parameter covariance estimate
    XTWX = X.T @ (w[:, None] * X)
    cov = np.linalg.pinv(XTWX) * (chi2 / ndof)
    perr = np.sqrt(np.diag(cov))

    # Save params
    fit_json = {
        "var": var, "channel": channel, "k3_fixed": float(k3_grid[i_k3_1]),
        "params": {"a": a, "b": b, "c": c, "d": d, "e": e},
        "errors": {"a": float(perr[0]), "b": float(perr[1]),
                   "c": float(perr[2]), "d": float(perr[3]), "e": float(perr[4])},
        "chi2": float(chi2), "ndof": int(ndof), "chi2_ndof": float(chi2/ndof)
    }
    out_fit = outdir / f"signal_reweight_heatmap/{var}_mean_slice_k4_fit.json"
    with open(out_fit, "w") as fh:
        json.dump(fit_json, fh, indent=2)

    # Plot slice + fit
    xx = np.linspace(k4_grid.min(), k4_grid.max(), 600)
    yy = (a + b*xx + c*xx**2) / (1 + d*xx + e*xx**2)


    # plot with a 3:1 main:ratio layout
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(6, 5),
        sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05),
    )
    ax.plot(k4_vals, mu, lw=2, color='black', label="Mean (points)")
    ax.plot(xx, yy, lw=2, ls="--", label="Fit: (a+b x+c x²)/(1+d x+e x²)")
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_ylabel(f"Mean {pretty_label}")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    txt = (rf"a={a:.3g}, b={b:.3g}, c={c:.3g}" "\n"
        rf"d={d:.3g}, e={e:.3g},  $\chi^2$/ndof={chi2/ndof:.2f}")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, va="bottom", ha="left")
    banner_heatmaps(ax)

    # ratio panel
    rax.plot(k4_vals, ratio_k4, lw=1.8)
    rax.axhline(1.0, color="grey", ls="--", lw=1)
    rax.set_xlabel(r"$\kappa_4$")
    rax.set_ylabel(r"$\sigma/\sigma_{\rm SM}$")

    finite = np.isfinite(ratio_k4)
    if finite.any():
        rmin, rmax = np.nanmin(ratio_k4[finite]), np.nanmax(ratio_k4[finite])
        pad = 0.05 * max(1e-12, rmax - rmin)
        rax.set_ylim(rmin - pad, rmax + pad)

    out_k4 = outdir / f"signal_reweight_heatmap/{var}_mean_slice_k4_with_fit.pdf"
    fig.savefig(out_k4, bbox_inches="tight")
    plt.close(fig)

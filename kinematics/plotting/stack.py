import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uproot
import awkward as ak

from config import XLIM_MAP, N_BINS_1D, SIGNAL, BACKGROUNDS, LUMINOSITY_PB, SELECTION, HADHAD
from aesthetics import LABEL_MAP, process_labels, process_colours, banner, channel_colors, channel_labels
from tools2 import  numeric, get_weights, build_mask_from_selection

import matplotlib.gridspec as gridspec
    
def stack_plot_weight(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    channel: str,
    comment: str = "",
    *,
    trees: dict[str, object] | None = None,
    masks: dict[str, ak.Array] | None = None,
):
    selection = SELECTION[channel]
    if trees is not None and SIGNAL in trees:
        t_sig = trees[SIGNAL]
        m_sig = masks[SIGNAL] if masks is not None and SIGNAL in masks else build_mask_from_selection(t_sig, selection)
        sig = numeric(t_sig[var].array(library="ak")[m_sig])
        w_sig = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig]) * LUMINOSITY_PB
    else:
        with uproot.open(files[SIGNAL]) as f_sig:
            t_sig = f_sig["events"]
            m_sig = build_mask_from_selection(t_sig, selection)
            sig = numeric(t_sig[var].array(library="ak")[m_sig])
            w_sig = numeric(t_sig["weight_xsec"].array(library="ak")[m_sig]) * LUMINOSITY_PB

    if sig.size == 0:
        return

    # Infer plot limits
    data_xmin, data_xmax = sig.min(), sig.max()
    if var in XLIM_MAP:
        user_xmin, user_xmax = XLIM_MAP[var]
        xmin = max(data_xmin, user_xmin) if var not in ["BDT_score", "NN_score"] else user_xmin
        xmax = min(data_xmax, user_xmax) if var not in ["BDT_score", "NN_score"] else user_xmax
        if xmin >= xmax:
            print(f"Skipping {var} - incompatible selection and XLIM range")
            return
    else:
        xmin, xmax = data_xmin, data_xmax

    edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    mask = (sig >= xmin) & (sig <= xmax)
    if len(mask) != len(w_sig):
        return

    cnt_sig = np.histogram(sig[mask], bins=edges, weights=w_sig[mask])[0]

    stacks, cols, labs = [], [], []
    for proc in BACKGROUNDS:
        if trees is not None and proc in trees:
            t = trees[proc]
            m_bkg = masks[proc] if masks is not None and proc in masks else build_mask_from_selection(t, selection)
            arr = numeric(t[var].array(library="ak")[m_bkg])
            weights = numeric(t["weight_xsec"].array(library="ak")[m_bkg]) * LUMINOSITY_PB
            if arr.size == 0:
                continue
            mask = (arr >= xmin) & (arr <= xmax)
            if len(mask) != len(weights):
                return
            hist = np.histogram(arr[mask], bins=edges, weights=weights[mask])[0]
            stacks.append(hist)
            cols.append(process_colours[proc])
            labs.append(proc)
            continue
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f_bkg:
            t = f_bkg["events"]
            m_bkg = build_mask_from_selection(t, selection)
            arr = numeric(t[var].array(library="ak")[m_bkg])
            weights = numeric(t["weight_xsec"].array(library="ak")[m_bkg]) * LUMINOSITY_PB         
            if arr.size == 0:
                continue
            mask = (arr >= xmin) & (arr <= xmax)
            if len(mask) != len(weights):
                return
            hist = np.histogram(arr[mask], bins=edges, weights=weights[mask])[0]
            stacks.append(hist)
            cols.append(process_colours[proc])
            labs.append(proc)

    if not stacks:
        return

    bkg_total = np.sum(stacks, axis=0)
    ratio = np.divide(cnt_sig, bkg_total, out=np.zeros_like(cnt_sig), where=bkg_total > 0)

    # === Plotting logic ===
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    x = np.append(edges[:-1], edges[-1])
    bottom = np.zeros_like(cnt_sig)
    stacks.reverse()
    cols.reverse()
    labs.reverse()
    for cts, col, lab in zip(stacks, cols, labs):
        step_bottom = np.append(bottom, bottom[-1])
        step_y = np.append(cts, cts[-1])
        ax.fill_between(x, step_bottom, step_bottom + step_y, step="post", color=col, alpha=1,
                        label=process_labels.get(lab, lab))
        bottom += cts

    ax.step(x, np.append(bottom, bottom[-1]), where="post", color="black", lw=1.2, linestyle="-", label="Total background")
    ax.step(x, np.append(cnt_sig, cnt_sig[-1]), where="post", color=process_colours[SIGNAL], lw=2, linestyle="--", label=process_labels[SIGNAL])

    ax.set_ylabel("Events / bin", loc="top")
    if np.ptp(bottom + cnt_sig) > 1e3:
        ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='both', direction='in')

    handles, labels = ax.get_legend_handles_labels()
    bg_handles_labels = [(h, l) for h, l in zip(handles, labels) if l not in [process_labels.get(SIGNAL, SIGNAL), "Total background"]]
    sig_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in [process_labels.get(SIGNAL, SIGNAL), "Total background"]]
    bg_handles_labels.reverse()
    handles, labels = zip(*bg_handles_labels + sig_handles_labels)
    ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=10, ncol=2)
    # Compute total stacked histogram
    total = bottom + cnt_sig
    nonzero_total = total[total > 0]

    if nonzero_total.size > 0:
        ymin = 0.1
        ymax = np.max(nonzero_total) * 500
    else:
        ymin, ymax = 0.1, 10

    if np.ptp(bottom + cnt_sig) > 1e2:
        ax.set_yscale("log")
        ax.set_ylim(0.1, ymax)
    else:
        ax.set_ylim(0, ymax)

    banner(ax, comment)
    ax.tick_params(labelbottom=False)

    # === Ratio subplot ===
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    step_x = np.append(edges[:-1], edges[-1])
    step_y = np.append(ratio, ratio[-1])
    ax_ratio.step(step_x, step_y, where="post", color="black", lw=1.5)
    ax_ratio.axhline(1, color="gray", linestyle="--", lw=1)
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("S/B", fontsize=10)
    ax_ratio.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax_ratio.tick_params(axis='both', direction='in', top=True, right=True)
    ax_ratio.set_xlim(edges[0], edges[-1])
        
    # Save
    out = outdir / f"{channel}/stacked/{var}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)
    
def normalised_overlay_plot_chan(var: str, files: dict[str, Path], outdir: Path, channels, k3: float = 1.0, k4: float = 1.0, comment: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    from config import XLIM_MAP

    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    else:
        edges = None

    for proc in BACKGROUNDS + [SIGNAL]:
        fp = files.get(proc)
        if not fp:
            continue
        for chan in channels:
            #if chan == "Total": continue
            selection = SELECTION[chan]

            with uproot.open(fp) as f:
                tree = f["events"]
                mask = build_mask_from_selection(tree, selection)
                arr = numeric(tree[var].array(library="ak")[mask])
                if var in XLIM_MAP:
                    arr = arr[(arr >= xmin) & (arr <= xmax)]

                if arr.size == 0:
                    continue
                
                weights = np.ones_like(arr)

                if edges is None:
                    edges = np.linspace(arr.min(), arr.max(), N_BINS_1D + 1)
                print(f"Process {proc}: plotting {var} with length {len(arr)} and weight length {len(weights)}")
                if len(arr) != len(weights):
                    return
                counts, _ = np.histogram(arr, bins=edges, weights=weights)
                if counts.sum() == 0:
                    continue
                norm_counts = counts / counts.sum()

                step_x = np.append(edges[:-1], edges[-1])
                step_y = np.append(norm_counts, norm_counts[-1])
                ax.step(step_x, step_y, where="post", color=channel_colors[chan],
                        lw=2 if proc == SIGNAL else 1.5,
                        linestyle="--" if proc == SIGNAL else "-",
                        label=f"{channel_labels[chan]}")

    if edges is None:
        return

    ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax.set_ylabel("Normalised events", loc="top")
    ax.set_ylim(bottom=0, top=0.125)
    if var in XLIM_MAP:
        ax.set_xlim(*XLIM_MAP[var])
    
    if len(files) > 4:
        ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=2)
    else:
        ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=1)
    banner(ax)

    out = outdir / f"compare/normalised/{var}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)



def overlay_plot(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    channel: str,
    *,
    normalise: bool = True,
    k3: float = 1.0,
    k4: float = 1.0,
    comment: str = "",
    trees: dict[str, object] | None = None,
    masks: dict[str, ak.Array] | None = None,
):
    if normalise:
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax_ratio = None
    from config import XLIM_MAP

    edges = None
    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    ymax = 0.0  # track max y actually plotted
    sig_counts = None
    bkg_counts = None
    ratio_sig_counts = None
    ratio_bkg_counts = None
    ratio_sig_w2 = None
    ratio_bkg_w2 = None

    for proc in BACKGROUNDS + [SIGNAL]:
        selection = SELECTION[channel]
        if trees is not None and proc in trees:
            tree = trees[proc]
            mask = masks[proc] if masks is not None and proc in masks else build_mask_from_selection(tree, selection)
            arr = numeric(tree[var].array(library="ak")[mask])
            mask_x = None

            if var in XLIM_MAP:
                mask_x = (arr >= xmin) & (arr <= xmax)
                arr = arr[mask_x]

            if arr.size == 0:
                continue

            # --- Optional signal EFT reweighting (kept as-is; enable if needed) ---
            # if proc == SIGNAL:
            #     if "reweight" not in tree:
            #         raise KeyError("Signal tree is missing 'reweight' branch.")
            #     all_weights = tree["reweight"].array(library="ak")[mask]
            #     all_weights = np.stack([np.asarray(x) for x in all_weights])
            #     weights = get_weights(k3, k4, all_weights)
            # else:
            weights = np.ones_like(arr)

            if edges is None:
                edges = np.linspace(arr.min(), arr.max(), N_BINS_1D + 1)

            if len(arr) != len(weights):
                raise ValueError(f"{proc}: len(arr)={len(arr)} != len(weights)={len(weights)}")

            counts, _ = np.histogram(arr, bins=edges, weights=weights)
            if counts.sum() == 0:
                continue

            if normalise:
                y = counts / counts.sum()
                if proc == SIGNAL:
                    sig_counts = counts
                else:
                    bkg_counts = counts if bkg_counts is None else bkg_counts + counts
            else:
                y = counts
            # update plotted max (use y, not step_y; they're identical for max)
            if y.size:
                ymax = max(ymax, float(np.max(y)))

            step_x = np.append(edges[:-1], edges[-1])
            step_y = np.append(y, y[-1])
            ax.step(
                step_x, step_y, where="post",
                color=process_colours[proc],
                lw=2 if proc == SIGNAL else 1.5,
                linestyle="-",
                label=process_labels.get(proc, proc),
            )
            if len(weights) != 0:
                print(f"Process {proc}: plotting {var} with length {len(arr)} and weight length {len(weights)}")

            if normalise:
                # Ratio uses weighted yields to be representative, top panel stays unweighted.
                try:
                    weights_ratio = numeric(tree["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB
                    if mask_x is not None:
                        weights_ratio = weights_ratio[mask_x]
                except Exception:
                    weights_ratio = np.ones_like(arr)
                if len(arr) == len(weights_ratio):
                    counts_ratio, _ = np.histogram(arr, bins=edges, weights=weights_ratio)
                    counts_ratio_w2, _ = np.histogram(arr, bins=edges, weights=weights_ratio * weights_ratio)
                    if counts_ratio.sum() > 0:
                        if proc == SIGNAL:
                            ratio_sig_counts = counts_ratio if ratio_sig_counts is None else ratio_sig_counts + counts_ratio
                            ratio_sig_w2 = counts_ratio_w2 if ratio_sig_w2 is None else ratio_sig_w2 + counts_ratio_w2
                        else:
                            ratio_bkg_counts = counts_ratio if ratio_bkg_counts is None else ratio_bkg_counts + counts_ratio
                            ratio_bkg_w2 = counts_ratio_w2 if ratio_bkg_w2 is None else ratio_bkg_w2 + counts_ratio_w2
            continue
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f:
            tree = f["events"]
            mask = build_mask_from_selection(tree, selection)
            arr = numeric(tree[var].array(library="ak")[mask])
            mask_x = None

            if var in XLIM_MAP:
                mask_x = (arr >= xmin) & (arr <= xmax)
                arr = arr[mask_x]

            if arr.size == 0:
                continue

            # --- Optional signal EFT reweighting (kept as-is; enable if needed) ---
            # if proc == SIGNAL:
            #     if "reweight" not in tree:
            #         raise KeyError("Signal tree is missing 'reweight' branch.")
            #     all_weights = tree["reweight"].array(library="ak")[mask]
            #     all_weights = np.stack([np.asarray(x) for x in all_weights])
            #     weights = get_weights(k3, k4, all_weights)
            # else:
            weights = np.ones_like(arr)

            if edges is None:
                edges = np.linspace(arr.min(), arr.max(), N_BINS_1D + 1)

            if len(arr) != len(weights):
                raise ValueError(f"{proc}: len(arr)={len(arr)} != len(weights)={len(weights)}")

            counts, _ = np.histogram(arr, bins=edges, weights=weights)
            if counts.sum() == 0:
                continue

            if normalise:
                y = counts / counts.sum()
                if proc == SIGNAL:
                    sig_counts = counts
                else:
                    bkg_counts = counts if bkg_counts is None else bkg_counts + counts
            else:
                y = counts
            # update plotted max (use y, not step_y; they're identical for max)
            if y.size:
                ymax = max(ymax, float(np.max(y)))

            step_x = np.append(edges[:-1], edges[-1])
            step_y = np.append(y, y[-1])
            ax.step(
                step_x, step_y, where="post",
                color=process_colours[proc],
                lw=2 if proc == SIGNAL else 1.5,
                linestyle="-",
                label=process_labels.get(proc, proc),
            )
            if len(weights) != 0:
                print(f"Process {proc}: plotting {var} with length {len(arr)} and weight length {len(weights)}")

            if normalise:
                # Ratio uses weighted yields to be representative, top panel stays unweighted.
                try:
                    weights_ratio = numeric(tree["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB
                    if mask_x is not None:
                        weights_ratio = weights_ratio[mask_x]
                except Exception:
                    weights_ratio = np.ones_like(arr)
                if len(arr) == len(weights_ratio):
                    counts_ratio, _ = np.histogram(arr, bins=edges, weights=weights_ratio)
                    counts_ratio_w2, _ = np.histogram(arr, bins=edges, weights=weights_ratio * weights_ratio)
                    if counts_ratio.sum() > 0:
                        if proc == SIGNAL:
                            ratio_sig_counts = counts_ratio if ratio_sig_counts is None else ratio_sig_counts + counts_ratio
                            ratio_sig_w2 = counts_ratio_w2 if ratio_sig_w2 is None else ratio_sig_w2 + counts_ratio_w2
                        else:
                            ratio_bkg_counts = counts_ratio if ratio_bkg_counts is None else ratio_bkg_counts + counts_ratio
                            ratio_bkg_w2 = counts_ratio_w2 if ratio_bkg_w2 is None else ratio_bkg_w2 + counts_ratio_w2



    if edges is None:
        plt.close(fig)
        return

    if not normalise:
        ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax.set_ylabel("Normalised events" if normalise else "Events", loc="top")

    # Y-range: only force for normalised plots (raw yields can vary widely)
    if normalise:
        # 1.5 * max, with a small fallback in case nothing was plotted
        top = 1.5 * ymax if ymax > 0 else 1.0
        ax.set_ylim(bottom=0, top=top)
    else:
        ax.set_ylim(bottom=0)
    if var in XLIM_MAP:
        ax.set_xlim(*XLIM_MAP[var])

    ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=2 if len(files) > 4 else 1)
    banner(ax, comment)

    if normalise and ax_ratio is not None:
        if ratio_sig_counts is None or ratio_bkg_counts is None:
            ratio_sig_counts = sig_counts
            ratio_bkg_counts = bkg_counts
            ratio_sig_w2 = sig_counts
            ratio_bkg_w2 = bkg_counts
        if ratio_sig_counts is not None and ratio_bkg_counts is not None:
            if ratio_sig_counts.sum() > 0 and ratio_bkg_counts.sum() > 0:
                sig_norm = ratio_sig_counts / ratio_sig_counts.sum()
                bkg_norm = ratio_bkg_counts / ratio_bkg_counts.sum()
                ratio = np.divide(sig_norm, bkg_norm, out=np.zeros_like(sig_norm), where=bkg_norm > 0)
                sig_err = np.divide(np.sqrt(ratio_sig_w2), ratio_sig_counts.sum(), out=np.zeros_like(sig_norm), where=ratio_sig_counts.sum() > 0)
                bkg_err = np.divide(np.sqrt(ratio_bkg_w2), ratio_bkg_counts.sum(), out=np.zeros_like(bkg_norm), where=ratio_bkg_counts.sum() > 0)
                sig_rel = np.divide(sig_err, sig_norm, out=np.zeros_like(sig_err), where=sig_norm > 0)
                bkg_rel = np.divide(bkg_err, bkg_norm, out=np.zeros_like(bkg_err), where=bkg_norm > 0)
                ratio_err = ratio * np.sqrt(sig_rel * sig_rel + bkg_rel * bkg_rel)
                step_x = np.append(edges[:-1], edges[-1])
                step_y = np.append(ratio, ratio[-1])
                ax_ratio.step(step_x, step_y, where="post", color="black", lw=1.5)
                valid = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio > 0)
                band_lo = np.where(valid, np.clip(ratio - ratio_err, 0, None), np.nan)
                band_hi = np.where(valid, ratio + ratio_err, np.nan)
                ax_ratio.fill_between(
                    step_x,
                    np.append(band_lo, band_lo[-1]),
                    np.append(band_hi, band_hi[-1]),
                    step="post",
                    color="black",
                    alpha=0.2,
                    linewidth=0,
                )
        ax_ratio.axhline(1, color="gray", linestyle="--", lw=1)
        ax_ratio.set_ylabel("S/B", fontsize=10)
        ax_ratio.set_xlabel(LABEL_MAP.get(var, var), loc="right")
        ax_ratio.tick_params(axis='both', direction='in', top=True, right=True)
        ax_ratio.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelbottom=False)

    subdir = "normalised" if normalise else "raw"
    out = outdir / f"{channel}/{subdir}/{var}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def normalised_overlay_plot(var: str, files: dict[str, Path], outdir: Path, channel: str,
                            k3: float = 1.0, k4: float = 1.0, comment: str = "",
                            trees: dict[str, object] | None = None,
                            masks: dict[str, ak.Array] | None = None):
    overlay_plot(var, files, outdir, channel, normalise=True, k3=k3, k4=k4, comment=comment, trees=trees, masks=masks)


def raw_overlay_plot(var: str, files: dict[str, Path], outdir: Path, channel: str,
                     k3: float = 1.0, k4: float = 1.0, comment: str = "",
                     trees: dict[str, object] | None = None,
                     masks: dict[str, ak.Array] | None = None):
    overlay_plot(var, files, outdir, channel, normalise=False, k3=k3, k4=k4, comment=comment, trees=trees, masks=masks)


def normalised_total_background_plot(
    var: str,
    files: dict[str, Path],
    outdir: Path,
    channel: str,
    k3: float = 1.0,
    k4: float = 1.0,
    comment: str = "",
    *,
    trees: dict[str, object] | None = None,
    masks: dict[str, ak.Array] | None = None,
):
    from config import XLIM_MAP

    selection = SELECTION[channel]
    edges = None
    xmin, xmax = None, None

    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    else:
        vmins, vmaxs = [], []
        for proc in BACKGROUNDS + [SIGNAL]:
            if trees is not None and proc in trees:
                tree = trees[proc]
                mask = masks[proc] if masks is not None and proc in masks else build_mask_from_selection(tree, selection)
                arr = numeric(tree[var].array(library="ak")[mask])
                if arr.size == 0:
                    continue
                vmins.append(float(np.min(arr)))
                vmaxs.append(float(np.max(arr)))
                continue
            fp = files.get(proc)
            if not fp:
                continue
            with uproot.open(fp) as f:
                tree = f["events"]
                mask = build_mask_from_selection(tree, selection)
                arr = numeric(tree[var].array(library="ak")[mask])
                if arr.size == 0:
                    continue
                vmins.append(float(np.min(arr)))
                vmaxs.append(float(np.max(arr)))
        if not vmins:
            print(f"[!] No entries found for '{var}' in channel '{channel}'.")
            return
        xmin, xmax = min(vmins), max(vmaxs)
        if xmin == xmax:
            eps = 1e-9 if xmin == 0.0 else 1e-6 * abs(xmin)
            xmin -= eps
            xmax += eps
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)

    bkg_total = np.zeros(N_BINS_1D, dtype=float)
    sig_total = np.zeros(N_BINS_1D, dtype=float)
    bkg_w2 = np.zeros(N_BINS_1D, dtype=float)
    sig_w2 = np.zeros(N_BINS_1D, dtype=float)
    has_bkg = False
    has_sig = False

    for proc in BACKGROUNDS + [SIGNAL]:
        if trees is not None and proc in trees:
            tree = trees[proc]
            mask = masks[proc] if masks is not None and proc in masks else build_mask_from_selection(tree, selection)
            arr_ak = tree[var].array(library="ak")[mask]
            if len(arr_ak) == 0:
                continue

            try:
                evt_weights = np.asarray(tree["weight_xsec"].array(library="ak")[mask], dtype=float) * LUMINOSITY_PB
            except Exception:
                evt_weights = np.ones(len(arr_ak), dtype=float)

            try:
                # Broadcast per-event weights to match jagged/flat variable structure.
                arr_b, w_b = ak.broadcast_arrays(arr_ak, evt_weights)
                arr = np.asarray(ak.flatten(arr_b, axis=None), dtype=float)
                weights = np.asarray(ak.flatten(w_b, axis=None), dtype=float)
            except Exception:
                arr = numeric(arr_ak)
                weights = np.ones_like(arr)

            finite = np.isfinite(arr) & np.isfinite(weights)
            arr = arr[finite]
            weights = weights[finite]
            if arr.size == 0:
                continue

            mask_x = (arr >= xmin) & (arr <= xmax)
            arr = arr[mask_x]
            weights = weights[mask_x]
            if arr.size == 0 or len(arr) != len(weights):
                continue

            counts, _ = np.histogram(arr, bins=edges, weights=weights)
            counts_w2, _ = np.histogram(arr, bins=edges, weights=weights * weights)
            if counts.sum() <= 0:
                continue

            if proc == SIGNAL:
                sig_total += counts
                sig_w2 += counts_w2
                has_sig = True
            else:
                bkg_total += counts
                bkg_w2 += counts_w2
                has_bkg = True
            continue
        fp = files.get(proc)
        if not fp:
            continue
        with uproot.open(fp) as f:
            tree = f["events"]
            mask = build_mask_from_selection(tree, selection)
            arr_ak = tree[var].array(library="ak")[mask]
            if len(arr_ak) == 0:
                continue

            try:
                evt_weights = np.asarray(tree["weight_xsec"].array(library="ak")[mask], dtype=float) * LUMINOSITY_PB
            except Exception:
                evt_weights = np.ones(len(arr_ak), dtype=float)

            try:
                # Broadcast per-event weights to match jagged/flat variable structure.
                arr_b, w_b = ak.broadcast_arrays(arr_ak, evt_weights)
                arr = np.asarray(ak.flatten(arr_b, axis=None), dtype=float)
                weights = np.asarray(ak.flatten(w_b, axis=None), dtype=float)
            except Exception:
                arr = numeric(arr_ak)
                weights = np.ones_like(arr)

            finite = np.isfinite(arr) & np.isfinite(weights)
            arr = arr[finite]
            weights = weights[finite]
            if arr.size == 0:
                continue

            mask_x = (arr >= xmin) & (arr <= xmax)
            arr = arr[mask_x]
            weights = weights[mask_x]
            if arr.size == 0 or len(arr) != len(weights):
                continue

            counts, _ = np.histogram(arr, bins=edges, weights=weights)
            counts_w2, _ = np.histogram(arr, bins=edges, weights=weights * weights)
            if counts.sum() <= 0:
                continue

            if proc == SIGNAL:
                sig_total += counts
                sig_w2 += counts_w2
                has_sig = True
            else:
                bkg_total += counts
                bkg_w2 += counts_w2
                has_bkg = True

    if not has_bkg or bkg_total.sum() <= 0:
        print(f"[!] No background entries for '{var}' in channel '{channel}'.")
        return
    if not has_sig or sig_total.sum() <= 0:
        print(f"[!] No signal entries for '{var}' in channel '{channel}'.")
        return

    bkg_norm = bkg_total / bkg_total.sum()
    sig_norm = sig_total / sig_total.sum()
    ratio = np.divide(sig_norm, bkg_norm, out=np.zeros_like(sig_norm), where=bkg_norm > 0)
    sig_err = np.sqrt(sig_w2) / sig_total.sum()
    bkg_err = np.sqrt(bkg_w2) / bkg_total.sum()
    sig_rel = np.divide(sig_err, sig_norm, out=np.zeros_like(sig_err), where=sig_norm > 0)
    bkg_rel = np.divide(bkg_err, bkg_norm, out=np.zeros_like(bkg_err), where=bkg_norm > 0)
    ratio_err = ratio * np.sqrt(sig_rel * sig_rel + bkg_rel * bkg_rel)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    step_x = np.append(edges[:-1], edges[-1])
    ax.step(step_x, np.append(bkg_norm, bkg_norm[-1]), where="post", color="black", lw=2, label="Total background")
    ax.step(
        step_x,
        np.append(sig_norm, sig_norm[-1]),
        where="post",
        color=process_colours[SIGNAL],
        lw=2,
        linestyle="--",
        label=process_labels.get(SIGNAL, SIGNAL),
    )

    ymax = max(float(np.max(bkg_norm)), float(np.max(sig_norm)))
    ax.set_ylim(0, 1.5 * ymax if ymax > 0 else 1.0)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Normalised events", loc="top")
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    banner(ax, comment)
    ax.tick_params(labelbottom=False)

    ax_ratio.step(step_x, np.append(ratio, ratio[-1]), where="post", color="black", lw=1.5)
    valid = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio > 0)
    band_lo = np.where(valid, np.clip(ratio - ratio_err, 0, None), np.nan)
    band_hi = np.where(valid, ratio + ratio_err, np.nan)
    ax_ratio.fill_between(
        step_x,
        np.append(band_lo, band_lo[-1]),
        np.append(band_hi, band_hi[-1]),
        step="post",
        color="black",
        alpha=0.2,
        linewidth=0,
    )
    ax_ratio.axhline(1, color="gray", linestyle="--", lw=1)
    ax_ratio.set_ylabel("S/B", fontsize=10)
    ax_ratio.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax_ratio.set_xlim(edges[0], edges[-1])
    ax_ratio.tick_params(axis='both', direction='in', top=True, right=True)
    ax_ratio.set_ylim(0, 6.5)

    out = outdir / f"{channel}/normalised_total_background/{var}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[✓]", out)

def normalised_slice_plot(
    var: str,
    slicer: str,
    files: dict[str, Path],
    outdir: Path,
    channel: str,
    k3: float = 1.0,
    k4: float = 1.0,
    comment: str = "",
    n_slices: int = 4,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from config import XLIM_MAP

    n_slices = int(n_slices)
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")

    # ---------------------------
    # 1) Prepare global binning for var (consistent across all slices/procs)
    # ---------------------------
    if var in XLIM_MAP:
        xmin, xmax = XLIM_MAP[var]
        edges = np.linspace(xmin, xmax, N_BINS_1D + 1)
    else:
        vmins, vmaxs = [], []
        for proc in BACKGROUNDS + [SIGNAL]:
            fp = files.get(proc)
            if not fp:
                continue
            selection = SELECTION[channel]
            with uproot.open(fp) as f:
                tree = f["events"]
                mask = build_mask_from_selection(tree, selection)
                try:
                    arr_var = numeric(tree[var].array(library="ak")[mask])
                except Exception:
                    continue
                if arr_var.size:
                    vmins.append(float(np.min(arr_var)))
                    vmaxs.append(float(np.max(arr_var)))
        if not vmins:
            print(f"[!] No entries found for variable '{var}' after selection in channel '{channel}'.")
            return
        vmin, vmax = min(vmins), max(vmaxs)
        if vmin == vmax:
            eps = 1e-9 if vmin == 0.0 else 1e-6 * abs(vmin)
            edges = np.linspace(vmin - eps, vmax + eps, N_BINS_1D + 1)
        else:
            edges = np.linspace(vmin, vmax, N_BINS_1D + 1)

    # ---------------------------
    # 2) Plot one figure per process: quantiles are computed per process
    # ---------------------------
    slice_styles = ["-", "--", ":", "-."]
    slice_cmap = ['black', 'blue', 'green', 'brown']
    for proc in BACKGROUNDS + [SIGNAL]:
        fp = files.get(proc)
        if not fp:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        any_plotted = False
        selection = SELECTION[channel]
        with uproot.open(fp) as f:
            tree = f["events"]
            mask_base = build_mask_from_selection(tree, selection)

            try:
                arr_var_all = numeric(tree[var].array(library="ak")[mask_base])
                arr_slice_all = numeric(tree[slicer].array(library="ak")[mask_base])
            except Exception:
                plt.close(fig)
                continue

            if len(arr_var_all) != len(arr_slice_all):
                print(f"[!] Length mismatch for {proc}: {var} vs {slicer}.")
                plt.close(fig)
                continue

            if arr_slice_all.size == 0:
                print(f"[i] No slicer entries for {proc}; nothing to save.")
                plt.close(fig)
                continue

            # Compute slicer quantile edges per process.
            q_edges = np.quantile(arr_slice_all, np.linspace(0.0, 1.0, n_slices + 1))
            if np.unique(q_edges).size < (n_slices + 1):
                vmin, vmax = float(np.min(arr_slice_all)), float(np.max(arr_slice_all))
                if vmin == vmax:
                    eps = 1e-9 if vmin == 0.0 else 1e-6 * abs(vmin)
                    q_edges = np.linspace(vmin - eps, vmax + eps, n_slices + 1)
                else:
                    q_edges = np.linspace(vmin, vmax, n_slices + 1)
            print(f"[info] {proc} {slicer} quantile edges ({n_slices} slices): {q_edges}")

            for qi in range(n_slices):
                s_lo, s_hi = q_edges[qi], q_edges[qi + 1]
                right_inclusive = (qi == n_slices - 1)

                if right_inclusive:
                    mask_slice = (arr_slice_all >= s_lo) & (arr_slice_all <= s_hi)
                else:
                    mask_slice = (arr_slice_all >= s_lo) & (arr_slice_all < s_hi)

                arr = arr_var_all[mask_slice]

                # Respect optional hard x-limits
                if var in XLIM_MAP:
                    arr = arr[(arr >= xmin) & (arr <= xmax)]

                if arr.size == 0:
                    continue

                counts, _ = np.histogram(arr, bins=edges)
                if counts.sum() == 0:
                    continue

                norm_counts = counts / counts.sum()
                step_x = np.append(edges[:-1], edges[-1])
                step_y = np.append(norm_counts, norm_counts[-1])

                lbl = f"slice {qi+1}: [{q_edges[qi]:.3g}, {q_edges[qi+1]:.3g}{']' if right_inclusive else ')'}"

                ax.step(
                    step_x,
                    step_y,
                    where="post",
                    color=slice_cmap[qi],
                    linestyle=slice_styles[qi % len(slice_styles)],
                    lw=2,
                    label=lbl,
                )
                any_plotted = True

        if not any_plotted:
            plt.close(fig)
            print(f"[i] No non-empty histograms for {proc}; nothing to save.")
            continue

        # Axes cosmetics
        ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
        ax.set_ylabel("Normalised events", loc="top")
        ax.set_ylim(bottom=0, top=0.14)
        if var in XLIM_MAP:
            ax.set_xlim(*XLIM_MAP[var])
        if slicer in LABEL_MAP:
            slicer_label = LABEL_MAP[slicer]
        else:
            slicer_label = slicer
        
        ax.set_title(process_labels.get(proc, proc), loc="right", fontsize=11)
        ax.legend(frameon=False, loc="upper right", fontsize=8, title=slicer_label)
        banner(ax, comment)

        slice_tag = "quartiles" if n_slices == 4 else f"{n_slices}slices"
        out = outdir / f"{channel}/normalised/slices/{proc}/{var}__{slicer}_{slice_tag}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓]", out)

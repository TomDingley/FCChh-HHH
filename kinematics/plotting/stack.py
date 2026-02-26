import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uproot

from config import XLIM_MAP, N_BINS_1D, SIGNAL, BACKGROUNDS, LUMINOSITY_PB, SELECTION, HADHAD
from aesthetics import LABEL_MAP, process_labels, process_colours, banner, channel_colors, channel_labels
from tools2 import  numeric, get_weights, build_mask_from_selection

import matplotlib.gridspec as gridspec
    
def stack_plot_weight(var: str, files: dict[str, Path], outdir: Path, channel: str, comment: str = ""):
    with uproot.open(files[SIGNAL]) as f_sig:
        t_sig = f_sig["events"]
        selection = SELECTION[channel]
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

    for proc in BACKGROUNDS + [SIGNAL]:
        fp = files.get(proc)
        if not fp:
            continue

        selection = SELECTION[channel]

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
                    if counts_ratio.sum() > 0:
                        if proc == SIGNAL:
                            ratio_sig_counts = counts_ratio
                        else:
                            ratio_bkg_counts = counts_ratio if ratio_bkg_counts is None else ratio_bkg_counts + counts_ratio



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
        if ratio_sig_counts is not None and ratio_bkg_counts is not None:
            if ratio_sig_counts.sum() > 0 and ratio_bkg_counts.sum() > 0:
                sig_norm = ratio_sig_counts / ratio_sig_counts.sum()
                bkg_norm = ratio_bkg_counts / ratio_bkg_counts.sum()
                ratio = np.divide(sig_norm, bkg_norm, out=np.zeros_like(sig_norm), where=bkg_norm > 0)
                step_x = np.append(edges[:-1], edges[-1])
                step_y = np.append(ratio, ratio[-1])
                ax_ratio.step(step_x, step_y, where="post", color="black", lw=1.5)
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
                            k3: float = 1.0, k4: float = 1.0, comment: str = ""):
    overlay_plot(var, files, outdir, channel, normalise=True, k3=k3, k4=k4, comment=comment)


def raw_overlay_plot(var: str, files: dict[str, Path], outdir: Path, channel: str,
                     k3: float = 1.0, k4: float = 1.0, comment: str = ""):
    overlay_plot(var, files, outdir, channel, normalise=False, k3=k3, k4=k4, comment=comment)

def normalised_slice_plot(
    var: str,
    slicer: str,
    files: dict[str, Path],
    outdir: Path,
    channel: str,
    k3: float = 1.0,
    k4: float = 1.0,
    comment: str = "",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from config import XLIM_MAP

    # ---------------------------
    # 1) Collect global slicer values (post-selection) to define quartile edges
    # ---------------------------
    all_slice_vals = []

    for proc in BACKGROUNDS + [SIGNAL]:
        fp = files.get(proc)
        if not fp:
            continue
        selection = SELECTION[channel]
        with uproot.open(fp) as f:
            tree = f["events"]
            mask = build_mask_from_selection(tree, selection)
            try:
                arr_slice = numeric(tree[slicer].array(library="ak")[mask])
            except Exception:
                continue
            if arr_slice.size:
                all_slice_vals.append(arr_slice)

    if not all_slice_vals:
        print(f"[!] No entries found for slicer '{slicer}' after selection in channel '{channel}'.")
        return

    all_slice_vals = np.concatenate(all_slice_vals)
    q_edges = np.quantile(all_slice_vals, [0.0, 0.25, 0.5, 0.75, 1.0])

    # Fallback if quartiles collapse
    if np.unique(q_edges).size < 5:
        vmin, vmax = float(np.min(all_slice_vals)), float(np.max(all_slice_vals))
        if vmin == vmax:
            eps = 1e-9 if vmin == 0.0 else 1e-6 * abs(vmin)
            q_edges = np.array([vmin - 1.5*eps, vmin - 0.5*eps, vmin, vmin + 0.5*eps, vmin + 1.5*eps])
        else:
            q_edges = np.linspace(vmin, vmax, 5)

    print(f"[info] {slicer} quartile edges: {q_edges}")

    # ---------------------------
    # 2) Prepare global binning for var (consistent across all slices/procs)
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
    # 3) Plot one figure per process: linestyles/colors = slices
    # ---------------------------
    slice_styles = ["-", "--", ":", "-."]  # q1..q4
    slice_colors = ["Black", "Blue", "green", "pink"]
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

            for qi in range(4):
                s_lo, s_hi = q_edges[qi], q_edges[qi + 1]
                right_inclusive = (qi == 3)

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

                lbl = f"q{qi+1}: [{q_edges[qi]:.3g}, {q_edges[qi+1]:.3g}{']' if right_inclusive else ')'}"

                ax.step(
                    step_x,
                    step_y,
                    where="post",
                    color=slice_colors[qi],
                    linestyle=slice_styles[qi],
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
        ax.set_ylim(bottom=0, top=0.3)
        if var in XLIM_MAP:
            ax.set_xlim(*XLIM_MAP[var])

        ax.set_title(process_labels.get(proc, proc), loc="right", fontsize=11)
        ax.legend(frameon=False, loc="upper right", fontsize=8, title=slicer)
        banner(ax, comment)

        out = outdir / f"{channel}/normalised/slices/{proc}/{var}__{slicer}_quartiles.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[✓]", out)

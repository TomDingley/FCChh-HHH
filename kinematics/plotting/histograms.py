import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot
from pathlib import Path

from config import XLIM_MAP, N_BINS_1D, OVERLAY_PAIRS, LUMINOSITY_PB
from aesthetics import LABEL_MAP, banner, process_labels, banner_heatmaps
from tools import numeric


def save_1d_hist(ax, outpath, comment):
    ax.set_ylabel("Events / bin", loc="top")
    banner_heatmaps(ax)
    ax.figure.savefig(outpath, bbox_inches="tight")
    
    plt.close(ax.figure)
    print("[✓]", outpath)


def plot_1d_histograms(proc, tree, mask, outdir, comment, channel):
    outdir.mkdir(parents=True, exist_ok=True)
    comment_chan = channel + f"\n Channel: {channel}"
    weight = tree["weight_xsec"].array(library="ak")[mask] * LUMINOSITY_PB
    for leaf in tree.keys():
        if "/" in leaf:
            continue
        arr = numeric(tree[leaf].array(library="ak")[mask])
        if arr.size == 0:
            continue
        counts, edges = np.histogram(arr, bins=N_BINS_1D, weights=weight)
        counts = ak.to_numpy(counts).astype(float)
        ax = plt.subplots(figsize=(6, 5))[1]
        x = np.append(edges[:-1], edges[-1])
        ax.step(x, np.append(counts, counts[-1]), where="post", color="black", label=process_labels.get(proc, proc))
        ax.set_xlabel(LABEL_MAP.get(leaf, leaf), loc="right")
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        ax.set_ylim(0, max(counts) * 1.5)
        save_1d_hist(ax, outdir / f"{leaf}_{channel}.pdf", comment_chan)

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
def overlay_histogram(tree, mask, proc, outdir, comment, channel, groups=None):
    """
    Overlay normalised 1D histograms for groups of variables.
    - groups: iterable of iterables, e.g. [("varA","varB"), ("x","y","z"), ...]
              If None, uses global OVERLAY_PAIRS (or OVERLAY_GROUPS).
    Adds a special-case for (pT_truth_hbb1, pT_truth_hbb2, pT_truth_htautau):
      counts events with max(pT_truth_hbb1, pT_truth_hbb2) > 250 GeV.
    """
    import numpy as np
    import awkward as ak
    import matplotlib.pyplot as plt
    from itertools import cycle

    comment_chan = comment + f"\nChannel: {channel}"
    outdir.mkdir(parents=True, exist_ok=True)

    var_groups = groups if groups is not None else OVERLAY_PAIRS

    for vars_in_group in var_groups:
        vars_in_group = tuple(vars_in_group)
        if len(vars_in_group) < 2:
            print(f"Skipping group {vars_in_group}: need at least 2 variables.")
            continue

        # Check presence
        missing = [v for v in vars_in_group if v not in tree.keys()]
        if missing:
            print(f"Skipping group {vars_in_group}. Missing in tree: {missing}")
            continue

        # Pull arrays (event-aligned) and clean
        arrays = {}
        for v in vars_in_group:
            arr = ak.to_numpy(tree[v].array(library="ak")[mask])
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                print(f"Skipping group {vars_in_group}: '{v}' has empty array after cleaning.")
                arrays = None
                break
            arrays[v] = arr
        if arrays is None:
            continue

        # --- shared binning from combined x-lims across the group ---
        mins, maxs = [], []
        for v, arr in arrays.items():
            xmin, xmax = XLIM_MAP.get(v, (arr.min(), arr.max()))
            mins.append(xmin); maxs.append(xmax)
        xlo, xhi = float(min(mins)), float(max(maxs))
        if (not np.isfinite(xlo)) or (not np.isfinite(xhi)) or xhi <= xlo:
            all_vals = np.concatenate(list(arrays.values()))
            xlo, xhi = np.nanmin(all_vals), np.nanmax(all_vals)
            if (not np.isfinite(xlo)) or (not np.isfinite(xhi)) or xhi <= xlo:
                print(f"Skipping group {vars_in_group}: invalid x-limits.")
                continue

        edges  = np.linspace(xlo, xhi, N_BINS_1D + 1)
        step_x = np.append(edges[:-1], edges[-1])  # for step plots

        # --- plot ---
        fig, ax = plt.subplots(figsize=(6, 5))
        ymax = 0.0
        styles = cycle(["-", "--", "-.", ":", "--"])
        colors = cycle(["black", "blue", "green", "gray", "brown"])
        

        for v in vars_in_group:
            counts, _ = np.histogram(arrays[v], bins=edges)
            label = LABEL_MAP.get(v, v).replace(" [GeV]", "")
            
            integral = counts.sum()
            if integral <= 0:
                print(f"Skipping '{v}' in group {vars_in_group}: zero integral after binning.")
                continue
            counts_norm = counts / integral
            ymax = max(ymax, counts_norm.max())
            step_y = np.append(counts_norm, counts_norm[-1])
            ax.step(step_x, step_y, where="post", linestyle=next(styles), label=label, color=next(colors))

        if not ax.get_legend_handles_labels()[0]:
            plt.close(fig)
            continue

        ax.legend(loc="upper right", frameon=False, fontsize=12)
        xlabel = LABEL_MAP.get(v, v)
        print(xlabel)
        if "m_{H_1" in xlabel:
            print("ere we are", xlabel)
            if "Reco-matched" in xlabel:
                xlabel = r"Higgs mass [GeV]"
                print("ere we are again", xlabel)
            else:
                xlabel = r"Leading Higgs candidate mass [GeV]"
                
        if "m_{H_2" in xlabel:
            print("ere we are", xlabel)
            if "Reco-match" in xlabel:
                xlabel = r"Higgs candidate mass [GeV]"
                print("ere we are again", xlabel)
            else:
                xlabel = r"Sub-leading Higgs candidate mass [GeV]"
        
        #if "Truth " in xlabel:
        #    xlabel = r"$m_{\tau\tau}^{\text{vis}}$ [GeV]"
            
        ax.set_xlabel(xlabel, loc="right")
        ax.set_ylabel("Normalised counts")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

        # ---------- SPECIAL CASE: count events with max(pT_b1, pT_b2) > 250 GeV ----------
        # Trigger if both b1/b2 are present in this group (order-insensitive).
        names = set(vars_in_group)
        pt_threshold = 312.5
        special_vars_hbb = {"pT_truth_hbb1", "pT_truth_hbb2"}
        y = 0.95


        if special_vars_hbb.issubset(names):
            # Event-aligned arrays
            b1 = ak.to_numpy(tree["pT_truth_hbb1"].array(library="ak")[mask])
            b2 = ak.to_numpy(tree["pT_truth_hbb2"].array(library="ak")[mask])

            b1_finite = np.isfinite(b1)
            b2_finite = np.isfinite(b2)
            valid_any  = b1_finite | b2_finite
            valid_both = b1_finite & b2_finite
            b1_safe = np.where(b1_finite, b1, -np.inf)
            b2_safe = np.where(b2_finite, b2, -np.inf)

            n_tot_any  = int(valid_any.sum())
            n_tot_both = int(valid_both.sum())

            # Hbb-only stats
            n_pass_1 = int(((np.maximum(b1_safe, b2_safe) > pt_threshold) & valid_any).sum())
            n_pass_2 = int(((b1_safe > pt_threshold) & (b2_safe > pt_threshold) & valid_both).sum())

            msg1 = rf"Events with ≥1 Hbb > {pt_threshold:.1f} GeV: {n_pass_1}/{n_tot_any} ({(n_pass_1/n_tot_any if n_tot_any>0 else 0):.1%})"
            msg2 = rf"Events with 2 Hbb > {pt_threshold:.1f} GeV: {n_pass_2}/{n_tot_both} ({(n_pass_2/n_tot_both if n_tot_both>0 else 0):.1%})"
            print("[info]", msg1); print("[info]", msg2)
            comment_this += "\n" + msg1 + "\n" + msg2

            msg3 = msg4 = msg5 = None

            # Optional Hττ-only stat
            if "pT_truth_htautau" in names:
                t = ak.to_numpy(tree["pT_truth_htautau"].array(library="ak")[mask])
                t_finite = np.isfinite(t)
                n_tot_t  = int(t_finite.sum())
                n_pass_t = int(((t > pt_threshold) & t_finite).sum())
                msg3 = rf"Events with Hττ > {pt_threshold:.1f} GeV: {n_pass_t}/{n_tot_t} ({(n_pass_t/n_tot_t if n_tot_t>0 else 0):.1%})"
                print("[info]", msg3)
                comment_this += "\n" + msg3

                # ---- Combined boosted cases ----
                # ≥1 Hbb boosted AND Hττ boosted (same event)
                valid_1tau = valid_any & t_finite
                n_tot_1tau = int(valid_1tau.sum())
                pass_1tau = (np.maximum(b1_safe, b2_safe) > pt_threshold) & (t > pt_threshold) & valid_1tau
                n_pass_1tau = int(pass_1tau.sum())
                msg4 = rf"Events with (≥1 Hbb) & (Hττ) > {pt_threshold:.1f} GeV: {n_pass_1tau}/{n_tot_1tau} ({(n_pass_1tau/n_tot_1tau if n_tot_1tau>0 else 0):.1%})"
                print("[info]", msg4)
                comment_this += "\n" + msg4

                # 2 Hbb boosted AND Hττ boosted (same event)
                valid_2tau = valid_both & t_finite
                n_tot_2tau = int(valid_2tau.sum())
                pass_2tau = (b1_safe > pt_threshold) & (b2_safe > pt_threshold) & (t > pt_threshold) & valid_2tau
                n_pass_2tau = int(pass_2tau.sum())
                msg5 = rf"Events with (2 Hbb) & (Hττ) > {pt_threshold:.1f} GeV: {n_pass_2tau}/{n_tot_2tau} ({(n_pass_2tau/n_tot_2tau if n_tot_2tau>0 else 0):.1%})"
                print("[info]", msg5)
                comment_this += "\n" + msg5

            # Put lines on the plot (stacked near top-left)
            for m in (msg1, msg2, msg3, msg4, msg5):
                if m is None:
                    continue
                ax.text(0.04, y, m, transform=ax.transAxes, va="top", ha="left", fontsize=9)
                y -= 0.06
        y = 0.75

        # ---- ΔR-based "boosted/collimated" counts (ΔR < 0.8) ----
        msg3_dr = msg4_dr = msg5_dr = None
        dr_threshold = 0.8

        if "dR_truth_tautau" in names:
            pass_any = None
            valid_any = None

            def _ensure_arrays(n):
                nonlocal pass_any, valid_any
                if pass_any is None:
                    pass_any  = np.zeros(n, dtype=bool)
                    valid_any = np.zeros(n, dtype=bool)
            # Check each branch; accumulate "valid" (has finite ΔR) and "pass" (ΔR<thr) per event
            for name in ("dR_truth_tautau", "dR_truth_bb1", "dR_truth_bb2"):
                if name in names:
                    arr = ak.to_numpy(tree[name].array(library="ak")[mask])
                    _ensure_arrays(len(arr))
                    finite = np.isfinite(arr)
                    valid_any |= finite
                    pass_any  |= (arr < dr_threshold) & finite

            # Summaries
            if pass_any is not None:
                n_tot_any  = int(valid_any.sum())   # events where at least one ΔR is finite
                n_pass_any = int(pass_any.sum())    # events where any ΔR < threshold
                msg_any_dr = (
                    rf"Events with ANY Higgs collimated (ΔR < {dr_threshold:.2f}): "
                    rf"{n_pass_any}/{n_tot_any} "
                    rf"({(n_pass_any/n_tot_any if n_tot_any>0 else 0):.1%})"
                )
                print("[info]", msg_any_dr)
                comment_this += "\n" + msg_any_dr

            # ---- Add to plot text stack (after your other msgs) ----
            for m in (msg3_dr, msg4_dr, msg5_dr, msg_any_dr):
                if m is None:
                    continue
                ax.text(0.04, y, m, transform=ax.transAxes, va="top", ha="left", fontsize=9)
                y -= 0.06
        # ----------------------------------------
        comment_this = comment_chan
        # Save
        vars_tag = "_".join(vars_in_group)
        outpath = outdir / f"overlay_{vars_tag}_{channel}_normalised.pdf"
        save_1d_hist(ax, outpath, comment_this)
        plt.close(fig)
        print("[✓]", outpath)

        

def compare_histogram(tree_ref, tree_cmp, mask, var, outdir, comment, proc_ref="Reference", proc_cmp="Compare", channel="Combined"):
    """
    Compare distributions of a single variable between two trees.

    """
    comment_chan = channel + f"\n Channel: {channel}"
    outdir.mkdir(parents=True, exist_ok=True)

    # weights
    w_ref = ak.to_numpy(tree_ref["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB
    w_cmp = ak.to_numpy(tree_cmp["weight_xsec"].array(library="ak")[mask]) * LUMINOSITY_PB

    if var not in tree_ref.keys() or var not in tree_cmp.keys():
        print(f"Variable {var} not found in both trees.")
        return

    x_ref = ak.to_numpy(tree_ref[var].array(library="ak")[mask])
    x_cmp = ak.to_numpy(tree_cmp[var].array(library="ak")[mask])
    if x_ref.size == 0 or x_cmp.size == 0:
        print(f"Skipping {var}: empty arrays after masking.")
        return

    # binning
    xlim = XLIM_MAP.get(var, (min(x_ref.min(), x_cmp.min()), max(x_ref.max(), x_cmp.max())))
    edges = np.linspace(xlim[0], xlim[1], N_BINS_1D + 1)
    x = np.append(edges[:-1], edges[-1])  # step edges

    # --- absolute counts ---
    counts_ref, _ = np.histogram(x_ref, bins=edges, weights=w_ref)
    counts_cmp, _ = np.histogram(x_cmp, bins=edges, weights=w_cmp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(x, np.append(counts_ref, counts_ref[-1]), where="post", color="black", label=proc_ref)
    ax.step(x, np.append(counts_cmp, counts_cmp[-1]), where="post", color="blue", label=proc_cmp)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax.set_xlim(xlim)
    ax.set_ylim(0, max(max(counts_ref), max(counts_cmp)) * 1.5)
    save_1d_hist(ax, outdir / f"compare_{var}_{channel}.pdf", comment_chan)

    # --- normalised ---
    int_ref = counts_ref.sum()
    int_cmp = counts_cmp.sum()
    counts_ref_norm = counts_ref / int_ref if int_ref > 0 else counts_ref
    counts_cmp_norm = counts_cmp / int_cmp if int_cmp > 0 else counts_cmp

    fig, ax_norm = plt.subplots(figsize=(6, 5))
    ax_norm.step(x, np.append(counts_ref_norm, counts_ref_norm[-1]), where="post", color="black", label=proc_ref)
    ax_norm.step(x, np.append(counts_cmp_norm, counts_cmp_norm[-1]), where="post", color="green", label=proc_cmp)
    ax_norm.legend(loc="upper right", frameon=False)
    ax_norm.set_xlabel(LABEL_MAP.get(var, var), loc="right")
    ax_norm.set_xlim(xlim)
    ax_norm.set_ylabel("Normalised counts")
    ax_norm.set_ylim(0, max(max(counts_ref_norm), max(counts_cmp_norm)) * 1.5)
    save_1d_hist(ax_norm, outdir / f"compare_{var}_{channel}_norm.pdf", comment_chan)

    print(f"[✓] Compared histograms for {var}")
from pathlib import Path
import argparse
from contextlib import ExitStack
import uproot
import numpy as np
from config import processes, SIGNAL, BACKGROUNDS, SELECTION, SKIP_VARS, LUMINOSITY_PB
from tools import numeric, build_mask_from_selection
from cutflow import write_cutflow_weighted_summary, write_region_yield_summary


from aesthetics import LABEL_MAP, channel_labels
from plotting.histograms import plot_1d_histograms, overlay_histogram, compare_histogram
from plotting.shape_cutflow import plot_shape_cutflow
from plotting.heatmaps import plot_heatmaps, plot_hist2d_heatmaps
from plotting.stack import normalised_overlay_plot, stack_plot_weight, normalised_overlay_plot_chan, normalised_slice_plot, raw_overlay_plot, normalised_total_background_plot
from plotting.fit_shapes import fit_signal_mass_shape
from plotting.signal_reweighter import compare_signal_reweight_points, heatmap_signal_reweight_efficiency, heatmap_signal_reweight_efficiency_LR, heatmap_signal_reweight_mean,  heatmap_signal_pairing_mean, heatmap_signal_reweight_mean_with_slices, heatmap_signal_reweight_efficiency_presel, heatmap_signal_reweight_xsm_after_selection
from fitting.fit import scan_k3k4_limits, plot_k3k4_limit_contours, plot_signal_yield_morphing_grid, plot_k3k4_limit_contours_comparison

def process_file(proc: str, path: Path, outdir: Path, comment: str, compDir: Path, vars_to_stack: list[str], channel: str, shape_var: str | None = None, shape_outdir: Path | None = None, shape_normalize: bool = True):
    with uproot.open(path) as f:
        tree = f["events"]
        
        mask = build_mask_from_selection(tree, SELECTION[channel])

        #plot_1d_histograms(proc, tree, mask, outdir, comment, channel)

        #if overlay_histogram:
        #    overlay_histogram(tree, mask, proc, outdir, comment, channel)
        #if plot_heatmaps:
        #    plot_heatmaps(tree, mask, proc, outdir, comment, channel)

        plot_hist2d_heatmaps(
            tree=tree,
            mask=mask,      # e.g. (tree["channel"] == 2)
            proc=proc,
            outdir=outdir,
            comment="",
            channel=channel,
            use_weights=False,
            weight_field="weight_xsec",
            lumi_scale=LUMINOSITY_PB,
            bins=50,
            cmap="RdBu_r",
            color_by="weight",        # or "weight"
            norm="lin"                 # or "linear"
        )

        
        shape_dir = (shape_outdir or outdir / "shape_cutflow") / channel
        print("Shape variable used: ", shape_var)
        
        plot_shape_cutflow(
            root_path=path,
            var=shape_var,
            outdir=shape_dir,
            channel=channel,
            selection=SELECTION[channel],
            tree_name="events",
                weight_branch=None,
                lumi_scale=None,
            bins=10,
            normalize=True,
        )
        
        if compDir:
            comp_path = Path(compDir) / f"{proc}.root"
            if comp_path.is_file():
                with uproot.open(comp_path) as f_cmp:
                    tree_cmp = f_cmp["events"]
                    for var in vars_to_stack:  # pick your variable list
                        compare_histogram(tree, tree_cmp, mask, var, outdir, comment,
                                        proc_ref=proc, proc_cmp=f"{proc} (compare)", channel=channel)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default="/data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_ttbb_fullstat/merged/weighted")
    ap.add_argument("--outdir", default="84TeV/test_run")
    ap.add_argument("--comment", default="")
    ap.add_argument("--channel", default="HadHad", choices=["LepHad", "HadHad"])
    ap.add_argument("--test", action='store_true')
    ap.add_argument("--signal", default="mgp8_pp_hhh_84TeV")
    ap.add_argument("--compare_dir", default="")
    ap.add_argument("--shape-cutflow", action="store_true")
    ap.add_argument("--shape-var", default="m_hhh_vis")
    ap.add_argument("--n-slices", type=int, default=4, help="Number of quantile slices for normalised slice plots.")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    out_base = Path(args.outdir).expanduser().resolve()
    comment = args.comment
    isTest = args.test
    compDir = args.compare_dir
    shape_var = args.shape_var.strip() if args.shape_var else ""
    shape_outdir = Path(args.outdir + "/" + "shape")
    out_base.mkdir(parents=True, exist_ok=True)
    fit_dir = out_base / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running channel: {args.channel}")
    
    channels = ["HadHad"]
    signal = args.signal
    if isTest:
        files = {"mgp8_pp_hhh_84TeV": "/data/atlas/users/dingleyt/FCChh/FCCAnalyses/output.root"}
        signal = list(files.keys())[0]
    else:
        files = {p: root / f"{p}.root" for p in processes if (root / f"{p}.root").is_file()}
        print(files)

    if (signal not in files) and (not isTest):
        raise FileNotFoundError(f"Signal file '{signal}.root' not found in {root}")
        
    with uproot.open(files[signal]) as f_sig:
        leaves = [k for k in f_sig["events"].keys() if k not in SKIP_VARS]
    vars_to_stack = [v for v in leaves if v in LABEL_MAP]
    
    # also make some slice plots, so var_1 in n_slices of var_2
    vars_to_slice = [
        ("m_h1", "pT_h1"),
        ("weighted_MMC_para_perp_vispTcal", "pT_tau1_tlv_LH"),
        ("weighted_MMC_para_perp_vispTcal", "MET"),
        ("weighted_MMC_para_perp_vispTcal", "m_tautau_vis_OS"),
        ("m_tautau_vis_OS", "weighted_MMC_para_perp_vispTcal"),
        ("weighted_MMC_para_perp_vispTcal", "pT_tau2_tlv_LH"),
        ("m_tautau_vis_OS", "metsig_derived"),
        ("m_tautau_vis_OS", "dR_tautau"),
        ("m_tautau_vis_OS", "MET"),
        ("m_tautau_vis_OS", "pT_tau2_tlv_LH"),
        ("m_tautau_vis_OS", "pT_tau1_tlv_LH"),
        ("weighted_MMC_para_perp_vispTcal", "dR_tautau"),
        ("m_hhh_vis", "mlp_score"),
        ("dR_tautau", "mlp_score"),
        ("m_hhh_truth", "mlp_score"),
        ("m_hhh_vis", "dR_tautau"),
        ("m_hhh_vis", "weighted_MMC_para_perp_vispTcal"),
        ("m_hhh_vis", "m_tautau_vis_OS"),
        ("m_hhh_vis", "m_h1"),
        ("m_hhh_vis", "m_h2"),
        ("weighted_MMC_para_perp_vispTcal", "m_hhh_vis")

    ]
    
    for chan in channels:
        chan_dir = out_base / f"{chan}"
        chan_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing channel: {chan}")
        
        comment = f"{channel_labels[chan]}"
        
        # make a summary cutflow of all processes
        for proc, fpath in files.items():
            print("[*]", proc)
            
            process_file(proc, fpath, out_base / proc, comment, compDir, vars_to_stack, chan, shape_var, shape_outdir, False)
            
        doPlots = True
        doWeight = True
        if chan != "Total":

            # write weighted cutflow for process
            csv_path, tex_path, pdf_path = write_cutflow_weighted_summary(
                files,
                outdir=out_base / chan,
                channel=chan,                  
                mode="chain",                 
                tree_name="events",
                weight_branch="weight_xsec",
                lumi=None,                     
                signal_sub=None,          
                ttbb_sub="ttbb",
                rel_sys_b=0.0,
                process_label_map=None,    
                step_size="200 MB",            
            )
        if doPlots:
            
            selection = SELECTION[chan]
            #write_cutflow_weighted_summary(files, out_base, chan)
                
            k3_points = np.linspace(-2.5, 6, 100)
            k4_points = np.linspace(-15, 30, 100)
            xsm_maps = True
            if xsm_maps:
                # make a contour map across the k3k4 space
                heatmap_signal_reweight_xsm_after_selection(
                    files=files,
                    outdir=out_base,
                    k3_grid=k3_points,
                    k4_grid=k4_points,
                    comment=comment,
                    channel=chan,
                    signal_key=signal,
                    cmap="viridis",
                    contour_levels=[1.0, 1.5, 2.0, 5.0],
                )
            maps = False
            if maps:
                heatmap_signal_reweight_efficiency_presel(
                    files=files,
                    outdir=out_base,
                    k3_grid=k3_points,
                    k4_grid=k4_points,
                    comment=comment,
                    channel=chan
                )
                heatmap_signal_pairing_mean(
                        files=files,
                            outdir=out_base,
                            k3_grid=k3_points,
                            k4_grid=k4_points,
                            comment=comment,
                            channel=chan,
                            type="dRminmax"
                )
                heatmap_signal_pairing_mean(
                        files=files,
                            outdir=out_base,
                            k3_grid=k3_points,
                            k4_grid=k4_points,
                            comment=comment,
                            channel=chan,
                            type="absmass"
                )
                heatmap_signal_pairing_mean(
                        files=files,
                            outdir=out_base,
                            k3_grid=k3_points,
                            k4_grid=k4_points,
                            comment=comment,
                            channel=chan,
                            type="squaremass"
                )
                
                heatmap_signal_reweight_mean(
                    var="Higgs_HT_truth",
                    files = files,
                    outdir = out_base,
                    k3_grid=k3_points,
                    k4_grid=k4_points,
                    comment=comment,
                    channel=chan
                )
                
                heatmap_signal_reweight_mean_with_slices(
                    var="Higgs_HT_truth",
                    files = files,
                    outdir = out_base,
                    k3_grid=k3_points,
                    k4_grid=k4_points,
                    comment=comment,
                    channel=chan
                )
                
                heatmap_signal_reweight_efficiency(
                    files=files,
                    outdir=out_base,
                    k3_grid=k3_points,
                    k4_grid=k4_points,
                    comment=comment,
                    channel=chan
                )

                heatmap_signal_reweight_efficiency_LR(
                            files=files,
                            outdir=out_base,
                            k3_grid=k3_points,
                            k4_grid=k4_points,
                            comment=comment,
                            channel=chan
                        )
            doSlice = False
            if doSlice:    
                for var, slicer in vars_to_slice:
                    normalised_slice_plot(
                        var,
                        slicer,
                        files,
                        out_base,
                        chan,
                        k3=1,
                        k4=1,
                        comment=comment,
                        n_slices=args.n_slices,
                    )
            with ExitStack() as stack:
                open_trees = {}
                selection_masks = {}
                for proc, fpath in files.items():
                    f = stack.enter_context(uproot.open(fpath))
                    tree = f["events"]
                    open_trees[proc] = tree
                    selection_masks[proc] = build_mask_from_selection(tree, selection)

                for var in vars_to_stack:
                    stack_plot_weight(var, files, out_base, chan, comment, trees=open_trees, masks=selection_masks)
                    #raw_overlay_plot(var, files, out_base, chan, k3 = 1, k4 = 1, comment=comment)
                    normalised_overlay_plot(
                        var,
                        files,
                        out_base,
                        chan,
                        k3=1,
                        k4=1,
                        comment=comment,
                        trees=open_trees,
                        masks=selection_masks,
                    )
                    normalised_total_background_plot(
                        var,
                        files,
                        out_base,
                        chan,
                        k3=1,
                        k4=1,
                        comment=comment,
                        trees=open_trees,
                        masks=selection_masks,
                    )
                    if doWeight:
                        compare_signal_reweight_points(
                            var=var,
                            files=files,
                            outdir=out_base,
                            k3k4_points=[(1, 1), (5, 1), (1, 20)],
                            comment=comment,
                            channel=chan
                        )
                    
        fit_vars = ["m_tautau_vis_OS","weighted_MMC_para_perp_vispTcal", "metRatio_mode_vispTcal"]
        
        # do asimov limits for a given variable
        #scan_k3k4_limits("m_hhh_vis_LR", files, out_base, k3_range=(-7.5, 10), k4_range=(-75, 75), nsteps=10, channel=chan)
        #plot_k3k4_limit_contours(out_base / f"fit/{chan}_significance_scan_m_hhh_viss.npz", out_base, chan)
        #scan_k3k4_limits("mlp_score", files, out_base, k3_range=(-5, 5), k4_range=(-20, 40), nsteps=40, channel=chan)
        #plot_k3k4_limit_contours(out_base / "fit/limit_scan_NNScore_OOF.npz", out_base, chan)
        
        k3_vals = np.linspace(-15.0, 20.0, 21)
        k4_vals = np.linspace(-300.0, 200.0, 21)
        #plot_signal_yield_morphing_grid(files, out_base, k3_vals, k4_vals)
        #for var in fit_vars:
        #    fit_signal_mass_shape(files, out_base, var, comment=comment, channel=chan)
    
    # comparing channels
    plot_k3k4_limit_contours_comparison(out_base)
    # compare kinematics across channels
    for proc, fpath in files.items():
        one_file = {proc: fpath}
        #for var in vars_to_stack:
        #  normalised_overlay_plot_chan(var, one_file, out_base / proc, channels, k3=1, k4=1, comment=comment)

    
    # getting xSM
    # At the end of cli(), after the for-chan loop finishes:
    summary_out = out_base / "summary"
    summary_out.mkdir(parents=True, exist_ok=True)

    write_region_yield_summary(
        files=files,
        outdir=summary_out,
        channels=channels,            # e.g. ["HadHad_resolved", "HadHad_1BB", ...]
        tree_name="events",
        weight_branch="weight_xsec",
        lumi=None,                    # will pick up config.LUMINOSITY_PB
        signal_sub=args.signal,       # or None to use config.SIGNAL
        process_label_map=None,       # will use aesthetics.process_labels if available
        step_size="200 MB",
    )

if __name__ == "__main__":
    cli()

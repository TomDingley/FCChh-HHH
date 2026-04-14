#!/usr/bin/env python3
import os
import argparse
from array import array
import ROOT

# code inspired entirely on ATLAS HHH fitting input prep


ROOT.ROOT.EnableImplicitMT()  # optional

# --- Defaults (override via CLI) ---
IN_DIR_DEFAULT   = "/data/atlas/users/dingleyt/FCChh/hhh/ML/clean/scored_220925_kfold2/hadhad"
OUT_DIR_DEFAULT  = "/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_lephad_hadhad_220925_kfold2_mhhh"   # TRExFitter reads these via HistoFile/HistoName
TREE_NAME        = "events"
VAR_NAME_DEFAULT = "m_hhh_vis"       # histogram variable (now m_hhh_vis)
EDGES_DEFAULT    = "200,650,1100,1550,2000"      # 4-bin mHHH fit
NN_BRANCH_DEF    = "mlp_score"
NN_CUT_DEF       = 0.975
WEIGHT_BRANCH    = "weight_xsec"    

# Sample -> filename mapping 
BKG_SAMPLES = {
    "ttbb": "mgp8_pp_ttbb_4f_84TeV.root",
    "tth" : "mgp8_pp_tth_5f_84TeV.root",
    "ttz" : "mgp8_pp_ttz_5f_84TeV.root",
    "tttt": "mgp8_pp_tttt_5f_84TeV.root",
    "zzz" : "mgp8_pp_zzz_5f_84TeV.root",
    "hh": "pwp8_pp_hh_k3_1_k4_1_84TeV.root"
}

def parse_edges(edges_str: str):
    try:
        edges = [float(x) for x in edges_str.split(",")]
    except Exception as e:
        raise ValueError(f"Could not parse --edges '{edges_str}': {e}")
    if len(edges) < 2:
        raise ValueError("Need at least two edges to define a histogram.")
    if any(edges[i] >= edges[i+1] for i in range(len(edges)-1)):
        raise ValueError("Edges must be strictly increasing.")
    return edges

def make_hist_for_sample(sample_key, infile, outdir, lumi_pb, extra_weight,
                         suffix, tree_name, var_name, edges, nn_branch, nn_cut):
    """Create a ROOT file with a single TH1D named 'hist' scaled to lumi."""
    os.makedirs(outdir, exist_ok=True)

    df = ROOT.RDataFrame(tree_name, infile)
    cols = set(df.GetColumnNames())

    # Sanity checks
    for needed in (WEIGHT_BRANCH, var_name, nn_branch):
        if needed not in cols:
            raise RuntimeError(f"[{sample_key}] Missing branch '{needed}' in {infile}")

    # Define final per-event weight
    if extra_weight:
        df = df.Define("w", f"({WEIGHT_BRANCH}) * ({lumi_pb}) * ({extra_weight})")
    else:
        df = df.Define("w", f"({WEIGHT_BRANCH}) * ({lumi_pb})")

    # Apply NN cut and keep only events within histogram range
    low, high = edges[0], edges[-1]
    df_sel = (df
              .Filter(f"{nn_branch} >= {nn_cut}")
              .Filter(f"{var_name} >= {low} && {var_name} < {high}"))

    # Histogram model (variable binning)
    edges_arr = array('d', edges)
    hmodel = ROOT.RDF.TH1DModel("hist", f"{var_name};{var_name};Events", len(edges)-1, edges_arr)
    h = df_sel.Histo1D(hmodel, var_name, "w")

    # Write out
    outpath = os.path.join(outdir, f"hist_{sample_key}_{suffix}.root" if suffix else f"hist_{sample_key}.root")
    f = ROOT.TFile(outpath, "RECREATE")
    h.Write()   # writes as 'hist'
    f.Close()
    print(f"[OK] {sample_key:5s}  -> {outpath}")

def main():
    ap = argparse.ArgumentParser(description="Make background histograms in m_hhh_vis with NN cut, scaled to lumi.")
    ap.add_argument("--in-dir",      default=IN_DIR_DEFAULT,  help="Input directory with ROOT ntuples")
    ap.add_argument("--out-dir",     default=OUT_DIR_DEFAULT, help="Output directory for histogram ROOT files")
    ap.add_argument("--lumi-ab",     type=float, default=30.0, help="Integrated luminosity in ab^-1 (default: 30)")
    ap.add_argument("--extra-weight", default="", help="Optional extra weight expression (e.g. 'sf_btag*sf_trig')")
    ap.add_argument("--suffix",       default="", help="Suffix for output ROOT files (e.g. channel tag)")

    ap.add_argument("--tree",        default=TREE_NAME, help=f"TTree name (default: {TREE_NAME})")
    ap.add_argument("--var-name",    default=VAR_NAME_DEFAULT, help=f"Variable to histogram (default: {VAR_NAME_DEFAULT})")
    ap.add_argument("--edges",       default=EDGES_DEFAULT, help=f"Comma-separated bin edges (default: '{EDGES_DEFAULT}')")
    ap.add_argument("--nn-branch",   default=NN_BRANCH_DEF, help=f"NN score branch (default: {NN_BRANCH_DEF})")
    ap.add_argument("--nn-cut",      type=float, default=NN_CUT_DEF, help=f"NN cut (>=) (default: {NN_CUT_DEF})")

    args = ap.parse_args()
    lumi_pb = args.lumi_ab * 1e6  # ab^-1 -> pb^-1
    edges = parse_edges(args.edges)

    print(f"Input dir : {args.in_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"Lumi      : {args.lumi_ab} ab^-1  ({lumi_pb:.3g} pb^-1)")
    print(f"Var/edges : {args.var_name} with edges {edges}")
    print(f"NN cut    : {args.nn_branch} >= {args.nn_cut}")
    if args.extra_weight:
        print(f"Extra wgt : {args.extra_weight}")

    for skey, fname in BKG_SAMPLES.items():
        inpath = os.path.join(args.in_dir, fname)
        if not os.path.isfile(inpath):
            print(f"[WARN] Missing input: {inpath}")
            continue
        make_hist_for_sample(
            sample_key=skey,
            infile=inpath,
            outdir=args.out_dir,
            lumi_pb=lumi_pb,
            extra_weight=args.extra_weight,
            suffix=args.suffix,
            tree_name=args.tree,
            var_name=args.var_name,
            edges=edges,
            nn_branch=args.nn_branch,
            nn_cut=args.nn_cut,
        )

if __name__ == "__main__":
    main()
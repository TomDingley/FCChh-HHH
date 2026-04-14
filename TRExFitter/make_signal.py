#!/usr/bin/env python3
import os
import argparse
from array import array
import ROOT

ROOT.ROOT.EnableImplicitMT()


# code inspired entirely on ATLAS HHH fitting input prep


# -------------------------
# Defaults
# -------------------------
DEFAULT_TREE       = "events"
DEFAULT_NN_BRANCH  = "mlp_score"
DEFAULT_NN_CUT     = 0.975
DEFAULT_MHHH_BR    = "m_hhh_vis"
# mHHH bins
DEFAULT_EDGES_STR  = "200,650,1100,1550,2000"

# Luminosity 
LUMI_PBINV = 3.0e7           # 30 ab^-1 in pb^-1
BASE_WEIGHT = "weight_xsec"  # per-event weight in pb

# Logical names for the 9 basis weights
LOGICAL = [
    "k3_0_k4_0",
    "k3_1_k4_m1",
    "k3_m1_k4_1",
    "k3_1_k4_0",
    "k3_0_k4_1",
    "k3_m1_k4_0",
    "k3_0_k4_m1",
    "k3_0p5_k4_0",
    "k3_m0p5_k4_0",
]

# mapping
ALIASES = {
    "k3_0_k4_0"    : ["weight_k3m1_k4m1"],
    "k3_1_k4_m1"   : ["weight_k30_k4m2"],
    "k3_m1_k4_1"   : [ "weight_k3m2_k40"],
    "k3_1_k4_0"    : ["weight_k30_k4m1"],
    "k3_0_k4_1"    : ["weight_k3m1_k40"],
    "k3_m1_k4_0"   : ["weight_k3m2_k4m1"],
    "k3_0_k4_m1"   : ["weight_k3m1_k4m2"],
    "k3_0p5_k4_0"  : ["weight_k3m0p5_k4m1"],
    "k3_m0p5_k4_0" : ["weight_k3m1p5_k4m1"],
}

# SM coefficients
COEF = {
    "k3_0_k4_0"    : -2.0,
    "k3_1_k4_m1"   : -1.0,
    "k3_m1_k4_1"   :  0.0,
    "k3_1_k4_0"    :  2.0,
    "k3_0_k4_1"    :  1.0,
    "k3_m1_k4_0"   :  0.0,
    "k3_0_k4_m1"   :  1.0,
    "k3_0p5_k4_0"  :  0.0,
    "k3_m0p5_k4_0" :  0.0,
}

def pick_existing_name(df, candidates, tree_name):
    cols = set(df.GetColumnNames())
    for c in candidates:
        if c in cols:
            return c
    raise RuntimeError(f"None of {candidates} found in TTree '{tree_name}'")

def parse_edges(edges_str):
    try:
        edges = [float(x) for x in edges_str.split(",")]
    except Exception:
        raise ValueError(f"Could not parse --edges '{edges_str}'. Use comma-separated floats, e.g. '0,900,2000'.")
    if len(edges) < 2:
        raise ValueError("Need at least two edges.")
    if any(edges[i] >= edges[i+1] for i in range(len(edges)-1)):
        raise ValueError("Edges must be strictly increasing.")
    return edges

def main():
    ap = argparse.ArgumentParser(description="Make HHH inputs: NN cut + 2-bin m_hhh_vis histograms with k3/k4 reweights")
    ap.add_argument("--input",   required=True, help="Path to scored signal ROOT (e.g. .../mgp8_pp_hhh_84TeV.root)")
    ap.add_argument("--tree",    default=DEFAULT_TREE, help=f"TTree name (default: {DEFAULT_TREE})")
    ap.add_argument("--outdir",  required=True, help="Output directory for ROOT histograms")
    ap.add_argument("--nn-branch", default=DEFAULT_NN_BRANCH, help=f"NN score branch (default: {DEFAULT_NN_BRANCH})")
    ap.add_argument("--nn-cut",    type=float, default=DEFAULT_NN_CUT, help=f"NN cut (>=) (default: {DEFAULT_NN_CUT})")
    ap.add_argument("--mhhh-branch", default=DEFAULT_MHHH_BR, help=f"m_hhh branch (default: {DEFAULT_MHHH_BR})")
    ap.add_argument("--edges",      default=DEFAULT_EDGES_STR,
                    help=f"Comma-separated bin edges for m_hhh (default: '{DEFAULT_EDGES_STR}')")
    ap.add_argument("--suffix")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    edges = parse_edges(args.edges)
    edges_arr = array('d', edges)
    hist_model = ROOT.RDF.TH1DModel("hist", "hist", len(edges)-1, edges_arr)
    suffix = args.suffix
    print(f"[INFO] Input:    {args.input}")
    print(f"[INFO] Tree:     {args.tree}")
    print(f"[INFO] Outdir:   {args.outdir}")
    print(f"[INFO] NN cut:   {args.nn_branch} >= {args.nn_cut}")
    print(f"[INFO] m_hhh:    {args.mhhh_branch} with edges {edges}")

    df = ROOT.RDataFrame(args.tree, args.input)
    cols = set(df.GetColumnNames())

    # Basic column checks
    for needed in (args.nn_branch, args.mhhh_branch, BASE_WEIGHT):
        if needed not in cols:
            raise RuntimeError(f"Column '{needed}' not found in tree '{args.tree}'")

    # Resolve actual branch names for the 9 benchmarks
    resolved = {key: pick_existing_name(df, ALIASES[key], args.tree) for key in LOGICAL}

    # Build SM denominator and scaled base weight
    sm_sum = " + ".join(f"{COEF[k]}*{resolved[k]}" for k in LOGICAL)
    df = (df
          .Define("wSM_raw", f"({sm_sum})")
          .Define("wSM", "wSM_raw == 0.0 ? 1e-12 : wSM_raw")
          .Define("w_base", f"{BASE_WEIGHT} * {LUMI_PBINV}")
          )

    # Selection: NN cut AND m_hhh within the histogram range to avoid under/overflow
    low, high = edges[0], edges[-1]
    df_sel = df.Filter(f"{args.nn_branch} >= {args.nn_cut}") \
               .Filter(f"{args.mhhh_branch} >= {low} && {args.mhhh_branch} < {high}")

    # 1) Nominal histogram (no kappa reweight): use w_base
    h_nom = df_sel.Histo1D(hist_model, args.mhhh_branch, "w_base")
    with ROOT.TFile(os.path.join(args.outdir, f"hist_nominal_k3k4_{suffix}.root"), "RECREATE") as f:
        f.WriteObject(h_nom.GetPtr(), "hist")

    # 2) One file per benchmark: w = w_base * (w_kappa / wSM)
    for key in LOGICAL:
        colname = f"w_{key}"
        df_w = df_sel.Define(colname, f"w_base * {resolved[key]} / wSM")
        h = df_w.Histo1D(hist_model, args.mhhh_branch, colname)
        fout = os.path.join(args.outdir, f"hist_{key}_k3k4_{suffix}.root")
        with ROOT.TFile(fout, "RECREATE") as f:
            f.WriteObject(h.GetPtr(), "hist")

    print(f"[OK] Wrote {1+len(LOGICAL)} ROOT files to {args.outdir} (lumi = {LUMI_PBINV:.2e} pb^-1)")

if __name__ == "__main__":
    main()
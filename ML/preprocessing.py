#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import json
from pathlib import Path
import argparse


# ----------------------------
# Selection functions
# ----------------------------
def apply_selection_lephadMMC(events):
    mu_cond = (events["n_sel_mu"] == 1) & (events["OS_taumu"] == 1 & (events["n_sel_el_0p2"] == 0))
    el_cond = (events["n_sel_el_0p2"] == 1) & (events["OS_taue"] == 1) & (events["n_sel_mu"] == 0)

    return (
        (events["n_tau_jets_medium"] == 1)
        & (events["n_b_jets_medium_tauprio"] == 4)
        & (mu_cond | el_cond)
        & (events["pT_b1"] > 40)
        & (events["pT_b2"] > 35)
        & (events["pT_b3"] > 30)
        & (events["pT_b4"] > 25)
        & (events["pT_tau1_tlv_LH"] > 25)
        & (events["pT_tau2_tlv_LH"] > 20)
        & (events["weighted_MMC_para_perp_vispTcal"] > 50)
        & (events["weighted_MMC_para_perp_vispTcal"] < 300)
        & (events["m_h1"] > 40)
        & (events["m_h2"] > 20) 
        & (events["m_h1"] < 175)
        & (events["m_h2"] < 160) 
    )



def apply_selection_hadhadMMC(events):
    return (
        (events["n_tau_jets_medium"] == 2)
        & (events["n_b_jets_medium_tauprio"] == 4)
        & (events["n_sel_el_0p2"] == 0)
        & (events["n_sel_mu"] == 0)
        & (events["OS_tau"] == 1)
        & (events["pT_tau1_tlv_LH"] > 25)
        & (events["pT_tau2_tlv_LH"] > 25)
        & (events["pT_b1"] > 40)
        & (events["pT_b2"] > 35)
        & (events["pT_b3"] > 30)
        & (events["pT_b4"] > 25)
        & (events["weighted_MMC_para_perp_vispTcal"] > 50)
        & (events["weighted_MMC_para_perp_vispTcal"] < 300)
        & (events["weight"] > 0)
        & (events["m_h1"] > 40)
        & (events["m_h2"] > 20) 
        & (events["m_h1"] < 175)
        & (events["m_h2"] < 160) 
    )
    


def selection_for_channel(channel: str):
    if "lephad" in channel:        
        return apply_selection_lephadMMC
    if "hadhad" in channel:
        return apply_selection_hadhadMMC
    raise ValueError(f"Unknown channel: {channel}")



# ----------------------------
# Processing function
# ----------------------------
def process_file(fpath: Path, outdir: Path, config: dict, channel: str):
    print(f"[*] Processing {fpath.name} for channel {channel}...")

    with uproot.open(fpath) as f:
        tree = f["events"]
        events = tree.arrays(library="ak")

    mask = selection_for_channel(channel)(events)

    events_sel = events[mask]

    # keep only training / weight vars (need weights for fitting after)
    vars_to_keep = config["channels"][channel]["features"]
    out_arrays = {v: ak.to_numpy(events_sel[v]) for v in vars_to_keep if v in events_sel.fields}

    # write new ROOT file
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{fpath.stem}_{channel}.root"
    with uproot.recreate(outpath) as fout:
        fout["events"] = out_arrays

    print(f"[✓] Saved {outpath}")


# ----------------------------
# Main CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input directory with ROOT files")
    ap.add_argument("--outdir", required=True, help="Output directory for reduced ROOT files")
    ap.add_argument("--config", default="config.json", help="Config JSON with training vars")
    ap.add_argument("--channels", nargs="+", default=None, help="Channels to process")
    ap.add_argument("--processes", nargs="+", default=None, help="Process names (without .root)")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    with open(args.config) as f:
        config = json.load(f)

    # list of processes to process
    processes = args.processes or [
        "mgp8_pp_ttbb_4f_84TeV",
        "mgp8_pp_tth_5f_84TeV",
        "mgp8_pp_ttz_5f_84TeV",
        "mgp8_pp_tttt_5f_84TeV",
        "mgp8_pp_zzz_5f_84TeV",
        "mgp8_pp_hhh_84TeV",
        "pwp8_pp_hh_k3_1_k4_1_84TeV"
    ]
    
    channels = args.channels

    for proc in processes:
        fpath = indir / f"{proc}.root"
        if not fpath.is_file():
            print(f"[!] Skipping {proc}, file not found")
            continue
        for channel in channels:
            process_file(fpath, outdir, config, channel)


if __name__ == "__main__":
    main()

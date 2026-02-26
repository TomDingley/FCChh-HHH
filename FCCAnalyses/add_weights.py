import os
import sys
import uproot
import awkward as ak
import numpy as np
from pathlib import Path

tree_path = "events"
output_dir = "weighted"
noDecay = False

BR = {}
BR['h'] = {"bb": 0.5792, "tautau": 0.0624, "yy": 0.00227}
BR['Z'] = {"tautau": 0.033696, "bb": 0.1512}
BR['W'] = {"taunu": 0.1138}

# k-factors the same for all
kFactors = {
    "mgp8_pp_tth_5f_84TeV": 1.45,
    "mgp8_pp_ttbb_4f_84TeV": 1.41,
    "mgp8_pp_ttz_5f_84TeV": 1.52,
    "mgp8_pp_zzz_5f_84TeV": 1.70,
    "mgp8_pp_tttt_5f_84TeV": 2.10,
    "mgp8_pp_hhh_84TeV": 2.25,
    "pwp8_pp_hh_k3_1_k4_1_84TeV": 1 # is an NLO sample, kl-dependent K-factors are already applied
}
    
    
# === Cross sections and per-process BRs/k-factors ===
if noDecay:
    print("********* Running with noDecay option enabled *********")
    cross_sections = {
        "mgp8_pp_ttbb_4f_84TeV": 155.77,
        "mgp8_pp_tth_5f_84TeV": 17.5041,
        "mgp8_pp_ttz_5f_84TeV": 28.15,
        "mgp8_pp_tttt_5f_84TeV": 1.57,
        "mgp8_pp_zzz_5f_84TeV": 0.151,
        "mgp8_pp_hhh_84TeV": 0.00156,   # inclusive pb at 84 TeV (LO)
    }
    # In noDecay mode, treat all final states as 100%
    BFs = {key: 1.0 for key in cross_sections}
    BFs["mgp8_pp_hhh_84TeV"] = 3 * BR['h']['bb']**2 * BR['h']['tautau']
    
else:
    cross_sections = {
        "mgp8_pp_ttbb_4f_84TeV": 155.77,
        "mgp8_pp_tth_5f_84TeV": 17.5041,
        "mgp8_pp_ttz_5f_84TeV": 28.15,
        "mgp8_pp_tttt_5f_84TeV": 1.57,
        "mgp8_pp_zzz_5f_84TeV": 0.151,
        "mgp8_pp_hhh_84TeV": 0.00156,   # inclusive pb at 84 TeV (LO),
        "pwp8_pp_hh_k3_1_k4_1_84TeV": 0.885
    }
    
    # Build per-process BRs from particle-level BR map
    BFs = {
        "mgp8_pp_tth_5f_84TeV": (BR['h']['bb'] + BR['h']['tautau']) * BR['W']['taunu']**2,
        "mgp8_pp_ttbb_4f_84TeV": (3*BR['W']['taunu'])**2,
        "mgp8_pp_ttz_5f_84TeV": BR['W']['taunu']**2 * (BR['Z']['bb'] + BR['Z']['tautau']),
        "mgp8_pp_zzz_5f_84TeV": (BR['Z']['bb'] + BR['Z']['tautau'])**3,
        "mgp8_pp_tttt_5f_84TeV": 1,
        "mgp8_pp_hhh_84TeV": 3 * BR['h']['bb']**2 * BR['h']['tautau'],
        "pwp8_pp_hh_k3_1_k4_1_84TeV": 4 * BR['h']['bb'] * BR['h']['tautau']

    }


SIGNAL_KEY = "mgp8_pp_hhh_84TeV"

def process_file(input_path, output_base):
    try:
        with uproot.open(input_path) as infile:
            if tree_path not in infile:
                print(f"  [!] Tree '{tree_path}' not found in {input_path}, skipping.")
                return

            tree = infile[tree_path]

            # avoid non-flat, funky variables
            available = tree.keys(filter_name="*")
            wanted = [b for b in available]  # keep simple ones
            branches = tree.arrays(wanted, library="ak")

            file_key = Path(input_path).stem
            if (file_key not in cross_sections) or (file_key not in BFs) or (file_key not in kFactors):
                print(f" File '{file_key}' missing in xsec/BF/k-factor dicts, skipping.")
                return

            if hasattr(tree, "num_entries"):
                total_events = int(tree.num_entries)
            else:
                if not branches.fields:
                    print(f"No branches found in {file_key}, skipping.")
                    return
                total_events = len(branches[branches.fields[0]])
            
            if total_events == 0:
                print(f"No events found in {file_key}, skipping.")
                return

            doForNow = False
            if file_key == SIGNAL_KEY or doForNow:
                w_base = np.ones(total_events, dtype=np.float64)
            else:
                if "weight" in branches.fields:
                    print("using weight from file")
                    w_base = branches["weight"]
                else:
                    ref = branches[branches.fields[0]]
                    w_base = np.ones(len(ref), dtype=np.float64)

            # Per-event normalisation
            if file_key == SIGNAL_KEY or doForNow:
                # signal uses cross-section * BR * k, distributed over N events
                norm = (cross_sections[file_key] * BFs[file_key] * kFactors[file_key]) / total_events
                kind = "Signal weighting"
            else:
                # backgrounds keep previous convention: (BF * k / N) multiplied by existing generator weight
                norm = (BFs[file_key] * kFactors[file_key]) / total_events
                kind = "Background weighting"

            # Final xsec-like weight
            branches["weight_xsec"] = w_base * norm

            sigma_incl = cross_sections[file_key]
            sigma_eff = sigma_incl * BFs[file_key] * kFactors[file_key]  # effective cross section (pb)
            print(
                f"  [{kind}] {file_key}: "
                f"events={total_events:,}, xsec_incl={sigma_incl:.6g} pb, "
                f"BF={BFs[file_key]:.6g}, k={kFactors[file_key]:.3f}, "
                f"xsec_eff={sigma_eff:.6g} pb, norm={norm:.3e}"
            )

            # Write out
            outdir = output_base / output_dir
            os.makedirs(outdir, exist_ok=True)
            output_path = outdir / f"{file_key}.root"
            with uproot.recreate(output_path) as outfile:
                outfile[tree_path] = branches

            print(f"Wrote weighted file: {output_path}")

    except Exception as e:
        print(f" Error processing {input_path}: {e}")

# === Entry Point ===
def main():
    if len(sys.argv) != 2:
        print("Usage: python filter_ntuples.py <input_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_base = input_dir

    if not input_dir.is_dir():
        print(f"Input path {input_dir} is not a directory.")
        sys.exit(1)

    root_files = list(input_dir.glob("*.root"))
    if not root_files:
        print(f"No .root files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(root_files)} ROOT files from: {input_dir}")

    for input_file in root_files:
        process_file(input_file, output_base)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Prepare HH -> bb gamma gamma inputs using the original bbaa ML selection."""
import argparse
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from root_io import open_root


PROCESSES = [
    "pwp8_pp_hh_k3_1_k4_1_84TeV",
    "mgp8_pp_jjaa_5f_84TeV",
    "mgp8_pp_h012j_5f_84TeV",
    "mgp8_pp_tth01j_5f_84TeV",
    "mgp8_pp_vbf_h01j_5f_84TeV",
]
SELECTION_COLUMNS = ["njets", "ngamma", "nele", "nmu", "nbjets", "m_yy", "m_bb"]


def apply_selection_bbaa(events):
    return (
        (events["njets"] > 1)
        & (events["ngamma"] > 1)
        & (events["nele"] == 0)
        & (events["nmu"] == 0)
        & (events["nbjets"] == 2)
        & (events["m_yy"] > 122)
        & (events["m_yy"] < 128)
        & (events["m_bb"] < 160)
        & (events["m_bb"] > 80)
    )


def process_file(fpath, outdir, config, channel="bbaa"):
    columns = list(dict.fromkeys(config["channels"][channel]["features"]))
    if "weight_xsec" not in columns:
        raise ValueError("Preprocessing config must retain weight_xsec.")
    required = list(dict.fromkeys(SELECTION_COLUMNS + columns))
    outpath = outdir / f"{fpath.stem}_{channel}.root"
    if outpath.exists():
        raise FileExistsError(outpath)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = 0
    with open_root(fpath) as source:
        tree = source["events"]
        missing = set(required) - set(tree.keys())
        if missing:
            raise ValueError(f"{fpath}: missing columns {sorted(missing)}")
        # Create an explicit TTree, including when no events pass selection.
        empty = tree.arrays(columns, entry_stop=0, library="ak")
        try:
            with uproot.recreate(outpath) as output:
                target = output.mktree("events", {
                    name: ak.type(empty[name]).content for name in columns
                })
                for events in tree.iterate(required, step_size="100 MB", library="ak"):
                    passing = events[apply_selection_bbaa(events)]
                    arrays = {name: ak.to_numpy(passing[name], allow_missing=False) for name in columns}
                    for name, values in arrays.items():
                        if values.ndim != 1 or not np.isfinite(values).all():
                            raise ValueError(f"{fpath}: selected {name} must be finite scalars")
                    if len(passing):
                        target.extend(arrays)
                    selected += len(passing)
        except Exception:
            outpath.unlink(missing_ok=True)
            raise
        print(f"{fpath.name}: {selected}/{tree.num_entries} events -> {outpath}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--channels", nargs="+", choices=["bbaa"], default=["bbaa"])
    parser.add_argument("--processes", nargs="+", default=PROCESSES)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Check every input before writing any output; missing samples are errors.
    inputs = [args.indir / f"{proc}.root" for proc in args.processes]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in inputs:
        for channel in args.channels:
            process_file(path, args.outdir, config, channel)


if __name__ == "__main__":
    main()

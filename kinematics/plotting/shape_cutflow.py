"""
Distribution shape cutflow for a single variable.

For each top-level AND term in the channel selection, this script plots the
variable distribution after cumulatively applying cuts. It also saves an
overlay summary of all steps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import re

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

from config import SELECTION, N_BINS_1D, XLIM_MAP, LUMINOSITY_PB
from aesthetics import LABEL_MAP, banner, channel_labels, banner_heatmaps
from tools import build_mask_from_selection


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s


def _split_top_level_and(expr: str) -> List[str]:
    s = _strip_outer_parens(expr)
    parts = []
    start = 0
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif depth == 0 and i + 1 < len(s) and s[i:i+2] == "&&":
            parts.append(_strip_outer_parens(s[start:i].strip()))
            start = i + 2
            i += 1
        i += 1
    parts.append(_strip_outer_parens(s[start:].strip()))
    return [p for p in parts if p and p.lower() != "true"]


def _flatten_with_weights(arr: ak.Array, weights: ak.Array | None) -> Tuple[np.ndarray, np.ndarray | None]:
    if isinstance(arr.type, ak.types.ListType):
        counts = ak.to_numpy(ak.num(arr, axis=1))
        flat = ak.to_numpy(ak.flatten(arr))
        if weights is not None:
            w = ak.to_numpy(weights)
            w = np.repeat(w, counts)
        else:
            w = None
    else:
        flat = ak.to_numpy(arr)
        w = ak.to_numpy(weights) if weights is not None else None

    finite = np.isfinite(flat)
    flat = flat[finite]
    if w is not None:
        w = w[finite]
    return flat.astype(float), w.astype(float) if w is not None else None


def _make_edges(values: np.ndarray, var: str, bins: int) -> np.ndarray:
    if var in XLIM_MAP:
        lo, hi = XLIM_MAP[var]
    else:
        lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid x-limits for {var}: {lo}, {hi}")
    return np.linspace(lo, hi, bins + 1)


def _short_label(term: str, max_len: int = 80) -> str:
    term = term.replace("  ", " ").strip()
    return term if len(term) <= max_len else term[: max_len - 3] + "..."

def _strip_units(label: str) -> str:
    # Remove bracketed units like " [GeV]" or " [MMC]"
    label = re.sub(r"\s*\[[^\]]+\]\s*", "", label)
    return label.strip()

def _prettify_cut_label(term: str) -> str:
    # Replace variable tokens with LABEL_MAP entries (minus units)
    pretty = term
    for tok in sorted(LABEL_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(tok)}\b", pretty):
            pretty_label = _strip_units(LABEL_MAP[tok])
            pretty = re.sub(rf"\b{re.escape(tok)}\b", lambda _m, pl=pretty_label: pl, pretty)
    # Normalise logical operators for readability
    pretty = pretty.replace("&&", " and ").replace("||", " or ")
    pretty = pretty.replace(">=", " ≥ ").replace("<=", " ≤ ")
    pretty = pretty.replace("!=", " ≠ ").replace("==", " = ")
    pretty = pretty.replace(">", " > ").replace("<", " < ")
    pretty = re.sub(r"\s+", " ", pretty).strip()
    return pretty


def plot_shape_cutflow(
    root_path: Path,
    var: str,
    outdir: Path,
    channel: str,
    selection: str,
    tree_name: str = "events",
    weight_branch: str | None = None,
    lumi_scale: float | None = LUMINOSITY_PB,
    bins: int = N_BINS_1D,
    normalize: bool = True,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cuts = _split_top_level_and(selection)
    steps = [("Total", "True", "Total")]
    cumulative = "True"
    for i, term in enumerate(cuts, start=1):
        cumulative = f"({cumulative}) && ({term})"
        pretty = _prettify_cut_label(term)
        steps.append((_short_label(pretty), cumulative, pretty))

    with uproot.open(root_path) as f:
        tree = f[tree_name]

        if var not in tree.keys():
            raise KeyError(f"Variable '{var}' not found in tree '{tree_name}'.")

        weights = None
        if weight_branch:
            if weight_branch not in tree.keys():
                raise KeyError(f"Weight branch '{weight_branch}' not found in tree '{tree_name}'.")
            weights = tree[weight_branch].array(library="ak")
            if lumi_scale is not None:
                weights = weights * float(lumi_scale)

        # Determine bin edges from the total selection (no cuts)
        base_vals, _ = _flatten_with_weights(tree[var].array(library="ak"), None)
        edges = _make_edges(base_vals, var, bins)
        step_x = np.append(edges[:-1], edges[-1])

        all_counts = []
        all_labels = []

        n_entries = tree.num_entries
        full_mask = np.ones(n_entries, dtype=bool)

        for idx, (label, expr, _label_full) in enumerate(steps):
            if expr.strip().lower() == "true":
                mask = full_mask
            else:
                mask = build_mask_from_selection(tree, expr)
            vals, w = _flatten_with_weights(tree[var].array(library="ak")[mask], weights[mask] if weights is not None else None)
            if vals.size == 0:
                print(f"[skip] {label}: no events after cut")
                continue

            counts, _ = np.histogram(vals, bins=edges, weights=w)
            counts = counts.astype(float)
            if normalize:
                total = counts.sum()
                if total > 0:
                    counts = counts / total

            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            ax.step(step_x, np.append(counts, counts[-1]), where="post", color="black", label=label)
            if all_counts:
                prev_counts = all_counts[-1]
                prev_label = all_labels[-1]
                ax.step(step_x, np.append(prev_counts, prev_counts[-1]), where="post", color="gray", linestyle="--", label=f"Prev: {prev_label}")
                ax.legend(frameon=False, fontsize=9)
            ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
            ax.set_ylabel("Normalised counts" if normalize else "Events / bin", loc="top")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, max(counts) * 1.25 if counts.size else 1)

            comment = f"{channel_labels.get(channel, channel)}\n{label}"
            banner_heatmaps(ax)
            outpath = outdir / f"{var}_cut{idx:02d}.pdf"
            fig.savefig(outpath, bbox_inches="tight")
            plt.close(fig)
            print("[✓]", outpath)

            all_counts.append(counts)
            all_labels.append(label)

        # Overlay summary
        if all_counts:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_counts)))
            overlay_labels = [s[2] for s in steps[: len(all_counts)]]
            for counts, label, color in zip(all_counts, overlay_labels, colors):
                ax.step(step_x, np.append(counts, counts[-1]), where="post", label=label, color=color, linewidth=1.4)

            ax.set_xlabel(LABEL_MAP.get(var, var), loc="right")
            ax.set_ylabel("Normalised counts" if normalize else "Events / bin", loc="top")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, max([c.max() for c in all_counts]) * 1.25)
            ax.legend(frameon=False, fontsize=9, ncol=2)
            banner_heatmaps(ax)

            outpath = outdir / f"{var}_cutflow_overlay.pdf"
            fig.savefig(outpath, bbox_inches="tight")
            plt.close(fig)
            print("[✓]", outpath)


def cli() -> None:
    ap = argparse.ArgumentParser(description="Plot a variable's distribution after each cut in a channel selection.")
    ap.add_argument("--root", required=True, help="Path to ROOT file")
    ap.add_argument("--var", required=True, help="Variable/branch name (e.g. m_hhh_vis)")
    ap.add_argument("--channel", default="HadHad", help="Channel key in config.SELECTION")
    ap.add_argument("--selection", default="", help="Override selection string (if set, channel is ignored)")
    ap.add_argument("--outdir", default="cutflow_shapes", help="Output directory")
    ap.add_argument("--tree", default="events", help="Tree name")
    ap.add_argument("--weight-branch", default="", help="Weight branch ('' to disable)")
    ap.add_argument("--lumi", type=float, default=LUMINOSITY_PB, help="Luminosity scale (pb^-1)")
    ap.add_argument("--bins", type=int, default=N_BINS_1D, help="Number of bins")
    ap.add_argument("--no-normalize", action="store_true", help="Plot raw weighted counts instead of normalized shapes")
    args = ap.parse_args()

    selection = args.selection.strip() if args.selection else ""
    if not selection:
        if args.channel not in SELECTION:
            raise KeyError(f"Channel '{args.channel}' not found in config.SELECTION.")
        selection = SELECTION[args.channel]

    weight_branch = args.weight_branch if args.weight_branch else None
    normalize = not args.no_normalize

    plot_shape_cutflow(
        root_path=Path(args.root),
        var=args.var,
        outdir=Path(args.outdir),
        channel=args.channel,
        selection=selection,
        tree_name=args.tree,
        weight_branch=weight_branch,
        lumi_scale=args.lumi,
        bins=args.bins,
        normalize=normalize,
    )


if __name__ == "__main__":
    cli()

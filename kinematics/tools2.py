from __future__ import annotations
import re
import operator as op
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np
import uproot

from pathlib import Path
from collections import OrderedDict, defaultdict
import re
import csv
import argparse
import math



from config import processes, SIGNAL, BACKGROUNDS, SELECTION, SKIP_VARS, LUMINOSITY_PB, HADHAD, LEPHAD_TAU, LEPHAD_ETAU, LEPHAD_MUTAU, LEPHAD_ANY, BJET_PRESEL, HADHAD_JET, HADHAD_OS, HADHAD_NOLEP, LEPHAD_BJETS
from aesthetics import process_labels, LABEL_MAP


import re
from typing import Dict, Iterable, Set

# cutflow:
CUTFLOW = {
    "LepHad": [
        {"label": "Total",      "expr": "True"},
        {"label": "Lep–Had",    "expr": "LEPHAD"},  # replace with your selection
        # Examples (uncomment/adapt):
        # {"label": "Njet ≥ 5",   "expr": "jets_n >= 5"},
        # {"label": "≥2 b-jets",  "expr": "bjets_n >= 2"},
        # {"label": "pT_b1 > 40", "expr": "pt_bjet_1 > 40"},
        # {"label": "Standalone @Lep–Had: NNScore>0.98", "expr": "NNScore_OOF > 0.98", "base": "Lep–Had"},
    ],
    "HadHad": [
        {"label": "Total",     "expr": "True"},
        {"label": "Had–Had",   "expr": "HADHAD"},    # replace with your selection
    ],
    "Combined": [
        {"label": "Total",                "expr": "True"},
        {"label": "LepHad ∪ HadHad",      "expr": "NNScore_OOF > 0.985"},
    ],
}


# Whitelisted helper functions usable in selection strings
SAFE_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
}

def _group_logical_terms(expr: str) -> str:
    """Recursively wrap terms separated by '&' and '|' at every parenthesis level.

    Ensures sub-expressions like "a==1 & b==0 & c==1" become
    "(a==1) & (b==0) & (c==1)", preventing Python's chained-comparison parse.
    """
    def process(level_expr: str) -> str:
        # 1) Recursively process nested parentheses first
        pieces: List[str] = []
        i = 0
        while i < len(level_expr):
            ch = level_expr[i]
            if ch == '(':
                j = i + 1
                depth = 1
                while j < len(level_expr) and depth:
                    if level_expr[j] == '(':
                        depth += 1
                    elif level_expr[j] == ')':
                        depth -= 1
                    j += 1
                inner = level_expr[i+1:j-1]
                pieces.append('(' + process(inner) + ')')
                i = j
            else:
                pieces.append(ch)
                i += 1
        s = ''.join(pieces)

        # 2) At this level, split on top-level '&' and '|'
        terms: List[str] = []
        ops: List[str] = []
        buf: List[str] = []
        depth = 0
        for ch in s:
            if ch == '(':
                depth += 1; buf.append(ch)
            elif ch == ')':
                depth -= 1; buf.append(ch)
            elif ch in ('&', '|') and depth == 0:
                terms.append(''.join(buf).strip()); buf = []
                ops.append(ch)
            else:
                buf.append(ch)
        terms.append(''.join(buf).strip())

        def wrap(t: str) -> str:
            # keep leading unary ~ outside
            u = t.lstrip()
            neg = 0
            while u.startswith('~'):
                neg += 1
                u = u[1:].lstrip()
            inner = f'({u})'
            return inner if neg % 2 == 0 else f'~{inner}'

        wrapped = [wrap(t) for t in terms]
        out_s = wrapped[0]
        for op, w in zip(ops, wrapped[1:]):
            out_s = f"{out_s} {op} {w}"
        return out_s

    return process(expr)

# Reserved words/operators we shouldn't treat as branch names
_RESERVED_TOKENS: Set[str] = {
    "and", "or", "not",  # just in case
    "True", "False",
}

_OP_TOKENS = {"&&", "||", "!", "==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%", "(", ")"}

_token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _translate_to_python(expr: str) -> str:
    """Translate ROOT/C style logical ops to Python/numexpr-friendly ops.

    - '&&' → '&', '||' → '|', '!' → '~'
    - Ensure we don't accidentally allow assignment '='
    """
    # quick check for accidental assignment
    if re.search(r"(?<![!<>=])=(?![=])", expr):
        raise ValueError("Selection contains '=' (assignment). Use '==' for comparison.")

    expr = expr.replace("&&", "&")
    expr = expr.replace("||", "|")
    # Replace logical NOT carefully: avoid turning '!=' into '~='
    expr = re.sub(r"!([^=])", r"~\1", expr)
    return expr


def _find_candidate_names(expr: str) -> Set[str]:
    """Return identifiers in expr that could be branch names."""
    names = set(_token_re.findall(expr))
    # remove known function names and booleans
    names -= set(SAFE_FUNCS.keys())
    names -= _RESERVED_TOKENS
    return names


def _ensure_boolean_mask(mask) -> ak.Array:
    # Coerce to Awkward
    if not isinstance(mask, ak.Array):
        mask = ak.Array(mask)
    # Ensure dtype is boolean
    if mask.type != np.bool_:
        mask = (mask != 0)
    return mask


def build_mask_from_selection(ttree, selection: str, *, cache: Dict[str, ak.Array] | None = None) -> ak.Array:
    """Build a boolean mask for `ttree` that satisfies `selection`.

    Parameters
    ----------
    ttree : uproot.behaviors.TTree.TTree
        The opened tree, e.g. f["events"].
    selection : str
        A selection string using C/ROOT-like syntax (&&, ||, !, parentheses, comparisons).
    cache : dict, optional
        Optional dict to reuse already-read branches across calls.

    Returns
    -------
    ak.Array (boolean)
        Mask with one element per entry.
    """
    cache = {} if cache is None else cache

    expr_py = _group_logical_terms(_translate_to_python(selection))
    # Determine which branch arrays we need
    candidates = _find_candidate_names(expr_py)

    # Load branches as awkward arrays (and keep in cache)
    vars_dict: Dict[str, ak.Array] = {}
    for name in sorted(candidates):
        if name in _RESERVED_TOKENS or name in SAFE_FUNCS:
            continue
        if name not in cache:
            try:
                cache[name] = ttree[name].array(library="ak")
            except Exception as e:
                raise KeyError(f"Branch '{name}' not found while parsing selection: {selection}\n{e}")
        vars_dict[name] = cache[name]

    # Evaluation namespace
    safe_locals = {**vars_dict, **SAFE_FUNCS, "np": np, "ak": ak}

    try:
        result = eval(expr_py, {"__builtins__": {}}, safe_locals)
    except Exception as e:
        raise ValueError(f"Failed to evaluate selection: {selection}\nTranslated: {expr_py}\nError: {e}")

    # Ensure boolean mask
    mask = _ensure_boolean_mask(result)

    # Sanity check: length compatibility
    if len(mask) != len(next(iter(vars_dict.values()))) if vars_dict else len(ttree):
        raise RuntimeError("Mask length does not match number of entries.")

    return mask

def _scalar(tree, var):
    """Return a flat array suitable for boolean selection logic."""
    arr = tree[var].array(library="ak")
    if isinstance(arr.type, ak.types.ListType):
        # Jagged array: ensure each entry has only one item
        if ak.any(ak.num(arr, axis=1) != 1):
            raise ValueError(f"'{var}' is jagged with multiple elements per event.")
        return ak.flatten(arr)
    else:
        # Already a flat array
        return arr

def numeric(arr: ak.Array) -> np.ndarray:
    """Convert an Awkward array to a clean NumPy array of finite floats."""
    flat = ak.flatten(arr, axis=None)
    num = np.asarray(flat, dtype=float)
    return num[np.isfinite(num)]

# --- Weighting Function ---
import numpy as np

# EFT basis definition (shared between helpers below)
BASIS_POINTS = [
    (0, 0),
    (1, -1),
    (-1, 1),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (0.5, 0),
    (-0.5, 0),
]

BASIS_KEYS = [
    "weight_k3m1_k4m1",
    "weight_k30_k4m2",
    "weight_k3m2_k40",
    "weight_k30_k4m1",
    "weight_k3m1_k40",
    "weight_k3m2_k4m1",
    "weight_k3m1_k4m2",
    "weight_k3m0p5_k4m1",
    "weight_k3m1p5_k4m1",
]

def basis_funcs(k3, k4):
    return np.array([
        1.0, k3, k3**2,
        k3**3, k3**4, k4,
        k4**2, k3*k4, k3**2*k4
    ])

def build_moments(weight_dict):
    """Project event weights into the morphing moment basis."""
    event_weights = np.vstack([weight_dict[key] for key in BASIS_KEYS]).T  # (N, 9)
    M = np.array([basis_funcs(k3b, k4b) for (k3b, k4b) in BASIS_POINTS])
    Minv = np.linalg.inv(M)
    return event_weights @ Minv.T  # (N, 9)

def evaluate_weights_from_moments(k3, k4, moments):
    """Evaluate per-event weights from precomputed moments."""
    target_vec = basis_funcs(k3, k4)
    sm_vec = basis_funcs(1, 1)
    target_weights = moments @ target_vec
    sm_weights = moments @ sm_vec
    return np.divide(target_weights, sm_weights, out=np.full_like(target_weights, np.nan), where=sm_weights != 0)

def get_weights(k3, k4, weight_dict):
    """
    Compute per-event reweighting factors for given EFT couplings (k3, k4),
    using named event weight branches. Normalises to SM (k3=1, k4=1).

    Parameters:
    - k3, k4: EFT coupling values
    - weight_dict: dict of arrays (one per named EFT weight leaf).
      Keys must match the expected basis order.

    Returns:
    - Array of shape (N_events,) with reweighting factors
    """

    moments = build_moments(weight_dict)
    return evaluate_weights_from_moments(k3, k4, moments)

# ---- adapters so external scripts can import these symbols ----

def _load_weights_with_mask(ttree, selection=None):
    weight_dict = {k: ttree[k].array(library="ak") for k in BASIS_KEYS}

    if selection:
        mask = build_mask_from_selection(ttree, selection)
        weight_dict = {k: arr[mask] for k, arr in weight_dict.items()}
    weight_dict = {k: ak.to_numpy(arr) for k, arr in weight_dict.items()}

    finite = np.ones(len(next(iter(weight_dict.values()))), dtype=bool)
    for arr in weight_dict.values():
        finite &= np.isfinite(arr)
    weight_dict = {k: arr[finite] for k, arr in weight_dict.items()}

    return weight_dict

def compute_xsec_grid(root_path, k3_vals, k4_vals, tree="events", selection=None):
    with uproot.open(root_path) as f:
        t = f[tree]
        weight_dict = _load_weights_with_mask(t, selection=selection)
        moments = build_moments(weight_dict)

    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    xsec = np.zeros_like(K3, dtype=float)

    for i in range(K3.shape[0]):
        for j in range(K3.shape[1]):
            k3, k4 = K3[i, j], K4[i, j]
            ratios = evaluate_weights_from_moments(k3, k4, moments)
            xsec[i, j] = np.nansum(ratios)

    return xsec

def _plot_contour(k3_vals, k4_vals, grid, outpath, label, cmap):
    import matplotlib.pyplot as plt

    K3, K4 = np.meshgrid(k3_vals, k4_vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    c = ax.contourf(K3, K4, grid, levels=60, cmap=cmap)
    fig.colorbar(c, ax=ax, label=label)
    ax.set_xlabel("$k_3$")
    ax.set_ylabel("$k_4$")
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def _plot_ratio_contour(k3_vals, k4_vals, num, den, outpath, label_num, label_den):
    ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)
    _plot_contour(k3_vals, k4_vals, ratio, outpath, f"{label_num}/{label_den}", cmap="coolwarm")

def _plot_slice(x_vals, ys_by_label, xlabel, ylabel, outpath):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for label, y in ys_by_label.items():
        ax.plot(x_vals, y, linewidth=1.7, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def plot_xsec_comparison(
    files_by_label,
    outdir,
    k3_vals,
    k4_vals,
    tree="events",
    selection=None,
    k3_slice=1.0,
    k4_slice=1.0,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grids = {}
    for label, path in files_by_label.items():
        grids[label] = compute_xsec_grid(path, k3_vals, k4_vals, tree=tree, selection=selection)
        _plot_contour(
            k3_vals,
            k4_vals,
            grids[label],
            outdir / f"xsec_contour_{label}.pdf",
            f"Sum of reweight ratios ({label})",
            cmap="viridis",
        )

    labels = list(grids.keys())
    if len(labels) == 2:
        _plot_ratio_contour(
            k3_vals,
            k4_vals,
            grids[labels[0]],
            grids[labels[1]],
            outdir / f"xsec_ratio_{labels[0]}_over_{labels[1]}.pdf",
            labels[0],
            labels[1],
        )

    k4_idx = int(np.argmin(np.abs(k4_vals - k4_slice)))
    k3_idx = int(np.argmin(np.abs(k3_vals - k3_slice)))

    slice_k3 = {label: grid[:, k4_idx] for label, grid in grids.items()}
    slice_k4 = {label: grid[k3_idx, :] for label, grid in grids.items()}

    _plot_slice(
        k3_vals,
        slice_k3,
        xlabel=f"$k_3$ (k4={k4_vals[k4_idx]:.2f})",
        ylabel="Sum of reweight ratios",
        outpath=outdir / f"xsec_slice_k3_k4_{k4_vals[k4_idx]:.2f}.pdf",
    )
    _plot_slice(
        k4_vals,
        slice_k4,
        xlabel=f"$k_4$ (k3={k3_vals[k3_idx]:.2f})",
        ylabel="Sum of reweight ratios",
        outpath=outdir / f"xsec_slice_k4_k3_{k3_vals[k3_idx]:.2f}.pdf",
    )

def _parse_args():
    ap = argparse.ArgumentParser(description="Plot EFT cross-section contours and slices.")
    ap.add_argument("--file-a", required=True, help="Path to COM A ROOT file")
    ap.add_argument("--file-b", required=True, help="Path to COM B ROOT file")
    ap.add_argument("--label-a", default="COMA", help="Label for COM A")
    ap.add_argument("--label-b", default="COMB", help="Label for COM B")
    ap.add_argument("--tree", default="events", help="TTree name")
    ap.add_argument("--selection", default=None, help="Optional selection string")
    ap.add_argument("--outdir", default="eft_xsec", help="Output directory")
    ap.add_argument("--k3-min", type=float, default=-20.0)
    ap.add_argument("--k3-max", type=float, default=20.0)
    ap.add_argument("--k4-min", type=float, default=-200.0)
    ap.add_argument("--k4-max", type=float, default=200.0)
    ap.add_argument("--k3-steps", type=int, default=41)
    ap.add_argument("--k4-steps", type=int, default=41)
    ap.add_argument("--k3-slice", type=float, default=1.0)
    ap.add_argument("--k4-slice", type=float, default=1.0)
    return ap.parse_args()

def main():
    args = _parse_args()
    k3_vals = np.linspace(args.k3_min, args.k3_max, args.k3_steps)
    k4_vals = np.linspace(args.k4_min, args.k4_max, args.k4_steps)
    files_by_label = {
        args.label_a: args.file_a,
        args.label_b: args.file_b,
    }
    plot_xsec_comparison(
        files_by_label,
        outdir=args.outdir,
        k3_vals=k3_vals,
        k4_vals=k4_vals,
        tree=args.tree,
        selection=args.selection,
        k3_slice=args.k3_slice,
        k4_slice=args.k4_slice,
    )
    print(f"[✓] Plots saved under: {args.outdir}")

if __name__ == "__main__":
    main()

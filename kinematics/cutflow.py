# cutflow_tables.py
# Minimal, importable cutflow table utility (no CLI).
# Usage from your main:
#   from cutflow_tables import write_cutflow_weighted_summary
#   csv_path, tex_path, pdf_path = write_cutflow_weighted_summary(files, outdir, channel)

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import awkward as ak
import uproot

from aesthetics import process_labels
# ---------------------------------------------------------------------
# Formatting helper (kept tiny; you can import if you like)
# ---------------------------------------------------------------------
def numeric(x, ndigits: int = 2) -> str:
    try:
        xf = float(x)
        if xf.is_integer():
            return f"{int(xf)}"
        return f"{xf:.{ndigits}f}"
    except Exception:
        return str(x)


# ---------------------------------------------------------------------
# Expression parsing / evaluation
# ---------------------------------------------------------------------
_ID_RE = re.compile(r"[A-Za-z_]\w*")

def _normalize_expr(expr: str) -> str:
    """Make expressions array-friendly: and/or/not, &&/||/! -> &/|/~."""
    s = expr
    s = re.sub(r"\band\b", "&", s)
    s = re.sub(r"\bor\b",  "|", s)
    s = re.sub(r"\bnot\b", "~", s)
    s = s.replace("&&", "&").replace("||", "|").replace("!", "~")
    return s

def _extract_vars(expr: str) -> set[str]:
    """Identifiers used in expr, excluding booleans and helper names."""
    s = _normalize_expr(expr)
    tokens = set(_ID_RE.findall(s))
    tokens -= {"True", "False", "np"}
    tokens -= {"abs", "min", "max", "clip"}
    return tokens

def _eval_mask(env: Dict[str, np.ndarray], expr: str) -> np.ndarray:
    """Evaluate boolean mask with only numpy + arrays in env."""
    s = _normalize_expr(expr)
    safe_globals = {"__builtins__": {}}
    safe_locals = {"np": np}
    safe_locals.update(env)
    out = eval(s, safe_globals, safe_locals)
    out = np.asarray(out)
    if out.dtype != bool:
        if np.issubdtype(out.dtype, np.number):
            out = out != 0
        else:
            out = out.astype(bool)
    return out


# ---------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------
def _asimov_Z(s: float, b: float, rel_sys: float = 0.0) -> float:
    """Cowan et al. Asimov significance with optional background relative systematic."""
    s = max(float(s), 1e-12)
    b = max(float(b), 1e-12)
    if rel_sys <= 0:
        return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))
    sb2 = (rel_sys * b) ** 2
    term1 = (s + b) * math.log((s + b) * (b + sb2) / (b * b + (s + b) * sb2))
    term2 = (b * b / sb2) * math.log(1.0 + (sb2 * s) / (b * (b + sb2)))
    z2 = 2.0 * (term1 - term2)
    return math.sqrt(max(z2, 0.0))


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _tree_branch_names(one_file: str, tree_name: str) -> set[str]:
    with uproot.open(one_file) as f:
        try:
            t = f[tree_name]
        except KeyError as e:
            available = ", ".join([k.split(";")[0] for k in f.keys()])
            raise FileNotFoundError(
                f"Tree '{tree_name}' not found in '{one_file}'. "
                f"Available top-level keys: {available}"
            ) from e
        return set(t.keys())

def _needed_branches(expressions: List[str], weight_branch: str, available: Optional[set[str]] = None) -> List[str]:
    """
    Branches required by expressions (+ weight branch if present in file).
    """
    names = set()
    for expr in expressions:
        names |= _extract_vars(expr)
    if available is not None:
        names = {n for n in names if n in available}
        if weight_branch in available:
            names.add(weight_branch)
    else:
        names.add(weight_branch)
    return sorted(names)

def _iter_arrays(process_files: List[str],
                 tree_name: str,
                 branches: List[str],
                 step: Optional[Union[int, str]] = None,
                 default_bytes: str = "200 MB"):
    """
    Stream arrays from files using uproot.iterate, only requested branches.
    step: int entries, or memory string like "200 MB". None -> default_bytes.
    """
    filespecs = [f"{str(p)}:{tree_name}" for p in process_files]
    step_size = step if step is not None else default_bytes

    for chunk in uproot.iterate(filespecs,
                                expressions=branches,
                                step_size=step_size,
                                library="ak"):
        arrays = {}
        for k in chunk.fields:
            arr = chunk[k]
            if isinstance(arr, ak.Array):
                arr = ak.to_numpy(arr)
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(f"Branch '{k}' is not 1D (shape={arr.shape})")
            arrays[k] = arr
        yield arrays


# Try to import pretty labels; fall back to empty map if unavailable.
try:
    from aesthetics import LABEL_MAP as _VAR_LABEL_MAP
except Exception:
    _VAR_LABEL_MAP = {}

_LATEX_RESERVED = {"True", "False", "np", "abs", "min", "max", "clip"}

def _latex_escape_text(s: str) -> str:
    # minimal safe escaping for LaTeX text mode
    return s.replace("_", r"\_")

# Pretty LaTeX labels pulled from aesthetics.LABEL_MAP
try:
    from aesthetics import LABEL_MAP as _VAR_LABEL_MAP
except Exception:
    _VAR_LABEL_MAP = {}

_LATEX_RESERVED = {"True", "False", "np", "abs", "min", "max", "clip"}

def _strip_dollars(s: str) -> str:
    # Remove any $ to avoid nested math when we wrap with $$...$$
    return s.replace("$", "")

def _strip_math_and_units(s: str, remove_units=("GeV",)) -> str:
    # remove inline math dollars and specified bracketed units like [GeV]
    s = s.replace("$", "")
    for u in remove_units:
        s = re.sub(rf"\s*\[{re.escape(u)}\]", "", s)
    return s.strip()

def _pretty_expr_for_latex(expr: str, label_map: Optional[Dict[str, str]] = None) -> str:
    """
    Pretty version of `expr` suitable for wrapping in $$...$$:
    - identifiers replaced via aesthetics.LABEL_MAP with $ removed and [GeV] stripped
    - unknown identifiers shown as \mathrm{foo_bar}
    """
    if label_map is None:
        label_map = _VAR_LABEL_MAP

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok in _LATEX_RESERVED:
            return tok
        if tok in label_map:
            return _strip_math_and_units(label_map[tok])
        return r"\mathrm{" + tok.replace("_", r"\_") + "}"

    s = _strip_outer_parens(expr)
    s = _ID_RE.sub(repl, s)
    return s


def _split_top_level_or(expr: str) -> list[str]:
    """Split a boolean expression at *top-level* ORs (| or 'or')."""
    s = _strip_outer_parens(_normalize_expr(expr))
    parts, start, depth = [], 0, 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == "|" and depth == 0:
            parts.append(_strip_outer_parens(s[start:i].strip()))
            start = i + 1
    parts.append(_strip_outer_parens(s[start:].strip()))
    # drop trivial True
    return [p for p in parts if p and p.lower() != "true"]

def _split_or_macro(expr: str) -> Optional[list[str]]:
    """
    If expr looks like OR("a","b",...), extract the string args.
    Mirrors _split_and_macro but for OR(...).
    """
    try:
        node = ast.parse(expr, mode="eval").body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.upper() == "OR":
            terms = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    terms.append(arg.value)
                else:
                    return None
            return terms
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------
# Cut rows from config.SELECTION
# ---------------------------------------------------------------------
def _rows_from_config(channel: str, mode: str = "chain") -> List[Dict[str, str]]:
    try:
        import importlib
        cfg = importlib.import_module("config")
        SELECTION = getattr(cfg, "SELECTION")
    except Exception as e:
        raise RuntimeError("config.SELECTION is required but could not be imported.") from e

    rows: List[Dict[str, str]] = []
    total_expr = str(SELECTION.get("Total", "True"))

    if mode == "chain":
        rows.append({"label": "Total", "expr": total_expr, "advance": True})
        if channel != "Total":
            if channel not in SELECTION:
                raise KeyError(f"SELECTION missing '{channel}'. Available: {list(SELECTION.keys())}")
            chan_expr = str(SELECTION[channel])

            terms = _split_and_macro(chan_expr)
            if not terms:
                terms = _split_top_level_and(chan_expr)

            if len(terms) <= 1:
                pretty = _pretty_expr_for_latex(chan_expr)
                rows.append({"label": f"$$ {pretty} $$", "expr": chan_expr, "advance": True})
            else:
                cumul: list[str] = []
                for t in terms:
                    t_clean = _strip_outer_parens(t)

                    or_terms = _split_or_macro(t_clean) or _split_top_level_or(t_clean)

                    if or_terms and len(or_terms) > 1:
                        # one row per OR branch (do NOT advance cum)
                        for ot in or_terms:
                            pretty_branch = _pretty_expr_for_latex(ot)
                            rows.append({
                                "label": f"$$ {pretty_branch} $$",
                                "expr": " & ".join(cumul + [f"({ot})"]),
                                "advance": False,                      # <-- no cum update here
                            })

                        # union row (advance cum)
                        union_expr = "(" + " | ".join(or_terms) + ")"
                        pretty_union = " \\lor ".join(_pretty_expr_for_latex(ot) for ot in or_terms)
                        rows.append({
                            "label": f"$$ {pretty_union} $$",
                            "expr": " & ".join(cumul + [union_expr]),
                            "advance": True,                          # <-- advance on the union
                        })
                        cumul.append(union_expr)
                    else:
                        pretty = _pretty_expr_for_latex(t_clean)
                        cumul.append(f"({t_clean})")
                        rows.append({
                            "label": f"$$ {pretty} $$",
                            "expr": " & ".join(cumul),
                            "advance": True,                          # normal step advances
                        })
        return rows

    # mode == "all": compare each selection vs Total (standalone)
    rows.append({"label": "Total", "expr": total_expr})
    for key, expr in SELECTION.items():
        if key == "Total":
            continue
        rows.append({"label": key, "expr": str(expr), "base": "Total"})
    return rows


# ---------------------------------------------------------------------
# Core: cutflow per process
# ---------------------------------------------------------------------
def _cutflow_for_process(files: List[str],
                         cut_rows: List[Dict[str, str]],
                         *,
                         tree_name: str,
                         weight_branch: str,
                         lumi: float,
                         mode: str,
                         step_size: Optional[Union[int, str]] = None) -> List[float]:
    """
    Return [sumW per row] for one process across files.
    Respects mode:
      - 'chain' cumulative: Total -> channel
      - 'all'   standalone rows: each row applied as (Total & row_expr)
    """
    # Inspect first file for available branches
    avail = _tree_branch_names(files[0], tree_name)
    expr_list = [r["expr"] for r in cut_rows]

    # Friendly error if selection references missing branches
    all_vars = set().union(*(_extract_vars(e) for e in expr_list))
    missing = [v for v in sorted(all_vars) if v not in avail]
    if missing:
        raise KeyError(f"Selection references missing branches: {missing}. "
                       f"Check config.SELECTION or your tree '{tree_name}'.")

    branches = _needed_branches(expr_list, weight_branch, available=avail)
    sums = np.zeros(len(cut_rows), dtype=float)

    for arrays in _iter_arrays(files, tree_name, branches, step=step_size):
        # weights (scaled by lumi)
        if weight_branch in arrays:
            w = arrays[weight_branch].astype(float) * float(lumi)
        else:
            first_key = next(iter(arrays.keys()))
            w = np.ones_like(arrays[first_key], dtype=float)

        # baseline
        total_mask = _eval_mask(arrays, cut_rows[0]["expr"]) if cut_rows and cut_rows[0]["label"] == "Total" else np.ones_like(w, bool)

        if mode == "chain":
            cum = None
            for i, row in enumerate(cut_rows):
                m = _eval_mask(arrays, row["expr"])
                sums[i] += float(w[m].sum())

                # only advance the cumulative reference on advancing rows
                if i == 0 or row.get("advance", True):
                    cum = m
        else:
            for i, row in enumerate(cut_rows):
                if row["label"] == "Total":
                    m = total_mask
                else:
                    m = total_mask & _eval_mask(arrays, row["expr"])
                sums[i] += float(w[m].sum())

    return sums.tolist()


# --- add near the other helpers ---

# --- add near your other helpers ---
import ast

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

def _split_top_level_and(expr: str) -> list[str]:
    """Split a boolean expression at *top-level* ANDs (& or 'and').
    Handles cases like '((a) && (b))' by first removing redundant outer parens.
    """
    s = _strip_outer_parens(_normalize_expr(expr))  # <-- strip outer () first
    parts, start, depth = [], 0, 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == "&" and depth == 0:
            parts.append(_strip_outer_parens(s[start:i].strip()))
            start = i + 1
    parts.append(_strip_outer_parens(s[start:].strip()))
    return [p for p in parts if p and p.lower() != "true"]

def _split_and_macro(expr: str) -> Optional[list[str]]:
    """
    If expr looks like AND("a","b",...), extract the string args.
    Works even if your config builds selections as a macro call string.
    """
    try:
        node = ast.parse(expr, mode="eval").body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.upper() == "AND":
            terms = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    terms.append(arg.value)
                else:
                    return None
            return terms
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------
# Table building + writing
# ---------------------------------------------------------------------
def _eff_step(arr: np.ndarray) -> np.ndarray:
    eff = np.ones_like(arr, dtype=float)
    for i in range(1, len(arr)):
        eff[i] = arr[i] / arr[i - 1] if arr[i - 1] > 0 else 0.0
    return eff

def _eff_cum(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    base = arr[0] if arr[0] > 0 else 1.0
    return np.array([x / base for x in arr], dtype=float)

def _make_table(
    cut_rows: List[Dict[str, str]],
    yields_by_proc: Dict[str, List[float]],
    signal_key_substr: str,
    ttbb_key_substr: Optional[str],
    rel_sys_b: float = 0.0,
):
    """
    Build rows with metrics for LaTeX-friendly output.
    Returns (columns, rows, process_order).
    """
    procs = list(yields_by_proc.keys())
    if not procs:
        raise ValueError("No processes provided.")

    # Signal first, others sorted by total yield (desc)
    sig = next((p for p in procs if signal_key_substr in p), procs[0])
    others = [p for p in procs if p != sig]
    others.sort(key=lambda p: -yields_by_proc[p][-1] if yields_by_proc[p] else 0.0)
    order = [sig] + others

    # Optional ttbb reference
    ttbb = next((p for p in procs if ttbb_key_substr and ttbb_key_substr in p), None)

    nrows = len(cut_rows)
    Y = np.array([yields_by_proc[p] for p in order], dtype=float)  # (P, R)

    S = Y[0]
    B = Y[1:].sum(axis=0) if Y.shape[0] > 1 else np.zeros(nrows)

    # Efficiencies
    S_eff_step = _eff_step(S)
    S_eff_cum = _eff_cum(S)

    if ttbb is not None and ttbb in yields_by_proc:
        T = np.array(yields_by_proc[ttbb], dtype=float)
        T_eff_step = _eff_step(T)
        T_eff_cum = _eff_cum(T)
    else:
        T_eff_step = np.full(nrows, np.nan)
        T_eff_cum = np.full(nrows, np.nan)

    # Derived metrics
    s_over_sqrtb = np.divide(S, np.sqrt(np.maximum(B, 1e-12)))
    Z_A = np.array([_asimov_Z(S[i], B[i], rel_sys_b) for i in range(nrows)])

    # Column headers (LaTeX-friendly)
    columns = (
        ["Cut"]
        + order
        + [
            r"$\frac{S}{\sqrt{B}}$",
            r"$\varepsilon_{\text{sig}}$",
            r"$\varepsilon_{\text{ttbb}}$",
        ]
    )

    # Table rows
    labels = [r["label"] for r in cut_rows]
    rows = []
    for i, lab in enumerate(labels):
        row_vals = [lab]
        for p in order:
            row_vals.append(yields_by_proc[p][i])
        row_vals += [
            s_over_sqrtb[i],
            S_eff_step[i],
            T_eff_step[i],
        ]
        rows.append(row_vals)

    return columns, rows, order

def _write_csv(path: Path, columns: List[str], rows: List[List[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(columns)
        for r in rows:
            w.writerow(r)

def _write_latex(path, columns, rows):
    """Write a minimal LaTeX table and (best-effort) compile to PDF.
    Uses string concatenation to avoid f-string brace issues with LaTeX.
    """
    import numpy as np
    import subprocess

    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return f"{int(x)}"
        if isinstance(x, float):
            ax = abs(x)
            if (ax != 0.0 and (ax >= 1e4 or ax < 1e-2)):
                exp = int(np.floor(np.log10(ax)))
                mant = x / (10 ** exp)
                # show sign, 2 decimal places on mantissa, LaTeX 10^{exp}
                return r"${:.2f}\times10^{{{}}}$".format(mant, exp)
            else:
                return f"{x:.2f}"
        return str(x)

    colspec = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\ \hline"
    body = "\n".join(" & ".join(fmt(v) for v in row) + r" \\" for row in rows)

    tex = (
        r"\documentclass[10pt]{article}" "\n"
        r"\usepackage[margin=0.8in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{siunitx}" "\n"
        r"\begin{document}" "\n"
        r"\small" "\n"
        r"\begin{center}" "\n"
        r"\begin{tabular}{" + colspec + r"}" "\n"
        r"\toprule" "\n" +
        header + "\n" +
        r"\midrule" "\n" +
        body + "\n" +
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{center}" "\n"
        r"\end{document}" "\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)

    # Best-effort compile to PDF
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", path.name],
            cwd=path.parent, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------
# Public entry point (import and call from your main)
# ---------------------------------------------------------------------
def write_cutflow_weighted_summary(
    files: Dict[str, Union[str, Path, List[Union[str, Path]]]],  # {process_key: path or [paths]}
    outdir: Union[str, Path],
    channel: str,               # "Total", "LepHad", "HadHad", or "Combined"
    *,
    mode: str = "chain",        # "chain" (Total→channel cumulative) or "all" (each selection vs Total)
    tree_name: str = "events",
    weight_branch: str = "weight_xsec",
    lumi: Optional[float] = None,           # None -> config.LUMINOSITY_PB or 1.0
    signal_sub: Optional[str] = None,       # None -> config.SIGNAL or first process key
    ttbb_sub: Optional[str] = "ttbb",       # identifies ttbb for efficiency cols
    rel_sys_b: float = 0.0,                 # background rel. syst. for Z_Asimov
    process_label_map: Optional[Dict[str, str]] = None,  # header prettifier; default to config.PROCESS_LABELS if available
    step_size: Optional[Union[int, str]] = None,         # entries or memory string; None -> "200 MB"
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Compute a luminosity-weighted cutflow table from config.SELECTION and write CSV+TeX (+PDF if pdflatex).
    Returns (csv_path, tex_path, pdf_path_or_None).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Normalize files dict -> list of files per process (strings)
    groups: Dict[str, List[str]] = {}
    for proc, p in files.items():
        if isinstance(p, (list, tuple)):
            groups[proc] = [str(Path(x)) for x in p]
        else:
            groups[proc] = [str(Path(p))]

    # Pull defaults from config if available
    try:
        import importlib
        cfg = importlib.import_module("config")
    except Exception:
        cfg = None

    if lumi is None:
        lumi = getattr(cfg, "LUMINOSITY_PB", 1.0) if cfg else 1.0
    if signal_sub is None:
        signal_sub = getattr(cfg, "SIGNAL", None) if cfg else None
        if signal_sub is None:
            signal_sub = next(iter(groups.keys()))  # fallback: first key

    # Build rows from config.SELECTION
    cut_rows = _rows_from_config(channel, mode=mode)
   

    # Compute per-process yields
    yields_by_proc: Dict[str, List[float]] = {}
    for proc, filelist in sorted(groups.items()):
        sums = _cutflow_for_process(
            filelist,
            cut_rows,
            tree_name=tree_name,
            weight_branch=weight_branch,
            lumi=float(lumi),
            mode=mode,
            step_size=step_size,  # may be None; iterate() defaults internally to "200 MB"
        )
        yields_by_proc[proc] = sums

    # Table
    cols, rows, order = _make_table(
        cut_rows=cut_rows,
        yields_by_proc=yields_by_proc,
        signal_key_substr=str(signal_sub),
        ttbb_key_substr=ttbb_sub,
        rel_sys_b=rel_sys_b,
    )

    # Optional pretty labels
    if process_label_map is None and cfg is not None:
        process_label_map = process_labels
    if process_label_map:
        cols = [cols[0]] + [process_label_map.get(p, p) for p in order] + cols[len(order) + 1 :]

    # Write outputs
    tag = f"{channel}_{mode}"
    csv_path = outdir / f"cutflow_{tag}.csv"
    tex_path = outdir / f"cutflow_{tag}.tex"
    _write_csv(csv_path, cols, rows)
    _write_latex(tex_path, cols, rows)

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        pdf_path = None

    return csv_path, tex_path, pdf_path

def write_region_yield_summary(
    files: Dict[str, Union[str, Path, List[Union[str, Path]]]],  # {process_key: path or [paths]}
    outdir: Union[str, Path],
    channels: List[str],            # e.g. ["HadHad_resolved", "HadHad_1BB", ...]
    *,
    tree_name: str = "events",
    weight_branch: str = "weight_xsec",
    lumi: Optional[float] = None,   # None -> config.LUMINOSITY_PB or 1.0
    signal_sub: Optional[str] = None,
    process_label_map: Optional[Dict[str, str]] = None,
    channel_label_map: Optional[Dict[str, str]] = None,
    step_size: Optional[Union[int, str]] = None,  # entries or memory string; None -> "200 MB"
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Make a summary table with the *final* yield of each channel
    for all processes.

    One row per `channel`, columns:
        Channel | signal | bkg1 | bkg2 | ... | S/sqrt(B)

    Returns (csv_path, tex_path, pdf_path_or_None).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Normalise files dict -> list of files per process (strings)
    groups: Dict[str, List[str]] = {}
    for proc, p in files.items():
        if isinstance(p, (list, tuple)):
            groups[proc] = [str(Path(x)) for x in p]
        else:
            groups[proc] = [str(Path(p))]

    # Pull defaults from config if available
    try:
        import importlib
        cfg = importlib.import_module("config")
    except Exception:
        cfg = None

    if lumi is None:
        lumi = getattr(cfg, "LUMINOSITY_PB", 1.0) if cfg else 1.0

    # Signal substring: same convention as write_cutflow_weighted_summary
    if signal_sub is None:
        if cfg is not None:
            signal_sub = getattr(cfg, "SIGNAL", None)
        if signal_sub is None:
            signal_sub = next(iter(groups.keys()))  # fallback: first key

    # Optional pretty process labels: default to aesthetics.process_labels
    if process_label_map is None and cfg is not None:
        try:
            from aesthetics import process_labels as _proc_labels
            process_label_map = _proc_labels
        except Exception:
            pass

    # Optional pretty channel labels: default to aesthetics.channel_labels
    _default_channel_labels: Dict[str, str] = {}
    try:
        from aesthetics import channel_labels as _chan_labels
        _default_channel_labels = _chan_labels
    except Exception:
        pass

    # Fixed process ordering: signal first, then others alphabetically
    all_procs = sorted(groups.keys())
    sig_proc = next((p for p in all_procs if signal_sub and signal_sub in p), all_procs[0])
    other_procs = [p for p in all_procs if p != sig_proc]
    proc_order = [sig_proc] + other_procs

    # Build rows: one per channel
    rows: List[List[Union[str, float]]] = []
    for chan in channels:
        # Build cut rows for this channel (cumulative chain)
        cut_rows = _rows_from_config(chan, mode="chain")

        # For each process, compute cutflow and take the final value
        final_yields: Dict[str, float] = {}
        for proc, filelist in sorted(groups.items()):
            sums = _cutflow_for_process(
                filelist,
                cut_rows,
                tree_name=tree_name,
                weight_branch=weight_branch,
                lumi=float(lumi),
                mode="chain",
                step_size=step_size,
            )
            final_yields[proc] = float(sums[-1]) if sums else 0.0

        # Channel label (pretty if available)
        if channel_label_map is not None:
            chan_label = channel_label_map.get(chan, chan)
        else:
            chan_label = _default_channel_labels.get(chan, chan)

        # Build row: channel, per-process yields in fixed order
        row: List[Union[str, float]] = [chan_label]
        proc_vals: List[float] = []
        for proc in proc_order:
            val = final_yields.get(proc, 0.0)
            proc_vals.append(val)
            row.append(val)

        # Compute S/sqrt(B) for this channel
        S = proc_vals[0] if proc_vals else 0.0
        B = sum(proc_vals[1:]) if len(proc_vals) > 1 else 0.0
        s_over_sqrtB = S / math.sqrt(B) if B > 0.0 else 0.0
        row.append(s_over_sqrtB)

        rows.append(row)

    # Build column headers
    # First column: "Channel", then processes with pretty labels if available, then S/sqrt(B)
    cols: List[str] = ["Channel"]
    for p in proc_order:
        if process_label_map:
            # process_label_map may be a dict or a function; handle both
            label = None
            if hasattr(process_label_map, "get"):
                label = process_label_map.get(p, p)
            else:
                try:
                    label = process_label_map(p)  # type: ignore[call-arg]
                except Exception:
                    label = p
            cols.append(label)
        else:
            cols.append(p)
    cols.append(r"$S/\sqrt{B}$")

    # Write outputs
    tag = "region_summary"
    csv_path = outdir / f"cutflow_{tag}.csv"
    tex_path = outdir / f"cutflow_{tag}.tex"
    _write_csv(csv_path, cols, rows)
    _write_latex(tex_path, cols, rows)

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        pdf_path = None

    return csv_path, tex_path, pdf_path
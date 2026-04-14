from __future__ import annotations

import ast
import csv
import importlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import awkward as ak
import numpy as np
import uproot

from aesthetics import LABEL_MAP as _VAR_LABEL_MAP
from aesthetics import process_labels


CutRow = Dict[str, Any]
PathLike = Union[str, Path]
ProcessFiles = Dict[str, Union[PathLike, List[PathLike]]]

_ID_RE = re.compile(r"[A-Za-z_]\w*")
_RESERVED_EXPR_NAMES = {"True", "False", "np", "abs", "min", "max", "clip"}


# ---------------------------------------------------------------------
# Formatting helper
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
# Config and input normalization
# ---------------------------------------------------------------------
def _load_optional_config():
    try:
        return importlib.import_module("config")
    except Exception:
        return None


def _selection_from_config() -> Dict[str, Any]:
    try:
        cfg = importlib.import_module("config")
        return getattr(cfg, "SELECTION")
    except Exception as e:
        raise RuntimeError("config.SELECTION is required but could not be imported.") from e


def _normalize_file_groups(files: ProcessFiles) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for proc, paths in files.items():
        if isinstance(paths, (list, tuple)):
            groups[proc] = [str(Path(path)) for path in paths]
        else:
            groups[proc] = [str(Path(paths))]
    return groups


def _resolve_lumi(lumi: Optional[float], cfg) -> float:
    if lumi is not None:
        return float(lumi)
    return float(getattr(cfg, "LUMINOSITY_PB", 1.0)) if cfg else 1.0


def _resolve_signal_sub(signal_sub: Optional[str], cfg, groups: Dict[str, List[str]]) -> str:
    if signal_sub is not None:
        return str(signal_sub)
    configured = getattr(cfg, "SIGNAL", None) if cfg else None
    return str(configured if configured is not None else next(iter(groups.keys())))


def _label_for(label_map, key: str) -> str:
    if not label_map:
        return key
    if hasattr(label_map, "get"):
        return label_map.get(key, key)
    try:
        return label_map(key)
    except Exception:
        return key


def _default_channel_labels() -> Dict[str, str]:
    try:
        from aesthetics import channel_labels

        return channel_labels
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Expression parsing / evaluation
# ---------------------------------------------------------------------
def _normalize_expr(expr: str) -> str:
    """Make expressions array-friendly: and/or/not, &&/||/! -> &/|/~."""
    s = expr
    s = re.sub(r"\band\b", "&", s)
    s = re.sub(r"\bor\b", "|", s)
    s = re.sub(r"\bnot\b", "~", s)
    s = s.replace("&&", "&").replace("||", "|").replace("!", "~")
    return s


def _extract_vars(expr: str) -> set[str]:
    """Identifiers used in expr, excluding booleans and helper names."""
    tokens = set(_ID_RE.findall(_normalize_expr(expr)))
    return tokens - _RESERVED_EXPR_NAMES


def _eval_mask(env: Dict[str, np.ndarray], expr: str) -> np.ndarray:
    """Evaluate boolean mask with only numpy + arrays in env."""
    safe_locals = {"np": np}
    safe_locals.update(env)

    out = eval(_normalize_expr(expr), {"__builtins__": {}}, safe_locals)
    out = np.asarray(out)
    if out.dtype == bool:
        return out
    if np.issubdtype(out.dtype, np.number):
        return out != 0
    return out.astype(bool)


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        wraps_whole_expr = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    wraps_whole_expr = False
                    break
        if not wraps_whole_expr:
            break
        s = s[1:-1].strip()
    return s


def _split_top_level(expr: str, operator: str) -> list[str]:
    s = _strip_outer_parens(_normalize_expr(expr))
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == operator and depth == 0:
            parts.append(_strip_outer_parens(s[start:i].strip()))
            start = i + 1
    parts.append(_strip_outer_parens(s[start:].strip()))
    return [p for p in parts if p and p.lower() != "true"]


def _split_top_level_and(expr: str) -> list[str]:
    return _split_top_level(expr, "&")


def _split_top_level_or(expr: str) -> list[str]:
    return _split_top_level(expr, "|")


def _split_macro(expr: str, name: str) -> Optional[list[str]]:
    try:
        node = ast.parse(expr, mode="eval").body
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.upper() == name
        ):
            return None

        terms = []
        for arg in node.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                return None
            terms.append(arg.value)
        return terms
    except Exception:
        return None


def _split_and_macro(expr: str) -> Optional[list[str]]:
    return _split_macro(expr, "AND")


def _split_or_macro(expr: str) -> Optional[list[str]]:
    return _split_macro(expr, "OR")


# ---------------------------------------------------------------------
# ROOT IO
# ---------------------------------------------------------------------
def _tree_branch_names(one_file: str, tree_name: str) -> set[str]:
    with uproot.open(one_file) as f:
        try:
            tree = f[tree_name]
        except KeyError as e:
            available = ", ".join(k.split(";")[0] for k in f.keys())
            raise FileNotFoundError(
                f"Tree '{tree_name}' not found in '{one_file}'. "
                f"Available top-level keys: {available}"
            ) from e
        return set(tree.keys())


def _needed_branches(
    expressions: List[str],
    weight_branch: str,
    available: Optional[set[str]] = None,
) -> List[str]:
    names = set()
    for expr in expressions:
        names |= _extract_vars(expr)

    if available is not None:
        names = {name for name in names if name in available}
        if weight_branch in available:
            names.add(weight_branch)
    else:
        names.add(weight_branch)

    return sorted(names)


def _iter_arrays(
    process_files: List[str],
    tree_name: str,
    branches: List[str],
    step: Optional[Union[int, str]] = None,
    default_bytes: str = "200 MB",
):
    filespecs = [f"{path}:{tree_name}" for path in process_files]
    step_size = step if step is not None else default_bytes

    for chunk in uproot.iterate(
        filespecs,
        expressions=branches,
        step_size=step_size,
        library="ak",
    ):
        arrays = {}
        for key in chunk.fields:
            arr = chunk[key]
            if isinstance(arr, ak.Array):
                arr = ak.to_numpy(arr)
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(f"Branch '{key}' is not 1D (shape={arr.shape})")
            arrays[key] = arr
        yield arrays


# ---------------------------------------------------------------------
# Cut rows from config.SELECTION
# ---------------------------------------------------------------------
def _strip_math_and_units(s: str, remove_units=("GeV",)) -> str:
    s = s.replace("$", "")
    for unit in remove_units:
        s = re.sub(rf"\s*\[{re.escape(unit)}\]", "", s)
    return s.strip()


def _pretty_expr_for_latex(expr: str, label_map: Optional[Dict[str, str]] = None) -> str:
    """
    Pretty version of `expr` suitable for wrapping in $$...$$.
    Identifiers in aesthetics.LABEL_MAP are reused; unknown identifiers are
    rendered as \mathrm{name}.
    """
    labels = _VAR_LABEL_MAP if label_map is None else label_map

    def repl(match: re.Match) -> str:
        tok = match.group(0)
        if tok in _RESERVED_EXPR_NAMES:
            return tok
        if tok in labels:
            return _strip_math_and_units(labels[tok])
        return r"\mathrm{" + tok.replace("_", r"\_") + "}"

    return _ID_RE.sub(repl, _strip_outer_parens(expr))


def _rows_from_config(channel: str, mode: str = "chain") -> List[CutRow]:
    selection = _selection_from_config()
    total_expr = str(selection.get("Total", "True"))

    if mode == "chain":
        return _chain_rows_from_selection(selection, channel, total_expr)

    rows: List[CutRow] = [{"label": "Total", "expr": total_expr}]
    for key, expr in selection.items():
        if key != "Total":
            rows.append({"label": key, "expr": str(expr), "base": "Total"})
    return rows


def _chain_rows_from_selection(
    selection: Dict[str, Any],
    channel: str,
    total_expr: str,
) -> List[CutRow]:
    rows: List[CutRow] = [{"label": "Total", "expr": total_expr, "advance": True}]
    if channel == "Total":
        return rows

    if channel not in selection:
        raise KeyError(f"SELECTION missing '{channel}'. Available: {list(selection.keys())}")

    channel_expr = str(selection[channel])
    terms = _split_and_macro(channel_expr) or _split_top_level_and(channel_expr)

    if len(terms) <= 1:
        pretty = _pretty_expr_for_latex(channel_expr)
        rows.append({"label": f"$$ {pretty} $$", "expr": channel_expr, "advance": True})
        return rows

    cumulative_terms: list[str] = []
    for term in terms:
        clean_term = _strip_outer_parens(term)
        or_terms = _split_or_macro(clean_term) or _split_top_level_or(clean_term)

        if or_terms and len(or_terms) > 1:
            _append_or_cut_rows(rows, cumulative_terms, or_terms)
            continue

        cumulative_terms.append(f"({clean_term})")
        pretty = _pretty_expr_for_latex(clean_term)
        rows.append({
            "label": f"$$ {pretty} $$",
            "expr": " & ".join(cumulative_terms),
            "advance": True,
        })

    return rows


def _append_or_cut_rows(
    rows: List[CutRow],
    cumulative_terms: list[str],
    or_terms: list[str],
) -> None:
    for term in or_terms:
        pretty_branch = _pretty_expr_for_latex(term)
        rows.append({
            "label": f"$$ {pretty_branch} $$",
            "expr": " & ".join(cumulative_terms + [f"({term})"]),
            "advance": False,
        })

    union_expr = "(" + " | ".join(or_terms) + ")"
    pretty_union = " \\lor ".join(_pretty_expr_for_latex(term) for term in or_terms)
    rows.append({
        "label": f"$$ {pretty_union} $$",
        "expr": " & ".join(cumulative_terms + [union_expr]),
        "advance": True,
    })
    cumulative_terms.append(union_expr)


# ---------------------------------------------------------------------
# Core cutflow
# ---------------------------------------------------------------------
def _cutflow_for_process(
    files: List[str],
    cut_rows: List[CutRow],
    *,
    tree_name: str,
    weight_branch: str,
    lumi: float,
    mode: str,
    step_size: Optional[Union[int, str]] = None,
) -> List[float]:
    """
    Return [sumW per row] for one process across files.
    In chain mode, row expressions are already cumulative.
    In all mode, non-Total rows are evaluated as Total & row_expr.
    """
    available = _tree_branch_names(files[0], tree_name)
    expressions = [str(row["expr"]) for row in cut_rows]

    all_vars = set().union(*(_extract_vars(expr) for expr in expressions))
    missing = [name for name in sorted(all_vars) if name not in available]
    if missing:
        raise KeyError(
            f"Selection references missing branches: {missing}. "
            f"Check config.SELECTION or your tree '{tree_name}'."
        )

    branches = _needed_branches(expressions, weight_branch, available=available)
    sums = np.zeros(len(cut_rows), dtype=float)

    for arrays in _iter_arrays(files, tree_name, branches, step=step_size):
        if weight_branch in arrays:
            weights = arrays[weight_branch].astype(float) * float(lumi)
        else:
            first_key = next(iter(arrays.keys()))
            weights = np.ones_like(arrays[first_key], dtype=float)

        if mode == "chain":
            for i, row in enumerate(cut_rows):
                mask = _eval_mask(arrays, str(row["expr"]))
                sums[i] += float(weights[mask].sum())
            continue

        if cut_rows and cut_rows[0]["label"] == "Total":
            total_mask = _eval_mask(arrays, str(cut_rows[0]["expr"]))
        else:
            total_mask = np.ones_like(weights, bool)

        for i, row in enumerate(cut_rows):
            if row["label"] == "Total":
                mask = total_mask
            else:
                mask = total_mask & _eval_mask(arrays, str(row["expr"]))
            sums[i] += float(weights[mask].sum())

    return sums.tolist()


# ---------------------------------------------------------------------
# Table building and writing
# ---------------------------------------------------------------------
def _eff_step(arr: np.ndarray) -> np.ndarray:
    eff = np.ones_like(arr, dtype=float)
    for i in range(1, len(arr)):
        eff[i] = arr[i] / arr[i - 1] if arr[i - 1] > 0 else 0.0
    return eff


def _make_table(
    cut_rows: List[CutRow],
    yields_by_proc: Dict[str, List[float]],
    signal_key_substr: str,
    ttbb_key_substr: Optional[str],
    rel_sys_b: float = 0.0,
):
    """
    Build rows with the existing output columns:
    Cut | processes... | S/sqrt(B) | signal step eff | ttbb step eff.
    """

    procs = list(yields_by_proc.keys())
    if not procs:
        raise ValueError("No processes provided.")

    signal_proc = next((proc for proc in procs if signal_key_substr in proc), procs[0])
    other_procs = [proc for proc in procs if proc != signal_proc]
    other_procs.sort(key=lambda proc: -yields_by_proc[proc][-1] if yields_by_proc[proc] else 0.0)
    proc_order = [signal_proc] + other_procs

    ttbb_proc = next((proc for proc in procs if ttbb_key_substr and ttbb_key_substr in proc), None)

    nrows = len(cut_rows)
    yields = np.array([yields_by_proc[proc] for proc in proc_order], dtype=float)
    signal = yields[0]
    background = yields[1:].sum(axis=0) if yields.shape[0] > 1 else np.zeros(nrows)

    signal_eff_step = _eff_step(signal)
    if ttbb_proc is not None and ttbb_proc in yields_by_proc:
        ttbb_eff_step = _eff_step(np.array(yields_by_proc[ttbb_proc], dtype=float))
    else:
        ttbb_eff_step = np.full(nrows, np.nan)

    s_over_sqrtb = np.divide(signal, np.sqrt(np.maximum(background, 1e-12)))
    columns = (
        ["Cut"]
        + proc_order
        + [
            r"$\frac{S}{\sqrt{B}}$",
            r"$\varepsilon_{\text{sig}}$",
            r"$\varepsilon_{\text{ttbb}}$",
        ]
    )

    rows = []
    for i, cut_row in enumerate(cut_rows):
        row = [cut_row["label"]]
        row.extend(yields_by_proc[proc][i] for proc in proc_order)
        row.extend([s_over_sqrtb[i], signal_eff_step[i], ttbb_eff_step[i]])
        rows.append(row)

    return columns, rows, proc_order


def _write_csv(path: Path, columns: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_latex(path: Path, columns: List[str], rows: List[List[Any]]) -> None:
    """Write a minimal LaTeX table and compile to PDF on a best-effort basis."""

    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return f"{int(x)}"
        if isinstance(x, float):
            ax = abs(x)
            if ax != 0.0 and (ax >= 1e4 or ax < 1e-2):
                exp = int(np.floor(np.log10(ax)))
                mant = x / (10 ** exp)
                return r"${:.2f}\times10^{{{}}}$".format(mant, exp)
            return f"{x:.2f}"
        return str(x)

    colspec = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\ \hline"
    body = "\n".join(" & ".join(fmt(value) for value in row) + r" \\" for row in rows)

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

    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", path.name],
            cwd=path.parent,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _write_outputs(
    outdir: Path,
    tag: str,
    columns: List[str],
    rows: List[List[Any]],
) -> Tuple[Path, Path, Optional[Path]]:
    csv_path = outdir / f"cutflow_{tag}.csv"
    tex_path = outdir / f"cutflow_{tag}.tex"
    _write_csv(csv_path, columns, rows)
    _write_latex(tex_path, columns, rows)

    pdf_path = tex_path.with_suffix(".pdf")
    return csv_path, tex_path, pdf_path if pdf_path.exists() else None


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------
def write_cutflow_weighted_summary(
    files: Dict[str, Union[str, Path, List[Union[str, Path]]]],
    outdir: Union[str, Path],
    channel: str,
    *,
    mode: str = "chain",
    tree_name: str = "events",
    weight_branch: str = "weight_xsec",
    lumi: Optional[float] = None,
    signal_sub: Optional[str] = None,
    ttbb_sub: Optional[str] = "ttbb",
    rel_sys_b: float = 0.0,
    process_label_map: Optional[Dict[str, str]] = None,
    step_size: Optional[Union[int, str]] = None,
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Compute a luminosity-weighted cutflow table from config.SELECTION and
    write CSV + TeX (+ PDF if pdflatex is available).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups = _normalize_file_groups(files)
    cfg = _load_optional_config()
    lumi_value = _resolve_lumi(lumi, cfg)
    signal_key = _resolve_signal_sub(signal_sub, cfg, groups)
    cut_rows = _rows_from_config(channel, mode=mode)

    yields_by_proc: Dict[str, List[float]] = {}
    for proc, filelist in sorted(groups.items()):
        yields_by_proc[proc] = _cutflow_for_process(
            filelist,
            cut_rows,
            tree_name=tree_name,
            weight_branch=weight_branch,
            lumi=lumi_value,
            mode=mode,
            step_size=step_size,
        )

    columns, rows, proc_order = _make_table(
        cut_rows=cut_rows,
        yields_by_proc=yields_by_proc,
        signal_key_substr=signal_key,
        ttbb_key_substr=ttbb_sub,
        rel_sys_b=rel_sys_b,
    )

    if process_label_map is None and cfg is not None:
        process_label_map = process_labels
    if process_label_map:
        columns = (
            [columns[0]]
            + [_label_for(process_label_map, proc) for proc in proc_order]
            + columns[len(proc_order) + 1 :]
        )

    return _write_outputs(outdir, f"{channel}_{mode}", columns, rows)


def write_region_yield_summary(
    files: Dict[str, Union[str, Path, List[Union[str, Path]]]],
    outdir: Union[str, Path],
    channels: List[str],
    *,
    tree_name: str = "events",
    weight_branch: str = "weight_xsec",
    lumi: Optional[float] = None,
    signal_sub: Optional[str] = None,
    process_label_map: Optional[Dict[str, str]] = None,
    channel_label_map: Optional[Dict[str, str]] = None,
    step_size: Optional[Union[int, str]] = None,
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Make a region summary table with one row per channel:
    Channel | signal | backgrounds... | S/sqrt(B).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups = _normalize_file_groups(files)
    cfg = _load_optional_config()
    lumi_value = _resolve_lumi(lumi, cfg)
    signal_key = _resolve_signal_sub(signal_sub, cfg, groups)

    if process_label_map is None and cfg is not None:
        process_label_map = process_labels

    default_channel_labels = _default_channel_labels()
    all_procs = sorted(groups.keys())
    signal_proc = next((proc for proc in all_procs if signal_key and signal_key in proc), all_procs[0])
    proc_order = [signal_proc] + [proc for proc in all_procs if proc != signal_proc]

    rows: List[List[Union[str, float]]] = []
    for channel in channels:
        cut_rows = _rows_from_config(channel, mode="chain")
        final_yields = _final_yields_by_process(
            groups,
            cut_rows,
            tree_name=tree_name,
            weight_branch=weight_branch,
            lumi=lumi_value,
            step_size=step_size,
        )

        label_source = channel_label_map if channel_label_map is not None else default_channel_labels
        row: List[Union[str, float]] = [_label_for(label_source, channel)]
        proc_values = []
        for proc in proc_order:
            value = final_yields.get(proc, 0.0)
            proc_values.append(value)
            row.append(value)

        signal = proc_values[0] if proc_values else 0.0
        background = sum(proc_values[1:]) if len(proc_values) > 1 else 0.0
        row.append(signal / math.sqrt(background) if background > 0.0 else 0.0)
        rows.append(row)

    columns = ["Channel"]
    columns.extend(_label_for(process_label_map, proc) for proc in proc_order)
    columns.append(r"$S/\sqrt{B}$")

    return _write_outputs(outdir, "region_summary", columns, rows)


def _final_yields_by_process(
    groups: Dict[str, List[str]],
    cut_rows: List[CutRow],
    *,
    tree_name: str,
    weight_branch: str,
    lumi: float,
    step_size: Optional[Union[int, str]],
) -> Dict[str, float]:
    final_yields: Dict[str, float] = {}
    for proc, filelist in sorted(groups.items()):
        sums = _cutflow_for_process(
            filelist,
            cut_rows,
            tree_name=tree_name,
            weight_branch=weight_branch,
            lumi=lumi,
            mode="chain",
            step_size=step_size,
        )
        final_yields[proc] = float(sums[-1]) if sums else 0.0
    return final_yields

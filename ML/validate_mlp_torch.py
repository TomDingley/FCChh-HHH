#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

import numpy as np
import awkward as ak
import uproot
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import shap
import importlib.util


# ------------------------------
# I/O helpers
# ------------------------------
def load_channel_dataframe(indir: Path, channel: str, features, signal_key: str):
    files = sorted([p for p in Path(indir).glob(f"*_{channel}.root") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No ROOT files like '*_{channel}.root' in {indir}")

    X_list, y_list, proc_list, w_list = [], [], [], []

    for fpath in files:
        proc = fpath.stem.replace(f"_{channel}", "")
        with uproot.open(fpath) as f:
            tree = f["events"]
            cols = [ak.to_numpy(tree[v].array(library="ak")) for v in features]
            X_list.append(np.column_stack(cols))

            if "weight_xsec" in tree.keys():
                w = ak.to_numpy(tree["weight_xsec"].array(library="ak"))
            else:
                w = np.ones(tree.num_entries, dtype=float)
            w_list.append(w)

            y = np.ones(tree.num_entries, dtype=int) if signal_key in proc else np.zeros(tree.num_entries, dtype=int)
            y_list.append(y)

            proc_list.append(np.array([proc] * tree.num_entries, dtype=object))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    proc_names = np.concatenate(proc_list, axis=0)
    w_xsec = np.concatenate(w_list, axis=0)
    return X, y, proc_names, w_xsec, files


def normalise_per_process_weights(proc_names: np.ndarray, w_xsec: np.ndarray) -> np.ndarray:
    # Use |weight_xsec| for sampling/split weights to avoid signed-weight
    # cancellations and invalid probabilities in balanced selection.
    w = np.abs(w_xsec.astype(float))
    w = np.where(np.isfinite(w), w, 0.0)
    for p in np.unique(proc_names):
        mask = (proc_names == p)
        S = float(w[mask].sum())
        N = int(mask.sum())
        w[mask] = (N / S) * w[mask] if S > 0 else 1.0
    return w


def weighted_stratified_kfold_indices(y: np.ndarray,
                                      proc: np.ndarray,
                                      w: np.ndarray,
                                      n_splits: int = 5,
                                      random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    idx_all = np.arange(n)

    strata = defaultdict(list)
    for i, (yi, pi) in enumerate(zip(y, proc)):
        strata[(int(yi), pi)].append(i)

    classes = np.unique(y)
    class_weight_total = {c: float(w[y == c].sum()) for c in classes}
    target_per_fold = {c: class_weight_total[c] / n_splits for c in class_weight_total}

    fold_class_w = [{c: 0.0 for c in class_weight_total} for _ in range(n_splits)]
    folds = [[] for _ in range(n_splits)]

    for (c, p), idxs in strata.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for i in idxs:
            deficits = np.array([target_per_fold[c] - fold_class_w[k][c] for k in range(n_splits)])
            k_best = int(np.argmax(deficits))
            folds[k_best].append(i)
            fold_class_w[k_best][c] += w[i]

    splits = []
    for k in range(n_splits):
        test_idx = np.array(sorted(folds[k]))
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = idx_all[mask]
        splits.append((train_idx, test_idx))
    return splits


def make_balanced_train_selection(tr_idx, y, w_norm, rng: np.random.Generator):
    tr_sig = tr_idx[y[tr_idx] == 1]
    tr_bkg = tr_idx[y[tr_idx] == 0]
    N_bal = max(1, min(tr_sig.size, tr_bkg.size))

    sig_sel = rng.choice(tr_sig, size=N_bal, replace=(tr_sig.size < N_bal))

    p_bkg = w_norm[tr_bkg].astype(float)
    p_bkg = np.full_like(p_bkg, 1 / len(p_bkg)) if p_bkg.sum() <= 0 else (p_bkg / p_bkg.sum())
    bkg_sel = tr_bkg[rng.choice(np.arange(tr_bkg.size), size=N_bal,
                                replace=(tr_bkg.size < N_bal), p=p_bkg)]

    tr_sel = np.concatenate([sig_sel, bkg_sel])
    ytr = np.concatenate([np.ones(N_bal, int), np.zeros(N_bal, int)])
    order = rng.permutation(tr_sel.size)
    return tr_sel[order], ytr[order]


def split_train_val(tr_sel, ytr, val_frac: float, rng: np.random.Generator):
    n = tr_sel.size
    n_val = max(1, int(val_frac * n))
    perm = rng.permutation(n)
    val_local = perm[:n_val]
    tr_local = perm[n_val:]
    return tr_sel[tr_local], ytr[tr_local], tr_sel[val_local], ytr[val_local]


def build_training_composition_rows(proc_names: np.ndarray, y: np.ndarray,
                                    w_xsec: np.ndarray, w_norm: np.ndarray,
                                    train_indices_per_fold,
                                    process_label_map=None):
    proc_names = np.asarray(proc_names, dtype=object)
    y = np.asarray(y, dtype=int)
    w_xsec = np.asarray(w_xsec, dtype=float)
    w_norm = np.asarray(w_norm, dtype=float)

    rows = []
    for proc in np.unique(proc_names):
        mask = (proc_names == proc)
        raw_abs_weights = np.abs(np.where(np.isfinite(w_xsec[mask]), w_xsec[mask], 0.0))
        train_counts = []
        train_weighted_yields = []

        for train_idx in train_indices_per_fold:
            train_idx = np.asarray(train_idx, dtype=int)
            if train_idx.size == 0:
                train_counts.append(0)
                train_weighted_yields.append(0.0)
                continue
            proc_train_idx = train_idx[proc_names[train_idx] == proc]
            train_counts.append(int(proc_train_idx.size))
            train_weighted_yields.append(float(np.sum(w_norm[proc_train_idx])))

        rows.append({
            "process": str(proc),
            "process_label": map_process_label(proc, process_label_map),
            "class": "signal" if np.any(y[mask] == 1) else "background",
            "raw_events": int(np.count_nonzero(mask)),
            "raw_abs_weight_sum": float(np.sum(raw_abs_weights)),
            "train_events_mean": float(np.mean(train_counts)) if train_counts else 0.0,
            "train_events_std": float(np.std(train_counts)) if train_counts else 0.0,
            "train_weighted_yield_mean": float(np.mean(train_weighted_yields)) if train_weighted_yields else 0.0,
            "train_weighted_yield_std": float(np.std(train_weighted_yields)) if train_weighted_yields else 0.0,
            "train_events_per_fold": train_counts,
            "train_weighted_yield_per_fold": train_weighted_yields,
        })

    rows.sort(
        key=lambda row: (
            row["class"] != "signal",
            -row["train_weighted_yield_mean"],
            -row["raw_events"],
            row["process"],
        )
    )
    return rows


def _latex_escape(text: str) -> str:
    text = str(text)
    if "\\" in text or "$" in text:
        return text
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_training_composition_latex(rows, outpath: Path):
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Process & Class & Raw events & Train events & Train weighted yield \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join([
                _latex_escape(row["process_label"]),
                _latex_escape(row["class"]),
                f"{int(row['raw_events'])}",
                f"{row['train_events_mean']:.1f}",
                f"{row['train_weighted_yield_mean']:.3f}",
            ]) + r" \\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
    ])
    outpath.write_text("\n".join(lines) + "\n")


# ------------------------------
# Torch dataset + model
# ------------------------------
class StandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        mean = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1], dtype=np.float32)
        if self.with_std:
            scale = np.sqrt(X.var(axis=0))
            scale[scale < self.eps] = 1.0
        else:
            scale = np.ones(X.shape[1], dtype=np.float32)
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform.")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_

    def state_dict(self) -> dict:
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        inst = cls(with_mean=state.get("with_mean", True),
                   with_std=state.get("with_std", True),
                   eps=state.get("eps", 1e-12))
        inst.mean_ = np.asarray(state["mean"], dtype=np.float32)
        inst.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return inst


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_activation_layer(name: str) -> nn.Module:
    key = str(name).strip().lower()
    factories = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "elu": nn.ELU,
    }
    if hasattr(nn, "GELU"):
        factories["gelu"] = nn.GELU
    if key not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported activation '{name}'. Supported activations: {supported}")
    return factories[key]()


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(128, 64), activation="relu", dropout=0.0):
        super().__init__()

        layers = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(make_activation_layer(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 8192):
    model.eval()
    ds = ArrayDataset(X, np.zeros(len(X), dtype=np.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ------------------------------
# Metrics
# ------------------------------
def weighted_roc_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    w_pos = float(sample_weight[y_true == 1].sum())
    w_neg = float(sample_weight[y_true == 0].sum())
    if w_pos <= 0.0 or w_neg <= 0.0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    sample_weight = sample_weight[order]

    is_new = np.concatenate(([True], y_score[1:] != y_score[:-1]))
    group_starts = np.where(is_new)[0]

    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    cum_pos = 0.0
    cum_neg = 0.0

    for start, end in zip(group_starts, list(group_starts[1:]) + [len(y_score)]):
        w = sample_weight[start:end]
        y = y_true[start:end]
        w_pos_grp = float(w[y == 1].sum())
        w_neg_grp = float(w[y == 0].sum())

        next_cum_pos = cum_pos + w_pos_grp
        next_cum_neg = cum_neg + w_neg_grp
        tpr_next = next_cum_pos / w_pos
        fpr_next = next_cum_neg / w_neg
        auc += (fpr_next - fpr) * (tpr + tpr_next) / 2.0

        cum_pos = next_cum_pos
        cum_neg = next_cum_neg
        tpr = tpr_next
        fpr = fpr_next

    return float(auc)


def weighted_pr_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    w_pos = float(sample_weight[y_true == 1].sum())
    if w_pos <= 0.0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    sample_weight = sample_weight[order]

    tp = np.cumsum(sample_weight * (y_true == 1))
    fp = np.cumsum(sample_weight * (y_true == 0))
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / w_pos

    # integrate precision-recall curve
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def weighted_roc_curve(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    w_pos = float(sample_weight[y_true == 1].sum())
    w_neg = float(sample_weight[y_true == 0].sum())
    if w_pos <= 0.0 or w_neg <= 0.0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    sample_weight = sample_weight[order]

    tp = np.cumsum(sample_weight * (y_true == 1))
    fp = np.cumsum(sample_weight * (y_true == 0))
    tpr = tp / w_pos
    fpr = fp / w_neg
    return fpr, tpr


def weighted_pr_curve(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    w_pos = float(sample_weight[y_true == 1].sum())
    if w_pos <= 0.0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    sample_weight = sample_weight[order]

    tp = np.cumsum(sample_weight * (y_true == 1))
    fp = np.cumsum(sample_weight * (y_true == 0))
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / w_pos
    return recall, precision


def plot_loss_curves(loss_files, outpath: Path, channel_comment: str = ""):
    def fold_number_from_path(loss_path: Path, default_fold: int) -> int:
        prefix, marker, suffix = loss_path.stem.rpartition("_fold")
        if prefix and marker and suffix.endswith("_loss"):
            fold_token = suffix[:-len("_loss")]
            if fold_token.isdigit():
                return int(fold_token)
        return default_fold

    histories = []
    for idx, lf in enumerate(loss_files, start=1):
        with open(lf) as jf:
            hist = json.load(jf)

        train_loss = np.asarray(hist.get("train_loss", []), dtype=float)
        val_loss = np.asarray(hist.get("val_loss", []), dtype=float)
        n_epochs = min(train_loss.size, val_loss.size)
        if n_epochs == 0:
            continue

        fold_num = fold_number_from_path(lf, idx)
        epochs = np.arange(1, n_epochs + 1)
        histories.append((fold_num, epochs, train_loss[:n_epochs], val_loss[:n_epochs]))

    if not histories:
        return

    histories.sort(key=lambda item: item[0])

    fig, (ax_loss, ax_ratio) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.2, 5.8),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(hspace=0.012)

    for plot_idx, (fold_num, epochs, train_loss, val_loss) in enumerate(histories):
        color = {1: "black", 2: "tab:blue"}.get(fold_num, f"C{plot_idx % 10}")
        ratio = np.divide(
            train_loss,
            val_loss,
            out=np.full(train_loss.shape, np.nan, dtype=float),
            where=np.abs(val_loss) > 1e-12,
        )

        ax_loss.plot(
            epochs,
            train_loss,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"K-fold {fold_num} (Train)",
        )
        ax_loss.plot(
            epochs,
            val_loss,
            color=color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"K-fold {fold_num} (Val)",
        )
        ax_ratio.plot(epochs, ratio, color=color, linewidth=1.4, alpha=0.95)

    ax_loss.set_ylabel("Loss", loc='top')
    ax_loss.legend(
        fontsize=10,
        ncol=2,
        frameon=False,
        loc="upper right",
    )

    ax_ratio.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    ax_ratio.set_xlabel("Epoch", loc='right')
    #ax_ratio.xaxis.set_label_coords(1.0, -0.10)
    #ax_ratio.xaxis.label.set_horizontalalignment("right")
    ax_ratio.set_ylabel("Train / Val")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.75, 1.25])

    for ax in (ax_loss, ax_ratio):
        ax.tick_params(which="both", direction="in", top=True, right=True)

    apply_banner_heatmaps(ax_loss)
    add_plot_comment(ax_loss, channel_comment)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_score_hist(y_true, y_score, sample_weight, outpath: Path, channel_comment: str = ""):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    bins = np.linspace(0.0, 1.0, 51)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, hspace=0.02, wspace=0.02)
    bkg_max = _plot_normalized_hist(
        ax,
        y_score[y_true == 0],
        sample_weight[y_true == 0],
        bins,
        "Background",
        color="black",
        density=True,
    )
    sig_max = _plot_normalized_hist(
        ax,
        y_score[y_true == 1],
        sample_weight[y_true == 1],
        bins,
        "Signal",
        color="tab:blue",
        density=True,
    )
    _style_plot_axis(ax, x_label="NN Score", y_label="Density")
    _set_hist_headroom(ax, [bkg_max, sig_max], scale=1.2)
    ax.legend(frameon=False, loc="upper right")
    apply_banner(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def _plot_normalized_hist(ax, scores, weights, bins, label, *,
                          color=None, linestyle="-", linewidth=1.6, alpha=1.0,
                          density: bool = False):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan
    scores = scores[mask]
    weights = weights[mask]
    total = float(weights.sum())
    if total <= 0.0:
        return np.nan
    hist_weights = weights if density else (weights / total)
    ax.hist(
        scores,
        bins=bins,
        weights=hist_weights,
        density=density,
        histtype="step",
        linewidth=linewidth,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )
    hist, _ = np.histogram(scores, bins=bins, weights=hist_weights, density=density)
    return float(np.nanmax(hist)) if hist.size else np.nan


def _set_hist_headroom(ax: plt.Axes, maxima, scale: float = 1.2):
    finite_maxima = [m for m in maxima if np.isfinite(m) and m > 0.0]
    if not finite_maxima:
        return
    ax.set_ylim(0.0, max(finite_maxima) * scale)


def plot_oof_score_by_process(y_true, y_score, sample_weight, proc_names,
                              outpath: Path, process_label_map=None,
                              target_class: int = 1, channel_comment: str = ""):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    proc_names = np.asarray(proc_names, dtype=object)
    bins = np.linspace(0.0, 1.0, 26)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.03, wspace=0.02)
    proc_palette = list(plt.get_cmap("tab20").colors)

    proc_list = [
        p for p in np.unique(proc_names[y_true == target_class])
        if np.any((proc_names == p) & (y_true == target_class))
    ]
    proc_list = sorted(
        proc_list,
        key=lambda p: float(sample_weight[(proc_names == p) & (y_true == target_class)].sum()),
        reverse=True,
    )
    maxima = []
    for idx, proc in enumerate(proc_list):
        mask = (proc_names == proc) & (y_true == target_class)
        label = map_process_label(proc, process_label_map)
        maxima.append(_plot_normalized_hist(
            ax,
            y_score[mask],
            sample_weight[mask],
            bins,
            label,
            color=proc_palette[idx % len(proc_palette)],
        ))
    _style_plot_axis(ax, x_label="NN Score", y_label="Fraction")
    _set_hist_headroom(ax, maxima, scale=1.2)
    if proc_list:
        legend_kwargs = {
            "frameon": False,
            "loc": "upper right",
            "fontsize": 10,
        }
        if target_class == 0:
            legend_kwargs.update({
                "fontsize": 10,
                "ncol": 2,
                "columnspacing": 1.0,
                "handlelength": 1.8,
            })
        ax.legend(**legend_kwargs)

    apply_banner(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_split_score_concordance(score_payloads, outpath: Path, channel_comment: str = ""):
    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.03, wspace=0.02)
    split_styles = {
        "train": {"color": "black", "label": "Train"},
        "val": {"color": "tab:blue", "label": "Val"},
        "test": {"color": "tab:orange", "label": "Test"},
    }
    class_styles = {
        0: {"label": "Background", "linestyle": "-", "linewidth": 1.6},
        1: {"label": "Signal", "linestyle": "--", "linewidth": 1.8},
    }

    maxima = []
    for split_name, split_style in split_styles.items():
        payload = score_payloads.get(split_name, {})
        y_true = np.asarray(payload.get("y", []), dtype=int)
        y_score = np.asarray(payload.get("prob", []), dtype=float)
        weights = np.asarray(payload.get("w", []), dtype=float)
        if y_true.size == 0 or y_score.size == 0 or weights.size == 0:
            continue
        for cls, cls_style in class_styles.items():
            mask = (y_true == cls)
            if not np.any(mask):
                continue
            maxima.append(_plot_normalized_hist(
                ax,
                y_score[mask],
                weights[mask],
                bins,
                f"{split_style['label']} ({cls_style['label']})",
                color=split_style["color"],
                linestyle=cls_style["linestyle"],
                linewidth=cls_style["linewidth"],
            ))

    _style_plot_axis(ax, x_label="NN Score", y_label="Fraction")
    _set_hist_headroom(ax, maxima, scale=1.2)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    apply_banner(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_roc_pr(y_true, y_score, sample_weight, outdir: Path, channel_comment: str = ""):
    fpr, tpr = weighted_roc_curve(y_true, y_score, sample_weight)
    auc = weighted_roc_auc(y_true, y_score, sample_weight)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    ax.plot(fpr, tpr, color="black", linewidth=1.9, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _style_plot_axis(ax, x_label="False positive", y_label="True positive")
    ax.legend(frameon=False, loc="lower right")
    apply_banner_heatmaps(ax)
    add_plot_comment(ax, channel_comment)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "roc_curve.pdf", format="pdf")
    plt.close(fig)

    recall, precision = weighted_pr_curve(y_true, y_score, sample_weight)
    pr_auc = weighted_pr_auc(y_true, y_score, sample_weight)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    ax.plot(recall, precision, color="black", linewidth=1.9, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _style_plot_axis(ax, x_label="Recall", y_label="Precision")
    ax.legend(frameon=False, loc="lower left")
    apply_banner_heatmaps(ax)
    add_plot_comment(ax, channel_comment)
    fig.savefig(outdir / "pr_curve.pdf", format="pdf")
    plt.close(fig)


def _plot_correlation_matrix(matrix: np.ndarray, labels, outpath: Path, title: str):
    labels = list(labels)
    n_features = len(labels)
    max_abs = float(np.nanmax(np.abs(matrix)))
    if not np.isfinite(max_abs):
        max_abs = 1.0
    max_abs = max(max_abs, 1e-6)
    vmin, vmax = -max_abs, max_abs
    cmap = LinearSegmentedColormap.from_list("blue_white_green", ["#2166ac", "#ffffff", "#1a9850"])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if (vmin < 0.0 < vmax) else Normalize(vmin=vmin, vmax=vmax)

    fig_w = float(np.clip(5.5 + 0.18 * n_features, 7.5, 11.5))
    fig_h = float(np.clip(4.8 + 0.18 * n_features, 6.5, 11.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(n_features)
    tick_fontsize = 7 if n_features <= 12 else 6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=tick_fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    ax.set_title(title)

    text_threshold = 0.55 * max_abs
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "--" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and abs(value) >= text_threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=5, color=color)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_correlation(X: np.ndarray, features, outpath: Path, max_samples: int = 20000,
                     title: str = "Feature Correlation", label_map=None):
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(123)
        sel = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[sel]
    corr = np.corrcoef(X, rowvar=False)
    labels = map_feature_labels(features, label_map)
    _plot_correlation_matrix(corr, labels, outpath, title)


def plot_correlation_delta(corr_sig: np.ndarray, corr_bkg: np.ndarray,
                           features, outpath: Path, label_map=None):
    delta = corr_sig - corr_bkg
    labels = map_feature_labels(features, label_map)
    _plot_correlation_matrix(delta, labels, outpath, "Signal - Background Correlation")


# for banners etc, get the aesthetics module from kinematics
@lru_cache(maxsize=1)
def load_aesthetics_module():
    try:
        base = Path(__file__).resolve().parents[2]
        mod_path = base / "kinematics" / "aesthetics.py"
        spec = importlib.util.spec_from_file_location("aesthetics", mod_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

# get label map which takes variable -> nice looking variable
def load_label_map() -> dict:
    module = load_aesthetics_module()
    if module is None:
        return {}
    return getattr(module, "LABEL_MAP", {}) or {}

# get process labels
def load_process_label_map() -> dict:
    module = load_aesthetics_module()
    if module is None:
        return {}
    return getattr(module, "process_labels", {}) or {}

# apply FCC-hh banner at the top-left of the top axis.
def apply_banner_heatmaps(ax: plt.Axes, comment: str = ""):
    module = load_aesthetics_module()
    if module is None:
        return
    banner_fn = getattr(module, "banner_heatmaps", None)
    if not callable(banner_fn):
        return
    try:
        banner_fn(ax, comment=comment)
    except TypeError:
        banner_fn(ax)

# apply FCC-hh banner top-left inside the axes.
def apply_banner(ax: plt.Axes, comment: str = ""):
    module = load_aesthetics_module()
    if module is None:
        return
    banner_fn = getattr(module, "banner", None)
    if callable(banner_fn):
        try:
            banner_fn(ax, comment=comment)
        except TypeError:
            banner_fn(ax)
        return
    apply_banner_heatmaps(ax, comment=comment)

# nice to remove the GeV label for ML plots, unit not needed
def map_feature_labels(features, label_map):
    labels = [label_map.get(f, f) for f in features] if label_map else list(features)
    return [str(label).replace(" [GeV]", "") for label in labels]

def map_process_label(proc_name, process_label_map):
    if not process_label_map:
        return proc_name
    return process_label_map.get(proc_name, proc_name)

# get nice channel labels
def map_channel_comment(channel: str) -> str:
    channel_map = {
        "hadhad_MMC": r"$\tau_{\mathrm{had}}\tau_{\mathrm{had}}$",
        "lephad_MMC": r"$\tau_{\mathrm{lep}}\tau_{\mathrm{had}}$",
    }
    return channel_map.get(str(channel), str(channel))


def add_plot_comment(ax: plt.Axes, comment: str = ""):
    if not comment:
        return
    ax.text(
        1.0, 1.0,
        comment,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=11,
    )


# ensure labels are where they should be
def _style_plot_axis(ax: plt.Axes, x_label: str = None, y_label: str = None):
    if y_label is not None:
        try:
            ax.set_ylabel(y_label, loc="top")
        except TypeError:
            ax.set_ylabel(y_label)
            #ax.yaxis.set_label_coords(-0.10, 1.0)
            ax.yaxis.label.set_verticalalignment("top")
    if x_label is not None:
        try:
            ax.set_xlabel(x_label, loc="right")
        except TypeError:
            ax.set_xlabel(x_label)
            ax.xaxis.set_label_coords(1.0, -0.08)
            ax.xaxis.label.set_horizontalalignment("right")
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _annotate_panel(ax: plt.Axes, text: str):
    ax.text(
        0.98, 0.84, text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )


def _coerce_shap_base_values(expected_value, n_samples: int) -> np.ndarray:
    if hasattr(expected_value, "detach"):
        expected_value = expected_value.detach().cpu().numpy()
    base = np.asarray(expected_value, dtype=float)
    if base.ndim == 0:
        return np.full(n_samples, float(base), dtype=float)
    base = base.reshape(-1)
    if base.size == 0:
        return np.zeros(n_samples, dtype=float)
    if base.size == 1:
        return np.full(n_samples, float(base[0]), dtype=float)
    if base.size != n_samples:
        return np.full(n_samples, float(base[0]), dtype=float)
    return base.astype(float, copy=False)


def compute_shap_values(model, scaler, X, features, device,
                        max_samples: int = 1000, background: int = 200,
                        return_indices: bool = False,
                        return_base_values: bool = False):
    rng = np.random.default_rng(123)
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_eval = X[idx]
    else:
        idx = np.arange(n, dtype=int)
        X_eval = X
    if X_eval.shape[0] > background:
        bg_idx = rng.choice(X_eval.shape[0], size=background, replace=False)
        X_bg = X_eval[bg_idx]
    else:
        X_bg = X_eval

    X_eval_s = scaler.transform(X_eval)
    X_bg_s = scaler.transform(X_bg)
    if X_eval_s.ndim == 1:
        X_eval_s = X_eval_s.reshape(-1, 1)
    if X_bg_s.ndim == 1:
        X_bg_s = X_bg_s.reshape(-1, 1)
    class _ShapWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return self.base(x).unsqueeze(-1)

    model.eval()
    model.to(device)
    wrapped = _ShapWrapper(model)

    bg = torch.from_numpy(X_bg_s).to(device)
    eval_data = torch.from_numpy(X_eval_s).to(device)
    eval_data.requires_grad_(True)

    try:
        explainer = shap.DeepExplainer(wrapped, bg)
        shap_vals = explainer.shap_values(eval_data)
    except Exception:
        explainer = shap.GradientExplainer(wrapped, bg)
        shap_vals = explainer.shap_values(eval_data)
    base_values = _coerce_shap_base_values(
        getattr(explainer, "expected_value", 0.0),
        X_eval_s.shape[0],
    )

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.cpu().numpy()
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[:, :, 0]
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)
    eval_np = X_eval_s
    eval_raw = X_eval
    if shap_vals.shape[1] != eval_np.shape[1]:
        n = min(shap_vals.shape[1], eval_np.shape[1])
        shap_vals = shap_vals[:, :n]
        eval_np = eval_np[:, :n]
        eval_raw = eval_raw[:, :n]
        features = features[:n]
    extras = []
    if return_indices:
        extras.append(np.asarray(idx, dtype=int))
    if return_base_values:
        extras.append(np.asarray(base_values, dtype=float))
    if extras:
        return shap_vals, eval_np, eval_raw, features, *extras
    return shap_vals, eval_np, eval_raw, features


def plot_shap_beeswarm(shap_vals, eval_np, features, outpath: Path, label_map=None):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    labels = map_feature_labels(features, label_map)
    shap.summary_plot(shap_vals, eval_np, feature_names=labels, show=False)
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()


def plot_shap_dependence(shap_vals, eval_np, eval_raw, features, outdir: Path,
                         top_n: int = 6, label_map=None):
    outdir.mkdir(parents=True, exist_ok=True)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    labels = map_feature_labels(features, label_map)
    for i in top_idx:
        fname = features[i]
        shap.dependence_plot(i, shap_vals, eval_raw, feature_names=labels, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"shap_dependence_{fname}.pdf", format="pdf")
        plt.close()


def plot_shap_dependence_with_interactions(shap_vals, eval_np, eval_raw, features, outdir: Path,
                                           target_feature: str, top_n: int = 10, label_map=None):
    outdir.mkdir(parents=True, exist_ok=True)
    if target_feature not in features:
        raise ValueError(f"Target feature not found: {target_feature}")
    target_idx = features.index(target_feature)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    ranked = np.argsort(mean_abs)[::-1]
    top_idx = [i for i in ranked if i != target_idx][:top_n]
    labels = map_feature_labels(features, label_map)
    for i in top_idx:
        other = features[i]
        shap.dependence_plot(target_idx, shap_vals, eval_raw, feature_names=labels,
                             interaction_index=i, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"shap_dependence_{target_feature}__{other}.pdf", format="pdf")
        plt.close()
        try:
            plot_shap_2d_dependence(
                shap_vals, eval_raw, features, outdir / "shap_2d",
                target_feature=target_feature, other_feature=other,
                label_map=label_map,
            )
        except Exception:
            pass


def plot_shap_2d_dependence(shap_vals, eval_raw, features, outdir: Path,
                            target_feature: str, other_feature: str,
                            label_map=None, max_points: int = 20000,
                            cmap: str = "blue_white_black"):
    outdir.mkdir(parents=True, exist_ok=True)
    if target_feature not in features or other_feature not in features:
        raise ValueError(f"Missing feature for SHAP plot: {target_feature}, {other_feature}")
    t_idx = features.index(target_feature)
    o_idx = features.index(other_feature)

    x = eval_raw[:, t_idx]
    y = eval_raw[:, o_idx]
    z = shap_vals[:, t_idx]

    if x.shape[0] > max_points:
        rng = np.random.default_rng(123)
        sel = rng.choice(x.shape[0], size=max_points, replace=False)
        x, y, z = x[sel], y[sel], z[sel]

    labels = map_feature_labels(features, label_map)
    plt.figure(figsize=(6.5, 5.5))
    if cmap == "blue_white_black":
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        cmap = LinearSegmentedColormap.from_list(
            "blue_white_black", ["#2b83ba", "#f7f7f7", "#000000"]
        )
        max_abs = float(np.max(np.abs(z))) if z.size else 1.0
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    else:
        norm = None
    sc = plt.scatter(x, y, c=z, cmap=cmap, norm=norm, s=8, alpha=0.6)
    plt.xlabel(labels[t_idx])
    plt.ylabel(labels[o_idx])
    plt.title(f"SHAP: {labels[t_idx]} vs {labels[o_idx]}")
    plt.colorbar(sc, label="SHAP value")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_2d_{target_feature}__{other_feature}.pdf", format="pdf")
    plt.close()


def plot_shap_fold_stability(fold_importances, features, outpath: Path,
                             label_map=None, top_n: int = 20):
    if not fold_importances:
        return
    importance_matrix = np.vstack(fold_importances)
    mean_abs = np.nanmean(importance_matrix, axis=0)
    std_abs = np.nanstd(importance_matrix, axis=0)
    finite = np.isfinite(mean_abs)
    if not np.any(finite):
        return

    order = np.argsort(np.where(finite, mean_abs, -np.inf))[-top_n:][::-1]
    order = [idx for idx in order if np.isfinite(mean_abs[idx])]
    if not order:
        return

    labels = map_feature_labels(features, label_map)
    order_rev = list(reversed(order))
    y_pos = np.arange(len(order_rev))
    x_vals = mean_abs[order_rev]
    x_err = std_abs[order_rev]
    y_labels = [labels[idx] for idx in order_rev]

    fig_h = max(4.5, 0.34 * len(order_rev) + 1.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_h), constrained_layout=True)
    ax.barh(y_pos, x_vals, xerr=x_err, color="tab:blue", alpha=0.85, ecolor="black", capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"Mean $|SHAP|$ across folds")
    ax.set_title("SHAP Fold Stability")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_target_feature_shap_by_process(target_payload, target_feature: str, outpath: Path,
                                        label_map=None, process_label_map=None, top_n: int = 14):
    shap_values = np.asarray(target_payload.get("shap", []), dtype=float)
    feature_values = np.asarray(target_payload.get("value", []), dtype=float)
    proc_names = np.asarray(target_payload.get("proc", []), dtype=object)
    weights = np.asarray(target_payload.get("w", []), dtype=float)
    labels = np.asarray(target_payload.get("y", []), dtype=int)
    if shap_values.size == 0 or proc_names.size == 0:
        return

    rows = []
    for proc in np.unique(proc_names):
        mask = (proc_names == proc)
        if not np.any(mask):
            continue
        w = weights[mask]
        if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 0.0:
            continue
        shap_sel = shap_values[mask]
        value_sel = feature_values[mask]
        label_sel = labels[mask]
        mean_abs = float(np.average(np.abs(shap_sel), weights=w))
        mean_signed = float(np.average(shap_sel, weights=w))
        mean_value = float(np.average(value_sel, weights=w))
        cls = int(np.round(np.average(label_sel, weights=w))) if label_sel.size else 0
        rows.append({
            "proc": proc,
            "mean_abs": mean_abs,
            "mean_signed": mean_signed,
            "mean_value": mean_value,
            "class": cls,
        })
    if not rows:
        return

    rows = sorted(rows, key=lambda row: row["mean_abs"], reverse=True)[:top_n]
    rows_rev = list(reversed(rows))
    y_pos = np.arange(len(rows_rev))
    proc_labels = [map_process_label(row["proc"], process_label_map) for row in rows_rev]
    colors = ["tab:blue" if row["class"] == 1 else "black" for row in rows_rev]
    target_label = map_feature_labels([target_feature], label_map)[0]

    fig_h = max(4.8, 0.38 * len(rows_rev) + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, fig_h), sharey=True, constrained_layout=True)

    axes[0].barh(y_pos, [row["mean_abs"] for row in rows_rev], color=colors, alpha=0.88)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(proc_labels)
    axes[0].set_xlabel(rf"Mean $|SHAP|$ for {target_label}")
    axes[0].set_title("Process Importance")

    axes[1].barh(y_pos, [row["mean_signed"] for row in rows_rev], color=colors, alpha=0.88)
    axes[1].axvline(0.0, color="0.4", linewidth=1.0)
    axes[1].set_xlabel(rf"Mean SHAP for {target_label}")
    axes[1].set_title("Signed Contribution")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_target_feature_shap_binned(target_payload, target_feature: str, outpath: Path,
                                    label_map=None, n_bins: int = 12,
                                    channel_comment: str = ""):
    shap_values = np.asarray(target_payload.get("shap", []), dtype=float)
    feature_values = np.asarray(target_payload.get("value", []), dtype=float)
    class_labels = np.asarray(target_payload.get("y", []), dtype=int)
    weights = np.asarray(target_payload.get("w", []), dtype=float)
    if shap_values.size == 0 or feature_values.size == 0:
        return

    mask = (
        np.isfinite(shap_values)
        & np.isfinite(feature_values)
        & np.isfinite(weights)
    )
    if not np.any(mask):
        return
    shap_values = shap_values[mask]
    feature_values = feature_values[mask]
    class_labels = class_labels[mask]
    weights = weights[mask]

    if np.unique(feature_values).size < 2:
        return
    quantiles = np.quantile(feature_values, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.unique(quantiles)
    if bins.size < 3:
        lo = float(np.nanmin(feature_values))
        hi = float(np.nanmax(feature_values))
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            return
        bins = np.linspace(lo, hi, min(n_bins, 6) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    class_styles = {
        0: {"label": "Background", "color": "black"},
        1: {"label": "Signal", "color": "tab:blue"},
    }

    for cls, style in class_styles.items():
        y_vals = np.full(centers.shape, np.nan, dtype=float)
        for i in range(len(centers)):
            if i == len(centers) - 1:
                bin_mask = (feature_values >= bins[i]) & (feature_values <= bins[i + 1])
            else:
                bin_mask = (feature_values >= bins[i]) & (feature_values < bins[i + 1])
            sel = (class_labels == cls) & bin_mask
            if not np.any(sel):
                continue
            w = weights[sel]
            if float(np.nansum(w)) <= 0.0:
                continue
            y_vals[i] = float(np.average(shap_values[sel], weights=w))
        valid = np.isfinite(y_vals)
        if np.any(valid):
            ax.plot(
                centers[valid],
                y_vals[valid],
                color=style["color"],
                linewidth=1.8,
                marker="o",
                markersize=4.0,
                label=style["label"],
            )

    ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
    target_label = map_feature_labels([target_feature], label_map)[0]
    _style_plot_axis(ax, x_label=target_label, y_label="Mean SHAP")
    ax.legend(frameon=False, loc="upper right")
    apply_banner(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_target_feature_shap_waterfalls(target_payload, features, target_feature: str, outdir: Path,
                                        label_map=None, process_label_map=None,
                                        max_display: int = 12):
    shap_rows = np.asarray(target_payload.get("shap_row", []), dtype=float)
    feature_rows = np.asarray(target_payload.get("feature_row", []), dtype=float)
    base_values = np.asarray(target_payload.get("base_value", []), dtype=float)
    target_shap = np.asarray(target_payload.get("shap", []), dtype=float)
    class_labels = np.asarray(target_payload.get("y", []), dtype=int)
    proc_names = np.asarray(target_payload.get("proc", []), dtype=object)
    if (
        shap_rows.ndim != 2
        or feature_rows.ndim != 2
        or shap_rows.shape[0] == 0
        or feature_rows.shape[0] != shap_rows.shape[0]
        or shap_rows.shape[1] != len(features)
    ):
        return

    feature_labels = map_feature_labels(features, label_map)
    target_label = map_feature_labels([target_feature], label_map)[0]
    outdir.mkdir(parents=True, exist_ok=True)

    selections = [
        (1, "signal", "positive", np.argmax, "Largest positive"),
        (1, "signal", "negative", np.argmin, "Largest negative"),
        (0, "background", "positive", np.argmax, "Largest positive"),
        (0, "background", "negative", np.argmin, "Largest negative"),
    ]
    used_indices = set()
    for cls, cls_name, suffix, selector, descriptor in selections:
        cls_idx = np.where(class_labels == cls)[0]
        if cls_idx.size == 0:
            continue
        local_idx = int(selector(target_shap[cls_idx]))
        event_idx = int(cls_idx[local_idx])
        if event_idx in used_indices:
            continue
        used_indices.add(event_idx)

        explanation = shap.Explanation(
            values=shap_rows[event_idx],
            base_values=float(base_values[event_idx]) if base_values.size else 0.0,
            data=feature_rows[event_idx],
            feature_names=feature_labels,
        )
        try:
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
        except Exception:
            try:
                shap.waterfall_plot(explanation, max_display=max_display, show=False)
            except TypeError:
                shap.waterfall_plot(explanation, max_display=max_display)

        fig = plt.gcf()
        fig.set_size_inches(8.6, 5.8)
        proc_label = map_process_label(proc_names[event_idx], process_label_map)
        fig.suptitle(
            f"{descriptor} {target_label} contribution ({cls_name})\n{proc_label}",
            y=0.99,
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            outdir / f"shap_waterfall_{target_feature}_{cls_name}_{suffix}.pdf",
            format="pdf",
        )
        plt.close(fig)


def safe_filename_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "item"


def plot_process_feature_shap_summaries(process_feature_rows, features, outdir: Path,
                                        label_map=None, process_label_map=None,
                                        top_n: int = 10):
    if not process_feature_rows:
        return

    labels = map_feature_labels(features, label_map)
    outdir.mkdir(parents=True, exist_ok=True)
    for row in process_feature_rows:
        mean_abs = np.asarray(row.get("mean_abs", []), dtype=float)
        if mean_abs.size == 0:
            continue
        finite = np.isfinite(mean_abs)
        if not np.any(finite):
            continue

        order = np.argsort(np.where(finite, mean_abs, -np.inf))[-top_n:][::-1]
        order = [idx for idx in order if np.isfinite(mean_abs[idx])]
        if not order:
            continue

        order_rev = list(reversed(order))
        y_pos = np.arange(len(order_rev))
        x_vals = mean_abs[order_rev]
        y_labels = [labels[idx] for idx in order_rev]
        color = "tab:blue" if int(row.get("class", 0)) == 1 else "black"
        proc_label = map_process_label(row.get("proc", "process"), process_label_map)

        fig_h = max(4.5, 0.34 * len(order_rev) + 1.2)
        fig, ax = plt.subplots(figsize=(7.8, fig_h), constrained_layout=True)
        ax.barh(y_pos, x_vals, color=color, alpha=0.88)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        _style_plot_axis(ax, x_label=r"Mean $|SHAP|$", y_label="Feature")
        apply_banner_heatmaps(ax, comment=proc_label)
        fig.savefig(
            outdir / f"shap_process_summary_{safe_filename_token(row.get('proc', 'process'))}.pdf",
            format="pdf",
        )
        plt.close(fig)


def plot_feature_importance(imp_rows, outpath: Path, top_n: int = 30):
    if not imp_rows:
        return
    rows = sorted(imp_rows, key=lambda r: r["importance_mean"], reverse=True)[:top_n]
    feats = [r["feature"] for r in rows][::-1]
    means = [r["importance_mean"] for r in rows][::-1]
    stds = [r["importance_std"] for r in rows][::-1]

    plt.figure(figsize=(7.0, max(4.0, 0.25 * len(feats))))
    plt.barh(feats, means, xerr=stds, alpha=0.8)
    plt.xlabel("Permutation Importance (Δ metric)")
    plt.title(f"Top {len(feats)} Features")
    plt.grid(True, axis="x", alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()


def find_feature_family_indices(features, token: str):
    token = str(token).lower()
    return np.array(
        [idx for idx, feat in enumerate(features) if token in str(feat).lower()],
        dtype=int,
    )


def permute_feature_group(X, feature_indices, rng, groups=None):
    feature_indices = np.asarray(feature_indices, dtype=int)
    if feature_indices.size == 0:
        return np.array(X, copy=True)

    X_perm = np.array(X, copy=True)
    if groups is None:
        perm = rng.permutation(X_perm.shape[0])
        X_perm[np.ix_(np.arange(X_perm.shape[0]), feature_indices)] = X_perm[np.ix_(perm, feature_indices)]
        return X_perm

    groups = np.asarray(groups, dtype=object)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        if idx.size <= 1:
            continue
        perm = idx[rng.permutation(idx.size)]
        X_perm[np.ix_(idx, feature_indices)] = X_perm[np.ix_(perm, feature_indices)]
    return X_perm


def compute_grouped_permutation_diagnostic(model, X, y, w, proc_names, feature_indices,
                                           feature_names, device, metric_fn, metric_name,
                                           n_repeats=5, batch_size=8192, rng=None,
                                           baseline_prob=None):
    feature_indices = np.asarray(feature_indices, dtype=int)
    if feature_indices.size == 0:
        return None

    rng = rng or np.random.default_rng(123)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    proc_names = np.asarray(proc_names, dtype=object)
    baseline_prob = np.asarray(
        predict_proba(model, X, device, batch_size) if baseline_prob is None else baseline_prob,
        dtype=float,
    )
    overall_baseline = metric_fn(y, baseline_prob, w)
    if not np.isfinite(overall_baseline):
        return None

    process_payload = {}
    background_processes = sorted(
        np.unique(proc_names[y == 0]),
        key=lambda proc: float(w[proc_names == proc].sum()),
        reverse=True,
    )
    for proc in background_processes:
        mask = (y == 1) | (proc_names == proc)
        if np.unique(y[mask]).size < 2:
            continue
        baseline = metric_fn(y[mask], baseline_prob[mask], w[mask])
        if not np.isfinite(baseline):
            continue
        process_payload[proc] = {
            "baseline": float(baseline),
            "drops": [],
        }

    overall_drops = []
    for _ in range(int(n_repeats)):
        X_perm = permute_feature_group(X, feature_indices, rng, groups=proc_names)
        perm_prob = predict_proba(model, X_perm, device, batch_size)
        perm_score = metric_fn(y, perm_prob, w)
        if np.isfinite(perm_score):
            overall_drops.append(float(overall_baseline - perm_score))

        for proc, payload in process_payload.items():
            mask = (y == 1) | (proc_names == proc)
            proc_score = metric_fn(y[mask], perm_prob[mask], w[mask])
            if np.isfinite(proc_score):
                payload["drops"].append(float(payload["baseline"] - proc_score))

    return {
        "metric": metric_name,
        "feature_names": [str(name) for name in feature_names],
        "overall_baseline": float(overall_baseline),
        "overall_drops": overall_drops,
        "per_process": process_payload,
    }


def aggregate_grouped_permutation_diagnostics(fold_payloads):
    payloads = [payload for payload in fold_payloads if payload]
    if not payloads:
        return None

    overall_baselines = []
    overall_drops = []
    process_baselines = defaultdict(list)
    process_drops = defaultdict(list)
    metric_name = payloads[0]["metric"]
    feature_names = payloads[0]["feature_names"]

    for payload in payloads:
        if np.isfinite(payload.get("overall_baseline", np.nan)):
            overall_baselines.append(float(payload["overall_baseline"]))
        overall_drops.extend([
            float(val) for val in payload.get("overall_drops", [])
            if np.isfinite(val)
        ])
        for proc, proc_payload in payload.get("per_process", {}).items():
            baseline = float(proc_payload.get("baseline", np.nan))
            if np.isfinite(baseline):
                process_baselines[proc].append(baseline)
            process_drops[proc].extend([
                float(val) for val in proc_payload.get("drops", [])
                if np.isfinite(val)
            ])

    rows = []
    for proc, drops in process_drops.items():
        if not drops:
            continue
        rows.append({
            "proc": proc,
            "baseline_mean": float(np.mean(process_baselines.get(proc, [np.nan]))),
            "baseline_std": float(np.std(process_baselines.get(proc, [np.nan]))),
            "drop_mean": float(np.mean(drops)),
            "drop_std": float(np.std(drops)),
            "n": int(len(drops)),
        })
    rows.sort(key=lambda row: row["drop_mean"], reverse=True)

    return {
        "metric": metric_name,
        "feature_names": feature_names,
        "overall": {
            "baseline_mean": float(np.mean(overall_baselines)) if overall_baselines else float("nan"),
            "baseline_std": float(np.std(overall_baselines)) if overall_baselines else float("nan"),
            "drop_mean": float(np.mean(overall_drops)) if overall_drops else float("nan"),
            "drop_std": float(np.std(overall_drops)) if overall_drops else float("nan"),
            "n": int(len(overall_drops)),
        },
        "per_process": rows,
    }


def plot_grouped_permutation_diagnostics(diagnostics, outpath: Path, process_label_map=None,
                                         channel_comment: str = ""):
    if not diagnostics:
        return
    rows = [row for row in diagnostics.get("per_process", []) if np.isfinite(row.get("drop_mean", np.nan))]
    if not rows:
        return

    rows_rev = list(reversed(rows))
    y_pos = np.arange(len(rows_rev))
    labels = [map_process_label(row["proc"], process_label_map) for row in rows_rev]
    means = [row["drop_mean"] for row in rows_rev]
    stds = [row["drop_std"] for row in rows_rev]
    overall_drop = float(diagnostics.get("overall", {}).get("drop_mean", np.nan))
    metric_name = str(diagnostics.get("metric", "auc")).lower()
    metric_label = "AUC" if metric_name == "auc" else "PR AUC"

    fig_h = max(4.8, 0.36 * len(rows_rev) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_h), constrained_layout=True)
    ax.barh(y_pos, means, xerr=stds, color="black", alpha=0.88, ecolor="tab:blue", capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    _style_plot_axis(ax, x_label=f"MMC family Δ {metric_label}", y_label="Background process")
    if np.isfinite(overall_drop):
        ax.axvline(overall_drop, color="tab:blue", linestyle="--", linewidth=1.3, label="All processes")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    apply_banner_heatmaps(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def weighted_calibration_curve(y_true, y_prob, sample_weight, n_bins: int = 8):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    pred_mean = []
    obs_frac = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        w = sample_weight[mask]
        if float(np.nansum(w)) <= 0.0:
            continue
        pred_mean.append(float(np.average(y_prob[mask], weights=w)))
        obs_frac.append(float(np.average(y_true[mask], weights=w)))
    return np.asarray(pred_mean, dtype=float), np.asarray(obs_frac, dtype=float)


def compute_feature_quantile_diagnostics(feature_values, y_true, y_prob, sample_weight,
                                         n_quantiles: int = 5, calibration_bins: int = 8):
    feature_values = np.asarray(feature_values, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    mask = (
        np.isfinite(feature_values)
        & np.isfinite(y_prob)
        & np.isfinite(sample_weight)
    )
    if not np.any(mask):
        return None
    feature_values = feature_values[mask]
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    sample_weight = sample_weight[mask]

    quantiles = np.quantile(feature_values, np.linspace(0.0, 1.0, int(n_quantiles) + 1))
    edges = np.unique(quantiles)
    if edges.size < 3:
        lo = float(np.nanmin(feature_values))
        hi = float(np.nanmax(feature_values))
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            return None
        edges = np.linspace(lo, hi, min(int(n_quantiles), 5) + 1)

    rows = []
    for idx in range(len(edges) - 1):
        if idx == len(edges) - 2:
            sel = (feature_values >= edges[idx]) & (feature_values <= edges[idx + 1])
        else:
            sel = (feature_values >= edges[idx]) & (feature_values < edges[idx + 1])
        if not np.any(sel):
            continue
        w_sel = sample_weight[sel]
        if float(np.nansum(w_sel)) <= 0.0:
            continue
        pred_mean, obs_frac = weighted_calibration_curve(
            y_true[sel], y_prob[sel], w_sel, n_bins=calibration_bins
        )
        rows.append({
            "quantile": int(len(rows) + 1),
            "lower": float(edges[idx]),
            "upper": float(edges[idx + 1]),
            "n_events": int(np.count_nonzero(sel)),
            "weight_sum": float(np.sum(w_sel)),
            "signal_fraction": float(np.average(y_true[sel], weights=w_sel)),
            "roc_auc": float(weighted_roc_auc(y_true[sel], y_prob[sel], w_sel)),
            "pr_auc": float(weighted_pr_auc(y_true[sel], y_prob[sel], w_sel)),
            "calibration": {
                "pred_mean": pred_mean.tolist(),
                "obs_frac": obs_frac.tolist(),
            },
        })
    if not rows:
        return None
    return {
        "edges": [float(edge) for edge in edges],
        "bins": rows,
    }


def plot_feature_performance_by_quantile(diagnostics, outpath: Path, feature_label: str,
                                         channel_comment: str = ""):
    if not diagnostics:
        return
    rows = diagnostics.get("bins", [])
    if not rows:
        return

    x = np.arange(len(rows))
    roc_vals = np.array([row.get("roc_auc", np.nan) for row in rows], dtype=float)
    pr_vals = np.array([row.get("pr_auc", np.nan) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    roc_mask = np.isfinite(roc_vals)
    pr_mask = np.isfinite(pr_vals)
    if np.any(roc_mask):
        ax.plot(x[roc_mask], roc_vals[roc_mask], color="black", linewidth=1.8, marker="o", label="ROC AUC")
    if np.any(pr_mask):
        ax.plot(x[pr_mask], pr_vals[pr_mask], color="tab:blue", linewidth=1.8, marker="o", label="PR AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{idx + 1}" for idx in range(len(rows))])
    ax.set_ylim(0.0, 1.02)
    _style_plot_axis(ax, x_label=f"{feature_label} quantile", y_label="Metric")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    apply_banner_heatmaps(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_feature_calibration_by_quantile(diagnostics, outpath: Path,
                                         channel_comment: str = ""):
    if not diagnostics:
        return
    rows = diagnostics.get("bins", [])
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = plt.get_cmap("cividis")(np.linspace(0.15, 0.9, max(len(rows), 2)))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--", linewidth=1.0)
    for color, row in zip(colors, rows):
        cal = row.get("calibration", {})
        pred_mean = np.asarray(cal.get("pred_mean", []), dtype=float)
        obs_frac = np.asarray(cal.get("obs_frac", []), dtype=float)
        if pred_mean.size == 0 or obs_frac.size == 0:
            continue
        ax.plot(
            pred_mean,
            obs_frac,
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=4.0,
            label=f"Q{row.get('quantile', 0)}",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _style_plot_axis(ax, x_label="Mean NN Score", y_label="Signal fraction")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    apply_banner_heatmaps(ax)
    add_plot_comment(ax, channel_comment)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def weighted_log_loss(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.average(loss, weights=sample_weight))


def weighted_brier(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    return float(np.average((y_prob - y_true) ** 2, weights=sample_weight))


def confusion_from_threshold(y_true, y_prob, sample_weight, threshold: float):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = float(sample_weight[(y_true == 1) & (y_pred == 1)].sum())
    fp = float(sample_weight[(y_true == 0) & (y_pred == 1)].sum())
    tn = float(sample_weight[(y_true == 0) & (y_pred == 0)].sum())
    fn = float(sample_weight[(y_true == 1) & (y_pred == 0)].sum())
    return tp, fp, tn, fn


def rate_metrics(tp, fp, tn, fn):
    tpr = tp / max(tp + fn, 1e-12)
    fpr = fp / max(fp + tn, 1e-12)
    tnr = tn / max(tn + fp, 1e-12)
    fnr = fn / max(fn + tp, 1e-12)
    precision = tp / max(tp + fp, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1e-12)
    bal_acc = 0.5 * (tpr + tnr)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
        "precision": precision, "accuracy": accuracy,
        "balanced_accuracy": bal_acc, "f1": f1,
    }


def ks_statistic(y_true, y_prob, sample_weight):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    sample_weight = sample_weight[order]

    w_pos = sample_weight * (y_true == 1)
    w_neg = sample_weight * (y_true == 0)
    cdf_pos = np.cumsum(w_pos) / max(w_pos.sum(), 1e-12)
    cdf_neg = np.cumsum(w_neg) / max(w_neg.sum(), 1e-12)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def best_threshold_by_metric(y_true, y_prob, sample_weight, metric: str):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    thresholds = np.unique(y_prob)
    best_t = 0.5
    best_val = -np.inf
    for t in thresholds:
        tp, fp, tn, fn = confusion_from_threshold(y_true, y_prob, sample_weight, float(t))
        rates = rate_metrics(tp, fp, tn, fn)
        if metric == "youden":
            val = rates["tpr"] - rates["fpr"]
        elif metric == "f1":
            val = rates["f1"]
        else:
            val = rates["balanced_accuracy"]
        if val > best_val:
            best_val = val
            best_t = float(t)
    return best_t, best_val


def compute_metrics(y_true, y_prob, sample_weight):
    auc = weighted_roc_auc(y_true, y_prob, sample_weight)
    pr_auc = weighted_pr_auc(y_true, y_prob, sample_weight)
    logloss = weighted_log_loss(y_true, y_prob, sample_weight)
    brier = weighted_brier(y_true, y_prob, sample_weight)
    ks = ks_statistic(y_true, y_prob, sample_weight)

    tp, fp, tn, fn = confusion_from_threshold(y_true, y_prob, sample_weight, 0.5)
    rates_05 = rate_metrics(tp, fp, tn, fn)

    t_youden, youden = best_threshold_by_metric(y_true, y_prob, sample_weight, "youden")
    tp, fp, tn, fn = confusion_from_threshold(y_true, y_prob, sample_weight, t_youden)
    rates_youden = rate_metrics(tp, fp, tn, fn)

    t_f1, f1_best = best_threshold_by_metric(y_true, y_prob, sample_weight, "f1")
    tp, fp, tn, fn = confusion_from_threshold(y_true, y_prob, sample_weight, t_f1)
    rates_f1 = rate_metrics(tp, fp, tn, fn)

    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "log_loss": logloss,
        "brier": brier,
        "ks": ks,
        "threshold_0p5": {"threshold": 0.5, "metrics": rates_05},
        "threshold_youden": {"threshold": t_youden, "youden": youden, "metrics": rates_youden},
        "threshold_f1": {"threshold": t_f1, "f1": f1_best, "metrics": rates_f1},
    }


def permutation_importance(model, X, y, w, device, metric_fn, n_repeats=5,
                           batch_size=8192, rng=None, feature_indices=None):
    rng = rng or np.random.default_rng(123)
    baseline = metric_fn(y, predict_proba(model, X, device, batch_size), w)
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    feature_indices = np.asarray(feature_indices, dtype=int)
    n_features = feature_indices.size
    drops = np.zeros((n_features, n_repeats), dtype=float)

    for j, feat_idx in enumerate(feature_indices):
        for r in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, feat_idx] = Xp[perm, feat_idx]
            score = metric_fn(y, predict_proba(model, Xp, device, batch_size), w)
            drops[j, r] = baseline - score
    return baseline, drops


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def load_fold_model_and_scaler(ch_model_dir: Path, model_name: str,
                               best_params: dict, n_in: int, device: torch.device):
    model = MLP(
        n_in=n_in,
        hidden=tuple(best_params["hidden"]),
        activation=best_params.get("activation", "relu"),
        dropout=best_params.get("dropout", 0.0),
    )
    state = torch.load(ch_model_dir / f"{model_name}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    scaler_path = ch_model_dir / f"{model_name}_scaler.pt"
    try:
        scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=True)
    except Exception:
        scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)
    return model, scaler


def concatenate_score_payload(payload: dict) -> dict:
    out = {}
    for key, chunks in payload.items():
        if not chunks:
            if key == "proc":
                out[key] = np.array([], dtype=object)
            elif key == "y":
                out[key] = np.array([], dtype=int)
            else:
                out[key] = np.array([], dtype=float)
            continue
        out[key] = np.concatenate(chunks, axis=0)
    return out


def collect_fold_shap_diagnostics(ch_model_dir: Path, tag: str, best_params: dict,
                                  X: np.ndarray, y: np.ndarray, proc_names: np.ndarray,
                                  w_norm: np.ndarray, splits, features, device,
                                  target_feature: str = "weighted_MMC_para_perp_vispTcal"):
    fold_importances = []
    target_payload = {
        "shap": [],
        "value": [],
        "y": [],
        "w": [],
        "proc": [],
        "shap_row": [],
        "feature_row": [],
        "base_value": [],
    }
    feature_index = {feat: idx for idx, feat in enumerate(features)}
    shap_features = None
    process_feature_accum = {}

    for fold, (_, te_idx) in enumerate(splits, start=1):
        model_name = f"{tag}_fold{fold}"
        model, scaler = load_fold_model_and_scaler(
            ch_model_dir, model_name, best_params, X.shape[1], device
        )
        shap_vals, _, shap_raw, shap_feats, eval_idx, base_values = compute_shap_values(
            model, scaler, X[te_idx], list(features), device,
            return_indices=True, return_base_values=True
        )
        if shap_vals.size == 0:
            continue
        if shap_features is None:
            shap_features = list(shap_feats)

        fold_mean_abs = np.full(len(features), np.nan, dtype=float)
        for local_idx, feat in enumerate(shap_feats):
            if feat in feature_index:
                fold_mean_abs[feature_index[feat]] = float(np.mean(np.abs(shap_vals[:, local_idx])))
        fold_importances.append(fold_mean_abs)

        eval_idx = np.asarray(eval_idx, dtype=int)
        eval_proc = np.asarray(proc_names[te_idx][eval_idx], dtype=object)
        eval_w = np.asarray(w_norm[te_idx][eval_idx], dtype=float)
        eval_y = np.asarray(y[te_idx][eval_idx], dtype=int)
        for proc in np.unique(eval_proc):
            mask = (eval_proc == proc)
            if not np.any(mask):
                continue
            w = eval_w[mask]
            weight_sum = float(np.nansum(w))
            if weight_sum <= 0.0:
                continue
            proc_payload = process_feature_accum.setdefault(
                proc,
                {
                    "weighted_abs_sum": np.zeros(len(features), dtype=float),
                    "weight_sum": 0.0,
                    "class": int(np.round(np.average(eval_y[mask], weights=w))) if np.any(mask) else 0,
                },
            )
            for local_idx, feat in enumerate(shap_feats):
                global_idx = feature_index.get(feat)
                if global_idx is None:
                    continue
                proc_payload["weighted_abs_sum"][global_idx] += float(
                    np.sum(np.abs(shap_vals[mask, local_idx]) * w)
                )
            proc_payload["weight_sum"] += weight_sum

        if target_feature in shap_feats:
            target_idx = shap_feats.index(target_feature)
            target_payload["shap"].append(np.asarray(shap_vals[:, target_idx], dtype=float))
            target_payload["value"].append(np.asarray(shap_raw[:, target_idx], dtype=float))
            target_payload["y"].append(eval_y)
            target_payload["w"].append(eval_w)
            target_payload["proc"].append(eval_proc)
            target_payload["shap_row"].append(np.asarray(shap_vals, dtype=float))
            target_payload["feature_row"].append(np.asarray(shap_raw, dtype=float))
            target_payload["base_value"].append(np.asarray(base_values, dtype=float))

    process_feature_rows = []
    for proc, payload in process_feature_accum.items():
        weight_sum = float(payload["weight_sum"])
        if weight_sum <= 0.0:
            continue
        process_feature_rows.append({
            "proc": proc,
            "class": int(payload["class"]),
            "mean_abs": (payload["weighted_abs_sum"] / weight_sum).tolist(),
            "weight_sum": weight_sum,
        })
    process_feature_rows.sort(
        key=lambda row: (
            row["class"] != 1,
            -float(np.nanmax(np.asarray(row["mean_abs"], dtype=float))),
            row["proc"],
        )
    )

    return (
        fold_importances,
        concatenate_score_payload(target_payload),
        shap_features,
        process_feature_rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing preprocessed ROOTs")
    ap.add_argument("--features-config", default="train_features.json",
                    help="JSON file with training variables per channel")
    ap.add_argument("--model-dir", default="trained_models_torch", help="Where models were saved")
    ap.add_argument("--outdir", default="validation_torch", help="Output directory for validation")
    ap.add_argument("--signal", default="mgp8_pp_hhh_84TeV", help="Substring to identify signal process")
    ap.add_argument("--tag", default="mlp_torch", help="Model tag/prefix for filenames")
    ap.add_argument("--splits", type=int, default=2, help="Number of folds")
    ap.add_argument("--seed", type=int, default=42, help="Random state")
    ap.add_argument("--channels", nargs="+", default=["hadhad", "lephad"])
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--perm-repeats", type=int, default=2)
    ap.add_argument("--perm-metric", choices=["auc", "pr"], default="auc")
    ap.add_argument("--perm-max-features", type=int, default=30,
                    help="Limit number of features for permutation importance")
    ap.add_argument("--quick", action="store_true",
                    help="Run fast validation only: metrics, ROC, and score plots")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    feat_cfg = None
    if args.features_config:
        with open(args.features_config) as jf:
            feat_cfg = json.load(jf)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metric_fn = weighted_roc_auc if args.perm_metric == "auc" else weighted_pr_auc
    label_map = load_label_map()
    process_label_map = load_process_label_map()

    for channel in args.channels:
        ch_model_dir = Path(args.model_dir).resolve() / channel
        meta_files = sorted(ch_model_dir.glob(f"{args.tag}_fold*.json"))
        if meta_files:
            with open(meta_files[0]) as jf:
                meta = json.load(jf)
            feats = list(meta["features"])
        else:
            if feat_cfg is None:
                raise FileNotFoundError(f"No model metadata found in {ch_model_dir} and no features config provided.")
            if "channels" not in feat_cfg or channel not in feat_cfg["channels"]:
                raise KeyError(f"Channel '{channel}' not found in {args.features_config}")
            feats = list(feat_cfg["channels"][channel]["features"])
        weighted_mmc_idx = feats.index("weighted_MMC_para_perp_vispTcal") if "weighted_MMC_para_perp_vispTcal" in feats else None
        mmc_family_idx = find_feature_family_indices(feats, "mmc")
        mmc_family_feature_names = [feats[idx] for idx in mmc_family_idx]

        X, y, proc_names, w_xsec, files = load_channel_dataframe(
            Path(args.indir).resolve(), channel, feats, args.signal
        )
        w_norm = normalise_per_process_weights(proc_names, w_xsec)
        splits = weighted_stratified_kfold_indices(
            y=y, proc=proc_names, w=w_norm, n_splits=args.splits, random_state=args.seed
        )

        best_params_path = ch_model_dir / f"{args.tag}_best_params.json"
        with open(best_params_path) as jf:
            best_params = json.load(jf)

        rng = np.random.default_rng(args.seed)
        fold_metrics = {"train": [], "val": [], "test": []}
        importance_per_fold = []
        oof_y = []
        oof_prob = []
        oof_w = []
        oof_proc = []
        oof_weighted_mmc = []
        mmc_permutation_per_fold = []
        train_indices_per_fold = []
        split_scores = {
            "train": {"y": [], "prob": [], "w": [], "proc": []},
            "val": {"y": [], "prob": [], "w": [], "proc": []},
            "test": {"y": [], "prob": [], "w": [], "proc": []},
        }
        shap_model = None
        shap_scaler = None
        shap_X = None

        for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
            model_name = f"{args.tag}_fold{fold}"
            model, scaler = load_fold_model_and_scaler(
                ch_model_dir, model_name, best_params, X.shape[1], device
            )

            tr_sel, ytr = make_balanced_train_selection(tr_idx, y, w_norm, rng)
            tr_sel2, ytr2, va_sel, yva = split_train_val(tr_sel, ytr, best_params["val_frac"], rng)
            train_indices_per_fold.append(np.asarray(tr_sel2, dtype=int))

            Xtr = scaler.transform(X[tr_sel2])
            Xva = scaler.transform(X[va_sel])
            Xte = scaler.transform(X[te_idx])

            ytr_prob = predict_proba(model, Xtr, device=device, batch_size=args.batch_size)
            yva_prob = predict_proba(model, Xva, device=device, batch_size=args.batch_size)
            yte_prob = predict_proba(model, Xte, device=device, batch_size=args.batch_size)

            fold_metrics["train"].append(
                compute_metrics(ytr2, ytr_prob, w_norm[tr_sel2])
            )
            split_scores["train"]["y"].append(ytr2)
            split_scores["train"]["prob"].append(ytr_prob)
            split_scores["train"]["w"].append(w_norm[tr_sel2])
            split_scores["train"]["proc"].append(proc_names[tr_sel2])
            fold_metrics["val"].append(
                compute_metrics(yva, yva_prob, w_norm[va_sel])
            )
            split_scores["val"]["y"].append(yva)
            split_scores["val"]["prob"].append(yva_prob)
            split_scores["val"]["w"].append(w_norm[va_sel])
            split_scores["val"]["proc"].append(proc_names[va_sel])
            fold_metrics["test"].append(
                compute_metrics(y[te_idx], yte_prob, w_norm[te_idx])
            )
            split_scores["test"]["y"].append(y[te_idx])
            split_scores["test"]["prob"].append(yte_prob)
            split_scores["test"]["w"].append(w_norm[te_idx])
            split_scores["test"]["proc"].append(proc_names[te_idx])
            oof_y.append(y[te_idx])
            oof_prob.append(yte_prob)
            oof_w.append(w_norm[te_idx])
            oof_proc.append(proc_names[te_idx])
            if weighted_mmc_idx is not None:
                oof_weighted_mmc.append(np.asarray(X[te_idx, weighted_mmc_idx], dtype=float))

            if shap_model is None:
                shap_model = model
                shap_scaler = scaler
                shap_X = X[te_idx]

            if not args.quick:
                if args.perm_max_features is not None:
                    feat_idx = np.arange(min(args.perm_max_features, Xte.shape[1]))
                else:
                    feat_idx = np.arange(Xte.shape[1])

                baseline, drops = permutation_importance(
                    model, Xte, y[te_idx], w_norm[te_idx], device, metric_fn,
                    n_repeats=args.perm_repeats, batch_size=args.batch_size,
                    rng=np.random.default_rng(args.seed + fold),
                    feature_indices=feat_idx,
                )
                importance_per_fold.append({"baseline": baseline, "drops": drops, "feat_idx": feat_idx})
                if mmc_family_idx.size:
                    mmc_payload = compute_grouped_permutation_diagnostic(
                        model,
                        Xte,
                        y[te_idx],
                        w_norm[te_idx],
                        proc_names[te_idx],
                        mmc_family_idx,
                        mmc_family_feature_names,
                        device,
                        metric_fn,
                        args.perm_metric,
                        n_repeats=args.perm_repeats,
                        batch_size=args.batch_size,
                        rng=np.random.default_rng(args.seed + 1000 + fold),
                        baseline_prob=yte_prob,
                    )
                    if mmc_payload is not None:
                        mmc_permutation_per_fold.append(mmc_payload)

        # Aggregate metrics
        summary = {
            "channel": channel,
            "features": feats,
            "files": [str(p) for p in files],
            "splits": args.splits,
            "seed": args.seed,
            "quick_mode": bool(args.quick),
            "metrics": {
                "train": aggregate_metrics([{
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "ks": m["ks"],
                    "accuracy": m["threshold_0p5"]["metrics"]["accuracy"],
                    "balanced_accuracy": m["threshold_0p5"]["metrics"]["balanced_accuracy"],
                    "f1": m["threshold_0p5"]["metrics"]["f1"],
                } for m in fold_metrics["train"]]),
                "val": aggregate_metrics([{
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "ks": m["ks"],
                    "accuracy": m["threshold_0p5"]["metrics"]["accuracy"],
                    "balanced_accuracy": m["threshold_0p5"]["metrics"]["balanced_accuracy"],
                    "f1": m["threshold_0p5"]["metrics"]["f1"],
                } for m in fold_metrics["val"]]),
                "test": aggregate_metrics([{
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "ks": m["ks"],
                    "accuracy": m["threshold_0p5"]["metrics"]["accuracy"],
                    "balanced_accuracy": m["threshold_0p5"]["metrics"]["balanced_accuracy"],
                    "f1": m["threshold_0p5"]["metrics"]["f1"],
                } for m in fold_metrics["test"]]),
            },
            "thresholds": {
                "val": [m["threshold_youden"] for m in fold_metrics["val"]],
                "test": [m["threshold_youden"] for m in fold_metrics["test"]],
            },
        }

        ch_out = outdir / channel
        ch_out.mkdir(parents=True, exist_ok=True)
        training_composition_rows = build_training_composition_rows(
            proc_names,
            y,
            w_xsec,
            w_norm,
            train_indices_per_fold,
            process_label_map=process_label_map,
        )
        with open(ch_out / "validation_summary.json", "w") as jf:
            json.dump(summary, jf, indent=2)
        with open(ch_out / "training_composition.json", "w") as jf:
            json.dump({
                "channel": channel,
                "n_folds": args.splits,
                "weight_definition": "train_weighted_yield uses normalise_per_process_weights(abs(weight_xsec)) on the actual training subset after the train/val split, averaged over folds",
                "rows": [{
                    "process_label": row["process_label"],
                    "process": row["process"],
                    "class": row["class"],
                    "raw_events": row["raw_events"],
                    "train_events_mean": row["train_events_mean"],
                    "train_weighted_yield_mean": row["train_weighted_yield_mean"],
                    "train_events_per_fold": row["train_events_per_fold"],
                    "train_weighted_yield_per_fold": row["train_weighted_yield_per_fold"],
                } for row in training_composition_rows],
            }, jf, indent=2)
        with open(ch_out / "training_composition.csv", "w", newline="") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "process_label",
                    "process",
                    "class",
                    "raw_events",
                    "train_events_mean",
                    "train_weighted_yield_mean",
                ],
            )
            writer.writeheader()
            for row in training_composition_rows:
                writer.writerow({
                    "process_label": row["process_label"],
                    "process": row["process"],
                    "class": row["class"],
                    "raw_events": row["raw_events"],
                    "train_events_mean": row["train_events_mean"],
                    "train_weighted_yield_mean": row["train_weighted_yield_mean"],
                })
        write_training_composition_latex(training_composition_rows, ch_out / "training_composition.tex")
        if not args.quick:
            feat_names = [feats[i] for i in range(len(feats))]
            imp_mean = np.zeros(len(feats), dtype=float)
            imp_std = np.zeros(len(feats), dtype=float)
            count = np.zeros(len(feats), dtype=int)

            for item in importance_per_fold:
                idx = item["feat_idx"]
                drops = item["drops"]
                mean = drops.mean(axis=1)
                std = drops.std(axis=1)
                imp_mean[idx] += mean
                imp_std[idx] += std
                count[idx] += 1

            count = np.maximum(count, 1)
            imp_mean /= count
            imp_std /= count

            imp_rows = []
            for i, name in enumerate(feat_names):
                imp_rows.append({
                    "feature": name,
                    "importance_mean": float(imp_mean[i]),
                    "importance_std": float(imp_std[i]),
                })

            with open(ch_out / "feature_importance.json", "w") as jf:
                json.dump(imp_rows, jf, indent=2)

            csv_lines = ["feature,importance_mean,importance_std"]
            for row in imp_rows:
                csv_lines.append(f"{row['feature']},{row['importance_mean']},{row['importance_std']}")
            (ch_out / "feature_importance.csv").write_text("\n".join(csv_lines))

        # Plots
        plots_dir = ch_out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        channel_comment = map_channel_comment(channel)
        y_all = None
        p_all = None
        w_all = None
        proc_all = None
        mmc_all = None

        if oof_y:
            y_all = np.concatenate(oof_y, axis=0)
            p_all = np.concatenate(oof_prob, axis=0)
            w_all = np.concatenate(oof_w, axis=0)
            proc_all = np.concatenate(oof_proc, axis=0)
            if oof_weighted_mmc:
                mmc_all = np.concatenate(oof_weighted_mmc, axis=0)
            plot_roc_pr(y_all, p_all, w_all, plots_dir, channel_comment=channel_comment)
            plot_score_hist(
                y_all, p_all, w_all, plots_dir / "score_hist.pdf",
                channel_comment=channel_comment,
            )
            plot_oof_score_by_process(
                y_all, p_all, w_all, proc_all,
                plots_dir / "score_hist_oof_signal_by_process.pdf",
                process_label_map=process_label_map,
                target_class=1,
                channel_comment=channel_comment,
            )
            plot_oof_score_by_process(
                y_all, p_all, w_all, proc_all,
                plots_dir / "score_hist_oof_background_by_process.pdf",
                process_label_map=process_label_map,
                target_class=0,
                channel_comment=channel_comment,
            )

        split_score_payloads = {
            split_name: concatenate_score_payload(payload)
            for split_name, payload in split_scores.items()
        }
        if any(payload["y"].size for payload in split_score_payloads.values()):
            plot_split_score_concordance(
                split_score_payloads,
                plots_dir / "score_hist_train_val_test.pdf",
                channel_comment=channel_comment,
            )

        if not args.quick:
            mmc_diagnostics = {}
            if mmc_permutation_per_fold:
                mmc_perm_summary = aggregate_grouped_permutation_diagnostics(mmc_permutation_per_fold)
                if mmc_perm_summary is not None:
                    mmc_diagnostics["family_permutation"] = mmc_perm_summary
                    plot_grouped_permutation_diagnostics(
                        mmc_perm_summary,
                        plots_dir / "mmc_family_permutation_by_process.pdf",
                        process_label_map=process_label_map,
                        channel_comment=channel_comment,
                    )
            if weighted_mmc_idx is not None and y_all is not None and p_all is not None and w_all is not None and mmc_all is not None:
                mmc_quantile_summary = compute_feature_quantile_diagnostics(
                    mmc_all,
                    y_all,
                    p_all,
                    w_all,
                )
                if mmc_quantile_summary is not None:
                    mmc_diagnostics["weighted_mmc_quantiles"] = mmc_quantile_summary
                    mmc_label = map_feature_labels(["weighted_MMC_para_perp_vispTcal"], label_map)[0]
                    plot_feature_performance_by_quantile(
                        mmc_quantile_summary,
                        plots_dir / "weighted_mmc_performance_by_quantile.pdf",
                        feature_label=mmc_label,
                        channel_comment=channel_comment,
                    )
                    plot_feature_calibration_by_quantile(
                        mmc_quantile_summary,
                        plots_dir / "weighted_mmc_calibration_by_quantile.pdf",
                        channel_comment=channel_comment,
                    )
            if mmc_diagnostics:
                with open(ch_out / "mmc_diagnostics.json", "w") as jf:
                    json.dump(mmc_diagnostics, jf, indent=2)

            try:
                plot_correlation(
                    X, feats, plots_dir / "correlation.pdf",
                    title="Feature Correlation (All)", label_map=label_map
                )
                X_sig = X[y == 1]
                X_bkg = X[y == 0]
                if X_sig.size and X_bkg.size:
                    plot_correlation(
                        X_sig, feats, plots_dir / "correlation_signal.pdf",
                        title="Feature Correlation (Signal)", label_map=label_map
                    )
                    plot_correlation(
                        X_bkg, feats, plots_dir / "correlation_background.pdf",
                        title="Feature Correlation (Background)", label_map=label_map
                    )
                    corr_sig = np.corrcoef(X_sig, rowvar=False)
                    corr_bkg = np.corrcoef(X_bkg, rowvar=False)
                    plot_correlation_delta(
                        corr_sig, corr_bkg, feats,
                        plots_dir / "correlation_signal_minus_background.pdf",
                        label_map=label_map,
                    )
            except Exception as exc:
                print(f"[!] Correlation plot failed: {exc}")

            loss_files = sorted(ch_model_dir.glob(f"{args.tag}_fold*_loss.json"))
            if loss_files:
                try:
                    plot_loss_curves(
                        loss_files,
                        plots_dir / "loss_curves.pdf",
                        channel_comment=channel_comment,
                    )
                except Exception as exc:
                    print(f"[!] Loss curve plot failed: {exc}")

            if shap_model is not None and shap_scaler is not None and shap_X is not None:
                try:
                    shap_vals, shap_eval, shap_raw, shap_feats = compute_shap_values(
                        shap_model, shap_scaler, shap_X, feats, device
                    )
                    plot_shap_beeswarm(
                        shap_vals, shap_eval, shap_feats, plots_dir / "shap_beeswarm.pdf", label_map=label_map
                    )
                    plot_shap_dependence(
                        shap_vals, shap_eval, shap_raw, shap_feats, plots_dir / "shap_dependence",
                        label_map=label_map
                    )
                    try:
                        plot_shap_dependence_with_interactions(
                            shap_vals, shap_eval, shap_raw, shap_feats,
                            plots_dir / "shap_dependence_weighted_MMC_para_perp_vispTcal",
                            target_feature="weighted_MMC_para_perp_vispTcal",
                            top_n=10,
                            label_map=label_map,
                        )
                        plot_shap_dependence_with_interactions(
                            shap_vals, shap_eval, shap_raw, shap_feats,
                            plots_dir / "shap_dependence_m_tautau_vis_OS",
                            target_feature="m_tautau_vis_OS",
                            top_n=10,
                            label_map=label_map,
                        )
                    except Exception as exc:
                        print(f"[!] SHAP dependence (targeted) failed: {exc}")

                    try:
                        (
                            fold_shap_importances,
                            weighted_mmc_payload,
                            shap_diag_features,
                            process_feature_rows,
                        ) = collect_fold_shap_diagnostics(
                            ch_model_dir, args.tag, best_params,
                            X, y, proc_names, w_norm, splits, feats, device,
                            target_feature="weighted_MMC_para_perp_vispTcal",
                        )
                        plot_shap_fold_stability(
                            fold_shap_importances,
                            feats,
                            plots_dir / "shap_fold_stability.pdf",
                            label_map=label_map,
                        )
                        if process_feature_rows:
                            plot_process_feature_shap_summaries(
                                process_feature_rows,
                                feats,
                                plots_dir / "shap_process_summaries",
                                label_map=label_map,
                                process_label_map=process_label_map,
                            )
                        if weighted_mmc_payload["shap"].size:
                            plot_target_feature_shap_by_process(
                                weighted_mmc_payload,
                                target_feature="weighted_MMC_para_perp_vispTcal",
                                outpath=plots_dir / "shap_weighted_MMC_by_process.pdf",
                                label_map=label_map,
                                process_label_map=process_label_map,
                            )
                            plot_target_feature_shap_binned(
                                weighted_mmc_payload,
                                target_feature="weighted_MMC_para_perp_vispTcal",
                                outpath=plots_dir / "shap_weighted_MMC_binned.pdf",
                                label_map=label_map,
                                channel_comment=channel_comment,
                            )
                            if shap_diag_features is not None:
                                plot_target_feature_shap_waterfalls(
                                    weighted_mmc_payload,
                                    shap_diag_features,
                                    target_feature="weighted_MMC_para_perp_vispTcal",
                                    outdir=plots_dir / "shap_waterfalls",
                                    label_map=label_map,
                                    process_label_map=process_label_map,
                                )
                    except Exception as exc:
                        print(f"[!] SHAP diagnostics failed: {exc}")
                except Exception as exc:
                    print(f"[!] SHAP plot failed: {exc}")

        try:
            plot_feature_importance(imp_rows, plots_dir / "feature_importance.pdf")
        except Exception as exc:
            print(f"[!] Feature importance plot failed: {exc}")

        print(f"[✓] {channel}: metrics + feature importance saved to {ch_out}")


if __name__ == "__main__":
    main()

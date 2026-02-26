#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import awkward as ak
import uproot
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    w = w_xsec.astype(float).copy()
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


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(128, 64), activation="relu", dropout=0.0):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh

        layers = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
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


def plot_loss_curves(loss_files, outpath: Path):
    plt.figure(figsize=(6.5, 4.5))
    for lf in loss_files:
        with open(lf) as jf:
            hist = json.load(jf)
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        label = lf.stem.replace("_loss", "")
        plt.plot(epochs, hist["train_loss"], alpha=0.7, label=f"{label} train")
        plt.plot(epochs, hist["val_loss"], alpha=0.7, label=f"{label} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()


def plot_score_hist(y_true, y_score, sample_weight, outpath: Path):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    bins = np.linspace(0.0, 1.0, 51)
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0],
             histtype="step", linewidth=1.5, label="Background")
    plt.hist(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1],
             histtype="step", linewidth=1.5, label="Signal")
    plt.xlabel("MLP score")
    plt.ylabel("Weighted events")
    plt.title("Score Distributions (OOF test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()


def plot_roc_pr(y_true, y_score, sample_weight, outdir: Path):
    fpr, tpr = weighted_roc_curve(y_true, y_score, sample_weight)
    auc = weighted_roc_auc(y_true, y_score, sample_weight)
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (OOF test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.pdf", format="pdf")
    plt.close()

    recall, precision = weighted_pr_curve(y_true, y_score, sample_weight)
    pr_auc = weighted_pr_auc(y_true, y_score, sample_weight)
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (OOF test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.pdf", format="pdf")
    plt.close()


def plot_correlation(X: np.ndarray, features, outpath: Path, max_samples: int = 20000,
                     title: str = "Feature Correlation"):
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(123)
        sel = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[sel]
    corr = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(features)), features, rotation=90, fontsize=6)
    plt.yticks(range(len(features)), features, fontsize=6)
    plt.title(title)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5, color="black")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, format="pdf")
    plt.close()


def plot_correlation_delta(corr_sig: np.ndarray, corr_bkg: np.ndarray,
                           features, outpath: Path):
    delta = corr_sig - corr_bkg
    plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(delta, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(features)), features, rotation=90, fontsize=6)
    plt.yticks(range(len(features)), features, fontsize=6)
    plt.title("Signal - Background Correlation")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            plt.text(j, i, f"{delta[i, j]:.2f}", ha="center", va="center", fontsize=5, color="black")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, format="pdf")
    plt.close()


def load_label_map() -> dict:
    try:
        base = Path(__file__).resolve().parents[2]
        mod_path = base / "kinematics" / "aesthetics.py"
        spec = importlib.util.spec_from_file_location("aesthetics", mod_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "LABEL_MAP", {}) or {}
    except Exception:
        return {}


def map_feature_labels(features, label_map):
    if not label_map:
        return features
    return [label_map.get(f, f) for f in features]


def compute_shap_values(model, scaler, X, features, device,
                        max_samples: int = 1000, background: int = 200):
    rng = np.random.default_rng(123)
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_eval = X[idx]
    else:
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
    return shap_vals, eval_np, eval_raw, features


def compute_shap_values_interventional(model, scaler, X, features, device,
                                       max_samples: int = 500, background: int = 200,
                                       nsamples: int = 200):
    rng = np.random.default_rng(123)
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_eval = X[idx]
    else:
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

    model.eval()
    model.to(device)

    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        preds = []
        bs = 1024
        for i in range(0, x_np.shape[0], bs):
            xb = torch.from_numpy(x_np[i:i + bs]).to(device)
            with torch.no_grad():
                logits = model(xb).detach().cpu().numpy()
            preds.append(logits)
        out = np.concatenate(preds, axis=0)
        return out

    explainer = shap.KernelExplainer(
        predict_fn, X_bg_s, feature_perturbation="interventional"
    )
    shap_vals = explainer.shap_values(X_eval_s, nsamples=nsamples)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals, dtype=float)
    eval_np = X_eval_s
    eval_raw = X_eval
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)
    if shap_vals.shape[1] != eval_np.shape[1]:
        n = min(shap_vals.shape[1], eval_np.shape[1])
        shap_vals = shap_vals[:, :n]
        eval_np = eval_np[:, :n]
        eval_raw = eval_raw[:, :n]
        features = features[:n]
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
        shap_model = None
        shap_scaler = None
        shap_X = None

        for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
            model_name = f"{args.tag}_fold{fold}"
            model = MLP(
                n_in=X.shape[1],
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
                # Torch >=2.6 defaults to weights_only=True; fall back for legacy pickle content.
                scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)
            scaler = StandardScaler.from_state_dict(scaler_state)

            tr_sel, ytr = make_balanced_train_selection(tr_idx, y, w_norm, rng)
            tr_sel2, ytr2, va_sel, yva = split_train_val(tr_sel, ytr, best_params["val_frac"], rng)

            Xtr = scaler.transform(X[tr_sel2])
            Xva = scaler.transform(X[va_sel])
            Xte = scaler.transform(X[te_idx])

            ytr_prob = predict_proba(model, Xtr, device=device, batch_size=args.batch_size)
            yva_prob = predict_proba(model, Xva, device=device, batch_size=args.batch_size)
            yte_prob = predict_proba(model, Xte, device=device, batch_size=args.batch_size)

            fold_metrics["train"].append(
                compute_metrics(ytr2, ytr_prob, w_norm[tr_sel2])
            )
            fold_metrics["val"].append(
                compute_metrics(yva, yva_prob, w_norm[va_sel])
            )
            fold_metrics["test"].append(
                compute_metrics(y[te_idx], yte_prob, w_norm[te_idx])
            )
            oof_y.append(y[te_idx])
            oof_prob.append(yte_prob)
            oof_w.append(w_norm[te_idx])

            if shap_model is None:
                shap_model = model
                shap_scaler = scaler
                shap_X = X[te_idx]

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

        # Aggregate metrics
        summary = {
            "channel": channel,
            "features": feats,
            "files": [str(p) for p in files],
            "splits": args.splits,
            "seed": args.seed,
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

        # Feature importance aggregation
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

        ch_out = outdir / channel
        ch_out.mkdir(parents=True, exist_ok=True)
        with open(ch_out / "validation_summary.json", "w") as jf:
            json.dump(summary, jf, indent=2)
        with open(ch_out / "feature_importance.json", "w") as jf:
            json.dump(imp_rows, jf, indent=2)

        csv_lines = ["feature,importance_mean,importance_std"]
        for row in imp_rows:
            csv_lines.append(f"{row['feature']},{row['importance_mean']},{row['importance_std']}")
        (ch_out / "feature_importance.csv").write_text("\n".join(csv_lines))

        # Plots
        plots_dir = ch_out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if oof_y:
            y_all = np.concatenate(oof_y, axis=0)
            p_all = np.concatenate(oof_prob, axis=0)
            w_all = np.concatenate(oof_w, axis=0)
            plot_roc_pr(y_all, p_all, w_all, plots_dir)
            plot_score_hist(y_all, p_all, w_all, plots_dir / "score_hist.pdf")

        try:
            plot_correlation(X, feats, plots_dir / "correlation.pdf", title="Feature Correlation (All)")
            X_sig = X[y == 1]
            X_bkg = X[y == 0]
            if X_sig.size and X_bkg.size:
                plot_correlation(X_sig, feats, plots_dir / "correlation_signal.pdf",
                                 title="Feature Correlation (Signal)")
                plot_correlation(X_bkg, feats, plots_dir / "correlation_background.pdf",
                                 title="Feature Correlation (Background)")
                corr_sig = np.corrcoef(X_sig, rowvar=False)
                corr_bkg = np.corrcoef(X_bkg, rowvar=False)
                plot_correlation_delta(corr_sig, corr_bkg, feats,
                                       plots_dir / "correlation_signal_minus_background.pdf")
        except Exception as exc:
            print(f"[!] Correlation plot failed: {exc}")

        loss_files = sorted(ch_model_dir.glob(f"{args.tag}_fold*_loss.json"))
        if loss_files:
            try:
                plot_loss_curves(loss_files, plots_dir / "loss_curves.pdf")
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
            except Exception as exc:
                print(f"[!] SHAP plot failed: {exc}")

            try:
                ishap_vals, ishap_eval, ishap_raw, ishap_feats = compute_shap_values_interventional(
                    shap_model, shap_scaler, shap_X, feats, device
                )
                plot_shap_beeswarm(
                    ishap_vals, ishap_eval, ishap_feats,
                    plots_dir / "shap_beeswarm_interventional.pdf", label_map=label_map
                )
                plot_shap_dependence(
                    ishap_vals, ishap_eval, ishap_raw, ishap_feats,
                    plots_dir / "shap_dependence_interventional", label_map=label_map
                )
                try:
                    plot_shap_dependence_with_interactions(
                        ishap_vals, ishap_eval, ishap_raw, ishap_feats,
                        plots_dir / "shap_dependence_weighted_MMC_para_perp_vispTcal_interventional",
                        target_feature="weighted_MMC_para_perp_vispTcal",
                        top_n=10,
                        label_map=label_map,
                    )
                    plot_shap_dependence_with_interactions(
                        ishap_vals, ishap_eval, ishap_raw, ishap_feats,
                        plots_dir / "shap_dependence_m_tautau_vis_OS_interventional",
                        target_feature="m_tautau_vis_OS",
                        top_n=10,
                        label_map=label_map,
                    )
                except Exception as exc:
                    print(f"[!] SHAP dependence (interventional targeted) failed: {exc}")
            except Exception as exc:
                print(f"[!] SHAP interventional failed: {exc}")

        try:
            plot_feature_importance(imp_rows, plots_dir / "feature_importance.pdf")
        except Exception as exc:
            print(f"[!] Feature importance plot failed: {exc}")

        print(f"[✓] {channel}: metrics + feature importance saved to {ch_out}")


if __name__ == "__main__":
    main()

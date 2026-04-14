#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import awkward as ak
import uproot
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform.")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_

    @classmethod
    def from_state_dict(cls, state: dict):
        inst = cls(with_mean=state.get("with_mean", True),
                   with_std=state.get("with_std", True),
                   eps=state.get("eps", 1e-12))
        inst.mean_ = np.asarray(state["mean"], dtype=np.float32)
        inst.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return inst


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i):
        return self.X[i]


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
    ds = ArrayDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for xb in dl:
        xb = xb.to(device)
        logits = model(xb)
        out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_models(channel_dir: Path, tag: str, n_features: int, device: torch.device):
    best_params_path = channel_dir / f"{tag}_best_params.json"
    if not best_params_path.is_file():
        raise FileNotFoundError(f"Missing best params: {best_params_path}")
    with open(best_params_path) as jf:
        best_params = json.load(jf)

    model_paths = sorted(
        p for p in channel_dir.glob(f"{tag}_fold*.pt")
        if not p.stem.endswith("_scaler")
    )
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {channel_dir} with tag {tag}")

    models = []
    for mp in model_paths:
        fold_name = mp.stem
        scaler_path = channel_dir / f"{fold_name}_scaler.pt"
        if not scaler_path.is_file():
            legacy = channel_dir / f"{fold_name}_scaler.joblib"
            if legacy.is_file():
                raise FileNotFoundError(
                    f"Missing torch scaler: {scaler_path}. Found legacy {legacy}; "
                    "retrain or convert the scaler to .pt."
                )
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

        model = MLP(
            n_in=n_features,
            hidden=tuple(best_params["hidden"]),
            activation=best_params.get("activation", "relu"),
            dropout=best_params.get("dropout", 0.0),
        )
        state = torch.load(mp, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)

        try:
            scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=True)
        except Exception:
            scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)
        scaler = StandardScaler.from_state_dict(scaler_state)

        fold_num = None
        m = re.search(r"_fold(\d+)$", fold_name)
        if m:
            fold_num = int(m.group(1))

        models.append((fold_num, fold_name, model, scaler))

    return models


def normalise_per_process_weights(proc_names: np.ndarray, w_xsec: np.ndarray) -> np.ndarray:
    # Match training-time split weighting (|weight_xsec|) for stable OOF fold
    # assignment in the presence of negative event weights.
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


def load_channel_arrays(indir: Path, channel: str, features, signal_key: str):
    files = sorted([p for p in Path(indir).glob(f"*_{channel}.root") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No ROOT files like '*_{channel}.root' in {indir}")

    X_list, y_list, proc_list, w_list, file_id_list = [], [], [], [], []

    for i, fpath in enumerate(files):
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
            file_id_list.append(np.full(tree.num_entries, i, dtype=int))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    proc_names = np.concatenate(proc_list, axis=0)
    w_xsec = np.concatenate(w_list, axis=0)
    file_ids = np.concatenate(file_id_list, axis=0)
    return X, y, proc_names, w_xsec, file_ids, files


def apply_to_file(fpath: Path,
                  outpath: Path,
                  scores: np.ndarray,
                  fold_ids: np.ndarray,
                  mask: np.ndarray):
    with uproot.open(fpath) as f:
        tree = f["events"]
        events = tree.arrays(library="ak")

    if mask is None or mask.dtype != bool:
        raise RuntimeError("Invalid file mask for output writing.")
    if int(mask.sum()) != len(events):
        raise RuntimeError(f"Mask/event mismatch for {fpath.name}: {int(mask.sum())} vs {len(events)}")

    out_arrays = {field: events[field] for field in events.fields}
    out_arrays["mlp_score"] = scores[mask]
    out_arrays["mlp_oof_fold"] = fold_ids[mask].astype(np.int32)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(outpath) as fout:
        fout["events"] = out_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing preprocessed ROOTs")
    ap.add_argument("--features-config", default="train_features.json",
                    help="JSON file with training variables per channel")
    ap.add_argument("--model-dir", default="trained_models_torch", help="Where models were saved")
    ap.add_argument("--outdir", default="scored_ntuples", help="Output directory for scored ROOTs")
    ap.add_argument("--tag", default="mlp_torch", help="Model tag/prefix for filenames")
    ap.add_argument("--channels", nargs="+", default=["hadhad", "lephad"])
    ap.add_argument("--suffix", default="", help="Suffix for output ROOT filenames")
    ap.add_argument("--signal", default="mgp8_pp_hhh_84TeV", help="Substring to identify signal process")
    ap.add_argument("--seed", type=int, default=42, help="Random state")
    args = ap.parse_args()

    device =  torch.device("cpu")

    feat_cfg = None
    if args.features_config:
        with open(args.features_config) as jf:
            feat_cfg = json.load(jf)

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    model_dir = Path(args.model_dir).resolve()

    for channel in args.channels:
        channel_dir = model_dir / channel
        meta_files = sorted(channel_dir.glob(f"{args.tag}_fold*.json"))
        if meta_files:
            with open(meta_files[0]) as jf:
                meta = json.load(jf)
            features = list(meta["features"])
        else:
            if feat_cfg is None:
                raise FileNotFoundError(f"No model metadata found in {channel_dir} and no features config provided.")
            if "channels" not in feat_cfg or channel not in feat_cfg["channels"]:
                raise KeyError(f"Channel '{channel}' not found in {args.features_config}")
            features = list(feat_cfg["channels"][channel]["features"])

        models = load_models(channel_dir, args.tag, len(features), device)
        models_sorted = sorted(models, key=lambda x: (x[0] is None, x[0]))

    
        X, y, proc_names, w_xsec, file_ids, files = load_channel_arrays(
            indir, channel, features, args.signal
        )
        w_norm = normalise_per_process_weights(proc_names, w_xsec)

        n_splits = len(models_sorted)
        if n_splits != len(models_sorted):
            raise ValueError(
                f"--splits={n_splits} but found {len(models_sorted)} models in {channel_dir}."
            )
        # create exactly the same splits as used in trianing
        splits = weighted_stratified_kfold_indices(
            y=y, proc=proc_names, w=w_norm, n_splits=n_splits, random_state=args.seed
        )

        # define scores / folds
        oof_scores = np.full(X.shape[0], np.nan, dtype=np.float32)
        oof_folds = np.full(X.shape[0], -1, dtype=np.int32)

        # run over each fold and only apply model not trained on
        for fold_num, fold_name, model, scaler in models_sorted:
            if fold_num is None:
                raise ValueError(f"Could not parse fold number from model name: {fold_name}")
            _, te_idx = splits[fold_num - 1]
            if te_idx.size == 0:
                continue
            Xte = scaler.transform(X[te_idx])
            yprob = predict_proba(model, Xte, device=device, batch_size=8192)
            oof_scores[te_idx] = yprob
            oof_folds[te_idx] = fold_num

        if np.isnan(oof_scores).any():
            raise RuntimeError("OOF scoring left NaNs; check splits and model fold numbering.")

        for i, fpath in enumerate(files):
            proc = fpath.stem.replace(f"_{channel}", "")
            outpath = outdir / channel / f"{proc}.root"
            apply_to_file(
                fpath, outpath, oof_scores, oof_folds, file_ids == i
            )
            print(f"{channel} OOF scored: {outpath}")


if __name__ == "__main__":
    main()

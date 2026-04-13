#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import awkward as ak
import uproot
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import Optional

"""
 brief description of training script:
 
 Inputs: ROOT files from preprocessing.py output, with preselection applied. Either lephad / hadhad.
 
 Training: 
 - Trains one MLP per channel, with various optimisation metrics included within Optuna hyperparameter-optimisation
 - For resutls included in thesis we have 50 iterations of hyperparameter optimisation, 
   but generally you should be able to get away with around 10 (or even fewer), the AUC is quite stable.
 - The networks are implemented using PyTorch, and the training set is weighted using stratified k-folds (default 2)
   Stratified folds ensure the networks learn proportionally to the relative background composition 
 -  
   
 
"""

# ------------------------------
# helpers
# ------------------------------
def load_channel_arrays(indir: Path, channel: str, features, signal_key: str):
    files = sorted([p for p in Path(indir).glob(f"*_{channel}.root") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No ROOT files like '*_{channel}.root' in {indir}")

    X_list, y_list, proc_list, w_list = [], [], [], []

    for fpath in files:
        # strip path of the channel name, retain only the process
        proc = fpath.stem.replace(f"_{channel}", "")
        with uproot.open(fpath) as f:
            tree = f["events"]
            
            # make awkward arrays of all features used for training (as defined in train_features.json)
            cols = [ak.to_numpy(tree[v].array(library="ak")) for v in features]
            
            # stack 1D arrays together to make large 2D array of (features x entries)
            X_list.append(np.column_stack(cols))

            # if we have xsec weights, we should definitely use them
            if "weight_xsec" in tree.keys():
                w = ak.to_numpy(tree["weight_xsec"].array(library="ak"))
            else:
                print("[WARNING]: Please beware that you're training without weights included :(((")
                w = np.ones(tree.num_entries, dtype=float)
            w_list.append(w)

            # make class 'truth' outcomes, 1 for signal and 0 for background
            y = np.ones(tree.num_entries, dtype=int) if signal_key in proc else np.zeros(tree.num_entries, dtype=int)
            y_list.append(y)
            
            # process labels per-event
            proc_list.append(np.array([proc] * tree.num_entries, dtype=object))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    proc_names = np.concatenate(proc_list, axis=0)
    w_xsec = np.concatenate(w_list, axis=0)
    return X, y, proc_names, w_xsec, files


def normalise_per_process_weights(proc_names: np.ndarray, w_xsec: np.ndarray) -> np.ndarray:
    # Use |weight| for ML training, had to be introduced after including POWHEG NLO HH sample with -ve weights
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
                                      n_splits: int = 2,
                                      random_state: int = 42):
    # Build k-fold train/test splits while keeping each fold as representative
    # as possible in terms of class label and process composition
    rng = np.random.default_rng(random_state)
    
    # Global event index array. The folds below are stored as event indices into
    # the concatenated X/y/proc/w arrays returned by load_channel_arrays.
    n = len(y)
    idx_all = np.arange(n)

    # Each stratum is one class/process combination
    # This prevents a fold from accidentally receiving all/none of a given
    # process just because of a random split.
    strata = defaultdict(list)
    for i, (yi, pi) in enumerate(zip(y, proc)):
        strata[(int(yi), pi)].append(i)

    # Work out the total training weight carried by each class, then set the
    # target class weight per fold. In this original script, w is w_norm, i.e.
    # per-process-normalized |weight_xsec| rather than physicial yield - this is a BUG :(
    # TODO: implement preserved proper physical weights
    classes = np.unique(y)
    class_weight_total = {c: float(w[y == c].sum()) for c in classes}
    target_per_fold = {c: class_weight_total[c] / n_splits for c in class_weight_total}


    # Tracking logic: keep track of the class-weight sum already assigned to each fold. This is used to
    # place the next event into the fold that is currently most deficient for that event's class (sig/bkg).
    fold_class_w = [{c: 0.0 for c in class_weight_total} for _ in range(n_splits)]
    folds = [[] for _ in range(n_splits)]

    # Loop over each process within each class. Events in the stratum are
    # shuffled, then assigned one by one to the fold with the largest remaining
    # class-weight deficit.
    for (c, p), idxs in strata.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for i in idxs:
            deficits = np.array([target_per_fold[c] - fold_class_w[k][c] for k in range(n_splits)])
            k_best = int(np.argmax(deficits))
            folds[k_best].append(i)
            fold_class_w[k_best][c] += w[i]

    # Convert each test fold into the usual (train_idx, test_idx) pair expected
    # downstream by Optuna tuning and final fold training.
    splits = []
    for k in range(n_splits):
        test_idx = np.array(sorted(folds[k]))
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = idx_all[mask]
        splits.append((train_idx, test_idx))
    return splits


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

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

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
        "gelu": nn.GELU
    }
    return factories[key]()


def supported_activation_names():
    names = ["relu", "leaky_relu", "elu", "tanh", "gelu"]
    return names


# now we can start to build the MLP
class MLP(nn.Module):
    # initialisation:
    def __init__(self, n_in: int, hidden=(128, 64), activation="relu", dropout=0.0):
        super().__init__()
        # layers of the MLP
        layers = []
        # previous iteration of the MLP
        prev = n_in
        # since we have a optuna hyperparameter of number of layers, we can build it with a for-loop each time
        for h in hidden:
            # hidden layers, e.g. 128, 64
            layers.append(nn.Linear(prev, h))
            # specify the activation layer per layer (is always the same for each)
            layers.append(make_activation_layer(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        # output layer
        layers.append(nn.Linear(prev, 1))  # logits
        # wrap layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [N]


def set_all_seeds(seed: int):
    # for reproducibility, set seeds accordingly
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_roc_auc(y_true: np.ndarray,
                     y_score: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # take absolute weights for ROC eval too
    sample_weight = np.asarray(sample_weight, dtype=float)
    sample_weight = np.where(np.isfinite(sample_weight), np.abs(sample_weight), 0.0)

    # ensure we have appropriate weights and output scores
    mask = np.isfinite(y_score) & np.isfinite(sample_weight)
    y_true = y_true[mask]
    y_score = y_score[mask]
    sample_weight = sample_weight[mask]

    # get weights for true-positives
    w_pos = float(sample_weight[y_true == 1].sum())
    # get weights for true-negatives
    w_neg = float(sample_weight[y_true == 0].sum())

    # sorting from highest to lowest (-yscore since argsort defaults to ascending)    
    order = np.argsort(-y_score, kind="mergesort")
    # apply ordering to all sets
    y_true = y_true[order]
    y_score = y_score[order]
    sample_weight = sample_weight[order]

    # likely not needed given float precision, have seen elsewhere for ROCs -> make sure each group for the ROC is unique    
    is_new = np.concatenate(([True], y_score[1:] != y_score[:-1]))
    group_starts = np.where(is_new)[0]

    # true positive, flase positive and AUC
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
        
        # trapezoid integration
        auc += (fpr_next - fpr) * (tpr + tpr_next) / 2.0

        cum_pos = next_cum_pos
        cum_neg = next_cum_neg
        tpr = tpr_next
        fpr = fpr_next

    return float(auc)


# no grad is faster here when we're just doing evaluation. Device is almost always cpu, left in-case of gpu usage. Not really needed.
@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 8192):
    # put model into eval mode, since we're using dropout this ensures all neurons are active to evaluate entire model.
    model.eval()
    # load dataset, dummy values for second argument (just for function to run, not needed later)
    ds = ArrayDataset(X, np.zeros(len(X), dtype=np.float32))
    # create batches of events
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    out = []
    for xb, _ in dl:
        # move batch to device, for cpu this is not needed
        xb = xb.to(device)
        # evaluate MLP
        logits = model(xb)
        # map logit to [0,1] with sigmoid
        out.append(torch.sigmoid(logits).detach().cpu().numpy())
    # output full array of scores
    return np.concatenate(out, axis=0)

# function to train one MLP iteration, called within optuna hyperparameter optimisation
def train_one(model: nn.Module,
              Xtr: np.ndarray, ytr: np.ndarray,
              Xva: np.ndarray, yva: np.ndarray,
              *,
              lr: float,
              weight_decay: float,
              batch_size: int,
              beta1: float,
              beta2: float,
              eps: float,
              max_epochs: int,
              patience: int,
              device: torch.device,
              return_history: bool = False):
    # only needed for GPU use
    model.to(device)

    # training dataset:
    tr_ds = ArrayDataset(Xtr, ytr)
    # validation dataset:
    va_ds = ArrayDataset(Xva, yva)
    
    # load into required format, shuffle on for training (not for valid)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # initialise adam optimisaer (https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                           betas=(beta1, beta2), eps=eps)
    
    # unweighted loss, need with Logits since we're outputting raw rather than probabilites.
    # this is the standard BCE but applies the sigmoid fn to map to [0,1]
    loss_fn = nn.BCEWithLogitsLoss()

    
    best_state = None
    best_val = float("inf")
    bad = 0
    history = {"train_loss": [], "val_loss": []}

    # start training over max_epochs (200 by default)
    for epoch in range(max_epochs):
        # enter training mode, required if using dropout
        model.train()
        # initialise counters for evaluating loss over batches
        tr_loss_sum = 0.0
        tr_count = 0
        
        
        for xb, yb in tr_dl:
            xb = xb.to(device); yb = yb.to(device)
            # clear out old
            opt.zero_grad(set_to_none=True)
            # forward pass, outputs logit
            logits = model(xb)
            # compute loss
            loss = loss_fn(logits, yb)
            # run backpropagation
            loss.backward()
            # update model parameters from adam
            opt.step()
            bs = int(xb.shape[0])
            tr_loss_sum += float(loss) * bs
            tr_count += bs

        # validation
        # change back to eval mode, no dropout
        model.eval()
        vloss_sum = 0.0
        vcount = 0
        # whilst running validation, more efficient running with no_grad:
        with torch.no_grad():
            for xb, yb in va_dl:
                # run model, no step / backward here since we're not updating the model, just applying.
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                bs = int(xb.shape[0])
                vloss_sum += float(loss) * bs
                vcount += bs
        vloss = vloss_sum / max(1, vcount)
        tr_loss = tr_loss_sum / max(1, tr_count)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(vloss))

        # early-stopping, if the validation loss doesn't improve for <patience> epochs then we stop.
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    # want to keep history for validation / diagnostic purposes
    if return_history:
        return best_val, history
    return best_val


# ------------------------------
# Balanced selection, want to have equal numbers of signal / background events in training.
# limiting factor here is the number of background events usually.
# ------------------------------
def make_balanced_train_selection(tr_idx, y, w_norm, rng: np.random.Generator):
    # separate based on labels:
    tr_sig = tr_idx[y[tr_idx] == 1]
    tr_bkg = tr_idx[y[tr_idx] == 0]
    
    # get minimum of the two, this sets the maximum possible even set of events.
    N_bal = min(tr_sig.size, tr_bkg.size)

    # uniform random draw of signal events of size N_bal
    sig_sel = rng.choice(tr_sig, size=N_bal, replace=False)

    # Sample backgrounds with probabilities proportional to the non-negative training weights.
    p_bkg = np.abs(w_norm[tr_bkg].astype(float))
    p_bkg = p_bkg / p_bkg.sum()
    bkg_sel = tr_bkg[rng.choice(np.arange(tr_bkg.size), size=N_bal,
                                replace=False, p=p_bkg)]

    tr_sel = np.concatenate([sig_sel, bkg_sel])
    ytr = np.concatenate([np.ones(N_bal, int), np.zeros(N_bal, int)])
    order = rng.permutation(tr_sel.size)
    return tr_sel[order], ytr[order]


# split into training and validation sets
def split_train_val(tr_sel, ytr, val_frac: float, rng: np.random.Generator):
    n = tr_sel.size
    n_val = max(1, int(val_frac * n))
    perm = rng.permutation(n)
    val_local = perm[:n_val]
    tr_local = perm[n_val:]
    return tr_sel[tr_local], ytr[tr_local], tr_sel[val_local], ytr[val_local]


# ------------------------------
# Optuna HPO
# ------------------------------
def tune_hyperparams_torch(X, y, proc_names, w_norm, splits, *,
                           random_state=42, n_trials=40, use_all_splits=False,
                           device=None):
    rng = np.random.default_rng(random_state)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # setup objectives to sample over
    def objective(trial: optuna.Trial):
        hidden_choices = [
            (256, 128, 64), (256, 128), (128, 128, 64),
            (128, 64), (128, 128), (64, 64, 32)
        ]
        params = {
            "hidden": trial.suggest_categorical("hidden", hidden_choices),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", -8, -2),
            "lr": 10 ** trial.suggest_float("lr_log10", -4, -1.5),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            "activation": trial.suggest_categorical("activation", supported_activation_names()),
            "patience": trial.suggest_int("patience", 12, 30),
            "val_frac": trial.suggest_float("val_frac", 0.10, 0.20),
            "beta1": trial.suggest_float("beta1", 0.85, 0.95),
            "beta2": trial.suggest_float("beta2", 0.995, 0.9999),
            "eps": 10 ** trial.suggest_float("eps_log10", -9, -7),
            "max_epochs": 200,
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        }

        aucs = []
        inner_splits = splits if use_all_splits else splits[:min(3, len(splits))]

        for (tr_idx, te_idx) in inner_splits:
            tr_sel, ytr = make_balanced_train_selection(tr_idx, y, w_norm, rng)
            tr_sel2, ytr2, va_sel, yva = split_train_val(tr_sel, ytr, params["val_frac"], rng)

            # scaler fit only on training selection
            # this just does: (x - mean) / std_dev
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(X[tr_sel2])
            Xva = scaler.transform(X[va_sel])
            Xte = scaler.transform(X[te_idx])

            model = MLP(n_in=X.shape[1], hidden=params["hidden"],
                        activation=params["activation"], dropout=params["dropout"])

            set_all_seeds(random_state)
            _ = train_one(
                model,
                Xtr, ytr2.astype(np.float32),
                Xva, yva.astype(np.float32),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                beta1=params["beta1"],
                beta2=params["beta2"],
                eps=params["eps"],
                max_epochs=params["max_epochs"],
                patience=params["patience"],
                device=device,
            )

            yprob = predict_proba(model, Xte, device=device, batch_size=8192)
            auc = weighted_roc_auc(y[te_idx], yprob, sample_weight=w_norm[te_idx])
            aucs.append(float(auc))

        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # get best from the full set of trials:
    bp = study.best_trial.params
    # save the best params
    best_params = {
        "hidden": bp["hidden"],
        "weight_decay": 10 ** bp["wd_log10"],
        "lr": 10 ** bp["lr_log10"],
        "batch_size": bp["batch_size"],
        "activation": bp["activation"],
        "patience": bp["patience"],
        "val_frac": bp["val_frac"],
        "beta1": bp["beta1"],
        "beta2": bp["beta2"],
        "eps": 10 ** bp["eps_log10"],
        "max_epochs": 200,
        "dropout": bp["dropout"],
    }
    return best_params, float(study.best_trial.value)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing preprocessed ROOTs")
    ap.add_argument("--features-config", default="train_features.json",
                    help="JSON file with training variables per channel")
    ap.add_argument("--outdir", default="trained_models_torch", help="Where to save models")
    ap.add_argument("--signal", default="mgp8_pp_hhh_84TeV", help="Substring to identify signal process")
    ap.add_argument("--tag", default="mlp_torch", help="Model tag/prefix for filenames")
    ap.add_argument("--splits", type=int, default=2, help="Number of folds")
    ap.add_argument("--seed", type=int, default=42, help="Random state")
    ap.add_argument("--n-trials", type=int, default=10, help="Optuna trials per channel")
    ap.add_argument("--use-all-splits", action="store_true", help="Use all outer splits inside HPO objective")
    ap.add_argument("--channels", nargs="+", default=["hadhad_MMC", "lephad_MMC"])
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    device = None
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    set_all_seeds(args.seed)

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.features_config) as jf:
        feat_cfg = json.load(jf)

    print(f"Device: {device}")
    print(f"Running {len(args.channels)} trainings")

    for channel in args.channels:
        print(f"\n=== Channel: {channel} ===")
        if "channels" not in feat_cfg or channel not in feat_cfg["channels"]:
            raise KeyError(f"Channel '{channel}' not found in {args.features_config}")
        feats = list(feat_cfg["channels"][channel]["features"])

        X, y, proc_names, w_xsec, files = load_channel_arrays(indir, channel, feats, args.signal)
        w_norm = normalise_per_process_weights(proc_names, w_xsec)
        print("  Using per-process normalised |weight_xsec| for training/sampling.")

        splits = weighted_stratified_kfold_indices(
            y=y, proc=proc_names, w=w_norm, n_splits=args.splits, random_state=args.seed
        )

        ch_out = outdir / channel
        ch_out.mkdir(parents=True, exist_ok=True)

        print("  Tuning hyperparameters with Optuna")
        # this is by far the most time consuming step:
        best_params, best_cv_auc = tune_hyperparams_torch(
            X, y, proc_names, w_norm, splits,
            random_state=args.seed, n_trials=args.n_trials,
            use_all_splits=args.use_all_splits,
            device=device
        )
        print("  Best params:", best_params)
        print(f"  Best mean CV AUC: {best_cv_auc:.4f}")

        # dump the best model parameters
        with open(ch_out / f"{args.tag}_best_params.json", "w") as jf:
            json.dump(best_params, jf, indent=2)
            
        # now that we have the best performing set of parameters, we train and save the full model.

        fold_aucs = []
        rng = np.random.default_rng(args.seed)
        
        for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
            print(f"\n  Fold {fold}/{args.splits}")

            tr_sel, ytr = make_balanced_train_selection(tr_idx, y, w_norm, rng)
            tr_sel2, ytr2, va_sel, yva = split_train_val(tr_sel, ytr, best_params["val_frac"], rng)

            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(X[tr_sel2])
            Xva = scaler.transform(X[va_sel])
            Xte = scaler.transform(X[te_idx])

            model = MLP(n_in=X.shape[1], hidden=best_params["hidden"],
                        activation=best_params["activation"], dropout=best_params.get("dropout", 0.0))

            set_all_seeds(args.seed)
            _, loss_hist = train_one(
                model,
                Xtr, ytr2.astype(np.float32),
                Xva, yva.astype(np.float32),
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
                batch_size=best_params["batch_size"],
                beta1=best_params["beta1"],
                beta2=best_params["beta2"],
                eps=best_params["eps"],
                max_epochs=best_params["max_epochs"],
                patience=best_params["patience"],
                device=device,
                return_history=True,
            )

            yprob = predict_proba(model, Xte, device=device, batch_size=8192)
            auc = weighted_roc_auc(y[te_idx], yprob, sample_weight=w_norm[te_idx])
            print(f"  Weighted ROC AUC (test): {auc:.4f}")
            fold_aucs.append(float(auc))

            model_name = f"{args.tag}_fold{fold}"
            
            # Save torch model + scaler
            torch.save(model.state_dict(), ch_out / f"{model_name}.pt")
            torch.save(scaler.state_dict(), ch_out / f"{model_name}_scaler.pt")
            with open(ch_out / f"{model_name}_loss.json", "w") as jf:
                json.dump(loss_hist, jf, indent=2)

            meta = {
                "channel": channel,
                "features": feats,
                "model_name": model_name,
                "n_splits": args.splits,
                "random_state": args.seed,
                "best_params": best_params,
                "test_auc_weighted": float(auc),
                "files": [str(p) for p in files],
                "signal_key": args.signal,
                "torch": {
                    "state_dict": f"{model_name}.pt",
                    "scaler": f"{model_name}_scaler.pt",
                }
            }
            with open(ch_out / f"{model_name}.json", "w") as jf:
                json.dump(meta, jf, indent=2)

        print(f"\n  Mean weighted AUC over {args.splits} folds: "
              f"{np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


if __name__ == "__main__":
    main()

"""Persist and verify the dataset and splits used for out-of-fold predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from root_io import open_root


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(files):
    records = []
    for path in files:
        with open_root(path) as root_file:
            entries = int(root_file['events'].num_entries)
        records.append({'name': Path(path).name, 'entries': entries,
                        'sha256': file_digest(path)})
    return records


def save_run(directory, tag, *, channel, features, signal_key, seed, files,
             manifest, splits, selections, arguments, versions):
    """Publish the run manifest last, only after all fold artifacts exist."""
    directory = Path(directory)
    if input_manifest(files) != manifest:
        raise RuntimeError('Training inputs changed during the run; outputs are not valid for OOF scoring.')
    n_events = sum(item['entries'] for item in manifest)
    fold_ids = np.zeros(n_events, dtype=np.int32)
    arrays = {}
    for fold, ((_, test), (train, val)) in enumerate(zip(splits, selections), 1):
        fold_ids[test] = fold
        arrays[f'train_{fold}'] = np.asarray(train, dtype=np.int64)
        arrays[f'val_{fold}'] = np.asarray(val, dtype=np.int64)
    arrays['fold_ids'] = fold_ids
    assignments = f'{tag}_assignments.npz'
    np.savez_compressed(directory / assignments, **arrays)
    artifacts = [assignments]
    for fold in range(1, len(splits) + 1):
        artifacts.extend(f'{tag}_fold{fold}{suffix}' for suffix in
                         ('.pt', '_scaler.pt', '.json', '_loss.json'))
    run = dict(schema_version=1, channel=channel, features=list(features),
               signal_key=signal_key, seed=seed, n_splits=len(splits),
               files=manifest, assignments=assignments, arguments=arguments,
               versions=versions,
               artifacts={name: file_digest(directory / name) for name in artifacts})
    (directory / f'{tag}_run.json').write_text(json.dumps(run, indent=2) + '\n')


def load_run(directory, tag):
    directory = Path(directory)
    path = directory / f'{tag}_run.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Retrain to save verified OOF assignments; '
                                'legacy folds cannot safely be reconstructed from current inputs.')
    run = json.loads(path.read_text())
    if run['schema_version'] != 1 or run['n_splits'] < 2:
        raise ValueError(f'Unsupported OOF manifest: {path}')
    expected = {f'{tag}_fold{k}.pt' for k in range(1, run['n_splits'] + 1)}
    actual = {p.name for p in directory.glob(f'{tag}_fold*.pt')
              if not p.stem.endswith('_scaler')}
    if actual != expected:
        raise ValueError(f'Model fold set differs from manifest: expected {sorted(expected)}, got {sorted(actual)}')
    for name, digest in run['artifacts'].items():
        artifact = directory / name
        if not artifact.is_file() or file_digest(artifact) != digest:
            raise ValueError(f'Missing or changed training artifact: {artifact}')
    for fold in range(1, run['n_splits'] + 1):
        meta = json.loads((directory / f'{tag}_fold{fold}.json').read_text())
        for key, expected_value in [('features', run['features']), ('channel', run['channel']),
                                    ('n_splits', run['n_splits']), ('random_state', run['seed']),
                                    ('signal_key', run['signal_key'])]:
            if meta[key] != expected_value:
                raise ValueError(f'Fold {fold} metadata differs from run: {key}')
    return run


def verified_splits(directory, run, files, *, channel, features, signal_key, seed=None,
                    n_splits=None):
    for key, value in [('channel', channel), ('features', list(features)), ('signal_key', signal_key)]:
        if run[key] != value:
            raise ValueError(f'OOF {key} differs from training: {value!r} != {run[key]!r}')
    if seed is not None and seed != run['seed']:
        raise ValueError(f'OOF seed {seed} differs from training seed {run["seed"]}')
    if n_splits is not None and n_splits != run['n_splits']:
        raise ValueError('Requested fold count differs from training.')
    if input_manifest(files) != run['files']:
        raise ValueError('OOF input files differ from training (names, event counts or contents). '
                         'Use the identical preprocessed dataset; no scores were written.')
    with np.load(Path(directory) / run['assignments'], allow_pickle=False) as saved:
        fold_ids = saved['fold_ids']
        n_events = sum(item['entries'] for item in run['files'])
        if fold_ids.shape != (n_events,) or not np.isin(fold_ids, range(1, run['n_splits'] + 1)).all():
            raise ValueError('Invalid saved OOF fold assignments.')
        selections, splits = [], []
        for fold in range(1, run['n_splits'] + 1):
            train, val = saved[f'train_{fold}'], saved[f'val_{fold}']
            test = np.flatnonzero(fold_ids == fold)
            for idx in (train, val):
                if idx.ndim != 1 or idx.dtype.kind not in 'iu' or np.any((idx < 0) | (idx >= n_events)):
                    raise ValueError('Invalid saved training/validation indices.')
                if np.unique(idx).size != idx.size or np.intersect1d(idx, test).size:
                    raise ValueError('Duplicate indices or overlap with the outer test fold.')
            if np.intersect1d(train, val).size or not len(train) or not len(val) or not len(test):
                raise ValueError('Invalid or overlapping train/validation/test sets.')
            splits.append((np.flatnonzero(fold_ids != fold), test))
            selections.append((train, val))
    return splits, selections

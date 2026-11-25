#!/usr/bin/env python3
"""Plot POWHEG ggHH_quartic LHE header cross sections on the k3–k4 plane."""
import argparse
from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import sys

def number(value):
    result = float(value.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(result):
        raise ValueError('non-finite value')
    return result


def read_lhe(path):
    """Read couplings, cross section and uncertainty (pb), and count events."""
    card, init = {}, []
    in_init = False
    events = ends = 0
    finished = False
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            text = line.strip()
            if text == '<init>':
                in_init = True
            elif text == '</init>':
                in_init = False
            elif in_init and text and not text.startswith('#'):
                init.append(text.split())
            elif not init:
                fields = text.split()
                if len(fields) >= 2 and fields[0] in ('chhh', 'ch4', 'numevts'):
                    card[fields[0]] = number(fields[1])
            if text == '<event>':
                events += 1
            if text == '</event>':
                ends += 1
            if text == '</LesHouchesEvents>':
                finished = True
    if not finished or not events or events != ends:
        raise ValueError('unfinished or empty LHE file')
    if 'numevts' in card and events != card['numevts']:
        raise ValueError(f"expected {card['numevts']:g} events, found {events}")
    if not {'chhh', 'ch4'} <= card.keys():
        raise ValueError('missing chhh/ch4 in the embedded POWHEG card')
    if not init or len(init[0]) != 10:
        raise ValueError('missing or invalid <init> beam row')
    nprocesses = int(init[0][9])
    if nprocesses < 1 or len(init) != nprocesses + 1:
        raise ValueError('invalid <init> process rows')
    processes = [(number(row[0]), number(row[1])) for row in init[1:]]
    if any(error < 0 for _, error in processes):
        raise ValueError('cross-section uncertainty is unspecified')
    xsec = sum(value for value, _ in processes)
    error = math.sqrt(sum(error**2 for _, error in processes))
    return dict(k3=card['chhh'], k4=card['ch4'], xsec_pb=xsec,
                error_pb=error, events=events)


def collect(directory, min_events=0):
    points = defaultdict(list)
    skipped = 0
    paths = sorted(p for p in directory.rglob('*') if p.is_file()
                   and (p.name.endswith('.lhe') or p.name.endswith('.lhe.gz')))
    for path in paths:
        try:
            data = read_lhe(path)
        except (ValueError, IndexError, OSError, EOFError) as exc:
            print(f'Skipping {path}: {exc}', file=sys.stderr)
            skipped += 1
            continue
        points[data['k3'], data['k4']].append(data)
    rows = []
    for (k3, k4), files in sorted(points.items()):
        events = sum(f['events'] for f in files)
        if events < min_events:
            continue
        # Seed files estimate the SAME cross section. Never sum their XSECUPs.
        # Shared integration grids mean their errors are correlated: do not divide
        # the header error by sqrt(number of files). Keep the largest header error.
        rows.append(dict(k3=k3, k4=k4,
                         xsec_pb=sum(f['xsec_pb'] for f in files) / len(files),
                         error_pb=max(f['error_pb'] for f in files),
                         xsec_min_pb=min(f['xsec_pb'] for f in files),
                         xsec_max_pb=max(f['xsec_pb'] for f in files),
                         files=len(files), events=events))
    if not rows:
        raise ValueError('no complete LHE points meet the event requirement')
    print(f"Read {len(paths)} files; skipped {skipped}; plotted {len(rows)} coupling points.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Search recursively for .lhe and .lhe.gz files')
    parser.add_argument('-o', '--output', type=Path, default=Path('hh_quartic_xsec.png'))
    parser.add_argument('--min-events', type=int, default=0,
                        help='Minimum total events per point (e.g. 1000 for the test scan)')
    parser.add_argument('--grid', type=Path, help='Optional two-column coupling list; show unfinished points')
    parser.add_argument('--log', action='store_true', help='Logarithmic colour scale (positive cross sections only)')
    args = parser.parse_args()
    if not args.directory.is_dir() or args.min_events < 0:
        parser.error('provide an existing directory and a nonnegative minimum event count')
    try:
        rows = collect(args.directory, args.min_events)
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    values = [row['xsec_pb'] for row in rows]
    if args.log and min(values) <= 0:
        parser.error('--log requires strictly positive cross sections')
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if args.grid:
        requested = {tuple(map(number, line.split())) for line in args.grid.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')}
        if any(len(point) != 2 for point in requested):
            parser.error('--grid requires two columns: k3 k4')
        missing = sorted(requested - {(r['k3'], r['k4']) for r in rows})
        if missing:
            ax.scatter([k3 for k3, k4 in missing], [k4 for k3, k4 in missing],
                       marker='s', s=90, facecolors='none', edgecolors='0.75', label='Unfinished')
            ax.legend(loc='best')
    points = ax.scatter([r['k3'] for r in rows], [r['k4'] for r in rows],
                        c=values, marker='s', s=90, cmap='viridis',
                        norm=LogNorm() if args.log else None)
    fig.colorbar(points, ax=ax, label=r'$\sigma(gg\to HH)$ [pb]')
    ax.set(xlabel=r'$\kappa_3$', ylabel=r'$\kappa_4$',
           title=f'POWHEG ggHH_quartic — {len(rows)} points')
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    pdf = args.output.with_suffix('.pdf')
    if pdf != args.output:
        fig.savefig(pdf)
    plt.close(fig)
    table = args.output.with_suffix('.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.output}, {pdf} and {table}')


if __name__ == '__main__':
    main()

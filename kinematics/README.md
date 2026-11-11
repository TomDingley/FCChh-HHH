# kinematics

Utilities for plotting, cutflows, and signal reweighting studies from output Ntuples (FCCAnalyses).
Mainly developed for my own studies, the code isn't beautiful - but is hopefully useful!

The code reads per-process ROOT files, applies channel selections defined in `config.py`, and writes analysis plots and summary tables. Main features:

- Automatically generated cutflows for each region
- Stacked, normalised and overlaid kinematic distributions
- 2D heatmaps, efficiencies and slice plots
- Signal reweighting studies
- likelihood and contour scans for Asimov approximations
- Plus various other features, mass resolution fitting for example

## Repository layout

- `main.py`: top-level driver, almost everything can be enabled / disabled from here
- `config.py`: process lists, signal/background split, selections, luminosity, and plotting ranges
- `cutflow.py`: weighted cutflow and region-yield table builders
- `tools.py`: selection parsing and moment morphing helpers
- `plotting/`: histogram, stack, heatmap, shape-cutflow, and signal-reweighting utilities
- `fitting/`: likelihood scans, MLP-cut optimization, and `k3`/`k4` contour tools
- `aesthetics.py`: labels, colors, and plotting text

## Expected inputs

The scripts assume a directory of ROOT files named after the process keys in `config.processes`, for example:

```text
<root-dir>/
  mgp8_pp_hhh_84TeV.root
  mgp8_pp_ttbb_4f_84TeV.root
  pwp8_pp_hh_k3_1_k4_1_84TeV.root
  ...
```

Each file is expected to contain a TTree called `events` and for weighted histograms & cutflows, the branch `weight_xsec`. For the signal reweighting, keys such as `weight_k3m1_k4m1`, `weight_k30_k4m2` are required.


## Setup
The repositories is made to run with the `Key4HEP` release, and is specified in `setup.sh`.

Run commands from the repository root:

```bash
cd FCChh_HHH/kinematics
source setup.sh
```

## Quick start

### Compare two HHH ntuple implementations

For two ntuples produced from the same signal ROOT input, compare the shared
event distributions and MMC values without requiring `weight_xsec`:

```bash
python compare_ntuples.py \
  --repository /path/to/repository_ntuple.root \
  --external /path/to/external_ntuple.root \
  --outdir /path/to/comparison_plots
```

This writes individual PNG histograms, a combined `comparison.pdf`, and a
`summary.csv` with event-level differences and non-finite counts. It draws both
all-event and exact 4b2tau selections. Ranked b-jet and tau pT values are only
used when that object exists; MMC mass plots require at least two reconstructed
tau jets. Both samples must contain the same events in the same order.

To compare two independently generated HHH samples with different nominal
couplings, use unit-normalized shapes instead:

```bash
python compare_signal_nominals.py \
  --default /path/to/c3_2_d4_3_ntuple.root \
  --sm /path/to/c3_0_d4_0_ntuple.root \
  --outdir /path/to/nominal_comparison_plots
```

This plots truth and reconstructed features for all events and exact 4b2tau
events. The default sample is a black histogram, the independently generated
SM sample is shown as red points with counting errors, and a muted blue curve
reweights the default events to the SM point using their nine LHE weights.
The lower panel shows differences in cumulative event fraction, which make
shape shifts easier to see. The CSV records event counts, non-finite values,
and shape differences. Both input ntuples must contain 1,000 events.

### Bulk plots and summaries
To get started, I've hard-coded the ntuple path on the Oxford data disk such that:
```bash
python main.py
```
Will output plots / tables for the hadhad channel. 

Typical outputs include:

- per-channel plot directories under `--outdir`
- weighted cutflow tables as `.csv`, `.tex`, and optionally `.pdf` (if you have pdflatex)
- region-yield summary tables under `--outdir/summary`

There are quite a number of functions I've added over time, with some optimisation scripts in the `fit` folder, where the NN score and bins can be checked in the asimov approximation.

## Configuration

Most analysis behavior is controlled in `config.py`:

- `processes`, `SIGNAL`, and `BACKGROUNDS` define which ROOT files are loaded
- `SELECTION` defines channel selections as expression strings
- `LUMINOSITY_PB` sets the scaling used for weighted yields
- `XLIM_MAP`, `OVERLAY_PAIRS`, and `HEATMAP_PAIRS` control plotting defaults

`aesthetics.py` holds variable labels, process labels, and channel labels used in plots and tables.

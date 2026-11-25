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


## HH quartic cross-section plane

`plot_hh_quartic_xsec.py` searches a run directory recursively for `.lhe` and
`.lhe.gz` files. It reads k3 (`chhh`) and k4 (`ch4`) from the embedded POWHEG card
and the cross section in pb from `<init>`, without guessing couplings from names.

```sh
python3 kinematics/plot_hh_quartic_xsec.py /path/to/scan \
    --min-events 1000 --grid /path/to/couplings.txt --log \
    -o plots/hh_quartic_xsec.png
```

The script writes a PNG (or the chosen image format), a vector PDF and a CSV.
Each square is a measured grid point; there is no interpolation. The optional
`--grid` list marks unfinished points with empty squares. `--min-events` counts
all seed files together at each point. Omit `--log` for linear colours.

Unfinished/empty LHE files and files with fewer events than their embedded card
requests are skipped with a message. Only couplings, cross sections and event
counts are read; other run settings are ignored. All files below the directory
are used: select one campaign rather than a parent containing old copies or
duplicated samples.

The CSV contains the arithmetic mean of seed-file header cross sections, their
minimum and maximum, the largest quoted header error, and file/event counts.
Seed-file cross sections are estimates of the same quantity and are never summed.
The error is not divided by the square root of the seed count, because POWHEG
workers can share integration grids. This plots the integrated header cross
section, not a new estimate from the finite sample of signed event weights.

Parser checks: `python3 -m unittest discover -s tests -p test_hh_quartic_xsec.py`.

### HH monomial fit and interpolated plots

Fit the extracted CSV and make separate smooth surfaces:

```sh
python3 kinematics/fit_hh_quartic_xsec.py plots/hh_quartic_xsec.csv \
    --grid /path/to/couplings.txt --log --outdir plots/hh_quartic_fit
```

The HH basis is

```
sigma [pb] = c0 + c1*k3 + c2*k4 + c3*k3^2 + c4*k3*k4 + c5*k4^2
             + c6*k3^2*k4 + c7*k3*k4^2 + c8*k3^2*k4^2
```

This spans the same nine monomials as Eq. (2.1) of
[arXiv:2402.03463](https://arxiv.org/html/2402.03463v2#S2), expressed here in
unshifted kappas rather than Delta kappa = kappa - 1. The polynomial follows from
squaring the amplitude components proportional to 1, k3, k4 and k3*k4 in the
local ggHH_quartic implementation. Coefficients are fitted to the supplied
84 TeV samples; no published coefficients at other energies are imported.

The script uses weighted linear least squares with the CSV header uncertainties.
`fit.json` records the coefficient order, powers, values in pb, formal covariance,
fit range and residual diagnostics. The covariance assumes independent grid
points, is not rescaled by chi2/ndof, and includes no theory/PDF/scale errors.
Common integration seeds can correlate different coupling points: a very small
chi2/ndof should not be interpreted as an independent precision measurement.

`fit.png/pdf` shows the fitted polynomial. `interpolated.png/pdf` shows a separate
piecewise-linear interpolation of the measured cross sections. Interpolation is
performed in sigma, even with logarithmic display colours. Both plots leave
regions outside the completed-point convex hull blank; the optional `--grid`
only expands the displayed axes. Neither plot extrapolates across the unfinished
half of a partial scan. `residuals.png/pdf` and `residuals.csv` compare data and
fit at the actual completed points. The original point-only plot is retained.

For evaluating the saved function in Python (within the sampled region):

```python
import json
from kinematics.fit_hh_quartic_xsec import design_matrix
model = json.load(open('plots/hh_quartic_fit/fit.json'))
coeff = [term['value_pb'] for term in model['coefficients']]
sigma_pb = design_matrix(-2.0, 10.0) @ coeff
```

Tests: `python3 -m unittest discover -s tests -p 'test_hh_quartic*.py'`.

To extend the fitted polynomial over the whole requested grid, opt in explicitly:

```sh
python3 kinematics/fit_hh_quartic_xsec.py plots/hh_quartic_xsec.csv \
    --grid /path/to/couplings.txt --extrapolate --log \
    --outdir plots/hh_quartic_full_grid
```

This produces `fit_extrapolated.png/pdf` and `grid_predictions.csv`, with an
`extrapolated` flag for each target point. A dashed line marks the boundary of
completed-run coverage. The coefficients still come only from the supplied CSV;
the predictions outside that coverage require validation as the remaining jobs
finish. The separate interpolation plot stays restricted to the completed-point
convex hull. The displayed colour scale includes the full fitted range.

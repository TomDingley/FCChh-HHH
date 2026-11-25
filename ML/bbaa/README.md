# HH -> bb gamma gamma ML

Run these commands from the repository root. This ports the selection and
configuration from `hhh/ML/pytorch/bbaa`, and uses the repository's shared PyTorch
training, validation and scoring scripts. The existing HHH configuration is unchanged.

Create the environment once with `bash ML/setup_env.sh`, then:

```bash
source ML/env/bin/activate

# Flat directory of unselected HH ntuples and the four background samples.
INPUT=/path/to/hh_ntuples
RUN=/path/to/new_hh_ml_run

# Add physics weights first. See FCCAnalyses/hh/bbaa/README.md for production.
# Use --hh-decays filtered instead for samples forced to bb gamma gamma.
python FCCAnalyses/add_weights_yy.py "$INPUT" --output-dir "$RUN/weighted" --hh-decays mixed

python ML/bbaa/preprocessing.py --indir "$RUN/weighted" --outdir "$RUN/ntuples"

python ML/train_mlp_torch.py --indir "$RUN/ntuples" \
  --features-config ML/bbaa/train_features.json --channels bbaa \
  --signal pwp8_pp_hh_k3_1_k4_1_84TeV --tag hh \
  --outdir "$RUN/models" --splits 2 --n-trials 10

python ML/validate_mlp_torch.py --indir "$RUN/ntuples" \
  --features-config ML/bbaa/train_features.json --channels bbaa \
  --signal pwp8_pp_hh_k3_1_k4_1_84TeV --tag hh \
  --model-dir "$RUN/models" --outdir "$RUN/validation" --quick

python ML/apply_mlp_torch.py --indir "$RUN/ntuples" \
  --features-config ML/bbaa/train_features.json --channels bbaa \
  --signal pwp8_pp_hh_k3_1_k4_1_84TeV --tag hh \
  --model-dir "$RUN/models" --outdir "$RUN/scored"
```

For a short functionality test, train with `--n-trials 1 --max-epochs 2 --device cpu`.
Remove `--quick` for the full validation diagnostics. Scored ROOT files are written
under `$RUN/scored/bbaa/`, retaining `weight_xsec` and adding the score and fold.
Even quick validation scans every distinct score threshold and can be slow on
large samples; this existing shared-script behaviour is unchanged.
Application here is out-of-fold scoring of the original training dataset: use the
same preprocessed files, unchanged. Model manifests check their contents and folds.
Existing legacy models cannot be substituted for models trained by this workflow.

The default processes are HH, prompt gamma gamma + jets, inclusive ggH, ttH and VBF H,
using the original filenames listed in `preprocessing.py`. For a smaller test use
`--processes` followed by the desired signal and background process names in both
weighting and preprocessing. VH was absent from the original configuration.

Preprocessing retains the original tight selection: at least two jets and photons,
exactly two b-jets, no electrons or muons, `122 < m_yy < 128` and
`80 < m_bb < 160` (masses in GeV). These cuts affect the ML inputs only; the
production ntuples remain unselected. Undefined quantities in events failing these
cuts are allowed; any nonfinite selected feature or weight raises an error.
Missing inputs/columns and existing output files also raise errors.

The original feature order is preserved, including its repeated `dphi_bb` and
`dphi_yy` inputs (20 input positions, 18 distinct variables). `weight_xsec` is
retained in preprocessing but excluded from network inputs. Signed physics weights
are preserved verbatim in preprocessing and scoring. The shared trainer uses their
absolute values, normalized per process, for training and sampling.

This is a port of the HH inputs to the current training workflow, rather than a
bitwise reproduction of the old HH models. The shared trainer uses nested tuning
within each outer training fold and saves exact fold assignments; its activation
search also supports more choices than the legacy HH trainer. The legacy code
tuned on outer evaluation folds and reconstructed assignments during application.
No environment, ntuples or trained models from the external directory are copied.

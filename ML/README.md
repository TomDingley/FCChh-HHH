# Machine Learning for HHH analysis
This repository contains all pre-processing, training, validation and application scripts needed to run the FCC-hh MLP.

All paths provided are absolute, so if you're wanting to run the entire chain locally please alter to your own paths.

## Environment
The code requirements are given in `requirements.txt`, and a virtual environment can be made:
`source setup_env.sh`.

## Preprocessing
Prior to training, Tight Preselection cuts are applied to ntuples from `FCCAnalyses`:

$\tau_{\text{had}}\tau_{\text{had}}$-channel:
`/data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_ttbb_fullstat/merged/weighted`

$\tau_{\text{lep}}\tau_{\text{had}}$-channel:
`/data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_ttbb_fullstat_fakes/merged/weighted`

### Selections:

| Variable | lephad | hadhad |
|---|---|---|
| Loose Preselection | Pass | Pass |
| $m_{H_1}$ | $40 < m_{H_1} < 175$ | $40 < m_{H_1} < 175$ |
| $m_{H_2}$ | $20 < m_{H_2} < 160$ | $20 < m_{H_2} < 160$ |
| MMC | Pass | Pass |
| $m_{\tau\tau}^{\mathrm{MMC}}$ | $50 < m_{\tau\tau}^{\mathrm{MMC}} < 300$ | $50 < m_{\tau\tau}^{\mathrm{MMC}} < 300$ |

To run the pre-processing, please run:

$\tau_{\text{had}}\tau_{\text{had}}$-channel:

`python preprocessing.py --indir /data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_ttbb_fullstat/merged/weighted --config config.json --channels hadhad_MMC`

$\tau_{\text{lep}}\tau_{\text{had}}$-channel:

`python preprocessing.py --indir /data/atlas/users/dingleyt/FCChh/FCCAnalyses/ATLASUK_MMC_ttbb_fullstat_fakes/merged/weighted --config config.json --channels lephad_MMC`


`config.json` contains all training features plus required weights / fitting variables.

### Absolute paths to thesis-inputs
If you're looking to quickly re-run trainings and check diagnostics, use the absolute paths to inputs used for the latest thesis ML section:
HHH: `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_ntuples`

The condor scripts in the following section are configured by default with these absolute paths so ***should*** run.



## Training
The condor scripts to submit trainings are configured by default with the absolute paths I provided above and so ***should*** run.

This will output the best-performing MLP (after Optuna hyperparameter optimisation) within `trained_models/Thesis_models`.

## Validation
A number of validation plots can be made via the `validate_mlp_torch.py` script.

With absolute-paths for the trained models used in the thesis, run:

`python validate_mlp_torch.py --indir /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/ntuples/Thesis_ntuples/ --model-dir trained_models/Thesis_models/ --outdir validation/Thesis_models/ --channels hadhad_MMC lephad_MMC`

This may take a while, so perhaps running one at a time would be advisable.
You should see training plots, final score plot , SHAP beeswarm summaries, correlation plots for signal, background and signal - background, SHAP waterfalls and 2D SHAP score plots for MMC distributions plus a few more.

## Application
Now we apply the models to the existing ntuples:

`python apply_mlp_torch.py --indir /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/ntuples/Thesis_ntuples --model-dir trained_models/Thesis_models/ --outdir scored_ntuples/Thesis_scored --channels lephad_MMC hadhad_MMC`
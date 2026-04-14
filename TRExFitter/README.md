# Fitting code

No modifications are made to central TRexFitter code, so we can use StatAnalysis releases to perform the fits.

Please source:
`setupATLAS && asetup StatAnalysis,0.4.0`

## Types of fit
All fit configurations are stored within the `configs` directory. 
Each sub-directory of configs (`HHH`, `HH`, `combination`) contain three separate fits:

| Source | Statistics-only (%) | Uncertainty Scenario I (%) | Uncertainty Scenario II (%)|
|---|--:|---:|---:|
| $b$-jet identification | $0\%$ |$1\%$ | $2\%$ |
| $\tau$ identification | $0\%$| $1\%$ | $2\%$ |
| Background norm | $0\%$| $1\%$ | $1\%$ |
| Luminosity | $0\%$ | $1\%$ | $1\%$ |

Results are then presented in various ways, and are summarised below:

| Fit | Parameter treatment | Asimov dataset |
|---|---|---|
| SM HHH upper limit | $\mu$ floated, corresponding to the SM signal hypothesis | $\mu_{HHH}=1$ |
| 2D coupling scan | $\kappa_3$ and $\kappa_4$ varied simultaneously | SM point: $(\kappa_3,\kappa_4)=(1,1)$ |
| 1D $\kappa_4$ scan | $\kappa_3=1$ fixed, $\kappa_4$ varied | SM point: $(\kappa_3,\kappa_4)=(1,1)$ |
| 1D $\kappa_3$ scan | $\kappa_4=1$ fixed, $\kappa_3$ varied | SM point: $(\kappa_3,\kappa_4)=(1,1)$ |

The `HHH` upperlimit scan is only included for the `HHH` config.

## Input preparation
Starting with the ntuples post-NN application:

HadHad:
`/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad/hadhad_MMC`

LepHad:
`/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad/lephad_MMC`

The fit-ready inputs can be generated using:

For the triple Higgs analysis:
`source prep_inputs_HHH.sh`

For the di-Higgs analysis:
`source prep_inputs_HH.sh`

### The absolute path for the inputs used for thesis results:
If you're looking to quickly run fits and check diagnostics, use the absolute paths to inputs used for the latest thesis results section. 

HHH: `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis`

HH: `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis_bbaa`


## Running fits
All fits can be run from the `run` directory, the bash scripts run the entire fit workflow and stores logs / errors in the `logs` directory according to the type of fit performed.

Diagnostic plots are found in the fit directory, taking `configs/HHH/hhh_k4Fit_hist_scenI.config` as an example, the outputs will be stored in `run/Thesis_k4Fit_scenI`. 

Most of these scripts also run the plotting scripts used for the final results figures and are stored in the `plots` directory.
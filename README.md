# FCC Triple Higgs 

End-to-end analysis repository for the FCC-hh triple-Higgs study: event generation -> Pythia8 + Delphes -> post-processing / plotting -> ML -> statistical interpretation.

First, clone this repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/TomDingley/FCChh-HHH.git
cd FCChh-HHH
```

---

##  overview

**Goal:** constrain Higgs self-couplings (e.g. $\kappa_3$, $\kappa_4$) using $HHH$ at the FCC-hh, with an analysis pipeline spanning:



| Step | Directory | Status | Notes |
|---:|---|---|---|
| 1 | `EventProducer` | Up to date | Event generation, showering, Delphes |
| 2 | `FCCAnalyses` | Up to date | Ntuple production |
| 3 | `kinematics` | Up to date | Plots & misc functions.  |
| 4 | `ML` | Up to date | Training, validation, scoring |
| 5 | `TRExFitter` | Up to date | Statistical interpretation & results |


There are READMEs in each directory with instructions:
- [`EventProducer`](EventProducer)
- [`FCCAnalyses`](FCCAnalyses)
- [`kinematics`](kinematics)
- [`ML`](ML)
- [`TRExFitter`](TRExFitter)


If you're using the repository with NTuples already generated, all you need are the kinematics, ML and TRExFitter directories.


Steps 1 and 2 are by far the most intensive and full production of Delphes ROOT files should take $\mathcal{O}(1-2)$ days on the batch system and a similar amount of time for the NTuple production (MMC is computationally very slow here). The last three are doable within a day, depending on the number of Optuna hyperparameter scans you choose to run. 


## Testing the repository
The repo is quite large, but there are a few smaller jobs that can be run to ensure the repository runs all of the  analysis.

### Event Generation
First, we setup the EventProducer repository. Run these commands from the top of this repo. Ideally, software paths in `EventProducer/config/param_FCChh.py` should point to your MG5, LHAPDF and Delphes installations - but using mine is also fine.

```bash
source EventProducer/init.sh
```
For a small local test, run:

```bash
bash HHH/local_test.sh
```

This makes ten HHH events, including reweighting and MadSpin, runs Delphes and makes the ntuples. Outputs are in `HHH/local_test/`. To repeat it, choose a new `test_dir` in `HHH/settings.sh`.

### Submitting jobs
For example, to submit one 100-event HHH signal job (you can use --dry-run before to check it's setup correctly):

```bash
python3 EventProducer/bin/run.py --FCChh --LHE --send --condor \
  --typelhe mg --process mg_pp_hhh_84TeV -N 1 -n 100
```

Once the LHE job has finished, check the output and submit Delphes:

```bash
python3 EventProducer/bin/run.py --FCChh --LHE --check --force \
  --process mg_pp_hhh_84TeV
python3 EventProducer/bin/run.py --FCChh --reco --send --condor \
  --type lhep8 --process mg_pp_hhh_84TeV --prodtag fcc_v07 --detector II \
  -N 1 -n 100 -q workday
```


The LHEs and Delphes files go into `EventProducer/LHE/` and `EventProducer/DelphesEvents/`. The ntuples go into `ntuples/`. 

### Ntuples
Build the FCCAnalyses code:

```bash
source FCCAnalyses/setup.sh
cmake -S FCCAnalyses -B FCCAnalyses/build \
  -DBUILD_TESTING=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/FCCAnalyses/install"
fccanalysis build -j 4
```



Once Delphes has finished, run the ntuples locally on the first signal file (up to 100 events):

```bash
fccanalysis run HHH/ntuples.py --test --nevents 100 --output ntuples/local_test.root
```

Or submit the ntuples:

```bash
mkdir -p ntuples/log/mgp8_pp_hhh_84TeV
fccanalysis submit HHH/ntuples.py
```

Some strange segfaults at the moment, which I'm working to try and fix at the moment...
---

## Available Inputs

There are inputs available at each stage, so the code can be tested independently of the preceding step.

<details>
<summary>Absolute paths for various steps!</summary>

| Step | Sample / purpose | Path |
|---|---|---|
| LHEs | Directory containing all LHEs | `/data/atlas/users/dingleyt/FCChh/LHE/lhe` |
| Pythia8 + Delphes | All samples, including `ttbar -> bbWW` with `W -> tau` decays | `/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/FullStatistics_tautau/fcc_v07/II` |
| Pythia8 + Delphes | `ttbb` samples with inclusive `W` decays | `/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ttbb_emutau_new/fcc_v07/II` |
| Processed ntuples | FCCAnalyses output ntuples | `/data/atlas/users/dingleyt/FCChh/FCCAnalyses/thesis_ntuples` |
| ML Models | Lephad / Hadhad channels | `/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/trained_models/Thesis_models` |
| Scored ntuples | Scored $\tau_{\text{lep}}\tau_{\text{had}}$ ntuples | `/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad` |
| Scored ntuples | Scored $\tau_{\text{had}}\tau_{\text{had}}$ ntuples | `/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad` |
| Fit-ready inputs | TRExFitter inputs | `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis` |
| Fit-ready inputs | TRExFitter $b\bar{b}\gamma\gamma$ inputs | `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis_bbaa` |

</details>

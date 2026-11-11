# FCC Triple Higgs 

End-to-end analysis repository for the FCC-hh triple-Higgs study: event generation -> Pythia8 + Delphes -> post-processing / plotting -> ML -> statistical interpretation.

First, clone this repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/TomDingley/FCChh-HHH.git
cd FCChh-HHH
```

---

## Project overview

**Goal:** constrain Higgs self-couplings (e.g. $\kappa_3$, $\kappa_4$) using $HHH$ at the FCC-hh, with an analysis pipeline spanning:



| Step | Directory | Status | Notes |
|---:|---|---|---|
| 1 | `EventProducer` | WIP | Event generation, showering, Delphes |
| 2 | `FCCAnalyses` | WIP | Ntuple production |
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

## Run production

The commands are collected in [`HHH/`](HHH/README.md). Set your software paths
in `EventProducer/config/param_FCChh.py`, build this checkout's FCCAnalyses, and
edit `HHH/settings.sh` for the process list, output directories and event counts.

```bash
bash HHH/local_test.sh                 # Ten fresh signal events, through ntuples; no submission
bash HHH/submit_lhe.sh --dry-run        # Preview generation for the six processes
bash HHH/submit_reco.sh --dry-run       # Preview reconstruction
bash HHH/submit_ntuples.sh --dry-run    # Preview native fccanalysis submit commands
```

Remove `--dry-run` to submit. Run each stage only after the previous jobs finish.
After all ntuples finish, run `bash HHH/prepare_ntuples.sh` to merge them and
create the `weight_xsec` inputs under `HHH/output/weighted/`.
The [HHH instructions](HHH/README.md) give the minimum setup, output paths and
settings. HHH includes MadSpin and nine reweights; backgrounds retain their
event weight. These scripts use the existing EventProducer and FCCAnalyses
submission machinery.

## Make and run the 100-event LHE smoke test

To generate **fresh** LHE events for all six processes on HTCondor, including
larger runs split into at most 10,000 events per file, follow the
[EventProducer Condor instructions](EventProducer/hhh/README.md).

The commands below extract the first 100 complete events from one existing production LHE file for each of HHH, ttbb, ttZ, ttH, tttt and ZZZ. The HHH file retains its nine LHE reweights. You need read access to the production LHE directory shown under **Available Inputs** below, a Key4hep installation, and a built `DelphesPythia8_EDM4HEP` executable. Run these commands from the repository root:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08
python3 tests/lhe_smoke/make_fixtures.py \
  --source-root /data/atlas/users/dingleyt/FCChh/LHE/lhe
python3 tests/lhe_smoke/run.py \
  --delphes /data/atlas02/users/dingleyt/FCChh/key4/k4SimDelphes_upstream_v00_08/install/bin/DelphesPythia8_EDM4HEP
```

The first command creates six compressed LHE files in `tests/lhe_smoke/inputs/`. The second showers and reconstructs each sample with its process-specific Pythia card, then checks that each ROOT file contains 100 events, finite nominal weights, and HHH reweights matching the LHE. Its ROOT files and per-process logs go to `tests/lhe_smoke/output/`. Both directories are ignored by Git. To rerun just the checks, use `python3 tests/lhe_smoke/run.py --check-only`. To regenerate the samples, choose a fresh `--output-dir` for the first command and pass that directory as `--input-dir` to the second; choose a fresh ROOT `--output-dir` too. Neither script overwrites existing output.

The background Pythia cards in `tests/lhe_smoke/cards/` come from the corresponding original event-production cards. The test exercises LHE extraction, Pythia, Delphes and event-weight preservation. It does not run ntuple production, ML training or statistical fits; their production configurations require larger samples and separate inputs. For those stages, follow the component READMEs.

For a larger signal-only input to compare ntuple implementations, replace the first two test commands with:

```bash
python3 tests/lhe_smoke/make_fixtures.py \
  --source-root /data/atlas/users/dingleyt/FCChh/LHE/lhe \
  --process mgp8_pp_hhh_84TeV --events 1000 \
  --output-dir tests/lhe_smoke/inputs/1000
python3 tests/lhe_smoke/run.py \
  --process mgp8_pp_hhh_84TeV --events 1000 \
  --input-dir tests/lhe_smoke/inputs/1000 \
  --output-dir tests/lhe_smoke/output/1000 \
  --delphes /data/atlas02/users/dingleyt/FCChh/key4/k4SimDelphes_upstream_v00_08/install/bin/DelphesPythia8_EDM4HEP
```


Steps 1 and 2 are by far the most intensive and full production of Delphes ROOT files should take $\mathcal{O}(1-2)$ days on the batch system and a similar amount of time for the NTuple production (MMC is computationally very slow here). The last three are doable within a day, depending on the number of Optuna hyperparameter scans you choose to run. 


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

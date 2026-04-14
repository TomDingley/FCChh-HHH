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
| Pythia8 + Delphes | `HH` sample | `/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ggHH_quartic_bbtata_march/fcc_v07/II/pwp8_pp_hh_k3_1_k4_1_84TeV` |
| Processed ntuples | FCCAnalyses output ntuples | `/data/atlas/users/dingleyt/FCChh/FCCAnalyses/thesis_ntuples` |
| Scored ntuples | Scored $\tau_{\text{lep}}\tau_{\text{had}}$ ntuples | `/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad` |
| Scored ntuples | Scored $\tau_{\text{had}}\tau_{\text{had}}$ ntuples | `/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad` |
| Fit-ready inputs | TRExFitter inputs | `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis` |
| Fit-ready inputs | TRExFitter $b\bar{b}\gamma\gamma$ inputs | `/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis_bbaa` |

</details>

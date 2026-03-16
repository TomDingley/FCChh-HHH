# FCChh-HHH

End-to-end analysis repository for the FCC-hh triple-Higgs study: event generation -> Delphes -> Pythia8 -> post-processing / plotting -> ML -> statistical interpretation.

---

## Contents

- [Project overview](#project-overview)
- [Repository layout](#repository-layout)
- [Environments](#environments)
- [Workflows](#workflows)
  - [Event generation](#event-generation)
  - [FCCAnalyses](#ntuples)
  - [Post-processing](#post-processing)
  - [Machine learning](#machine-learning)
  - [Statistical interpretation](#statistical-interpretation)


---

## Project overview

**Goal:** constrain Higgs self-couplings (e.g. $\kappa_3$, $\kappa_4$) using $HHH$ at the FCC-hh, with an analysis pipeline spanning:

- generator-level production (MadGraph / POWHEG)
- Parton-shower + detector simulation
- ntuple production with FCCAnalyses
- MMC calibration
- Plotting repository / analysis optimisations
- Machine learning training / validation / application
- Statistical modelling and limit/contour extraction, both with yields-approximation (in Kinematics) and TRExFitter

There are READMEs in each directory with instructions!
If you're using the repository with NTuples already generated, all you need are the kinematics, ML and TRExFitter directories.


**Status:** under active development, please contact thomas.dingley@cern.ch for more information / to report bugs and new features!

---

## Repository layout

```text
FCChh-HHH/
  README.md
  setup.sh

  eventProducer/                   # generator configs / cards / submission helpers
  delphes/              # Delphes cards and run configs
  FCCAnalyses/               # ntuple definitions / production scripts
  Kinematics/           # plotting, fitting, kinematics studies
  ML/                   # training, evaluation, diagnostics
  MMC/                  # MMC studies 
  TrexFitter/           # trexfitter workflow


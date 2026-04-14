# FCC Triple Higgs 

End-to-end analysis repository for the FCC-hh triple-Higgs study: event generation -> Delphes -> Pythia8 -> post-processing / plotting -> ML -> statistical interpretation.

---

## Project overview

**Goal:** constrain Higgs self-couplings (e.g. $\kappa_3$, $\kappa_4$) using $HHH$ at the FCC-hh, with an analysis pipeline spanning:

- Generator-level production (MadGraph / POWHEG)
- Parton-shower + detector simulation Pythia8
- Ntuple production with FCCAnalyses
- MMC calibration
- Plotting repository / analysis optimisations
- Machine learning training / validation / application
- Statistical interpretation and limit/contour extraction, both with yields-approximation (in Kinematics) and TRExFitter

There are READMEs in each directory with instructions!
If you're using the repository with NTuples already generated, all you need are the kinematics, ML and TRExFitter directories.



---


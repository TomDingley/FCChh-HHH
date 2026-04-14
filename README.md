# FCC Triple Higgs 
<p><strong>Timer:</strong> 10:00</p>
End-to-end analysis repository for the FCC-hh triple-Higgs study: event generation -> Delphes -> Pythia8 -> post-processing / plotting -> ML -> statistical interpretation.

---

## Project overview

**Goal:** constrain Higgs self-couplings (e.g. $\kappa_3$, $\kappa_4$) using $HHH$ at the FCC-hh, with an analysis pipeline spanning:

- Generator-level production (MadGraph / POWHEG) [Work-in-progress]
- Parton-shower + detector simulation Pythia8 [Work-in-progress]
- Ntuple production with FCCAnalyses [Work-in-progress]
- Plotting repository / analysis optimisations [Up-to-date]
- Machine learning training / validation / application [Up-to-date]
- Statistical interpretation and limit/contour extraction, both with yields-approximation (in Kinematics) and TRExFitter [Up-to-date]

There are READMEs in each directory with instructions!
If you're using the repository with NTuples already generated, all you need are the kinematics, ML and TRExFitter directories.

---

There are various inputs that are provided at each step so the code can be tested at each stage, independently of the preceeding one:

### LHEs:

`/data/atlas/users/dingleyt/FCChh/LHE/lhe`

A number of processes are contained in this sub-directory, with a few samples that were used for testing purposes.

### Pythia + Delphes:

All samples (with ttbar -> bbWW, with Ws to taus):

`/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/FullStatistics_tautau/fcc_v07/II`

ttbb samples with inclusive W decays:
`/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ttbb_emutau_new/fcc_v07/II`

HH sample:
`/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ggHH_quartic_bbtata_march/fcc_v07/II/pwp8_pp_hh_k3_1_k4_1_84TeV`

### Processed ntuples:
`/data/atlas/users/dingleyt/FCChh/FCCAnalyses/thesis_ntuples`

### Scored ntuples:
`/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad`
`/data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad`

### Fit-ready inputs:
`/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis`

`/data/atlas/users/dingleyt/FCChh/trex/TRExFitter/inputs_thesis_bbaa`


With these paths you should hopefully be able to run the full analysis chain. 
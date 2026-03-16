# MMC calibration

This directory contains the ingredients and workflow used to calibrate the inputs to the FCC Missing Mass Calculator (MMC) algorithm.

The calibration has two main components:

1. **Missing transverse momentum resolution ($\sigma\left(p_{\text{T}}^{\text{miss}}\right)$) calibration**
   - determination of the detector response and resolution relevant for the MMC treatment of the missing transverse momentum
   - for the purposes of a feasibility study targeting a mostly hadronic final state, only the dominant jet term is considered

2. **Input PDF calibration for the MMC**
   - construction of probability-density inputs used by the MMC scan
   - this includes the angular term, energy fraction and missing mass (for $\tau_{\text{lep}}\tau_{\text{had}}$ only)

The purpose of this directory is to provide a reproducible chain from input ntuples to final calibration products that can be used within FCCAnalyses to produce final ntuples.

---

## Getting setup
For the $\sigma\left(p_{\text{T}}^{\text{miss}}\right)$ calibration, either find the input ntuples on the Oxford data disk (`/data/atlas/users/dingleyt/FCChh/MMC/inputs`) or run the FCCAnalyses ntuple code (`analysis_HHH_4b2tau_ntuples.py`) --- this outputs the truth 
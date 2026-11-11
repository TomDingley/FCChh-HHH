# Running HHH production

These scripts cover HHH, ttbb, ttZ, ttH, tttt and ZZZ generation, reconstruction
and ntuple production, followed by merging and cross-section weighting. They call EventProducer's original workers and
`fccanalysis submit`; settings are written directly in `settings.sh`.
Run the commands below from the repository root. The scripts also work from
other directories when called by their full path.

## Minimum setup

1. Clone with `--recurse-submodules`.
2. Set the MG5, LHAPDF and upstream Delphes paths for your installation in
   `EventProducer/config/param_FCChh.py`. The checked-in paths currently refer
   to Oxford installations. The required PDF set is NNPDF30_lo_as_0130
   (LHAPDF ID 260000). Key4hep and the generator environment come from CVMFS.
3. Build this checkout's FCCAnalyses once:

   ```bash
   source EventProducer/init.sh
   source FCCAnalyses/setup.sh
   fccanalysis build -j 4
   ```

HTCondor and a shared disk visible to the workers are needed for submissions.
The local test needs no running Condor scheduler. No separate local-mode flag
or FCCAnalyses submission wrapper is required.

## Small local test

```bash
bash HHH/local_test.sh
```

This generates ten fresh SM HHH events, reweights them, applies MadSpin 4b2tau
decays, runs Pythia/Delphes, produces a ten-event ntuple including MMC, and checks its cross-section weights.
It checks the nominal couplings, decay content, all nine reweights against the
reconstructed weights, and the ntuple event count and required branches.
Ten events may still take several minutes because MG5 compiles the process.
MMC is only evaluated when its input requirements are met; invalid inputs keep
the existing null (`NaN`) result. Missing objects and invalid pairings can also
have null values. These are retained without filling or masking, and the test
applies no event selection. Background angle calculations can print ROOT
NaN warnings from the existing analysis.

Outputs and logs go to `HHH/local_test/`. A failed command stops the script;
inspect `lhe.log`, `reco.log` or `ntuples.log` there. Choose a fresh `test_dir`
in `settings.sh` to repeat the test. Existing test output is not overwritten.

## Submit production and prepare weighted inputs

Edit `settings.sh`: choose the process list, directories, job count and events
per job. Defaults are one 100-event job per process. HHH uses the bundled
MadSpin/reweight cards, with nominal SM couplings set manually in its MG5
card. Backgrounds have their ordinary event weight. Edit `condor.sub` to
change ntuple memory/disk requests (defaults: 8 GB / 10 GB).

Preview the commands first:

```bash
bash HHH/submit_lhe.sh --dry-run
bash HHH/submit_reco.sh --dry-run
bash HHH/submit_ntuples.sh --dry-run
```

Submit each stage **after all jobs from the previous stage have finished**:

```bash
bash HHH/submit_lhe.sh
# Wait for LHE generation to finish; check condor_q and the job logs.
bash HHH/submit_reco.sh
# Wait for reconstruction to finish and check the job logs.
bash HHH/submit_ntuples.sh
# Wait for every ntuple job to finish.
bash HHH/prepare_ntuples.sh
```

The reconstruction script checks completed LHE files and writes the original
YAML records before submitting. The ntuple configuration uses one job per
reconstructed file and reuses `FCCAnalyses/hhh/HHH_ntuples.py`. It retains
signal reweights; background ntuples omit the nine HHH-only weight columns.
Ntuple previews print commands only; the native submission tool has no dry-run
option. For LHE and reconstruction, `--prepare-only` writes native job files
without submitting them.

Each LHE is limited to 10,000 events. For five million events per selected
process, set `jobs=500` and `events=10000`. To submit signal only, keep just
`mg_pp_hhh_84TeV` in `processes`.

Default output locations:

| Stage | Location |
|---|---|
| LHE | `EventProducer/LHE/<process>/` |
| Reconstruction | `EventProducer/DelphesEvents/fcc_v07/II/<process>/` |
| EventProducer scripts/logs | `EventProducer/BatchOutputs/` |
| Ntuples | `HHH/output/ntuples/<process>/` |
| Merged ntuples | `HHH/output/merged/<process>.root` |
| Weighted analysis inputs | `HHH/output/weighted/<process>.root` |
| Ntuple logs/scripts | `HHH/output/ntuples/log/` and `HHH/output/batch-submission-files/` |

## Cross-section weights

`prepare_ntuples.sh` merges all chunks for each selected process, then calls
`FCCAnalyses/add_weights.py`, the script from the previous analysis. Run it on
**unselected** ntuples: normalization divides by the full merged event count.
Existing merged/weighted outputs are not overwritten; choose a fresh output
folder for another production. The script applies no event selection and keeps
all original event branches, including null kinematic values and signal reweights.

The previous normalization is retained:

- HHH: `weight_xsec = 0.00156 * 2.25 * (3 * 0.5792**2 * 0.0624) / N` pb.
  The sum is `0.00022042967851008` pb for the 4b2tau signal.
- Backgrounds: `weight_xsec = weight * BF * k / N`. The generator weight is
  retained, including its sign. Background reference cross sections are not
  substituted for the generator weight.
- The approved ttbb update uses `BF = 0.1138**2` for the current tau-only W
  decay card, replacing the previous all-leptonic `(3 * 0.1138)**2` factor.
  Other cross sections, branching fractions and k-factors are unchanged.

`weight_xsec` is in pb; plotting/fitting applies the luminosity separately. Only
HHH has the nine `weight_k3*` reweight branches, and weighting leaves them unchanged.
Missing/non-finite event weights are errors, rather than filled or masked entries.

For an existing flat directory of unselected, merged files, call it directly:

```bash
source EventProducer/init.sh
python3 FCCAnalyses/add_weights.py /path/to/merged --output-dir /path/to/weighted
```

The old one-argument command still writes to `INPUT/weighted`. `--processes`
selects specific process names when needed. Output is an explicit ROOT TTree.

Point ML preprocessing and plotting at `HHH/output/weighted`. The standard ML
and fit configurations also include the separate HH background, which is not
produced by the six-process EventProducer list. Supply it before running their
default seven-process setup, or explicitly select the six available processes
where a process-list argument is supported. Do not invent an HH sample.

This is the production entry point. Plotting, ML training
and fitting have their own instructions in `kinematics/`, `ML/` and
`TRExFitter/`; this small test does not validate those stages.

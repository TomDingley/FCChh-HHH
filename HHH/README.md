# HHH production

These scripts run HHH, ttbb, ttZ, ttH, tttt and ZZZ. Run the commands from the top of the repo, after the [setup](../README.md).

Edit `HHH/settings.sh` to choose the processes, output folders, number of jobs and events per job. By default, we run one 100-event job for each process. HHH includes MadSpin and reweighting.

## Local test

```bash
bash HHH/local_test.sh
```

This runs ten HHH events through generation, Delphes and ntuple production. Outputs and logs are in `HHH/local_test/`. Choose a new `test_dir` in `settings.sh` to repeat it.

## Submit jobs

For LHE generation, use EventProducer directly (`submit_lhe.sh` currently only previews):

```bash
source EventProducer/init.sh
source HHH/settings.sh
for process in "${processes[@]}"; do
    python3 EventProducer/bin/run.py --FCChh --LHE --send --condor \
        --typelhe mg --process "$process" --output-dir "$production_dir" \
        -N "$jobs" -n "$events" -q "$queue"
done
```

Once all LHE jobs have finished, submit Delphes:

```bash
bash HHH/submit_reco.sh
```

Once Delphes has finished, submit the ntuples:

```bash
bash HHH/submit_ntuples.sh
```

Use `condor_q` to check the jobs. Add `--dry-run` to the submission commands to preview them.

Once all ntuples have finished, merge them and add the cross-section weights:

```bash
bash HHH/prepare_ntuples.sh
```

LHEs and Delphes files are in `EventProducer/LHE/` and `EventProducer/DelphesEvents/`. Ntuples are in `HHH/output/ntuples/`, merged files in `HHH/output/merged/`, and weighted files in `HHH/output/weighted/`.

Each LHE can contain at most 10,000 events. For 5M signal events, set `jobs=500`, `events=10000` and keep only `mg_pp_hhh_84TeV` in the process list.

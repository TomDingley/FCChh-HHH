#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1

# Ten fresh HHH events through the same LHE worker, reconstruction and ntuple code.
# No jobs are submitted. Use a fresh test_dir for each run.
mkdir "$test_dir" || exit 1
producer="$repo_dir/EventProducer"
mapfile -t tools < <(python3 - "$producer" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
from EventProducer.config import param_FCChh as config
print(config.mg5_dir)
print(config.lhapdf_config)
print(config.generator_setup)
PY
)

echo "Generating ten HHH events (this can take several minutes): $test_dir/lhe.log"
(
    cd "$test_dir" || exit 1
    bash "$producer/bin/submitMG_v3.sh" \
        "$producer/hhh/cards/mg_pp_hhh_84TeV.mg5" mg_pp_hhh_84TeV \
        "$test_dir/LHE" 000000001 10 "" "$producer/hhh/cards/loop_sm_c3d4.tar" "" \
        "$producer/hhh/cards/reweight_card.dat" "$producer/hhh/cards/madspin_card.dat" \
        "${tools[0]}" "${tools[1]}" "${tools[2]}"
) > "$test_dir/lhe.log" 2>&1 || exit 1

if [ -z "${KEY4HEP_STACK:-}" ]; then
    source "$producer/init.sh" || exit 1
fi
python3 "$producer/bin/run.py" --FCChh --LHE --check --force \
    --process mg_pp_hhh_84TeV --output-dir "$test_dir" || exit 1
python3 "$producer/bin/run.py" --FCChh --reco --send --condor --type lhep8 \
    --process mg_pp_hhh_84TeV --prodtag fcc_v07 --detector II \
    --output-dir "$test_dir" -N 1 -n 10 --prepare-only || exit 1
echo "Running Pythia/Delphes: $test_dir/reco.log"
(
    cd "$test_dir" || exit 1
    env -u KEY4HEP_STACK bash "$test_dir/BatchOutputs/FCC/fcc_v07/II/mgp8_pp_hhh_84TeV/job000000001.sh"
) > "$test_dir/reco.log" 2>&1 || exit 1

lhe="$test_dir/LHE/mg_pp_hhh_84TeV/events_000000001.lhe.gz"
reco="$test_dir/DelphesEvents/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000000001.root"
python3 - "$producer/hhh" "$lhe" "$reco" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from produce import inspect_lhe
from check_root import inspect
report = inspect_lhe(Path(sys.argv[2]), 10)
assert report['events'] == 10, report
assert report['nominal_c3'] == report['nominal_d4'] == 0, report
print(report)
print(inspect(sys.argv[3], sys.argv[2]))
PY
if [ "$?" -ne 0 ]; then exit 1; fi

source "$repo_dir/FCCAnalyses/setup.sh" || exit 1
echo "Producing ntuples: $test_dir/ntuples.log"
fccanalysis run "$hhh_dir/ntuples.py" --input "$reco" \
    --output "$test_dir/ntuples.root" --nevents 10 -- \
    --production-dir "$test_dir" --process mgp8_pp_hhh_84TeV \
    > "$test_dir/ntuples.log" 2>&1 || exit 1
python3 - "$test_dir/ntuples.root" <<'PY'
import sys
import ROOT
with ROOT.TFile.Open(sys.argv[1]) as source:
    events = source.Get('events')
    assert events.GetEntries() == 10, 'Expected all ten events in the ntuple'
    for name in ('weight', 'weight_k3m1_k4m1', 'weighted_MMC_para_perp_vispTcal'):
        assert events.GetBranch(name), f'Missing output: {name}'
print('PASS: ten HHH events through generation, reconstruction and ntuples; LHE reweights preserved.')
PY

echo "Merging and weighting the unselected signal ntuple"
mkdir "$test_dir/merged" || exit 1
hadd "$test_dir/merged/mgp8_pp_hhh_84TeV.root" "$test_dir/ntuples.root" || exit 1
python3 "$repo_dir/FCCAnalyses/add_weights.py" "$test_dir/merged" \
    --processes mgp8_pp_hhh_84TeV || exit 1
python3 - "$test_dir/merged/weighted/mgp8_pp_hhh_84TeV.root" <<'PY'
import math
import sys
import ROOT
with ROOT.TFile.Open(sys.argv[1]) as source:
    events = source.Get('events')
    assert events.GetEntries() == 10, 'Weighting changed the event count'
    total = sum(float(event.weight_xsec) for event in events)
    expected = 0.00156 * 2.25 * 3 * 0.5792**2 * 0.0624
    assert math.isclose(total, expected, rel_tol=1e-12), (total, expected)
print('PASS: signal weight_xsec sums to the previous NNLO-normalized 4b2tau cross section.')
PY

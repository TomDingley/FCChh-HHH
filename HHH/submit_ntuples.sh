#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1

# Run after reconstruction finishes. Native fccanalysis submit handles the jobs.
if [ "${1:-}" != "--dry-run" ]; then
    if [ -z "${KEY4HEP_STACK:-}" ]; then
        source "$repo_dir/EventProducer/init.sh" || exit 1
    fi
    source "$repo_dir/FCCAnalyses/setup.sh" || exit 1
    mkdir -p "$ntuple_dir" || exit 1
    cd "$ntuple_dir" || exit 1
fi

for process in "${processes[@]}"; do
    reco_process="${process/mg_/mgp8_}"
    command=(fccanalysis submit "$hhh_dir/ntuples.py" --
             --production-dir "$production_dir" --process "$reco_process"
             --threads "$threads" --queue "$queue")
    if [ "${1:-}" = "--dry-run" ]; then
        printf 'From %s: ' "$ntuple_dir"
        printf '%q ' "${command[@]}"
        printf '\n'
    elif [ "$#" -ne 0 ]; then
        echo "Usage: bash HHH/submit_ntuples.sh [--dry-run]" >&2
        exit 2
    else
        mkdir -p "ntuples/log/$reco_process" || exit 1
        "${command[@]}" || exit 1
    fi
done

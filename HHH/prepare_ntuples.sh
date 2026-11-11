#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1
if [ -z "${KEY4HEP_STACK:-}" ]; then
    source "$repo_dir/EventProducer/init.sh" || exit 1
fi

# Run after all ntuple jobs finish. Merge each process before normalizing by N.
mkdir -p "$ntuple_dir/merged" || exit 1
reco_processes=()
for process in "${processes[@]}"; do
    reco_process="${process/mg_/mgp8_}"
    files=("$ntuple_dir/ntuples/$reco_process/"*.root)
    if [ ! -f "${files[0]}" ]; then
        echo "No ntuple chunks for $reco_process" >&2
        exit 1
    fi
    hadd "$ntuple_dir/merged/$reco_process.root" "${files[@]}" || exit 1
    reco_processes+=("$reco_process")
done

python3 "$repo_dir/FCCAnalyses/add_weights.py" "$ntuple_dir/merged" \
    --output-dir "$ntuple_dir/weighted" --processes "${reco_processes[@]}"

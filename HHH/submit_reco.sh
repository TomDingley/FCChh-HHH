#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1
if [ -z "${KEY4HEP_STACK:-}" ]; then
    source "$repo_dir/EventProducer/init.sh" || exit 1
fi

# Run after the LHE jobs finish. --dry-run previews without checking/writing YAML.
for process in "${processes[@]}"; do
    if [ "${1:-}" != "--dry-run" ]; then
        python3 "$repo_dir/EventProducer/bin/run.py" \
            --FCChh --LHE --check --force --process "$process" \
            --output-dir "$production_dir" || exit 1
    fi
    python3 "$repo_dir/EventProducer/bin/run.py" \
        --FCChh --reco --send --condor --type lhep8 --process "$process" \
        --prodtag fcc_v07 --detector II --output-dir "$production_dir" \
        -N "$jobs" -n "$events" -q "$queue" "$@" || exit 1
done

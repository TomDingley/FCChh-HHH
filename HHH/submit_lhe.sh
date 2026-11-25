#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1
if [ -z "${KEY4HEP_STACK:-}" ]; then
    source "$repo_dir/EventProducer/init.sh" || exit 1
fi

# Pass --dry-run to preview, or --prepare-only to write jobs without submitting.
for process in "${processes[@]}"; do
    python3 "$repo_dir/EventProducer/bin/run.py" \
        --FCChh --LHE --dry-run --send --condor --typelhe mg --process "$process" \
        --output-dir "$production_dir" -N "$jobs" -n "$events" -q "$queue" \
        "$@" || exit 1
done

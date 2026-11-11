#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh" || exit 1

# Pass --dry-run to preview, or --prepare-only to write jobs without submitting.
for process in "${processes[@]}"; do
    python3 "$repo_dir/EventProducer/bin/run.py" \
        --FCChh --LHE --send --condor --typelhe mg --process "$process" \
        --output-dir "$production_dir" -N "$jobs" -n "$events" -q "$queue" \
        "$@" || exit 1
done

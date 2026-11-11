# Shared settings for the scripts in this directory. Edit these before submitting.
hhh_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$hhh_dir/.." && pwd)

production_dir="$repo_dir/EventProducer"
ntuple_dir="$hhh_dir/output"
test_dir="$hhh_dir/local_test"

jobs=1
events=100                 # Events per job; maximum 10,000. 5M = 500 jobs x 10,000.
queue="workday"
threads=2

processes=(mg_pp_hhh_84TeV mg_pp_ttbb_4f_84TeV mg_pp_ttz_5f_84TeV
           mg_pp_tth_5f_84TeV mg_pp_tttt_5f_84TeV mg_pp_zzz_5f_84TeV)

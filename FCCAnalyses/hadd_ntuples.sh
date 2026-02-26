
# get directory from user input
dir=${1}
mkdir -p ${dir}/merged

#
for process in mgp8_pp_z_4f_84TeV mgp8_pp_h_5f_84TeV mgp8_pp_tttt_5f_84TeV mgp8_pp_ttbb_4f_84TeV mgp8_pp_tth_5f_84TeV mgp8_pp_ttz_5f_84TeV mgp8_pp_zzz_5f_84TeV mgp8_pp_hhh_84TeV mgp8_pp_z0123j_4f_84TeV mgp8_pp_h012j_5f_84TeV pwp8_pp_hh_k3_1_k4_1_84TeV mgp8_pp_jjaa_5f_84TeV mgp8_pp_tth01j_5f_84TeV mgp8_pp_vbf_h01j_5f_84TeV
do
  echo "Merging ntuples for ${process}"
  hadd ${dir}/merged/${process}.root ${dir}/${process}/*.root
done

echo "Merging complete. Output files are located in ${dir}/merged/"
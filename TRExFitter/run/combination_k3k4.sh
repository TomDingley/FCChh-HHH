# create workspaces (output: RooStats)
trex-fitter h ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_h.log 2>&1
trex-fitter h ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_h.log 2>&1
trex-fitter h ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_h.log 2>&1

# create workspaces (output: RooStats)
trex-fitter w ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_w.log 2>&1
trex-fitter w ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_w.log 2>&1
trex-fitter w ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_w.log 2>&1

# create pre-fit plots (output: Plots)
trex-fitter d ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_d.log 2>&1
trex-fitter d ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_d.log 2>&1
trex-fitter d ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_d.log 2>&1

# run the fit 
trex-fitter f ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_f.log 2>&1
trex-fitter f ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_f.log 2>&1
trex-fitter f ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_f.log 2>&1

# create post-fit plots (output: Plots)
trex-fitter p ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_p.log 2>&1
trex-fitter p ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_p.log 2>&1
trex-fitter p ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_p.log 2>&1

# run upper-limit scan on signal strength (output: Limits)
trex-fitter l ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_l.log 2>&1
trex-fitter l ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_l.log 2>&1
trex-fitter l ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_l.log 2>&1

# obs and expected significance (output: Significance)
trex-fitter s ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_s.log 2>&1
trex-fitter s ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_s.log 2>&1
trex-fitter s ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_s.log 2>&1

# produce ranking plots of nuisance parameters (output: Ranking)
trex-fitter r ../configs/combination/combination_hh_hhh_k3k4Fit_stat.config > combination_hh_hhh_k3k4Fit_stat_log_r.log 2>&1
trex-fitter r ../configs/combination/combination_hh_hhh_k3k4Fit_scenI.config > combination_hh_hhh_k3k4Fit_scenI_log_r.log 2>&1
trex-fitter r ../configs/combination/combination_hh_hhh_k3k4Fit_scenII.config > combination_hh_hhh_k3k4Fit_scenII_log_r.log 2>&1

python make_contour.py --in-dir Thesis_combination_hh_hhh_k3k4_stat --xmin 0 --xmax 4.5 --ymin -15 --ymax 30 --comb
python make_contour.py --in-dir Thesis_combination_hh_hhh_k3k4_scenI --xmin 0 --xmax 4.5 --ymin -15 --ymax 30 --comb
python make_contour.py --in-dir Thesis_combination_hh_hhh_k3k4_scenII --xmin 0 --xmax 4.5 --ymin -15 --ymax 30 --comb

python 2D_scenarios.py --stat /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3k4_stat/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenI /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3k4_scenI/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenII /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3k4_scenII/LHoodPlots/NLLscan_k3_k4_histo.root \
    --xmin 0.84 --xmax 1.23 --ymin -8 --ymax 23 --out thesis_k3k4_contours_compare_combination_restricted.pdf --comb

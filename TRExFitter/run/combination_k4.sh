# create workspaces (output: RooStats)
trex-fitter h ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_h.log 2>&1
trex-fitter h ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_h.log 2>&1
trex-fitter h ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_h.log 2>&1

# create workspaces (output: RooStats)
trex-fitter w ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_w.log 2>&1
trex-fitter w ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_w.log 2>&1
trex-fitter w ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_w.log 2>&1

# create pre-fit plots (output: Plots)
trex-fitter d ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_d.log 2>&1
trex-fitter d ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_d.log 2>&1
trex-fitter d ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_d.log 2>&1

# run the fit 
trex-fitter f ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_f.log 2>&1
trex-fitter f ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_f.log 2>&1
trex-fitter f ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_f.log 2>&1

# create post-fit plots (output: Plots)
trex-fitter p ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_p.log 2>&1
trex-fitter p ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_p.log 2>&1
trex-fitter p ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_p.log 2>&1

# run upper-limit scan on signal strength (output: Limits)
trex-fitter l ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_l.log 2>&1
trex-fitter l ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_l.log 2>&1
trex-fitter l ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_l.log 2>&1

# obs and expected significance (output: Significance)
trex-fitter s ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_s.log 2>&1
trex-fitter s ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_s.log 2>&1
trex-fitter s ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_s.log 2>&1

# produce ranking plots of nuisance parameters (output: Ranking)
trex-fitter r ../configs/combination_hh_hhh_k4Fit_stat.config > combination_hh_hhh_k4Fit_stat_log_r.log 2>&1
trex-fitter r ../configs/combination_hh_hhh_k4Fit_scenI.config > combination_hh_hhh_k4Fit_scenI_log_r.log 2>&1
trex-fitter r ../configs/combination_hh_hhh_k4Fit_scenII.config > combination_hh_hhh_k4Fit_scenII_log_r.log 2>&1

python make_LH_scenarios.py --labels "No syst" "Uncertainty Scenario I" "Uncertainty Scenario II" --out Theeeesis_combination_k4_scenarios.pdf  \
    Thesis_combination_hh_hhh_k4_stat/LHoodPlots/NLLscan_k4_curve.root \
    Thesis_combination_hh_hhh_k4_scenI/LHoodPlots/NLLscan_k4_curve.root \
    Thesis_combination_hh_hhh_k4_scenII/LHoodPlots/NLLscan_k4_curve.root \
    --xmin -10 --xmax 22 --ymax 2.5 --ymin 0 --comb
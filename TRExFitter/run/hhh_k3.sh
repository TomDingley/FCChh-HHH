# create workspaces (output: RooStats)
trex-fitter h ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_h.log 2>&1
trex-fitter h ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_h.log 2>&1
trex-fitter h ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_h.log 2>&1

# create workspaces (output: RooStats)
trex-fitter w ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_w.log 2>&1
trex-fitter w ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_w.log 2>&1
trex-fitter w ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_w.log 2>&1

# create pre-fit plots (output: Plots)
trex-fitter d ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_d.log 2>&1
trex-fitter d ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_d.log 2>&1
trex-fitter d ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_d.log 2>&1

# run the fit 
trex-fitter f ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_f.log 2>&1
trex-fitter f ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_f.log 2>&1
trex-fitter f ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_f.log 2>&1

# create post-fit plots (output: Plots)
trex-fitter p ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_p.log 2>&1
trex-fitter p ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_p.log 2>&1
trex-fitter p ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_p.log 2>&1

# run upper-limit scan on signal strength (output: Limits)
trex-fitter l ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_l.log 2>&1
trex-fitter l ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_l.log 2>&1
trex-fitter l ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_l.log 2>&1

# obs and expected significance (output: Significance)
trex-fitter s ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_s.log 2>&1
trex-fitter s ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_s.log 2>&1
trex-fitter s ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_s.log 2>&1

# produce ranking plots of nuisance parameters (output: Ranking)
trex-fitter r ../configs/HHH/hhh_k3Fit_hist_stat.config > hhh_k3Fit_hist_stat_log_r.log 2>&1
trex-fitter r ../configs/HHH/hhh_k3Fit_hist_scenI.config > hhh_k3Fit_hist_scenI_log_r.log 2>&1
trex-fitter r ../configs/HHH/hhh_k3Fit_hist_scenII.config > hhh_k3Fit_hist_scenII_log_r.log 2>&1

python make_LH_scenarios_k3.py --labels "No syst" "Uncertainty Scenario I" "Uncertainty Scenario II" --out Theeeesis_k3_scenarios.pdf  \
    Thesis_k3Fit_stat/LHoodPlots/NLLscan_k3_curve.root \
    Thesis_k3Fit_scenI/LHoodPlots/NLLscan_k3_curve.root \
    Thesis_k3Fit_scenII/LHoodPlots/NLLscan_k3_curve.root
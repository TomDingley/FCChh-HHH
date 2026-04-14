mkdir -p logs

# create workspaces (output: RooStats)
trex-fitter h ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_h.log 2>&1
trex-fitter h ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_h.log 2>&1
trex-fitter h ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_h.log 2>&1

# create workspaces (output: RooStats)
trex-fitter w ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_w.log 2>&1
trex-fitter w ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_w.log 2>&1
trex-fitter w ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_w.log 2>&1

# create pre-fit plots (output: Plots)
trex-fitter d ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_d.log 2>&1
trex-fitter d ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_d.log 2>&1
trex-fitter d ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_d.log 2>&1

# run the fit 
trex-fitter f ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_f.log 2>&1
trex-fitter f ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_f.log 2>&1
trex-fitter f ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_f.log 2>&1

# create post-fit plots (output: Plots)
trex-fitter p ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_p.log 2>&1
trex-fitter p ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_p.log 2>&1
trex-fitter p ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_p.log 2>&1

# run upper-limit scan on signal strength (output: Limits)
trex-fitter l ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_l.log 2>&1
trex-fitter l ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_l.log 2>&1
trex-fitter l ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_l.log 2>&1

# obs and expected significance (output: Significance)
trex-fitter s ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_s.log 2>&1
trex-fitter s ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_s.log 2>&1
trex-fitter s ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_s.log 2>&1

# produce ranking plots of nuisance parameters (output: Ranking)
trex-fitter r ../configs/HH/hh_k3Fit_stat.config > logs/hh_k3Fit_stat_log_r.log 2>&1
trex-fitter r ../configs/HH/hh_k3Fit_scenI.config > logs/hh_k3Fit_scenI_log_r.log 2>&1
trex-fitter r ../configs/HH/hh_k3Fit_scenII.config > logs/hh_k3Fit_scenII_log_r.log 2>&1

python make_LH_scenarios_k3.py --labels "No syst" "Uncertainty Scenario I" "Uncertainty Scenario II" --out hh_k3_scenarios.pdf  \
    Thesis_diHiggs_k3_stat/LHoodPlots/NLLscan_k3_curve.root \
    Thesis_diHiggs_k3_scenI/LHoodPlots/NLLscan_k3_curve.root \
    Thesis_diHiggs_k3_scenII/LHoodPlots/NLLscan_k3_curve.root \
    --xmin 0.8 --xmax 1.2 --ymin 0 --ymax 2.5 --hh
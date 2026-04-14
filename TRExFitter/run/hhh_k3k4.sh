mkdir -p logs

# create workspaces (output: RooStats)
trex-fitter h ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_h.log 2>&1
trex-fitter h ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_h.log 2>&1
trex-fitter h ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_h.log 2>&1

# create workspaces (output: RooStats)
trex-fitter w ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_w.log 2>&1
trex-fitter w ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_w.log 2>&1
trex-fitter w ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_w.log 2>&1

# create pre-fit plots (output: Plots)
trex-fitter d ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_d.log 2>&1
trex-fitter d ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_d.log 2>&1
trex-fitter d ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_d.log 2>&1

# run the fit 
trex-fitter f ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_f.log 2>&1
trex-fitter f ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_f.log 2>&1
trex-fitter f ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_f.log 2>&1

# create post-fit plots (output: Plots)
trex-fitter p ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_p.log 2>&1
trex-fitter p ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_p.log 2>&1
trex-fitter p ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_p.log 2>&1

# run upper-limit scan on signal strength (output: Limits)
trex-fitter l ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_l.log 2>&1
trex-fitter l ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_l.log 2>&1
trex-fitter l ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_l.log 2>&1

# obs and expected significance (output: Significance)
trex-fitter s ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_s.log 2>&1
trex-fitter s ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_s.log 2>&1
trex-fitter s ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_s.log 2>&1

# produce ranking plots of nuisance parameters (output: Ranking)
trex-fitter r ../configs/HHH/hhh_k3k4Fit_hist_MMC_stat.config > logs/hhh_k3k4Fit_hist_MMC_stat_log_r.log 2>&1
trex-fitter r ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenI.config > logs/hhh_k3k4Fit_hist_MMC_scenI_log_r.log 2>&1
trex-fitter r ../configs/HHH/hhh_k3k4Fit_hist_MMC_scenII.config > logs/hhh_k3k4Fit_hist_MMC_scenII_log_r.log 2>&1


python make_contour.py --in-dir Thesis_k3k4_stat
python make_contour.py --in-dir Thesis_k3k4_scenI --xmin -1.5 --xmax 5 --ymin -15 --ymax 30
python make_contour.py --in-dir Thesis_k3k4_scenII

python 2D_scenarios.py --stat Thesis_k3k4_stat/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenI Thesis_k3k4_scenI/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenII Thesis_k3k4_scenII/LHoodPlots/NLLscan_k3_k4_histo.root \
    --xmin -1 --xmax 4.5 --ymin -10 --ymax 30 --out k3k4_hhh.pdf

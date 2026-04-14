# code to compare different setups in k3, k4 and k3k4 configurations
python 2D_scenarios.py --stat /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k3k4_stat/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenI /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k3k4_stat/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenII /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3k4_stat/LHoodPlots/NLLscan_k3_k4_histo.root \
    --xmin -0.75 --xmax 4.5 --ymin -8 --ymax 25 --out k3k4_comparisons.pdf \
    --labels "HHH" "HH" "Combination" --comp
    
# with k4
python make_LH_scenarios.py --labels "HHH" "HH" "Combination" --out k4_comparisons.pdf  \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k4Fit_stat/LHoodPlots/NLLscan_k4_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k4_stat/LHoodPlots/NLLscan_k4_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k4_stat/LHoodPlots/NLLscan_k4_curve.root \
    --xmin -20 --xmax 30 --ymax 2.5 --ymin 0 --comp

# now with k3
python make_LH_scenarios_k3.py --labels "HHH" "HH" "Combination" --out k3_comparisons.pdf  \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k3Fit_stat/LHoodPlots/NLLscan_k3_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k3_stat/LHoodPlots/NLLscan_k3_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3_stat/LHoodPlots/NLLscan_k3_curve.root \
    --xmin -1 --xmax 3.5 --ymax 2.5 --ymin 0 --comp


# code to compare different setups in k3, k4 and k3k4 configurations in scenario I
python 2D_scenarios.py --stat /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k3k4_scenI/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenI /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k3k4_scenI/LHoodPlots/NLLscan_k3_k4_histo.root \
    --scenII /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3k4_scenI/LHoodPlots/NLLscan_k3_k4_histo.root \
    --xmin -0.75 --xmax 4.5 --ymin -8 --ymax 25 --out k3k4_comparisons_scenI.pdf \
    --labels "HHH" "HH" "Combination" --comp
    
# with k4
python make_LH_scenarios.py --labels "HHH" "HH" "Combination" --out k4_comparisons_scenI.pdf  \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k4Fit_scenI/LHoodPlots/NLLscan_k4_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k4_scenI/LHoodPlots/NLLscan_k4_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k4_scenI/LHoodPlots/NLLscan_k4_curve.root \
    --xmin -20 --xmax 30 --ymax 2.5 --ymin 0 --comp

# now with k3
python make_LH_scenarios_k3.py --labels "HHH" "HH" "Combination" --out k3_comparisons_scenI.pdf  \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_k3Fit_scenI/LHoodPlots/NLLscan_k3_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_diHiggs_k3_scenI/LHoodPlots/NLLscan_k3_curve.root \
    /data/atlas/users/dingleyt/FCChh/trex/TRExFitter/Thesis_combination_hh_hhh_k3_scenI/LHoodPlots/NLLscan_k3_curve.root \
    --xmin -1 --xmax 3.5 --ymax 2.5 --ymin 0 --comp

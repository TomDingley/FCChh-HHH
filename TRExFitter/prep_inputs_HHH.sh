
# make backgorunds
python make_backgrounds_mHHH.py \
    --in-dir /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad/lephad_MMC \
    --out-dir inputs/inputs_thesis \
    --suffix lephad_MMC


# make signals
python make_rootfiles_2binMHHH.py \
    --input /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_lephad/lephad_MMC/mgp8_pp_hhh_84TeV.root \
    --outdir inputs/inputs_thesis \
    --suffix lephad_MMC



python make_backgrounds_mHHH.py \
    --in-dir /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad/hadhad_MMC \
    --out-dir inputs/inputs_thesis \
    --suffix hadhad_MMC

python make_rootfiles_2binMHHH.py \
    --input /data/atlas/users/dingleyt/FCChh/hhh/ML/pytorch/scored_ntuples/Thesis_hadhad/hadhad_MMC/mgp8_pp_hhh_84TeV.root \
    --outdir inputs/inputs_thesis \
    --suffix hadhad_MMC
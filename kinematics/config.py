from typing import List, Dict, Tuple

# --------------------------------------------------
# Dataset selection and flags
# --------------------------------------------------
doSignal = False
doHiggs = False
newprod = True
newcard = False
doTtbb = False
noDecay = False
MMC = True
Truth= False

bbaa = False

newsamples = False
# --- Common preselection: 4 b-jets with pT thresholds ---
from collections import OrderedDict

# ---------- tiny helpers to keep strings tidy ----------
# ---------- tiny helpers ----------
def AND(*parts: str) -> str:
    # Wrap each part in (...) and join with &&
    return "(" + ") && (".join(parts) + ")"

def OR(*parts: str) -> str:
    return "(" + ") || (".join(parts) + ")"

# ---------- common preselection ----------
BJET_PRESEL = (
    "n_b_jets_medium_tauprio == 4",
)

# ---------- channel building blocks ----------
LEPHAD_TAU = ("n_tau_jets_medium == 1")
LEPHAD_ETAU = AND("n_tau_jets_medium == 1", "n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0")
LEPHAD_MUTAU = AND("n_tau_jets_medium == 1", "n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0")
LEPHAD_ANY = AND("n_tau_jets_medium == 1", "pT_tau1 > 20", "pT_tau2 > 20", OR(
    AND("n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0"),
    AND("n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0"),
))
LEPHAD_BJETS = AND("n_tau_jets_medium == 1", "n_b_jets_medium_tauprio == 4", OR(
    AND("n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0"),
    AND("n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0"),
))

HADHAD_JET    = AND("n_tau_jets_medium == 2")
HADHAD_OS     = AND("n_tau_jets_medium == 2", "OS_tau == 1")
HADHAD_NOLEP  = AND("n_tau_jets_medium == 2", "OS_tau == 1", "n_sel_el_0p2 == 0", "n_sel_mu == 0")
HADHAD_2B = AND("n_tau_jets_medium == 2", "n_b_jets_medium_tauprio == 2", "n_bb_tagged_08Jets == 1", "OS_tau == 1")
HADHAD_2LRB = AND("n_bb_tagged_08Jets == 2")

HADHAD_4B = AND("n_tau_jets_medium == 2", "n_b_jets_medium_tauprio == 4", "pT_b3 > 30", "pT_b4 > 25", "OS_tau == 1")
HADHAD_1TAU = AND("n_tautau_tagged_08Jets == 1")
HADHAD_1TAU2B = AND("n_tautau_tagged_08Jets == 1", "n_b_jets_medium_tauprio == 2", "n_bb_tagged_08Jets == 1")
HADHAD_BOOST_SEL = AND("n_tautau_tagged_08Jets == 1", "n_bb_tagged_08Jets == 2")
newSel = True
if newSel:
    LEPHAD_JETSEL = ("n_jets_sel >= 5")
    LEPHAD_TAU = ("n_tau_jets_medium == 1")
    LEPHAD_EMU = AND("n_tau_jets_medium == 1", OR(
        AND("n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0"),
        AND("n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0"),
    ))
    LEPHAD_ETAU = AND("n_tau_jets_medium == 1", "n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0")

    LEPHAD_PT = AND("pT_tau1 > 20", "pT_tau2 > 20")
    HADHAD_JETSEL = ("n_jets_sel >= 6")


    BJET_PRESEL = (
        AND("n_b_jets_medium_tauprio == 4","pT_b1 > 40", "pT_b2 > 35", "pT_b3 > 30", "pT_b4 > 25")
    )
    BJET_TAG = AND("n_b_jets_medium_tauprio == 4")

    # new sel
    LEPHAD_BJETS = AND("n_tau_jets_medium == 1", "n_b_jets_medium_tauprio == 4","pT_b1 > 40", "pT_b2 > 35", "pT_b3 > 30", "pT_b4 > 25", "pT_tau1 > 20", "pT_tau2 > 20",
    )
    
    LEPHAD_PTS = AND("pT_tau1 > 20", "pT_tau2 > 20")

    HADHAD_JET    = AND("n_tau_jets_medium == 2")
    HADHAD_OS     = AND("n_tau_jets_medium == 2", "OS_tau == 1")
    HADHAD_NOLEP  = AND("n_tau_jets_medium == 2", "OS_tau == 1", "n_sel_el_0p2 == 0", "n_sel_mu == 0")
    HADHAD     = AND("n_b_jets_medium_tauprio == 4", "pT_b1 > 40", "pT_b2 > 35", "pT_b3 > 30", "pT_b4 > 25", "n_tau_jets_medium == 2", "n_sel_el_0p2 == 0", "n_sel_mu == 0", "OS_tau == 1")
    #HADHAD = AND("n_b_jets_medium_tauprio == 4", "n_tau_jets_medium == 2")

    LEPHAD_ETAU = AND("n_tau_jets_medium == 1", "n_b_jets_medium_tauprio == 4","pT_b1 > 40", "pT_b2 > 35", "pT_b3 > 30", "pT_b4 > 25", "n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0","pT_tau1 > 20", "pT_tau2 > 20")
    
    LEPHAD_MUTAU = AND("n_tau_jets_medium == 1", "n_b_jets_medium_tauprio == 4","pT_b1 > 40", "pT_b2 > 35", "pT_b3 > 30", "pT_b4 > 25", "n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0", "pT_tau1 > 20", "pT_tau2 > 20")

LEPHAD_ANY = AND("n_tau_jets_medium == 1", OR(
    AND("n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0"),
    AND("n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0"),
))

HADHAD_TAUS    = AND( "n_tau_jets_medium == 2", "OS_tau == 1", "n_sel_el_0p2 ==0", "n_sel_mu == 0")

# --- boosted large-R objects ---
# tau tau resolved:
# --- LR-bb + resolved ττ (semi-boosted bb) ---
HADHAD_1BB      = AND(
    HADHAD_TAUS,
    "n_b_jets_removed == 2",
    "n_bb_tagged_08Jets == 1",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

LEPHAD_1BB      = AND(
    LEPHAD_ANY,
    "n_b_jets_removed == 2",
    "n_bb_tagged_08Jets == 1",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

HADHAD_2BB      = AND(
    HADHAD_TAUS,
    "n_b_jets_removed == 0",
    "n_bb_tagged_08Jets == 2",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

LEPHAD_2BB      = AND(
    LEPHAD_ANY,
    "n_b_jets_removed == 0",
    "n_bb_tagged_08Jets == 2",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

HADHAD_0BB      = AND(
    HADHAD_TAUS,
    "n_b_jets_removed == 4",
    "n_bb_tagged_08Jets == 0",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

HADHAD_OLDANA      = AND(HADHAD_TAUS,
    "n_b_jets_medium_tauprio == 4"
)

HADHAD_OLDANA_FAILEDMMC      = AND(HADHAD_OLDANA,
    "mass_MH_MMC_unweighted < 20"
)

HADHAD_OLDANA_PASSMMC      = AND(HADHAD_OLDANA,
    "mass_MH_MMC_unweighted > 20"
)


HADHAD_OLDANA_DR      = AND(HADHAD_TAUS,
    "n_b_jets_medium_tauprio == 4", "m_h1 > 25", "m_h1 < 175", "m_h1 > 40", "m_h2 > 20", "m_h2 < 160"
)

HADHAD_OLDANA_SQUARE      = AND(HADHAD_TAUS,
    "n_b_jets_medium_tauprio == 4",  "m_h1_squaremass > 70", "m_h1_squaremass < 160", "m_h2_squaremass > 60", "m_h2_squaremass < 175"
)

LEPHAD_0BB      = AND(
    LEPHAD_ANY,
    "n_b_jets_removed == 4",
    "n_bb_tagged_08Jets == 0",     # LR bb present
    "n_tautau_tagged_08Jets == 0" # no LR ττ (avoid overlap with 2LRB / 1TAU cats)
)

LEPHAD_OLDANA     = AND(
    LEPHAD_ANY,
    "n_b_jets_medium_tauprio == 4", "pT_b1 > 40", "pT_b2 > 35", 
    "pT_b3 > 30", "pT_b4 > 25"
)

HADHAD_2LRB1LRTAU    = AND(
    "n_bb_tagged_08Jets == 2",     # ≥1 LR bb-tagged jet
    "n_tautau_tagged_08Jets == 1" # ≥1 LR ττ-tagged jet
)

HADHAD_2LRB    = AND(
    "n_bb_tagged_08Jets == 2",     # ≥1 LR bb-tagged jet
    "n_tau_jets_medium == 2", "OS_tau == 1",
    "n_tautau_tagged_08Jets == 0" # ≥1 LR ττ-tagged jet
)

# --- fully resolved (2τ + ≥4b) ---
HADHAD_4B      = AND(
    "n_tau_jets_medium == 2", "OS_tau == 1",
    "n_b_jets_medium_tauprio == 4", "pT_b3 > 30", "pT_b4 > 25",
    "n_bb_tagged_08Jets == 0",      # no LR objects
    "n_tautau_tagged_08Jets == 0"
)

# --- LR-tautau + resolved bs (semi-boosted tau) ---
HADHAD_1TAU    = AND(
    "n_tautau_tagged_08Jets == 1", 
    "n_bb_tagged_08Jets == 0",      # no LR bb (keeps it exclusive from 2LRB/2B)
    "n_b_jets_medium_tauprio == 4", "pT_b3 > 30", "pT_b4 > 25"
)

# --- LR(tautau) + exactly 2 resolved b’s ---
HADHAD_1TAU2B  = AND(
    "n_tautau_tagged_08Jets == 1", 
    "n_bb_tagged_08Jets == 1",      # avoid overlap with 2LRB
    "n_b_jets_medium_tauprio == 2"
)

HADHAD            = AND( "n_sel_el_0p2 == 0", "n_sel_mu == 0",
                        OR(HADHAD_0BB, HADHAD_1BB, HADHAD_2BB))

LEPHAD            = AND(
                        OR(LEPHAD_0BB, LEPHAD_1BB, LEPHAD_2BB))

HADHAD_BB         = AND("pT_b1 > 40", "pT_b2 > 35", "n_sel_el_0p2 == 0", "n_sel_mu == 0",
                        OR(HADHAD_2B, HADHAD_2LRB))     # bb-focused (resolved or semi-boosted)

HADHAD_TAUTAU     = AND("pT_b1 > 40", "pT_b2 > 35", "n_sel_el_0p2 == 0", "n_sel_mu == 0",
                        OR(HADHAD_1TAU, HADHAD_1TAU2B))  # tau-boost

HADHAD_RESOLVED   = AND("pT_b1 > 40", "pT_b2 > 35", "n_sel_el_0p2 == 0", "n_sel_mu == 0",
                        HADHAD_4B)


HADHAD_RESOLVED_DR   = AND(HADHAD_4B, "m_h1 > 40", "m_h1 < 175", "m_h1 > 40", "m_h2 > 30" "m_h2 < 175")
HADHAD_RESOLVED_SQUARE   = AND(HADHAD_4B, "m_h1_squaremass > 70", "m_h1_squaremass < 175", "m_h2_squaremass > 70", "m_h2_squaremass < 175")

HADHAD_BOOST      = AND("n_sel_el_0p2 == 0", "n_sel_mu == 0",
                        HADHAD_2LRB1LRTAU)  # fully boosted (LR-bb & LR-ττ)

TRUTHTAU = AND("pT_reco_matched_tau_tau_jet_2 > 0", "pT_reco_matched_tau_tau_jet_1 > 0")
TRUTH4B = AND("pT_reco_matched_b_jet_1 > 0", "pT_reco_matched_b_jet_2 > 0", "pT_reco_matched_b_jet_3 > 0", "pT_reco_matched_b_jet_4 > 0")
TRUTHALL = AND(TRUTHTAU, TRUTH4B)


HH4BMATCH = AND("n_b_jets_medium_tauprio == 4", "n_reco_matched_B_hadrons == 4")


doNN = False
if doNN:
        HADHAD     = AND("mlp_score > 0.0")
        LEPHAD     = AND("mlp_score > 0.0")

        COMBINED   = AND("mlp_score > 0.0")
        SELECTION = {
            "Total":        AND("mlp_score > 0"),
            "LepHad":       AND(LEPHAD),
            "HadHad":       AND(HADHAD),
            "Combined":     AND(COMBINED),
            "bbaa": AND("mlp_score > 0")
        }
else:
    SELECTION = {
        "Total":        AND("njets > 0"),
        "bbaa": AND("m_yy > 122", "m_yy < 127", "m_bb < 160"),
        "LepHad":       AND(LEPHAD),
        "HadHad":       AND(HADHAD),
        "HadHad_bb":    AND(HADHAD_BB),
        "HadHad_tautau": AND(HADHAD_TAUTAU),
        "HadHad_boosted": OR(HADHAD_1BB, HADHAD_2BB),
        "Combined":     OR(HADHAD, LEPHAD),
        "HadHad_resolved": AND(HADHAD_0BB),
        "HadHad_1BB": AND(HADHAD_1BB),
        "HadHad_2BB": AND(HADHAD_2BB),
        "LepHad_resolved": AND(LEPHAD_0BB),
        "LepHad_1BB": AND(LEPHAD_1BB),
        "LepHad_2BB": AND(LEPHAD_2BB),
        "HadHad_old": AND(HADHAD_OLDANA),
        "HadHad_old_dRCuts": AND(HADHAD_OLDANA_DR),
        "HadHad_old_squareCuts": AND(HADHAD_OLDANA_SQUARE),
        "HadHad_resolved_dRCuts": AND(HADHAD_RESOLVED_DR),
        "HadHad_resolved_squareCuts": AND(HADHAD_RESOLVED_SQUARE),
        "LepHad_old": AND(LEPHAD_OLDANA),
        "TruthRecoTau": AND(TRUTHTAU),
        "TruthRecob": AND(TRUTH4B),
        "TruthRecoall": AND(TRUTHALL),
        "HH4bMatch": AND(HH4BMATCH),
        "HadHad_failedMMC": AND(HADHAD_OLDANA_FAILEDMMC),
        "HadHad_passMMC": AND(HADHAD_OLDANA_PASSMMC)
    }

        
if MMC:
    
    LEPHAD = AND("n_b_jets_medium_tauprio == 4", "n_tau_jets_medium == 1", OR(
        AND("n_sel_el_0p2 == 1", "OS_taue == 1", "n_sel_mu == 0"),
        AND("n_sel_mu == 1", "OS_taumu == 1", "n_sel_el_0p2 == 0"),
        
    ), "weighted_MMC_para_perp_vispTcal > 0",  "weighted_MMC_para_perp_vispTcal < 300", "m_h1 < 175", "m_h1 > 40", "m_h2 > 20", "m_h2 < 160")
    
    #HADHAD = AND("n_genmatch_truth_tau1 == 1", "n_genmatch_truth_tau2 == 1")
    HADHAD = AND("n_tau_jets_medium == 2","n_b_jets_medium_tauprio == 4",  "OS_tau == 1", "n_sel_el_0p2 == 0", "n_sel_mu == 0", "weighted_MMC_para_perp_vispTcal > 0",  "weighted_MMC_para_perp_vispTcal < 300", "m_h1 < 175", "m_h1 > 40", "m_h2 > 20", "m_h2 < 160",)

    SELECTION = {
        "Total": AND("n_jets_sel > -1"),
        "LepHad": AND(LEPHAD),
        "HadHad": AND(HADHAD)        
    }


if Truth:
    HADHAD = AND("pT_B_had_1 > 25", "pT_B_had_2 > 25", "pT_B_had_3 > 25", "pT_B_had_4 > 25", "n_visible_tauhad == 2", "pT_visible_tauhad_1 > 25", "pT_visible_tauhad_2 > 25")
    SELECTION = {
        "Total": AND("n_jets_sel > -1"),
        "LepHad": AND(LEPHAD),
        "HadHad": AND(HADHAD)        
    }

print(f"SELECTION: {SELECTION}")

# --------------------------------------------------
# Physics constants
# --------------------------------------------------
LUMINOSITY_PB = 30000000  # 30 ab⁻1
N_BINS_1D = 30
N_BINS_2D = 50
SKIP_VARS = {"weight", "reweight", "eventNumber"}


if doSignal:
    processes: List[str] = ["mg_pp_hhh_4b2tau"]
elif doHiggs:
    processes: List[str] = ["mgp8_pp_hhh_84TeV"]
elif newprod:
    if doTtbb:
        processes: List[str] = [
        "mgp8_pp_hhh_84TeV", "mgp8_pp_ttbb_4f_84TeV"
    ]
    else:
        processes: List[str] = [
            "mgp8_pp_hhh_84TeV",
            "mgp8_pp_ttbb_4f_84TeV",
            "pwp8_pp_hh_k3_1_k4_1_84TeV",
            "mgp8_pp_tth_5f_84TeV",
            "mgp8_pp_tttt_5f_84TeV",
            "mgp8_pp_ttz_5f_84TeV",
            "mgp8_pp_zzz_5f_84TeV",
            "mgp8_pp_hh01j_5f_84TeV",
            "mgp8_pp_zz012j_4f_84TeV",
            "mgp8_pp_z_4f_84TeV",
            "mgp8_pp_h_5f_84TeV",
            "mgp8_pp_hhjj_5f_84TeV",
            "mgp8_pp_jjaa_5f_84TeV",
            "mgp8_pp_h012j_5f_84TeV",
            "mgp8_pp_tth01j_5f_84TeV",
            "mgp8_pp_vbf_h01j_5f_84TeV"
        ]
else:
    processes: List[str] = [
        "mg_pp_hhh_4b2tau",
        "mg_pp_zzz_5f",
        #"mg_pp_hh01j_5f",
        "mg_pp_tttt_5f",
        "mg_pp_tth01j_5f",
        "mg_pp_ttz_5f",
        "mg_pp_ttbb_4f"
    ]
if newcard:
    processes = ["mgp8_pp_hhh_84TeV","mgp8_pp_hhh_84TeV_fixcard"]

testingfakes = False
if testingfakes:
    processes = ["mgp8_pp_ttbb_4f_84TeV"]

if newsamples:
    processes = ["mgp8_pp_zz012j_4f_84TeV", "mgp8_pp_hhjj_5f_84TeV"]

compare_energies = False

if compare_energies:
    processes = ["mgp8_pp_hhh_14TeV", "mgp8_pp_hhh_84TeV"]

if bbaa:
    processes = ["pwp8_pp_hh_k3_1_k4_1_84TeV",
            "mgp8_pp_jjaa_5f_84TeV",
            "mgp8_pp_h012j_5f_84TeV",
            "mgp8_pp_tth01j_5f_84TeV",
            "mgp8_pp_vbf_h01j_5f_84TeV"]
    
SIGNAL = processes[0]
BACKGROUNDS = processes[1:]

# --------------------------------------------------
# Selection and variables
# --------------------------------------------------


# (n_tau_jets_medium >= 1 and (n_sel_el_0p2 >= 1 or n_sel_mu >= 1) and (OS_taue or OS_taumu))



OVERLAY_PAIRS: List[Tuple[str, str]] = [
    ("dR_h1", "dR_h2"),
    ("m_h1", "m_h1_dR"),
    ("m_h1", "m_h2"),
    ("m_h1_absmass", "m_h1_squaremass","m_h1", "m_h1_random"),
    ("m_h2_absmass", "m_h2_squaremass", "m_h2", "m_h2_random"),
    ("m_h1_absmass_massordered", "m_h1_squaremass_massordered"),
    ("m_h2_absmass_massordered", "m_h2_squaremass_massordered"),
    ("m_h1_absmass", "m_h1_absmass_massordered"),
    ("m_h2_absmass", "m_h2_absmass_massordered"),
    ("m_h1_squaremass", "m_h1_squaremass_massordered"),
    ("m_h2_squaremass", "m_h2_squaremass_massordered"),
    ("m_tautau_vis_OS", "m_tautau_mT", "m_tautau_smin", "m_tautau_col"),
    ("pT_B_had_1", "pT_B_had_2", "pT_B_had_3", "pT_B_had_4"),
    ("pT_visible_tauhad_1", "pT_visible_tauhad_2"),
    ("pT_leading_taumu_fromH", "pT_leading_tauel_fromH"),
    ("m_h1_LR", "m_h2_LR"),
    ("m_Htautau_reco", "m_h1_recomatch", "m_h2_recomatch"),
    ("m_h1_recomatch", "m_h1_absmass"),
    ("m_h2_recomatch", "m_h2_absmass"),
    ("metsig_est", "metsig_derived"),
    ("m_h1", "m_h1_LR"),
    ("m_h2", "m_h2_LR"),
    ("sigma_12", "sigma_23"),
    ("sigma_23", "sigma_13"),
    ("sigma_12", "sigma_13"),
    ("m_tautau_vis_OS", "MMC_mass_lepHad"),
    ("m_Htautau_reco", "m_tautau_vis_OS"),
    ("m_tautau_vis_OS", "metOnly_mode", "unweighted_MMC_para_perp"),
    ("m_tautau_vis_OS", "metRatio_mode", "unweighted_MMC_para_perp"),
    ("m_tautau_vis_OS", "metRatio_mode_vispTcal", "unweighted_MMC_para_perp_vispTcal"),
    ("m_tautau_vis_OS", "ratioOnly_mode_vispTcal", "unweighted_MMC_para_perp_vispTcal"),
    ("unweighted_MMC_para_perp_vispTcal", "full_mode_vispTcal", "ratioOnly_mode_vispTcal", "angleOnly_mode_vispTcal", "metOnly_mode_vispTcal"),
    ("unweighted_MMC_para_perp_vispTcal", "full_mode_vispTcal", "weighted_MMC_para_perp_vispTcal"),

    ("m_tautau_vis_OS", "ratioOnly_mode", "unweighted_MMC_para_perp"),
    ("m_tautau_vis_OS", "metOnly_mode", "unweighted_MMC_para_perp"),
    ("unweighted_MMC_para_perp", "full_mode", "ratioOnly_mode", "angleOnly_mode", "metOnly_mode"),
    ("m_Htautau_reco", "m_Htautau_MC"),
    ("m_Htautau_reco", "m_h1_recomatch"),
    ("m_Htautau_reco", "m_h2_recomatch"),
    ("unweighted_MMC_para_perp_vispTcal", "weighted_MMC_para_perp_vispTcal"),

    ("unweighted_MMC_para_perp", "weighted_MMC_para_perp"),
    ("weighted_MMC_para_perp", "m_tautau_vis_OS"),
    ("weighted_MMC_para_perp_vispTcal", "m_tautau_vis_OS"),

    ("unweighted_MMC_para_perp", "m_tautau_vis_OS"),
    ("pT_truth_hbb1", "pT_truth_hbb2", "pT_truth_htautau"),
    ("dR_truth_tautau", "dR_truth_bb2", "dR_truth_bb1"),
    ("MMC_unweighted_mass_80", "MMC_unweighted_mass_60", "MMC_unweighted_mass"),
    ("MMC_unweighted_mass_80", "MMC_weighted_mass_80_1_10"),
    ("m_tautau_vis_OS", "MMC_mass_angular_weighted"),
    ("m_tautau_vis_OS", "MMC_unweighted_mass_80"),
    ("m_tautau_vis_OS",  "MMC_mass_METScan_weighted_80_10"),
    ("MMC_unweighted_mass_80", "MMC_mass_METScan_weighted_80_10", "MMC_mass_METScan_angular_weighted_80_10"),
    ("MMC_unweighted_mass_80", "MMC_METScan_unweighted", "MMC_mass_METScan_angular_weighted_40_20"),
    ("m_h1", "m_h1_square"),
    ("m_h2", "m_h2_dR"),
    ("m_h2", "m_h2_square"),
    ("m_tautau_vis_OS", "smin"),
    ("m_tautau_vis_OS", "MMC_mass"),
    ("m_tautau_vis_OS", "m_tautau_col"),
    ("m_tautau_vis_OS", "m_tautau_vis"),
    ("m_tautau_vis_OS", "m_tautau_vis_OS_nob"),
    ("m_tautau_col", "smin"),
    ("m_tautau_vis_OS", "smin_exp"),
    ("m_tautau_vis_OS", "tautau_mmc"),
    ("m_tautau_vis_OS", "MH_MMC_mass"),
    ("m_tautau_col", "tautau_mmc"),
    ("m_hhh_truth", "m_hhh_vis"),
    ("m_hhh_vis", "m_hhh_vis_dR"),
    ("m_h_recobb_lead", "m_h_recobb_lead_mucorr")
]

HEATMAP_PAIRS: List[Tuple[str, str]] = [
    ("m_h1", "m_h2"),
    ("pT_h1", "dR_h1"),
    ("pT_h2", "dR_h2"),
    ("MET", "met_res"),
    ("sigma_12", "sigma_13"),
    ("sigma_23", "sigma_13"),
    ("sigma_12", "sigma_23"),
    ("sigma_x", "sigma_y"),
    ("m_tautau_vis_OS", "weighted_MMC_para_perp_vispTcal"),
    ("m_tautau_vis_OS", "unweighted_MMC_para_perp_vispTcal"),
    ("m_h1", "m_tautau_vis_OS"),
    ("m_h2", "m_tautau_vis_OS"),
    ("pT_Htautau_cand", "m_tautau_vis_OS"),
    ("m_h1", "m_tautau_vis_OS_nob"),
    ("pT_tau1", "dphi_sig_met_real_met"),
    ("pT_h1", "pT_h2"),
    ("dEta_h1", "pT_h1"),
    ("pT_h1", "dPhi_h1"),
    ("pT_h2", "dEta_h2"),
    ("pT_b1", "pT_h1"),
    ("pT_b2", "pT_h1"),
    ("pT_b3", "pT_h2"),
    ("pT_b4", "pT_h2"),
    ("pT_tau1", "pT_Htautau_cand"),
    ("pT_tau2", "pT_Htautau_cand"),
    ("dphi_met_tautau", "m_tautau_col"),
    ("m_tautau_vis_OS", "m_tautau_col"),
    ("m_tautau_col", "dphi_sig_met_real_met"),
    ("m_h1", "m_h1_dR"),
    ("m_h2", "m_h2_dR"),
    ("m_h1", "m_h1_square"),
    ("m_h2", "m_h2_square"),
    ("m_h1_dR", "m_h1_square"),
    ("m_h2_dR", "m_h2_square"),
    ("pT_h1", "isRecoMatchedHHH"),
    ("sigma_12", "sigma_23"),
    ("sigma_13", "sigma_23")

]

XLIM_MAP: Dict[str, Tuple[float, float]] = {
    "m_tautau_col": [25, 600],
    "m_tautau_mT": [25, 200],
    "m_tautau_smin": [25, 200],
    "m_tautau_vis": [25, 200],
    "m_tautau_vis_OS": [25, 200],
    "m_h1_recomatch": [60, 170],
    "m_h2_recomatch": [60, 170],
    "m_Htautau_reco": [20, 150],
    "m_Htautau_MC": [20, 150],
    "pT_leading_taumu_fromH": [0.001, 250],
    "pT_leading_tauel_fromH": [0.001, 250],
    "MH_MMC_mass": [30, 300],
    "MH_MMC_mass_sat": [30, 300],
    "m_tautau_vis_OS_nob": [25, 150],
    "mtautau_new": [0, 300],
    "m_h1": [40, 175],
    "m_h1_LR": [0, 200],
    "asym_x_y": [-60, 60],
    "m_h2_LR": [0, 200],
    "m_h2": [20, 160],
    "m_h1_absmass": [25, 300],
    "m_h1_absmass_massordered": [25, 300],
    "pT_B_had_1": [0, 400],
    "pT_B_had_2": [0, 350],
    "pT_B_had_3": [0, 250],
    "pT_B_had_4": [0, 150],
    "pT_visible_tauhad_1": [0, 300],
    "pT_visible_tauhad_2": [0, 300],
    "m_h1_squaremass_massordered": [25, 300],
    "m_h1_random": [25, 300],

    "m_h2_absmass": [25, 300],
    "m_h2_absmass_massordered": [25, 300],
    "m_h2_random": [25, 300],
    "m_h2_squaremass_massordered": [25, 300],

    "dphi_truth_vismis_1": [-0.4, 0.4],
    "dphi_truth_vismis_2": [-0.4, 0.4],
    "m_h2_squaremass": [25, 200],
    "m_h1_squaremass": [25, 200],
    "m_h_recobb_lead": [50, 200],
    "m_h_recobb_lead_mucorr": [50, 200],
    "m_h_recobb_sublead": [50, 200],
    "m_h_recotautau": [50, 200],
    "m_tautau_col_truthmatch": [50, 200],
    "m_hhh_vis": [300, 1500],
    "m_hhh_vis_LR": [300, 2000],
    "m_hhh_vis_dR": [300, 2000],
    "m_hhh_truth": [300, 2000],
    "pT_truth_hbb1": [0, 800],
    "pT_truth_hbb2": [0, 800],
    "pT_truth_htautau": [0, 800],
    "MET": [0, 400],
    "gen_MET_y": [-200, 200],
    "gen_MET_x": [-200, 200],
    "d_MET_x": [-100, 100],
    "d_MET_y": [-100, 100],
    "d_MET": [-100, 100],
    "metOnly_mode": [40, 200],
    "ratioOnly_mode": [40, 200],
    "metRatio_mode": [40, 250],
    "metRatio_mode_vispTcal": [40, 250],
    "met_res": [0, 40],
    "pT_h1": [20, 400],
    "pT_h2": [20, 400],
    "pT_b1": [40, 400],
    "pT_b2": [35, 200],
    "pT_b3": [30, 100],
    "pT_b4": [25, 100],
    "pT_tau1": [20, 600],
    "pT_tau2": [20, 500],
    "x1": [0, 1],
    "x2": [0, 1],
    "sqrt_x1x2": [0, 1],
    "dR_h1": [0, 5.0],
    "dR_tautau": [0.4, 5.0],
    "smin": [0, 200],
    "smin_exp": [0, 200],
    "mmc_peak": [50, 200],
    "MMC_mass": [0, 200],
    "MMC_unweighted_mass_80": [50, 200],
    "MMC_weighted_mass_80_1_10": [50, 200],
    "sigma_x": [0, 1],
    "sigma_y": [0, 1],
    "unweighted_MMC_para_perp": [10, 1000],
    "weighted_MMC_para_perp": [40, 250],
    "weighted_MMC_para_perp_vispTcal": [50, 300],
    "unweighted_MMC_para_perp_vispTcal": [0, 400],

    "mass_again": [20, 200],
    "BDT_score": [0.5, 1],
    "NN_score": [0.9, 1],
    "RMS_mjj": [0, 500],
    "RMS_deta": [0, 4],
    "RMS_dR": [0, 5],
    "dR_h1": [0, 5],
    "dR_h2": [0,5],
    "xwt": [0, 11],
    "metsig_est": [0, 15],
    "metsig_derived": [0, 15],

    "mTb_min": [0, 150],
    "dm_h2_absmassdR": [-200, 200],
    "dm_h1_absmassdR": [-200, 200],
    "x_tau2": [0, 1],
    "x_tau1": [0, 1],
    "angle3D_vis_mis_tau1": [0, 0.2],
    "angle3D_vis_mis_tau2": [0, 0.2],
    "p_mis_1": [0, 1000],
    "p_mis_2": [0, 1000],
    "p_vis_1": [0 , 1000],
    "p_vis_2": [0 , 1000],
    "m_vis_truthHadronicTaus_2": [0 , 2],
    "m_vis_truthHadronicTaus_1": [0 , 2],
    "unweighted_MMC_para_perp": [25, 250],
    "weighted_MMC_para_perp": [30, 200],
    
    # bbaa samples
    "m_hh": [200, 1500],
    "m_bb": [80, 200],
    "m_yy": [110, 140],
    "max_dr_yb": [1, 7],
    "min_dr_yb": [0.4, 4],
    "dR_hh": [0, 6],
    "dR_hh": [0, 6],
    "pT_hyy": [0, 800],
    "pT_hbb": [0, 800],

}


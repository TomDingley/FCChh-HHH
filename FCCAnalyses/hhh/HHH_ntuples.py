'''
Analysis example for FCC-hh, using gg->HHH->4b2tau production events to check the Delphes b-tagging efficiencies
'''
from argparse import ArgumentParser
import ROOT
# Mandatory: Analysis class where the user defines the operations on the
# dataframe.
class Analysis():
    '''
    Validation of Delphes b-tagging efficiencies in HH->tautauyy events.
    '''
    def __init__(self, cmdline_args):
        parser = ArgumentParser(
            description='Additional analysis arguments',
            usage='Provide additional arguments after analysis script path')
        # parser.add_argument('--bjet-pt', default='10.', type=float,
        #                     help='Minimal pT of the selected b-jets.')
        # Parse additional arguments not known to the FCCAnalyses parsers
        # All command line arguments know to fccanalysis are provided in the
        # `cmdline_arg` dictionary.
        #self.ana_args, _ = parser.parse_known_args(cmdline_args['unknown'])

        # Mandatory: List of processes to run over
        self.process_list = {
            # # Add your processes like this: 
            ## '<name of process>':{'fraction':<fraction of events to run over>, 'chunks':<number of chunks to split the output into>, 'output':<name of the output file> }, 
            # # - <name of process> needs to correspond either the name of the input .root file, or the name of a directory containing root files 
            # # If you want to process only part of the events, split the output into chunks or give a different name to the output use the optional arguments
            # # or leave blank to use defaults = run the full statistics in one output file named the same as the process:
            #'mgp8_pp_ttz_5f_84TeV':{'output': 'mgp8_pp_ttz_5f_84TeV', 'chunks': 510},
            #'mgp8_pp_zzz_5f_84TeV':{'output': 'mgp8_pp_zzz_5f_84TeV', 'chunks': 509},
            #'mgp8_pp_tttt_5f_84TeV':{'output': 'mgp8_pp_tttt_5f_84TeV', 'chunks': 509},
            #'mgp8_pp_tth_5f_84TeV': {'output': 'mgp8_pp_tth_5f_84TeV', 'chunks': 600},
            #'mgp8_pp_ttbb_4f_84TeV':{'output': 'mgp8_pp_ttbb_4f_84TeV', 'chunks': 1000, 'fraction': 1.0},
            #'mgp8_pp_hhh_84TeV':{'output': 'mgp8_pp_hhh_84TeV', 'chunks': 437, 'fraction': 1.0},
            'pwp8_pp_hh_k3_1_k4_1_84TeV':{'output': 'pwp8_pp_hh_k3_1_k4_1_84TeV', 'chunks': 500, 'fraction': 1.0},
            #'mgp8_pp_h012j_5f_84TeV':{'output': 'mgp8_pp_h012j_5f_84TeV', 'chunks': 5},
            #'mgp8_pp_z0123j_4f_84TeV':{'output': 'mgp8_pp_z0123j_4f_84TeV', 'chunks': 20}
            #'mgp8_pp_z_4f_84TeV':{'output': 'mgp8_pp_z_4f_84TeV', 'chunks': 404, 'fraction': 1.0},
            #'mgp8_pp_h_5f_84TeV':{'output': 'mgp8_pp_h_5f_84TeV', 'chunks': 300}
        }

        # Mandatory: Input directory where to find the samples, or a production tag when running over the centrally produced
        # samples (this points to the yaml files for getting sample statistics)
        self.input_dir = '/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/LRjets_tautau/fcc_v07/II'
        #self.input_dir = '/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/14TeV_6b/fcc_v07/II'
        #self.input_dir = '/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_calibration_200925_check/fcc_v07/II'
        #self.input_dir = '/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_decayed_271225/fcc_v07/II'
        
        # ttbb emutau sample:
        self.input_dir = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ttbb_emutau_new/fcc_v07/II"
        self.input_dir = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/FullStatistics_tautau/fcc_v07/II"
        
        # hh + jet samples
        self.input_dir = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ggHH_quartic/fcc_v07/II"
        # noDecays
        #self.input_dir = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/noDecayttbar/fcc_v07/II"
        # Optional: output directory, default is local running directory
        self.output_dir = "hhjets_tautau"
        self.comp_group = "dingleyt"

        
        # Optional: analysisName, default is ''
        # self.analysis_name = 'My Analysis'

        # Optional: number of threads to run on, default is 'all available'
        # self.n_threads = 4

        # Optional: running on HTCondor, default is False
        self.run_batch = False

        # Optional: Use weighted events
        self.do_weighted = True 

        # Optional: test file that is used if you run with the --test argument (fccanalysis run ./examples/FCChh/ggHH_taubyy/analysis_stage1.py --test)
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/root_Decay_nonisolated/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/nonIso_Nodecay_0p3_fixCards_fakes/fcc_v07/II/mgp8_pp_ttbb_4f_84TeV/events_000396011.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/nonIso_Nodecay_0p3_fixCards_fakes/fcc_v07/II/mgp8_pp_ttz_5f_84TeV/events_002759849.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_calibration/fcc_v07/II/mgp8_pp_h012j_5f_84TeV/events_000112396.root"
        # ZZ
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/Object_calibration_GenET_Decay/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_02.root"

        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/newsamples_GenET_hh_ZZ/fcc_v07/II/mgp8_pp_zz012j_4f_84TeV/events_009743350.root"
        
        # HH+1j
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/newsamples_GenET_hh_ZZ/fcc_v07/II/mgp8_pp_hhj_5f_84TeV/events_054179512.root"
        # test for higher tau pT thresholds
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/GenET_METres_calib/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_01.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_calibration_200925/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_012559928.root"
        #
        # 100 events
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_calibration_200925_check/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_1159670.root"
        # 10k events
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_calibration_200925_check/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_013135542.root"

        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/LRjets_tautau/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"
       # truth jets
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/14TeV_6b/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"
       
        # 14 TeV
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/14TeV_6b/fcc_v07/II/mgp8_pp_hhh_14TeV/events_003945503.root"
        # 84 TeV
        self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/LRjets_tautau/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"

        # also define test file to check rejection of ttbb (emu)
        
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ttbb_emutau/fcc_v07/II/mgp8_pp_ttbb_4f_84TeV/events_000396011.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/nonIso_decay_0p3flavas/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"
        
        # 20GeV
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/nonIso_decay_0p3flavas_20GeVTaupT/fcc_v07/II/mgp8_pp_hhh_84TeV/events_000910581.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/newsamples_5050/fcc_v07/II/mgp8_pp_hh1j_5f_84TeV/events_01.root"
        # new batch for xmas MMC calibration
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_decayed_271225/fcc_v07/II/mgp8_pp_z_4f_84TeV/events_4Mzg.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_decayed_271225/fcc_v07/II/mgp8_pp_h_5f_84TeV/events_000063338.root"
        
        # testing hh+jets:
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/MMC_decayed_271225/fcc_v07/II/mgp8_pp_hhj_5f_84TeV/events_01.root"
        
        # ttbbemutau
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/LRjets_tautau/fcc_v07/II/mgp8_pp_ttbb_4f_84TeV/events_000396011.root"
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ttbb_emutau_new/fcc_v07/II/mgp8_pp_ttbb_4f_84TeV/events_000396011.root"
        
        # testing hhjj:
        #self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/FullStatistics_tautau/fcc_v07/II/mgp8_pp_hhjj_5f_84TeV/events_21.root"
        
        # hh quartic powheg
        self.test_file = "/data/atlas/users/dingleyt/FCChh/eventProd/EventProducer/ggHH_quartic/fcc_v07/II/pwp8_pp_hh_k3_1_k4_1_84TeV/events_547.root"
    # Mandatory: analyzers function to define the analysis graph, please make
    # sure you return the dataframe, in this example it is dframe2
    def analyzers(self, dframe):
        '''
        Analysis graph.
        '''
        dframe2 = (
            dframe
            
                .Define("eventNumber", "EventHeader.eventNumber")
                # can we filter on the event number to save comp time?
                
                # weights
                # event-based information
                .Define("weight",  "EventHeader.weight[0]")
                
                # define 9 separate weights for the k3k4 scanning
                .Define("weight_k3m1_k4m1",     "_EventHeader_weights[0]")
                .Define("weight_k30_k4m2",      "_EventHeader_weights[1]")
                .Define("weight_k3m2_k40",      "_EventHeader_weights[2]")
                .Define("weight_k30_k4m1",      "_EventHeader_weights[3]")
                .Define("weight_k3m1_k40",      "_EventHeader_weights[4]")
                .Define("weight_k3m2_k4m1",     "_EventHeader_weights[5]")
                .Define("weight_k3m1_k4m2",     "_EventHeader_weights[6]")
                .Define("weight_k3m0p5_k4m1",   "_EventHeader_weights[7]")
                .Define("weight_k3m1p5_k4m1",   "_EventHeader_weights[8]")
                
                
                # first define all truth particles we'll use
                #------------------------------------------------------------------------------------------------------------
                #                                       Truth Particle Preparation
                #------------------------------------------------------------------------------------------------------------
                .Define("mc_particles", "Particle")
                .Alias("mc_parents", "_Particle_parents.index")
                .Alias("mc_daughters", "_Particle_daughters.index")
                
                #  ---- truth electrons ----
                .Define("truth_e",       "FCCAnalyses::MCParticle::sel_pdgID(11, true)(mc_particles)")
                .Define("truth_e_eta",   "FCCAnalyses::MCParticle::sel_eta(4)(truth_e)")
                .Define("truth_e_selpt", "FCCAnalyses::MCParticle::sel_pt(0)(truth_e_eta)")
                .Define("n_truth_el",     "FCCAnalyses::MCParticle::get_n(truth_e_selpt)")
                .Define("pT_truth_e",    "FCCAnalyses::MCParticle::get_pt(truth_e_selpt)")
                .Define("sorted_truth_el", "AnalysisFCChh::SortMCByPt(truth_e_selpt)")

                .Define("truth_el1", "sorted_truth_el[0]")
                .Define("pT_truth_el1", "FCCAnalyses::MCParticle::get_pt(sorted_truth_el)[0]")
                
                # define stable electrons: (1 = stable, 2 = intermediate, 4 = beam)
                .Define("electrons_truth_prompt", "FCCAnalyses::MCParticle::sel_genStatus(1)(truth_e_selpt)") 
                .Define("n_truth_prompt_e",     "FCCAnalyses::MCParticle::get_n(electrons_truth_prompt)")
                
                # ---- truth muons (fiducial) ----
                .Define("truth_mu",       "FCCAnalyses::MCParticle::sel_pdgID(13, true)(mc_particles)")
                .Define("truth_mu_eta",   "FCCAnalyses::MCParticle::sel_eta(4)(truth_mu)")
                .Define("truth_mu_selpt", "FCCAnalyses::MCParticle::sel_pt(0)(truth_mu_eta)")
                .Define("n_truth_mu",     "FCCAnalyses::MCParticle::get_n(truth_mu_selpt)")
                .Define("pt_truth_mu",    "FCCAnalyses::MCParticle::get_pt(truth_mu_selpt)")
                
                .Define("sorted_truth_mu", "AnalysisFCChh::SortMCByPt(truth_mu_selpt)")

                .Define("truth_mu1", "sorted_truth_mu[0]")
                .Define("pT_truth_mu1", "FCCAnalyses::MCParticle::get_pt(sorted_truth_mu)[0]")
                
                # define stable muons: (1 = stable, 2 = intermediate, 4 = beam)
                .Define("muons_truth_prompt", "FCCAnalyses::MCParticle::sel_genStatus(1)(truth_mu_selpt)") 
                .Define("n_truth_prompt_mu",     "FCCAnalyses::MCParticle::get_n(muons_truth_prompt)")

                # Truth taus
                .Define("truth_tau", "FCCAnalyses::MCParticle::sel_pdgID(15, true)(mc_particles)") 
                .Define("truth_tau_eta",   "FCCAnalyses::MCParticle::sel_eta(4)(truth_tau)")
                .Define("truth_tau_selpt", "FCCAnalyses::MCParticle::sel_pt(0)(truth_tau_eta)")
                .Define("n_truth_tau",     "FCCAnalyses::MCParticle::get_n(truth_tau_selpt)")
                .Define("pt_truth_tau",    "FCCAnalyses::MCParticle::get_pt(truth_tau_selpt)")
                .Define("eta_truth_tau",    "FCCAnalyses::MCParticle::get_eta(truth_tau_selpt)")
                .Define("phi_truth_tau",    "FCCAnalyses::MCParticle::get_phi(truth_tau_selpt)")
                
                .Define("sorted_truth_tau", "AnalysisFCChh::SortMCByPt(truth_tau_selpt)")

                
                .Define("truth_tau1", "sorted_truth_tau[0]")
                .Define("pT_truth_tau1", "FCCAnalyses::MCParticle::get_pt(sorted_truth_tau)[0]")
                
                .Define("truth_tau2", "sorted_truth_tau[1]")
                .Define("pT_truth_tau2", "FCCAnalyses::MCParticle::get_pt(sorted_truth_tau)[1]")
                
                #---------------------------
                # Truth B-Hadrons
                #---------------------------
                .Define("truth_b", "AnalysisFCChh::getBhadron(mc_particles, mc_parents)")
                .Define("truth_b_sel", "FCCAnalyses::MCParticle::sel_eta(4)(truth_b)")
                .Define("truth_b_selpt", "FCCAnalyses::MCParticle::sel_pt(0)(truth_b_sel)")
                .Define("n_truth_b",     "FCCAnalyses::MCParticle::get_n(truth_b_selpt)")
                
                .Define("sorted_truth_b", "AnalysisFCChh::SortMCByPt(truth_b_selpt)")

                .Define("truth_b1", "sorted_truth_b[0]")
                .Define("pT_truth_b1", "FCCAnalyses::MCParticle::get_pt(sorted_truth_b)[0]")
                
                .Define("truth_b2", "sorted_truth_b[1]")
                .Define("pT_truth_b2", "FCCAnalyses::MCParticle::get_pt(sorted_truth_b)[1]")
                
                .Define("truth_b3", "sorted_truth_b[2]")
                .Define("pT_truth_b3", "FCCAnalyses::MCParticle::get_pt(sorted_truth_b)[2]")
                
                .Define("truth_b4", "sorted_truth_b[3]")
                .Define("pT_truth_b4", "FCCAnalyses::MCParticle::get_pt(sorted_truth_b)[3]")

                #------------------------------------------------------------------------------------------------------------
                #                                       Reconsructed Particles
                #------------------------------------------------------------------------------------------------------------                
                # ------------------ jets ------------------
                # #selected jets above a pT threshold of 25 GeV
                .Define("selected_jets_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(Jet)") 
                .Define("selected_jets", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selected_jets_pt)") 
                .Define("n_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_n(selected_jets)")
                .Define("pT_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_pt(selected_jets)")
                .Define("eta_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_eta(selected_jets)")
                .Define("phi_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_phi(selected_jets)")
                
                
                # now also look for truth-jets
                #
                #.Define("selected_truth_jets_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(GenJet04)") 
                #.Define("selected_truth_jets", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selected_truth_jets_pt)") 
                #.Define("n_truth_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_n(selected_truth_jets)")
                #.Define("pT_truth_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_pt(selected_truth_jets)")
                #.Define("eta_truth_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_eta(selected_truth_jets)")
                #.Define("phi_truth_jets_sel",  "FCCAnalyses::ReconstructedParticle::get_phi(selected_truth_jets)")
                #.Define("sorted_truth_jets", "AnalysisFCChh::SortParticleCollection(selected_truth_jets)")
                ## leading truth jet
                #.Define("truth_jet_1", "sorted_truth_jets[0]")
                #.Define("pT_truth_jet_1",  "FCCAnalyses::ReconstructedParticle::get_pt({truth_jet_1})[0]")
                ##.Filter("pT_truth_jet_1 > 0")
                #.Define("phi_truth_jet_1", "FCCAnalyses::ReconstructedParticle::get_phi({truth_jet_1})[0]")
                #.Define("eta_truth_jet_1", "FCCAnalyses::ReconstructedParticle::get_eta({truth_jet_1})[0]")
                #.Define("matched_truth_jet_1",
                #        "AnalysisFCChh::find_reco_reco_matches_unique({truth_jet_1}, selected_jets, 0.3)")
                #.Define("pT_matched_truth_jet_1",
                #        "FCCAnalyses::ReconstructedParticle::get_pt(matched_truth_jet_1)[0]")
                #.Define("phi_matched_truth_jet_1",
                #        "FCCAnalyses::ReconstructedParticle::get_phi(matched_truth_jet_1)[0]")
                ##.Filter("pT_matched_truth_jet_1 > 0")
                #.Define("pT_response_truth_jet_1", "pT_matched_truth_jet_1 / pT_truth_jet_1")
                #.Define("dpT_jet_1",  "pT_truth_jet_1  - pT_matched_truth_jet_1")
                #.Define("dphi_jet_1", "phi_truth_jet_1 - phi_matched_truth_jet_1")

                ## second truth jet
                #.Define("truth_jet_2", "sorted_truth_jets[1]")
                #.Define("pT_truth_jet_2",  "FCCAnalyses::ReconstructedParticle::get_pt({truth_jet_2})[0]")
                #.Define("phi_truth_jet_2", "FCCAnalyses::ReconstructedParticle::get_phi({truth_jet_2})[0]")
                #.Define("eta_truth_jet_2", "FCCAnalyses::ReconstructedParticle::get_eta({truth_jet_2})[0]")
                #.Define("matched_truth_jet_2",
                #        "AnalysisFCChh::find_reco_reco_matches_unique({truth_jet_2}, selected_jets, 0.3)")
                #.Define("pT_matched_truth_jet_2",
                #        "FCCAnalyses::ReconstructedParticle::get_pt(matched_truth_jet_2)[0]")
                #.Define("phi_matched_truth_jet_2",
                #        "FCCAnalyses::ReconstructedParticle::get_phi(matched_truth_jet_2)[0]")
                #.Define("pT_response_truth_jet_2", "pT_matched_truth_jet_2 / pT_truth_jet_2")
                #.Define("dpT_jet_2",  "pT_truth_jet_2  - pT_matched_truth_jet_2")
                #.Define("dphi_jet_2", "phi_truth_jet_2 - phi_matched_truth_jet_2")
#
                ## third truth jet
                #.Define("truth_jet_3", "sorted_truth_jets[2]")
                #.Define("pT_truth_jet_3",  "FCCAnalyses::ReconstructedParticle::get_pt({truth_jet_3})[0]")
                #.Define("phi_truth_jet_3", "FCCAnalyses::ReconstructedParticle::get_phi({truth_jet_3})[0]")
                #.Define("eta_truth_jet_3", "FCCAnalyses::ReconstructedParticle::get_eta({truth_jet_3})[0]")
                #.Define("matched_truth_jet_3",
                #        "AnalysisFCChh::find_reco_reco_matches_unique({truth_jet_3}, selected_jets, 0.3)")
                #.Define("pT_matched_truth_jet_3",
                #        "FCCAnalyses::ReconstructedParticle::get_pt(matched_truth_jet_3)[0]")
                #.Define("phi_matched_truth_jet_3",
                #        "FCCAnalyses::ReconstructedParticle::get_phi(matched_truth_jet_3)[0]")
                #.Define("pT_response_truth_jet_3", "pT_matched_truth_jet_3 / pT_truth_jet_3")
                #.Define("dpT_jet_3",  "pT_truth_jet_3  - pT_matched_truth_jet_3")
                #.Define("dphi_jet_3", "phi_truth_jet_3 - phi_matched_truth_jet_3")
        #
                ## fourth truth jet
                #.Define("truth_jet_4", "sorted_truth_jets[3]")
                #.Define("pT_truth_jet_4",  "FCCAnalyses::ReconstructedParticle::get_pt({truth_jet_4})[0]")
                #.Define("phi_truth_jet_4", "FCCAnalyses::ReconstructedParticle::get_phi({truth_jet_4})[0]")
                #.Define("eta_truth_jet_4", "FCCAnalyses::ReconstructedParticle::get_eta({truth_jet_4})[0]")
                #.Define("matched_truth_jet_4",
                #        "AnalysisFCChh::find_reco_reco_matches_unique({truth_jet_4}, selected_jets, 0.3)")
                #.Define("pT_matched_truth_jet_4",
                #        "FCCAnalyses::ReconstructedParticle::get_pt(matched_truth_jet_4)[0]")
                #.Define("phi_matched_truth_jet_4",
                #        "FCCAnalyses::ReconstructedParticle::get_phi(matched_truth_jet_4)[0]")
                #.Define("pT_response_truth_jet_4", "pT_matched_truth_jet_4 / pT_truth_jet_4")
                #.Define("dpT_jet_4",  "pT_truth_jet_4  - pT_matched_truth_jet_4")
                #.Define("dphi_jet_4", "phi_truth_jet_4 - phi_matched_truth_jet_4")
                                
                
                # ------------------ electrons ------------------
                # first we define "analysis electrons" passing default isolation < 0.1
                .Define("electrons",  "FCCAnalyses::ReconstructedParticle::get(Electron_objIdx.index, ReconstructedParticles)")
                .Define("selpt_el",   "FCCAnalyses::ReconstructedParticle::sel_pt(20)(electrons)")
                .Define("sel_el_unsort", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_el)")
                .Define("sel_el",  "AnalysisFCChh::SortParticleCollection(sel_el_unsort)") #sort by pT
                .Define("n_sel_el",  "FCCAnalyses::ReconstructedParticle::get_n(sel_el)") 
                

                # reco-matching
                .Define("reco_match_el", "AnalysisFCChh::find_reco_matches_unique(truth_e_selpt, sel_el, 0.1)") # truth-reco match with dR 0.1
                .Define("n_reco_match_el",  "FCCAnalyses::ReconstructedParticle::get_n(reco_match_el)") 
                
                # also with prompt electrons
                .Define("reco_match_prompt_el", "AnalysisFCChh::find_reco_matches_unique(electrons_truth_prompt, sel_el, 0.1)")
                .Define("n_reco_match_prompt_el",  "FCCAnalyses::ReconstructedParticle::get_n(reco_match_prompt_el)") 

                # now define all electrons with the same set of selections
                .Define("el_all",
                    "FCCAnalyses::ReconstructedParticle::get(ElectronNoIso_objIdx.index, ReconstructedParticles)")
                .Define("pt_el_all", "FCCAnalyses::ReconstructedParticle::get_pt(el_all)")
                .Define("el_all_seleta", "FCCAnalyses::ReconstructedParticle::sel_eta(4.)(el_all)")
                .Define("el_all_sel", "FCCAnalyses::ReconstructedParticle::sel_pt(1)(el_all_seleta)")
                .Define("n_el_all_sel", "FCCAnalyses::ReconstructedParticle::get_n(el_all_sel)")
                
                # ------------------ muons ------------------
                # again, start with analysis muons passing default isolation of 0.2
                .Define("muons",  "FCCAnalyses::ReconstructedParticle::get(Muon_objIdx.index, ReconstructedParticles)")
                .Define("selpt_mu",   "FCCAnalyses::ReconstructedParticle::sel_pt(20)(muons)")
                .Define("sel_mu_unsort", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_mu)")
                .Define("sel_mu",  "AnalysisFCChh::SortParticleCollection(sel_mu_unsort)") #sort by pT
                .Define("n_sel_mu", "FCCAnalyses::ReconstructedParticle::get_n(sel_mu)")
                
                # reco-matching
                .Define("reco_match_mu", "AnalysisFCChh::find_reco_matches_unique(truth_mu_selpt, sel_mu, 0.1)") # truth-reco match with dR 0.1
                .Define("n_reco_match_mu",  "FCCAnalyses::ReconstructedParticle::get_n(reco_match_mu)") 
                
                # also with prompt muons
                .Define("reco_match_prompt_mu", "AnalysisFCChh::find_reco_matches_unique(muons_truth_prompt, sel_mu, 0.1)")
                .Define("n_reco_match_prompt_mu",  "FCCAnalyses::ReconstructedParticle::get_n(reco_match_prompt_mu)")
                
                # also all muons
                .Define("mu_all",
                        "FCCAnalyses::ReconstructedParticle::get(MuonNoIso_objIdx.index, ReconstructedParticles)")
                .Define("mu_all_seleta", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(mu_all)")
                .Define("mu_all_sel",    "FCCAnalyses::ReconstructedParticle::sel_pt(20)(mu_all_seleta)")
                .Define("n_mu_all_sel", "FCCAnalyses::ReconstructedParticle::get_n(mu_all_sel)")
                                
                
                
        )
        dframe0 = (
                dframe2
                # now define HF tagging
                # ---------- b (Medium WP) ----------
                # first b-tagged jets (no overlap considered)
                .Define('dR_match', "0.3")
                .Define("b_tagged_jets_medium_general",
                        "AnalysisFCChh::get_tagged_jets(Jet, Jet_HF_tags, _Jet_HF_tags_particle, _Jet_HF_tags_parameters, 1)") #bit 1 = medium WP (0- loose, 2-tight)
                # selections on general b-tagged jets
                .Define("selpt_b_med_general", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(b_tagged_jets_medium_general)")
                .Define("bjet_sel_gen", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_b_med_general)")
                .Define("n_b_jets_medium", "FCCAnalyses::ReconstructedParticle::get_n(bjet_sel_gen)")
                
                # reco matching bs to jets
                .Define("reco_match_b_jets", "AnalysisFCChh::find_reco_matches_unique(truth_b_selpt, selected_jets, dR_match)")
                .Define("n_reco_match_b_jets", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_b_jets)") 
                
                
                # reco matching to general b-tagged jet collection
                .Define("reco_match_b_bjets", "AnalysisFCChh::find_reco_matches_unique(truth_b_selpt, bjet_sel_gen, dR_match)")
                .Define("n_reco_match_b_bjets", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_b_bjets)")    

                # now, b-tagged jets with overlap considered
                .Define("b_tagged_jets_medium",
                    "AnalysisFCChh::get_btagged_not_tau_tagged("
                    "  Jet, "
                    "  Jet_HF_tags, _Jet_HF_tags_particle, _Jet_HF_tags_parameters, "
                    "  Jet_tau_tags, _Jet_tau_tags_particle, _Jet_tau_tags_parameters, "
                    "  1, 1)")  # btagIndex=1, tauIndex=1, medium WP for each
                
                
                .Define("tau_tagged_jets_medium_nob",
                    "AnalysisFCChh::get_btagged_not_tau_tagged("
                    "  Jet, "
                    "  Jet_tau_tags, _Jet_tau_tags_particle, _Jet_tau_tags_parameters, "
                    "  Jet_HF_tags, _Jet_HF_tags_particle, _Jet_HF_tags_parameters, "
                    "  1, 1)")  # btagIndex=1, tauIndex=1, medium WP for each
                
                .Define("selpt_tau_med_nob", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(tau_tagged_jets_medium_nob)")
                .Define("sel_eta_tau_med_nob", "FCCAnalyses::ReconstructedParticle::sel_eta(4.)(selpt_tau_med_nob)")
                .Define("n_tau_jet_not_b", "FCCAnalyses::ReconstructedParticle::get_n(sel_eta_tau_med_nob)")
                # define delphes' isolation variable
                .Define("mu_all_iso_var", "MuonNoIso_IsolationVar")
                # non-isolated muons
                .Define("mu_isoVar_fail", "AnalysisFCChh::sel_by_iso_fail(mu_all, mu_all_iso_var, 0.2f)")

                # selections on exclusive b-jets
                #.Define("jets_muCorr", "AnalysisFCChh::add_muons_to_jets(b_tagged_jets_medium, mu_isoVar_fail, 0.4f)")
                .Define("selpt_b_med", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(b_tagged_jets_medium)")
                .Define("sel_eta_b_med", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_b_med)")
                .Define("n_b_jets_medium_tauprio", "FCCAnalyses::ReconstructedParticle::get_n(sel_eta_b_med)")
                .Define("sorted_sel_eta_b_med",  "AnalysisFCChh::SortParticleCollection(sel_eta_b_med)") #sort by pT
                .Define("pT_bs", "FCCAnalyses::ReconstructedParticle::get_pt(sorted_sel_eta_b_med)")
                .Define("pT_b1", "pT_bs[0]")
                .Define("pT_b2", "pT_bs[1]")
                .Define("pT_b3", "pT_bs[2]")
                .Define("pT_b4", "pT_bs[3]")

                # reco matching bs to bjets
                .Define("reco_match_b_jets_tauprio", "AnalysisFCChh::find_reco_matches_unique(truth_b_selpt, sel_eta_b_med, dR_match)")
                .Define("n_reco_match_b_jets_tauprio", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_b_jets_tauprio)") 
                
                # reco matching taus to bs
                .Define("reco_match_tau_b", "AnalysisFCChh::find_reco_matches_unique(truth_tau_selpt, sel_eta_b_med, dR_match)")
                .Define("n_reco_match_tau_b", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_tau_b)") 
                
                
                # ---------- tau (Medium WP) ----------
                # normal taus - take priority over bs :)
                .Define("tau_tagged_jets_medium", 
                    "AnalysisFCChh::get_tagged_jets(Jet, Jet_tau_tags, _Jet_tau_tags_particle, _Jet_tau_tags_parameters, 1)") #bit 1 = medium WP (0- loose, 2-tight)
                .Define("selpt_tau_med", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(tau_tagged_jets_medium)")
                .Define("sel_eta_tau_med", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_tau_med)")

                .Define("n_tau_jets_medium", "FCCAnalyses::ReconstructedParticle::get_n(sel_eta_tau_med)")
                
                # reco matching taus
                .Define("reco_match_taus", "AnalysisFCChh::find_reco_matches_unique(truth_tau_selpt, sel_eta_tau_med, dR_match)")
                .Define("n_reco_match_tau_jets", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_taus)")
                .Define("selected_PF08_jets_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25.)(ParticleFlowJet08)")
                .Define("selected_PF08_jets", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selected_PF08_jets_pt)")
                .Define("n_PF08_jets_sel", "FCCAnalyses::ReconstructedParticle::get_n(selected_PF08_jets)")
                .Define("pT_PF08_jets_sel", "FCCAnalyses::ReconstructedParticle::get_pt(selected_PF08_jets)")
                .Define("eta_PF08_jets_sel", "FCCAnalyses::ReconstructedParticle::get_eta(selected_PF08_jets)")
                .Define("phi_PF08_jets_sel", "FCCAnalyses::ReconstructedParticle::get_phi(selected_PF08_jets)")

                # akt 0.2 track jets
                .Define("selected_trackjets_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25)(TrackJet02)")
                .Define("selected_trackjets", "FCCAnalyses::ReconstructedParticle::sel_eta(4.0)(selected_trackjets_pt)")
                .Define("n_selected_tracks", "FCCAnalyses::ReconstructedParticle::get_n(selected_trackjets)")

                

                .Define("b_tagged_trackJets_medium",
                "AnalysisFCChh::get_btagged_not_tau_tagged("
                " TrackJet02, "
                " TrackJet02_HF_tags, _TrackJet02_HF_tags_particle, _TrackJet02_HF_tags_parameters, "
                " TrackJet02_tau_tags, _TrackJet02_tau_tags_particle, _TrackJet02_tau_tags_parameters, "
                " 1, 1)") # btagIndex=1, tauIndex=1, medium WP for each

                .Define("tau_tagged_trackJets_medium", "AnalysisFCChh::get_tagged_jets(TrackJet02, TrackJet02_tau_tags, _TrackJet02_tau_tags_particle, _TrackJet02_tau_tags_parameters, 1)") #bit 1 = medium WP (0- loose, 2-tight)
                .Define("selected_b_tagged_trackJets_medium_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25)(b_tagged_trackJets_medium)")
                .Define("selected_b_tagged_trackJets_medium", "FCCAnalyses::ReconstructedParticle::sel_eta(4.0)(selected_b_tagged_trackJets_medium_pt)")
                .Define("n_selected_b_tagged_trackJets_medium", "FCCAnalyses::ReconstructedParticle::get_n(selected_b_tagged_trackJets_medium)")

                

                .Define("selected_tau_tagged_trackJets_medium_pt", "FCCAnalyses::ReconstructedParticle::sel_pt(25)(tau_tagged_trackJets_medium)")

                .Define("selected_tau_tagged_trackJets_medium", "FCCAnalyses::ReconstructedParticle::sel_eta(4.0)(selected_tau_tagged_trackJets_medium_pt)")

                .Define("n_selected_tau_tagged_trackJets_medium", "FCCAnalyses::ReconstructedParticle::get_n(selected_tau_tagged_trackJets_medium)")

                
                .Define("bb_PF08_trackJet02", "AnalysisFCChh::find_bb_tagged(selected_PF08_jets, selected_b_tagged_trackJets_medium)")
                # now some selections on LR jets
                .Define("sel_pt_bb_PF08_trackJet02", "FCCAnalyses::ReconstructedParticle::sel_pt(50)(bb_PF08_trackJet02)")
                .Define("sel_eta_bb_PF08_trackJet02", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(sel_pt_bb_PF08_trackJet02)")
                .Define("n_bb_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_n(sel_eta_bb_PF08_trackJet02)")
                .Define("pT_bb_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_pt(sel_eta_bb_PF08_trackJet02)")
                .Define("eta_bb_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_eta(sel_eta_bb_PF08_trackJet02)")
                .Define("phi_bb_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_phi(sel_eta_bb_PF08_trackJet02)")

                # now we've defined our b objects, find higgs candidates
                .Define("hh_candidates", "AnalysisFCChh::find_4b_hh(sel_eta_bb_PF08_trackJet02, sel_eta_b_med)")

                # leading
                .Define("TLV_h1", "hh_candidates.first")
                .Define("pT_h1_LR", "TLV_h1.Pt()")
                .Define("eta_h1_LR", "TLV_h1.Phi()")
                .Define("phi_h1_LR", "TLV_h1.Eta()")
                .Define("m_h1_LR", "TLV_h1.M()")

                # sub-leading
                .Define("TLV_h2", "hh_candidates.second")
                .Define("pT_h2_LR", "TLV_h2.Pt()")
                .Define("eta_h2_LR", "TLV_h2.Phi()")
                .Define("phi_h2_LR", "TLV_h2.Eta()")
                .Define("m_h2_LR", "TLV_h2.M()")

                # define relative kinematics between Higgs'
                .Define("dphi_h1h2_LR", "ROOT::Math::VectorUtil::DeltaPhi(TLV_h1, TLV_h2)")
                .Define("deta_h1h2_LR", "TLV_h1.Eta() - TLV_h2.Eta()")
                .Define("dR_h1h2_LR", "ROOT::Math::VectorUtil::DeltaR(TLV_h1, TLV_h2)")
                # compute kinematics between Higgs decay products
                .Define("hh_cand_pairs", "AnalysisFCChh::find_4b_hh_pairs(sel_eta_bb_PF08_trackJet02, sel_eta_b_med)")
                .Define("h1_cand", "hh_cand_pairs[0][0]")
                .Define("h1b1_cand", "hh_cand_pairs[0].size()>1 ? hh_cand_pairs[0][1] : TLorentzVector()")
                .Define("h1b2_cand", "hh_cand_pairs[0].size()>2 ? hh_cand_pairs[0][2] : TLorentzVector()")
                .Define("pt_h1b2_cand","h1b2_cand.Pt()")

                

                # H2 candidate (TLV) and its constituents
                .Define("h2_cand", "hh_cand_pairs[1][0]")
                .Define("h2b1_cand", "hh_cand_pairs[1].size()>1 ? hh_cand_pairs[1][1] : TLorentzVector()")
                .Define("h2b2_cand", "hh_cand_pairs[1].size()>2 ? hh_cand_pairs[1][2] : TLorentzVector()")

                

                
                
                
        )
        dframe3 = (
                dframe0
                # some handy derived vars
                .Define("m_h1_L", "h1_cand.M()")
                .Define("m_h2_L", "h2_cand.M()")
                .Define("dR_h1_bb", "hh_cand_pairs[0].size()>2 ? hh_cand_pairs[0][1].DeltaR(hh_cand_pairs[0][2]) : -1.0")
                .Define("dR_h2_bb", "hh_cand_pairs[1].size()>2 ? hh_cand_pairs[1][1].DeltaR(hh_cand_pairs[1][2]) : -1.0")

        
                .Define("tautau_PF08_trackJet02", "AnalysisFCChh::find_bb_tagged(selected_PF08_jets, selected_tau_tagged_trackJets_medium)")
                .Define("sel_pt_tautau_PF08_trackJet02", "FCCAnalyses::ReconstructedParticle::sel_pt(50)(tautau_PF08_trackJet02)")
                .Define("sel_eta_tautau_PF08_trackJet02", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(sel_pt_tautau_PF08_trackJet02)")
                .Define("n_tautau_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_n(sel_eta_tautau_PF08_trackJet02)")
                .Define("pT_tautau_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_pt(sel_eta_tautau_PF08_trackJet02)")
                .Define("eta_tautau_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_eta(sel_eta_tautau_PF08_trackJet02)")
                .Define("phi_tautau_tagged_08Jets", "FCCAnalyses::ReconstructedParticle::get_phi(sel_eta_tautau_PF08_trackJet02)")
                # now we define the final higgs candidate
                .Define("Htautau_LR", "AnalysisFCChh::find_tautau_h(sel_eta_tautau_PF08_trackJet02, sel_eta_tau_med)")
                .Define("m_Htautau_LR", "Htautau_LR.M()")
                .Define("eta_Htautau_LR", "Htautau_LR.Eta()")
                .Define("phi_Htautau_LR", "Htautau_LR.Phi()")
                .Define("pT_Htautau_LR", "Htautau_LR.Pt()")
                # with this we can define the triple-Higgs TLV
                .Define("hhh_TLV", "Htautau_LR + TLV_h1 + TLV_h2")
                .Define("m_hhh_vis_LR", "hhh_TLV.M()")
                # define sigma parameters
                .Define("sigma", "AnalysisFCChh::compute_sigma(TLV_h1, TLV_h2, Htautau_LR)")
                .Define("sigma_12", "sigma[0]")
                .Define("sigma_23", "sigma[1]")
                .Define("sigma_13", "sigma[2]")
                .Define("sigma_x", "sigma_23 + 0.5 * sigma_13")
                .Define("sigma_y", "0.866 * sigma_13")
                # now look at tau-tagged bs
                .Define("tau_tagged_bs", "AnalysisFCChh::find_reco_matches_unique(truth_b_selpt, sel_eta_tau_med, dR_match)")
                .Define("n_tau_tagged_bs", "FCCAnalyses::ReconstructedParticle::get_n(tau_tagged_bs)")
                
                # also look for jets that contain both b and taus
                .Define("b_tau_shared_jet", "AnalysisFCChh::find_reco_matches_both(truth_b_selpt, truth_tau_selpt, selected_jets, dR_match)")
                .Define("n_b_tau_shared_jet", "FCCAnalyses::ReconstructedParticle::get_n(b_tau_shared_jet)")
                
                .Define("b_tau_shared_jet_0p3", "AnalysisFCChh::find_reco_matches_both(truth_b_selpt, truth_tau_selpt, selected_jets, 0.3)")
                .Define("n_b_tau_shared_jet_0p3", "FCCAnalyses::ReconstructedParticle::get_n(b_tau_shared_jet_0p3)")
                
                # reco matching bs to jets at 0.3
                .Define("reco_match_b_jets_0p3", "AnalysisFCChh::find_reco_matches_unique(truth_b_selpt, selected_jets, 0.3)")
                .Define("n_reco_match_b_jets_0p3", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_b_jets_0p3)") 

                # tau-lep channel
                .Define("sort_sel_el",  "AnalysisFCChh::SortParticleCollection(sel_el)") #sort by pT
                .Define("leading_electron" , "sort_sel_el[0]")

                .Define("taulep_fromH", "AnalysisFCChh::getTruthTauLeps(mc_particles, mc_daughters, mc_parents, \"from_higgs\")")
                .Define("tauel_fromH", "AnalysisFCChh::getTruthTauEmu(mc_particles, mc_daughters, mc_parents, \"from_higgs_el\")")
                .Define("n_tauel_fromH", "FCCAnalyses::MCParticle::get_n(tauel_fromH)")
                
                
                # define some selections on the electrons 
                .Define("sel_eta_tauel_fromH",  "FCCAnalyses::MCParticle::sel_eta(4)(tauel_fromH)")
                .Define("sel_pt_tauel_fromH",  "FCCAnalyses::MCParticle::sel_pt(10)(sel_eta_tauel_fromH)")
                .Define("n_sel_e_fromH", "FCCAnalyses::MCParticle::get_n(sel_pt_tauel_fromH)")
                .Define("sorted_sel_pt_tauel_fromH", "AnalysisFCChh::SortMCByPt(sel_pt_tauel_fromH)")
                .Define("leading_tauel_fromH", "sorted_sel_pt_tauel_fromH[0]")
                .Define("pT_leading_tauel_fromH", "FCCAnalyses::MCParticle::get_pt(sorted_sel_pt_tauel_fromH)[0]")
                .Define("phi_leading_tauel_fromH", "FCCAnalyses::MCParticle::get_phi(sorted_sel_pt_tauel_fromH)[0]")
                .Define("eta_leading_tauel_fromH", "FCCAnalyses::MCParticle::get_eta(sorted_sel_pt_tauel_fromH)[0]")


                .Define("truthHadronicTaus", "AnalysisFCChh::getTruthTauHadronic(mc_particles, mc_daughters, mc_parents, \"from_Z\")")
                #.Define("n_truthHadronicTaus", "truthHadronicTaus.size()")     
                .Define("TLV_pvis_truthHadronicTaus_1", "std::get<0>(truthHadronicTaus)[0]") 
                .Define("TLV_pmis_truthHadronicTaus_1", "std::get<1>(truthHadronicTaus)[0]") 
                .Define("dphi_truth_vismis_1", "ROOT::Math::VectorUtil::DeltaPhi(TLV_pvis_truthHadronicTaus_1, TLV_pmis_truthHadronicTaus_1)")

                .Define("pvis_truthHadronicTaus_1", "TLV_pvis_truthHadronicTaus_1.P()")
                .Define("pmis_truthHadronicTaus_1", "TLV_pmis_truthHadronicTaus_1.P()")
                .Define("angle3D_vis_mis_tau1",
                        "TLV_pvis_truthHadronicTaus_1.Vect().Angle("
                        " TLV_pmis_truthHadronicTaus_1.Vect())")

                .Define("n_charged_truthHadronicTaus_1", "std::get<2>(truthHadronicTaus)[0]") 
                .Define("n_neutral_truthHadronicTaus_1", "std::get<3>(truthHadronicTaus)[0]") 
                .Define("m_vis_truthHadronicTaus_1", "std::get<4>(truthHadronicTaus)[0]") 
                .Define("m_mis_truthHadronicTaus_1", "std::get<5>(truthHadronicTaus)[0]") 
                .Define("n_neutrinos_truthHadronicTaus_1", "std::get<6>(truthHadronicTaus)[0]") 

                .Define("TLV_pvis_truthHadronicTaus_2", "std::get<0>(truthHadronicTaus)[1]") 
                .Define("TLV_pmis_truthHadronicTaus_2", "std::get<1>(truthHadronicTaus)[1]") 
                
                
                
                .Define("dphi_truth_vismis_2", "ROOT::Math::VectorUtil::DeltaPhi(TLV_pvis_truthHadronicTaus_2, TLV_pmis_truthHadronicTaus_2)")
                
                .Define("pvis_truthHadronicTaus_2", "TLV_pvis_truthHadronicTaus_2.P()")
                .Define("pmis_truthHadronicTaus_2", "TLV_pmis_truthHadronicTaus_2.P()")
                .Define("angle3D_vis_mis_tau2",
                        "TLV_pvis_truthHadronicTaus_2.Vect().Angle("
                        " TLV_pmis_truthHadronicTaus_2.Vect())")

                .Define("n_charged_truthHadronicTaus_2", "std::get<2>(truthHadronicTaus)[1]") 
                .Define("n_neutral_truthHadronicTaus_2", "std::get<3>(truthHadronicTaus)[1]") 
                .Define("n_neutrinos_truthHadronicTaus_2", "std::get<6>(truthHadronicTaus)[1]") 

                .Define("m_vis_truthHadronicTaus_2", "std::get<4>(truthHadronicTaus)[1]") 
                .Define("m_mis_truthHadronicTaus_2", "std::get<5>(truthHadronicTaus)[1]") 

                # --- τ1 ---
                .Define("x_visE_1",
                        "TLV_pvis_truthHadronicTaus_1.E() / "
                        "(TLV_pvis_truthHadronicTaus_1.E() + TLV_pmis_truthHadronicTaus_1.E())")
                .Define("x_visE_2",
                        "TLV_pvis_truthHadronicTaus_2.E() / "
                        "(TLV_pvis_truthHadronicTaus_2.E() + TLV_pmis_truthHadronicTaus_2.E())")                
                
                .Define("x_misP_1",
                        "TLV_pmis_truthHadronicTaus_1.P() / "
                        "((TLV_pvis_truthHadronicTaus_1 + TLV_pmis_truthHadronicTaus_1).P())")   
                               
                .Define("x_misP_2",
                        "TLV_pmis_truthHadronicTaus_2.P() / "
                        "((TLV_pvis_truthHadronicTaus_2 + TLV_pmis_truthHadronicTaus_2).P())")   
                
        
                
                # isolation requirements
                # define delphes' isolation variable
                .Define("el_all_iso_var", "ElectronNoIso_IsolationVar")
                .Define("el_iso_var", "Electron_IsolationVar")

                # now define selections using this isolation
                .Define("el_iso_0p2", "AnalysisFCChh::select_with_mask(el_all_sel, el_all_iso_var < 0.2f)")
                .Define("selpt_el_0p2",   "FCCAnalyses::ReconstructedParticle::sel_pt(20)(el_iso_0p2)")
                .Define("sel_el_unsort_0p2", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_el_0p2)")
                .Define("sel_el_0p2",  "AnalysisFCChh::SortParticleCollection(sel_el_unsort_0p2)") #sort by pT
                .Define("leading_sel_el_0p2", "sel_el_0p2[0]")
                .Define("n_sel_el_0p2",  "FCCAnalyses::ReconstructedParticle::get_n(sel_el_0p2)") 
                
                .Define("el_iso_0p3", "AnalysisFCChh::select_with_mask(el_all_sel, el_all_iso_var < 0.3f)")
                .Define("selpt_el_0p3",   "FCCAnalyses::ReconstructedParticle::sel_pt(20)(el_iso_0p3)")
                .Define("sel_el_unsort_0p3", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_el_0p3)")
                .Define("sel_el_0p3",  "AnalysisFCChh::SortParticleCollection(sel_el_unsort_0p3)") #sort by pT
                .Define("leading_sel_el_0p3", "sel_el_0p3[0]")
                .Define("n_sel_el_0p3",  "FCCAnalyses::ReconstructedParticle::get_n(sel_el_0p3)") 
                
                # define leading MC muon from H for reco-matching
                .Define("leading_el_fromH", "sorted_sel_pt_tauel_fromH[0]")
                .Define("n_leading_el_fromH", "FCCAnalyses::MCParticle::get_n({leading_el_fromH})")

                # reco-match to entire collection
                .Define("reco_matched_leading_el_fromH_allel", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, el_all_sel, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_allel", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_allel)")
                .Define("pT_reco_matched_leading_el_fromH_allel", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_allel)[0]")

                .Define("reco_matched_leading_el_fromH_leading_el_0p1", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_leading_el_0p1", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_leading_el_0p1)")
                .Define("pT_reco_matched_leading_el_fromH_leading_el_0p1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_leading_el_0p1)[0]")

                .Define("reco_matched_leading_el_fromH_leading_el_0p2", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el_0p2, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_leading_el_0p2", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_leading_el_0p2)")
                .Define("pT_reco_matched_leading_el_fromH_leading_el_0p2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_leading_el_0p2)[0]")
                .Define("phi_reco_matched_leading_el_fromH_leading_el_0p2", "FCCAnalyses::ReconstructedParticle::get_phi(reco_matched_leading_el_fromH_leading_el_0p2)[0]")

                .Define("reco_matched_leading_el_fromH_leading_el_0p3", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el_0p3, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_leading_el_0p3", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_leading_el_0p3)")
                .Define("pT_reco_matched_leading_el_fromH_leading_el_0p3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_leading_el_0p3)[0]")
                .Define("phi_reco_matched_leading_el_fromH_leading_el_0p3", "FCCAnalyses::ReconstructedParticle::get_phi(reco_matched_leading_el_fromH_leading_el_0p3)[0]")

                # reco-match to isolated collection
                .Define("reco_matched_leading_el_fromH_selel", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_selel", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_selel)")

                # also the 0.2 working point
                .Define("reco_matched_leading_el_fromH_selel_0p2", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el_0p2, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_selel_0p2", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_selel_0p2)")
                .Define("pT_reco_matched_leading_el_fromH_selel_0p2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_selel_0p2)[0]")

                # also the 0.3 working point
                .Define("reco_matched_leading_el_fromH_selel_0p3", "AnalysisFCChh::find_reco_matches_unique({leading_el_fromH}, sel_el_0p3, 0.1)")
                .Define("n_reco_matched_leading_el_fromH_selel_0p3", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_el_fromH_selel_0p3)")
                .Define("pT_reco_matched_leading_el_fromH_selel_0p3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_el_fromH_selel_0p3)[0]")

                # also muons
                .Define("taumu_fromH", "AnalysisFCChh::getTruthTauEmu(mc_particles, mc_daughters, mc_parents, \"from_higgs_mu\")")
                .Define("n_taumu_fromH", "FCCAnalyses::MCParticle::get_n(taumu_fromH)")
                .Define("sel_eta_taumu_fromH",  "FCCAnalyses::MCParticle::sel_eta(4)(taumu_fromH)")
                .Define("sel_pt_taumu_fromH",  "FCCAnalyses::MCParticle::sel_pt(1)(sel_eta_taumu_fromH)")
                .Define("sorted_sel_pt_taumu_fromH", "AnalysisFCChh::SortMCByPt(sel_pt_taumu_fromH)")
                .Define("n_taulep_fromH", "n_tauel_fromH + n_taumu_fromH")
                
                
                


        )
        # higgs truth & kinematics
        dframe4 = (
                dframe3
                
                # define leading MC muon from H for reco-matching
                .Define("leading_mu_fromH", "sorted_sel_pt_taumu_fromH[0]")
                .Define("pT_leading_taumu_fromH", "FCCAnalyses::MCParticle::get_pt({leading_mu_fromH})[0]")
                .Define("phi_leading_taumu_fromH", "FCCAnalyses::MCParticle::get_phi({leading_mu_fromH})[0]")
                .Define("eta_leading_taumu_fromH", "FCCAnalyses::MCParticle::get_eta({leading_mu_fromH})[0]")
                
                # define higher muon working point (really to show how muons are okay already)
                .Define("mu_iso_0p3", "AnalysisFCChh::select_with_mask(mu_all_sel, mu_all_iso_var < 0.3f)")
                .Define("selpt_mu_0p3",   "FCCAnalyses::ReconstructedParticle::sel_pt(10)(mu_iso_0p3)")
                .Define("sel_mu_unsort_0p3", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(selpt_mu_0p3)")
                .Define("sel_mu_0p3",  "AnalysisFCChh::SortParticleCollection(sel_mu_unsort_0p3)") #sort by pT
                .Define("leading_sel_mu_0p3", "sel_mu_0p3[0]")
                .Define("n_sel_mu_0p3",  "FCCAnalyses::ReconstructedParticle::get_n(sel_mu_0p3)") 
                
                .Define("reco_matched_leading_mu_fromH_leading_mu_0p3", "AnalysisFCChh::find_reco_matches_unique({leading_mu_fromH}, sel_mu_0p3, 0.1)")
                .Define("n_reco_matched_leading_mu_fromH_leading_mu_0p3", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_mu_fromH_leading_mu_0p3)")
                .Define("pT_reco_matched_leading_mu_fromH_leading_mu_0p3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_mu_fromH_leading_mu_0p3)[0]")
                .Define("phi_reco_matched_leading_mu_fromH_leading_mu_0p3", "FCCAnalyses::ReconstructedParticle::get_phi(reco_matched_leading_mu_fromH_leading_mu_0p3)[0]")

                
                # also the 0.3 working point
                .Define("reco_matched_leading_mu_fromH_selmu_0p3", "AnalysisFCChh::find_reco_matches_unique({leading_mu_fromH}, sel_mu_0p3, 0.1)")
                .Define("n_reco_matched_leading_mu_fromH_selmu_0p3", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_mu_fromH_selmu_0p3)")
                .Define("pT_reco_matched_leading_mu_fromH_selmu_0p3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_mu_fromH_selmu_0p3)[0]")

                

                # reco-match to entire collection
                .Define("reco_matched_leading_mu_fromH_allmu", "AnalysisFCChh::find_reco_matches_unique({leading_mu_fromH}, mu_all_sel, 0.1)")
                .Define("pT_reco_matched_leading_mu_fromH_allmu", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_mu_fromH_allmu)[0]")
                .Define("phi_reco_matched_leading_mu_fromH_allmu", "FCCAnalyses::ReconstructedParticle::get_phi(reco_matched_leading_mu_fromH_allmu)[0]")
                .Define("n_reco_matched_leading_mu_fromH_allmu", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_mu_fromH_allmu)")
                
                # reco-match to isolated collection
                .Define("reco_matched_leading_mu_fromH_selmu", "AnalysisFCChh::find_reco_matches_unique({leading_mu_fromH}, sel_mu, 0.1)")
                .Define("n_reco_matched_leading_mu_fromH_selmu", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_leading_mu_fromH_selmu)")
                .Define("pT_reco_matched_leading_mu_fromH_selmu", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_leading_mu_fromH_selmu)[0]")
                .Define("phi_reco_matched_leading_mu_fromH_selmu", "FCCAnalyses::ReconstructedParticle::get_phi(reco_matched_leading_mu_fromH_selmu)[0]")

                
                .Define("truth_higgs", "AnalysisFCChh::get_truth_Higgs(mc_particles, mc_daughters, \"hhh\")")
                .Define("truth_hhh_higgs", "std::get<0>(truth_higgs)")
                .Define("truth_hhh_pairs", "std::get<1>(truth_higgs)")

                .Define("h1_bb", "truth_hhh_pairs[0]")
                .Define("h1_b1", "h1_bb.first") # these shoooould be MCParticles
                .Define("h1_b2", "h1_bb.second")
                .Define("h3_tautau", "ROOT::VecOps::RVec<edm4hep::MCParticleData>{truth_hhh_pairs[2].first, truth_hhh_pairs[2].second}")
                
                # Higgs HT with truth
                # define TLorentzVectors to be used in Dalitz analysis
                .Define("h1_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_higgs[0])")
                .Define("h1b1_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[0].first)")
                .Define("h1b2_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[0].second)")
                .Define("eta_h1b1", "h1b1_truth_TLV.Eta()")
                .Define("eta_h1b2", "h1b2_truth_TLV.Eta()")
                .Define("dR_truth_bb1", "ROOT::Math::VectorUtil::DeltaR(h1b1_truth_TLV, h1b2_truth_TLV)")
                
                .Define("h2_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_higgs[1])")
                .Define("h2b1_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[1].first)")
                .Define("h2b2_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[1].second)")
                .Define("eta_h2b1", "h2b1_truth_TLV.Eta()")
                .Define("eta_h2b2", "h2b2_truth_TLV.Eta()")
                .Define("dR_truth_bb2", "ROOT::Math::VectorUtil::DeltaR(h2b1_truth_TLV, h2b2_truth_TLV)")

                # H(tautau)
                .Define("h3_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_higgs[2])")                                
                .Define("h3tau1_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[2].first)")
                .Define("h3tau2_truth_TLV", "AnalysisFCChh::getTLV_MC(truth_hhh_pairs[2].second)")
                .Define("eta_htau1", "h3tau1_truth_TLV.Eta()")
                .Define("eta_htau2", "h3tau2_truth_TLV.Eta()")

                .Define("dR_truth_tautau", "ROOT::Math::VectorUtil::DeltaR(h3tau1_truth_TLV, h3tau2_truth_TLV)")
                # matching truth particles to reco
                .Define("truth_b_decay_products",
                        "ROOT::VecOps::RVec<edm4hep::MCParticleData>{ "
                        "truth_hhh_pairs[0].first, truth_hhh_pairs[0].second, "
                        "truth_hhh_pairs[1].first, truth_hhh_pairs[1].second}")
                .Define("truth_h1_products", "ROOT::VecOps::RVec<edm4hep::MCParticleData>{ "
                        "truth_hhh_pairs[0].first, truth_hhh_pairs[0].second}")
                .Define("truth_h2_products", "ROOT::VecOps::RVec<edm4hep::MCParticleData>{ "
                        "truth_hhh_pairs[1].first, truth_hhh_pairs[1].second}")
                
                .Define("truth_tau_decay_products",
                        "ROOT::VecOps::RVec<edm4hep::MCParticleData>{ "
                        "truth_hhh_pairs[2].first, truth_hhh_pairs[2].second}")
                .Define("sel_truth_tau_decay_products", "FCCAnalyses::MCParticle::sel_pt(1.0)(truth_tau_decay_products)")
                
                
                .Define("sort_truth_b_decay_products", "AnalysisFCChh::SortMCByPt(truth_b_decay_products)")
                
                .Define("reco_match_b", "AnalysisFCChh::find_reco_matches_unique(truth_b_decay_products, selected_jets, 0.3)")
                .Define("n_reco_matched_b", "FCCAnalyses::ReconstructedParticle::get_n(reco_match_b)")

                .Define("truth_b_1", "truth_hhh_pairs[0].first")
                .Define("pT_truth_b_1", "FCCAnalyses::MCParticle::get_pt({truth_b_1})[0]")

                .Define("reco_matched_b_jet_1", "AnalysisFCChh::find_reco_matches_unique({truth_b_1}, bjet_sel_gen, 0.3)")
                .Define("reco_matched_b_1", "AnalysisFCChh::find_reco_matches_unique({truth_b_1}, selected_jets, 0.3)")

                .Define("pT_reco_matched_b_jet_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_jet_1)[0]")
                .Define("pT_reco_matched_jet_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_1)[0]")
                

                .Define("truth_b_2", "truth_hhh_pairs[0].second")
                .Define("pT_truth_b_2", "FCCAnalyses::MCParticle::get_pt({truth_b_2})[0]")
                
                #.Filter("pT_truth_b_1 > pT_truth_b_2") 
                # define z-splitting function for plot in soft-drop section
                .Define("truth_b12_z", "pT_truth_b_2 / (pT_truth_b_2 + pT_truth_b_1)")
                #.Filter("truth_b12_z > 0.1")               
                
                .Define("reco_matched_b_2", "AnalysisFCChh::find_reco_matches_unique({truth_b_2}, selected_jets, 0.3)")
                .Define("reco_matched_b_jet_2", "AnalysisFCChh::find_reco_matches_unique({truth_b_2}, bjet_sel_gen, 0.3)")
                .Define("pT_reco_matched_b_jet_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_jet_2)[0]")
                .Define("pT_reco_matched_jet_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_2)[0]")
                .Define("truth_b_3", "truth_hhh_pairs[1].first")
                .Define("reco_matched_b_3", "AnalysisFCChh::find_reco_matches_unique({truth_b_3}, selected_jets, 0.3)")
                .Define("reco_matched_b_jet_3", "AnalysisFCChh::find_reco_matches_unique({truth_b_3}, bjet_sel_gen, 0.3)")
                .Define("pT_reco_matched_b_jet_3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_jet_3)[0]")
                .Define("pT_reco_matched_jet_3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_3)[0]")

                .Define("truth_b_4", "truth_hhh_pairs[1].second")
                .Define("reco_matched_b_4", "AnalysisFCChh::find_reco_matches_unique({truth_b_4}, selected_jets, 0.3)")
                .Define("reco_matched_b_jet_4", "AnalysisFCChh::find_reco_matches_unique({truth_b_4}, bjet_sel_gen, 0.3)")
                .Define("pT_reco_matched_b_jet_4", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_jet_4)[0]")
                .Define("pT_reco_matched_jet_4", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_b_4)[0]")
                # define tlorentzvecoors for each higgs
                .Define("b1_tlv_reco", "AnalysisFCChh::getTLV_reco(reco_matched_b_jet_1[0])")
                .Define("b2_tlv_reco", "AnalysisFCChh::getTLV_reco(reco_matched_b_jet_2[0])")
                
                .Define("b3_tlv_reco", "AnalysisFCChh::getTLV_reco(reco_matched_b_jet_3[0])")
                .Define("b4_tlv_reco", "AnalysisFCChh::getTLV_reco(reco_matched_b_jet_4[0])")
                # combined first and second tlvs
                .Define("m_h1_recomatch", "(b1_tlv_reco + b2_tlv_reco).M()")
                .Define("m_h2_recomatch", "(b3_tlv_reco + b4_tlv_reco).M()")

        
        )
        
        dframe5 = (
                dframe4 
                
                
                # look at B-hadrons instead
                .Define("B_hadrons_fromH_final", "AnalysisFCChh::getBhadron_final_fromH(mc_particles, mc_parents, mc_daughters, 0)")
                .Define("sel_pt_B_hadrons_fromH_final", "FCCAnalyses::MCParticle::sel_pt(5)(B_hadrons_fromH_final)")
                .Define("sel_eta_B_hadrons_fromH_final", "FCCAnalyses::MCParticle::sel_eta(4)(sel_pt_B_hadrons_fromH_final)")
                
                .Define("n_sel_eta_B_hadrons_fromH_final", "FCCAnalyses::MCParticle::get_n(sel_eta_B_hadrons_fromH_final)")
                .Define("pt_sel_eta_B_hadrons_fromH_final", "FCCAnalyses::MCParticle::get_pt(sel_eta_B_hadrons_fromH_final)")
                .Define("sel_B_had_sorted",  "AnalysisFCChh::SortMCByPt(sel_eta_B_hadrons_fromH_final)") #sort by pT
                .Define("reco_matched_B_hadrons", "AnalysisFCChh::find_reco_matches_unique(sel_B_had_sorted, selected_jets, 0.3)")
                .Define("n_reco_matched_B_hadrons", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_B_hadrons)")
                
                .Define("reco_matched_B_hadrons_bjets", "AnalysisFCChh::find_reco_matches_unique(sel_B_had_sorted, bjet_sel_gen, 0.3)")
                .Define("n_reco_matched_B_hadrons_bjets", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_B_hadrons_bjets)")
                
                
                .Define("B_had_1", "sel_B_had_sorted[0]")
                .Define("pT_B_had_1", "FCCAnalyses::MCParticle::get_pt({B_had_1})[0]")
                .Define("reco_match_B_had_1", "AnalysisFCChh::find_reco_matches_unique({B_had_1}, selected_jets, 0.3)")
                .Define("reco_match_B_had_bjet_1", "AnalysisFCChh::find_reco_matches_unique({B_had_1}, bjet_sel_gen, 0.3)")
                .Define("pt_reco_match_B_had_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_1)[0]")
                .Define("pt_reco_match_B_had_bjet_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_bjet_1)[0]")
                # also check if this jet overlaps iwth a signal tau
                .Define("is_contam_Bhad_matched_jet_1", "AnalysisFCChh::isSignalContaminated({B_had_1}, sel_truth_tau_decay_products, selected_jets, 0.3)")
                # check if this is exclusively tagged
                .Define("reco_match_B_had_bjet_not_tau_1", "AnalysisFCChh::find_reco_matches_unique({B_had_1}, sel_eta_b_med, 0.3)")
                .Define("pt_reco_match_B_had_bjet_not_tau_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_bjet_not_tau_1)[0]")
                .Define("B_had_2", "sel_B_had_sorted[1]")
                .Define("pT_B_had_2", "FCCAnalyses::MCParticle::get_pt({B_had_2})[0]")

                .Define("reco_match_B_had_2", "AnalysisFCChh::find_reco_matches_unique({B_had_2}, selected_jets, 0.3)")
                .Define("reco_match_B_had_bjet_2", "AnalysisFCChh::find_reco_matches_unique({B_had_2}, bjet_sel_gen, 0.3)")
                .Define("pt_reco_match_B_had_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_2)[0]")
                .Define("pt_reco_match_B_had_bjet_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_bjet_2)[0]")

                .Define("B_had_3", "sel_B_had_sorted[2]")
                .Define("pT_B_had_3", "FCCAnalyses::MCParticle::get_pt({B_had_3})[0]")
                .Define("reco_match_B_had_3", "AnalysisFCChh::find_reco_matches_unique({B_had_3}, selected_jets, 0.3)")
                .Define("reco_match_B_had_bjet_3", "AnalysisFCChh::find_reco_matches_unique({B_had_3}, bjet_sel_gen, 0.3)")
                .Define("pt_reco_match_B_had_3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_3)[0]")
                .Define("pt_reco_match_B_had_bjet_3", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_bjet_3)[0]")

                .Define("B_had_4", "sel_B_had_sorted[3]")
                .Define("pT_B_had_4", "FCCAnalyses::MCParticle::get_pt({B_had_4})[0]")

                .Define("reco_match_B_had_4", "AnalysisFCChh::find_reco_matches_unique({B_had_4}, selected_jets, 0.3)")
                .Define("reco_match_B_had_bjet_4", "AnalysisFCChh::find_reco_matches_unique({B_had_4}, bjet_sel_gen, 0.3)")
                .Define("pt_reco_match_B_had_4", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_4)[0]")
                .Define("pt_reco_match_B_had_bjet_4", "FCCAnalyses::ReconstructedParticle::get_pt(reco_match_B_had_bjet_4)[0]")

                .Define("visible_tauhad", "std::get<0>(AnalysisFCChh::visible_tauhad(mc_particles, mc_daughters, mc_parents, TString(\"from_higgs\")))")
                .Define("vistau_had_1", "visible_tauhad[0]")
                .Define("vistau_had_2", "visible_tauhad[1]")
                
                .Define("vis_tauhad_coll", "ROOT::VecOps::RVec<edm4hep::MCParticleData>{ "
                        "vistau_had_1, vistau_had_2}")
                .Define("sorted_vis_tauhad_coll", "AnalysisFCChh::SortMCByPt(vis_tauhad_coll)")
                .Define("leading_tauhad_vis", "sorted_vis_tauhad_coll[0]")
                .Define("reco_matched_leading_tauhad_vis", "AnalysisFCChh::find_reco_matches_unique({leading_tauhad_vis}, sel_eta_tau_med, 0.3)")
                
                
                # also define some fun quantities, like signal dR. Define the tau jet as just the leading tau jet and we'll require truth tau-had
                .Define("min_dr_signal_elec", "AnalysisFCChh::min_dr_signal(reco_matched_leading_el_fromH_allel, reco_matched_B_hadrons_bjets, reco_matched_leading_tauhad_vis)")
                .Define("min_dr_reco_elec", "AnalysisFCChh::min_dr_signal(reco_matched_leading_el_fromH_allel, selected_jets, sel_eta_tau_med)")

                
                # now use function to check the degree of signal overlap, for now use boolean for Jet contains signal b/tau candidates
                .Define("isContam_0p1", "AnalysisFCChh::isSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.1)")
                .Define("isContam_0p2", "AnalysisFCChh::isSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.2)")
                .Define("isContam_0p3", "AnalysisFCChh::isSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.3)")
                .Define("isContam_0p4", "AnalysisFCChh::isSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.4)")
                .Define("isContam_0p5", "AnalysisFCChh::isSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.5)")

                .Define("n_contam_0p3", "AnalysisFCChh::nSignalContaminated(sel_eta_B_hadrons_fromH_final, truth_tau_decay_products, selected_jets, 0.3)")
                .Define("pT_visible_tauhad_1", "FCCAnalyses::MCParticle::get_pt(visible_tauhad)[0]")
                .Define("pT_visible_tauhad_2", "FCCAnalyses::MCParticle::get_pt(visible_tauhad)[1]")
                
                .Define("vis1_tlv", "AnalysisFCChh::getTLV_MC(vistau_had_1)")
                .Define("vis2_tlv", "AnalysisFCChh::getTLV_MC(vistau_had_2)")
                .Define("m_Htautau_MC", "(vis1_tlv + vis2_tlv).M()")

                .Define("truth_tau_1", "truth_hhh_pairs[2].first")
                .Define("reco_matched_tau_jet_1", "AnalysisFCChh::find_reco_matches_unique({vistau_had_1}, selected_jets, 0.3)")
                .Define("pT_reco_matched_tau_jet_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_tau_jet_1)[0]")
                
                .Define("reco_matched_tau_tau_jet_1", "AnalysisFCChh::find_reco_matches_unique({vistau_had_1}, sel_eta_tau_med, 0.3)")
                .Define("pT_reco_matched_tau_tau_jet_1", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_tau_tau_jet_1)[0]")
                .Define("n_reco_matched_tau_tau_jet_1", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_tau_tau_jet_1)")
                
                
                .Define("truth_tau_2", "truth_hhh_pairs[2].second")
                .Define("reco_matched_tau_jet_2", "AnalysisFCChh::find_reco_matches_unique({vistau_had_2}, selected_jets, 0.3)")
                .Define("pT_reco_matched_tau_jet_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_tau_jet_2)[0]")
                
                .Define("reco_matched_tau_tau_jet_2", "AnalysisFCChh::find_reco_matches_unique({vistau_had_2}, sel_eta_tau_med, 0.3)")
                .Define("pT_reco_matched_tau_tau_jet_2", "FCCAnalyses::ReconstructedParticle::get_pt(reco_matched_tau_tau_jet_2)[0]")
                .Define("n_reco_matched_tau_tau_jet_2", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_tau_tau_jet_2)")

                
                .Define("vis1_reco_tlv", "AnalysisFCChh::getTLV_reco(reco_matched_tau_tau_jet_1[0])")
                .Define("vis2_reco_tlv", "AnalysisFCChh::getTLV_reco(reco_matched_tau_tau_jet_2[0])")
                
                #.Filter("pT_reco_matched_tau_tau_jet_2 > 0 and pT_reco_matched_tau_tau_jet_1 > 0")
                .Define("m_Htautau_reco", "(vis1_reco_tlv + vis2_reco_tlv).M()")
                
                
                .Define("pT_truth_matched_h1_b1", "FCCAnalyses::MCParticle::get_pt({truth_b_1})[0]")
                .Define("pT_truth_matched_h1_b2", "FCCAnalyses::MCParticle::get_pt({truth_b_2})[0]")
                .Define("pT_truth_matched_h2_b1", "FCCAnalyses::MCParticle::get_pt({truth_b_3})[0]")
                .Define("pT_truth_matched_h2_b2", "FCCAnalyses::MCParticle::get_pt({truth_b_4})[0]")
                .Define("pT_truth_matched_h3_tau1", "FCCAnalyses::MCParticle::get_pt({truth_tau_1})[0]")
                .Define("pT_truth_matched_h3_tau2", "FCCAnalyses::MCParticle::get_pt({truth_tau_2})[0]")
                
                .Define("reco_matched_tau", "AnalysisFCChh::find_reco_matches_unique(visible_tauhad, sel_eta_tau_med, 0.3)")
                .Define("sorted_reco_matched_tau", "AnalysisFCChh::SortParticleCollection(sel_eta_tau_med)")
                .Define("n_reco_matched_tau", "FCCAnalyses::ReconstructedParticle::get_n(sorted_reco_matched_tau)")
                .Define("pT_reco_matched_tau", "FCCAnalyses::ReconstructedParticle::get_pt(sorted_reco_matched_tau)")

                .Define("pT_truth_bb1", "h1_truth_TLV.Pt()")
                .Define("pT_truth_bb2", "h2_truth_TLV.Pt()")
                .Define("pT_truth_tautau1", "h3_truth_TLV.Pt()")

                .Define("m_hhh_truth", "(h1_truth_TLV + h2_truth_TLV + h3_truth_TLV).M()")
                
                # pts of higgs
                .Define("pT_truth_hbb1", "h1_truth_TLV.Pt()")
                .Define("pT_truth_hbb2", "h2_truth_TLV.Pt()")
                .Define("pT_truth_htautau", "h3_truth_TLV.Pt()")
                .Define("Higgs_HT_truth", "pT_truth_hbb1 + pT_truth_hbb2 + pT_truth_htautau")

                # filter for ttbb / ttZ / ttH
                .Define("WlWlFilter", "AnalysisFCChh::WWlvlvFilter(mc_particles, mc_daughters, mc_parents)")
                

        )
        
        dframe6 = (
                dframe5
        
                .Define("eta_ReconstructedParticles", "FCCAnalyses::ReconstructedParticle::get_eta(ReconstructedParticles)")
                .Define("phi_ReconstructedParticles", "FCCAnalyses::ReconstructedParticle::get_phi(ReconstructedParticles)")
                .Define("pt_ReconstructedParticles", "FCCAnalyses::ReconstructedParticle::get_pt(ReconstructedParticles)")
                
        )
        
        # start to construct event based features 
        dframe7 = (
                dframe6
                #.Filter("n_b_jets_medium > 3 and n_tau_jets_medium > 1")
                # missing energy
                .Define("MET", "FCCAnalyses::ReconstructedParticle::get_pt(MissingET)[0]")
                .Define("MET_x", "FCCAnalyses::ReconstructedParticle::get_px(MissingET)[0]")
                .Define("MET_y", "FCCAnalyses::ReconstructedParticle::get_py(MissingET)[0]")
                .Define("MET_phi", "FCCAnalyses::ReconstructedParticle::get_phi(MissingET)[0]")
                .Define("MET_eta", "FCCAnalyses::ReconstructedParticle::get_eta(MissingET)[0]")
                .Define("recoHT", "ScalarHT[0]")
                
                # met resolution in para perp
                .Define("metres_ax", "AnalysisFCChh::get_perp_para_metres(selected_jets, MissingET)")
                .Define("metres_para", "metres_ax[0]")
                .Define("metres_perp", "metres_ax[1]")
                .Define("d_MET_x_derived", "metres_ax[2]")
                .Define("d_MET_y_derived", "metres_ax[3]")
                
                # now we can also compute the MET-significance
                .Define("metsig_derived", "MET / metres_para")
                .Define("gen_MET", "FCCAnalyses::ReconstructedParticle::get_pt(GenMissingET)[0]")
                .Define("gen_MET_x", "FCCAnalyses::ReconstructedParticle::get_px(GenMissingET)[0]")
                .Define("gen_MET_y", "FCCAnalyses::ReconstructedParticle::get_py(GenMissingET)[0]")
                .Define("gen_MET_phi", "FCCAnalyses::ReconstructedParticle::get_phi(GenMissingET)[0]")
                .Define("gen_MET_eta", "FCCAnalyses::ReconstructedParticle::get_eta(GenMissingET)[0]")
                
                # also look at differences
                .Define("d_MET", "MET - gen_MET")
                .Define("d_MET_x", "MET_x - gen_MET_x")
                
                .Define("d_MET_x_ratio", "d_MET_x_derived/std::abs(d_MET_x)")
                .Define("d_MET_y", "MET_y - gen_MET_y")
                .Define("d_MET_y_ratio", "d_MET_y_derived/std::abs(d_MET_y)")
                
                .Define("asym_x_y", "d_MET_x_derived - d_MET_y_derived")

                .Define("d_MET_phi", "MET_phi - gen_MET_phi")
                .Define("d_MET_eta", "MET_eta - gen_MET_eta")

                .Define("metsig_est", "MET/sqrt(recoHT)")
                .Define("metres_est", "ROOT::Math::sqrt(recoHT)")
                
                .Define("metres_njet", "AnalysisFCChh::metres(n_jets_sel)")
                
                .Define("reco_matched_Htau1", "AnalysisFCChh::find_reco_matches_unique({truth_hhh_pairs[2].first}, sel_eta_tau_med, 0.3)")
                .Define("reco_matched_Htau2", "AnalysisFCChh::find_reco_matches_unique({truth_hhh_pairs[2].second}, sel_eta_tau_med, 0.3)")
                
                # define TLVs for MMC
                .Define("TLV_reco_matched_Htau1", "AnalysisFCChh::getTLV_reco(reco_matched_Htau1[0])")
                .Define("TLV_reco_matched_Htau2", "AnalysisFCChh::getTLV_reco(reco_matched_Htau2[0])")

                .Define("reco_matched_Htautau", "AnalysisFCChh::find_reco_matches_unique(h3_tautau, sel_eta_tau_med, 0.3)")
                .Define("n_reco_matched_Htautau", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_Htautau)")
                
                .Define("sort_visible_tauhad", "AnalysisFCChh::SortMCByPt(visible_tauhad)")
                .Define("pT_leading_visible_tauhad", "FCCAnalyses::MCParticle::get_pt(sort_visible_tauhad)[0]")
                .Define("pT_subleading_visible_tauhad", "FCCAnalyses::MCParticle::get_pt(sort_visible_tauhad)[1]")

                .Define("n_visible_tauhad", "FCCAnalyses::MCParticle::get_n(visible_tauhad)")

                # return untagged jets to be used in pairing functions for cases where n_b != 4
                .Define("untagged_jets", "AnalysisFCChh::get_untagged_jets_exclusive(Jet, Jet_HF_tags, _Jet_HF_tags_particle, _Jet_HF_tags_parameters, Jet_tau_tags, _Jet_tau_tags_particle, _Jet_tau_tags_parameters, 1, 2)")
                .Define("sel_pt_untagged_jets", "FCCAnalyses::ReconstructedParticle::sel_pt(25)(untagged_jets)")
                .Define("sel_eta_untagged_jets", "FCCAnalyses::ReconstructedParticle::sel_eta(4)(sel_pt_untagged_jets)")
                
                # pairing b-jet candidates
                .Define("hres_dR",
                        "AnalysisFCChh::getHiggsCandidateDoublet(sel_eta_b_med, sel_eta_untagged_jets, TString(\"dRminmax\"), 125.f, 120.f)")
                .Define("higgs_pairs_dR",      "hres_dR.pairs")
                
                # also by mass pairings
                .Define("hres_absmass",
                        "AnalysisFCChh::getHiggsCandidateDoublet(sel_eta_b_med, sel_eta_untagged_jets, TString(\"linearmass\"), 125.f, 120.f)")
                .Define("higgs_pairs_absmass",      "hres_absmass.pairs")

                .Define("hres_squaremass",
                        "AnalysisFCChh::getHiggsCandidateDoublet(sel_eta_b_med, sel_eta_untagged_jets, TString(\"squaremass\"), 125.f, 120.f)")
                .Define("higgs_pairs_squaremass",      "hres_squaremass.pairs")
                .Define("m_h1_absmass",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_absmass[0].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_absmass[0].particle_2)).M()")
                .Define("m_h2_absmass",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_absmass[1].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_absmass[1].particle_2)).M()")

                .Define("m_h1_squaremass",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass[0].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass[0].particle_2)).M()")
                .Define("m_h2_squaremass",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass[1].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass[1].particle_2)).M()")

                .Define("h1_tlv_dR",          "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[0].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_dR[0].particle_2)")
                .Define("h2_tlv_dR",          "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[1].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_dR[1].particle_2)")
                .Define("pT_h1", "h1_tlv_dR.Pt()")
                .Define("pT_h2", "h2_tlv_dR.Pt()")
                
                # now also try mass-ordering
                .Define("hres_absmass_massordered",
                        "AnalysisFCChh::getHiggsCandidateDoubletMassOrdered(sel_eta_b_med, TString(\"linearmass\"), 117.f, 115.f)")
                .Define("higgs_pairs_absmass_massordered",      "hres_absmass_massordered.pairs")

                .Define("hres_squaremass_massordered",
                        "AnalysisFCChh::getHiggsCandidateDoubletMassOrdered(sel_eta_b_med, TString(\"squaremass\"), 115.f, 113.f)")
                .Define("higgs_pairs_squaremass_massordered",      "hres_squaremass_massordered.pairs")
                .Define("m_h1_absmass_massordered",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_absmass_massordered[0].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_absmass_massordered[0].particle_2)).M()")
                .Define("m_h2_absmass_massordered",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_absmass_massordered[1].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_absmass_massordered[1].particle_2)).M()")

                .Define("m_h1_squaremass_massordered",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass_massordered[0].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass_massordered[0].particle_2)).M()")
                .Define("m_h2_squaremass_massordered",          "(AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass_massordered[1].particle_1) + AnalysisFCChh::getTLV_reco(higgs_pairs_squaremass_massordered[1].particle_2)).M()")

       
                .Define("sort_sel_eta_tau_med",  "AnalysisFCChh::SortParticleCollection(sel_eta_tau_med)") #sort by pT

                # now for lephad we have to 
                .Define("tautau_pairs_unmerged", "AnalysisFCChh::getPairs(sel_eta_tau_med)") #currently gets only leading pT pair, as a RecoParticlePair
                .Define("charge_tau_1", "FCCAnalyses::ReconstructedParticle::get_charge(sort_sel_eta_tau_med)[0]")
                .Define("charge_tau_2", "FCCAnalyses::ReconstructedParticle::get_charge(sort_sel_eta_tau_med)[1]")
                
                .Define("tau_1_tlv", "AnalysisFCChh::getTLV_reco(sort_sel_eta_tau_med[0])")
                .Define("tau_2_tlv", "AnalysisFCChh::getTLV_reco(sort_sel_eta_tau_med[1])")
                
                
                # define the visible masses of each too, to check how poorly calibrated they are
                .Define("m_vis_tau1_reco", "tau_1_tlv.M()")
                .Define("m_vis_tau2_reco", "tau_2_tlv.M()") 

                # define leptons to be used in lephad (leading isolated lepton)
                # sort collections
                .Define("sort_sel_mu",  "AnalysisFCChh::SortParticleCollection(sel_mu)") #sort by pT
                .Define("mu1_e",  "FCCAnalyses::ReconstructedParticle::get_e(sort_sel_mu)[0]")
                .Define("mu1_pt",  "FCCAnalyses::ReconstructedParticle::get_pt(sort_sel_mu)[0]")
                .Define("mu1_eta",  "FCCAnalyses::ReconstructedParticle::get_eta(sort_sel_mu)[0]")
                .Define("mu1_phi",  "FCCAnalyses::ReconstructedParticle::get_phi(sort_sel_mu)[0]")
                .Define("mu1_charge",  "FCCAnalyses::ReconstructedParticle::get_charge(sort_sel_mu)[0]")

                # using the 0.2 isolation working point
                .Define("el1_e",  "FCCAnalyses::ReconstructedParticle::get_e(sel_el_0p2)[0]")
                .Define("el1_pt",  "FCCAnalyses::ReconstructedParticle::get_pt(sel_el_0p2)[0]")
                .Define("el1_eta",  "FCCAnalyses::ReconstructedParticle::get_eta(sel_el_0p2)[0]")
                .Define("el1_phi",  "FCCAnalyses::ReconstructedParticle::get_phi(sel_el_0p2)[0]")
                .Define("el1_charge",  "FCCAnalyses::ReconstructedParticle::get_charge(sel_el_0p2)[0]")
                
                # OS pairs
                # opposite sign pairs
                .Define("OS_tau", "charge_tau_1 * charge_tau_2 < 0") # opposite sign tau pair
                .Define("OS_taumu", "charge_tau_1 * mu1_charge < 0") # opposite sign mu-tau pair
                .Define("OS_taue", "charge_tau_1 * el1_charge < 0") # opposite sign e-tau pair
                #.Filter("OS_taumu == 1 or OS_taue == 1")
                #.Filter("n_b_jets_medium_tauprio == 4 and ((n_tau_jets_medium == 1 and n_sel_el_0p2 == 1 and n_sel_mu == 0) or (n_tau_jets_medium == 1 and n_sel_el_0p2 == 0 and n_sel_mu == 1))")
                
        

                #.Define("constits_taus",
                 #       "FCCAnalyses::JetConstituentsUtils::build_constituents(sel_eta_tau_med, ReconstructedParticles)[0]")

                # Per-jet, per-constituent quantities (jagged)
                #.Define("constits_pt",    "FCCAnalyses::JetConstituentsUtils::get_pt(constits_taus)")
                #.Define("constits_charge","FCCAnalyses::JetConstituentsUtils::get_charge(constits_taus)")
                
                
                # now do siilar just with RecoParticle2Trakc
                #.Define("charge_rptrack", "FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_charge(sel_eta_tau_med,  _EFlowTrack_trackStates)[0]")
                #.Define("n_charge_rptrack", "FCCAnalyses::ReconstructedParticle2Track::countRP2TRK_ncharged(constits_taus,  _EFlowTrack_trackStates, 0)")

                
                .Define("best_tautau_TLVs", "AnalysisFCChh::get_Htautau_vis_exclusive_TLVs(sel_eta_tau_med, n_tau_jets_medium, sel_el_0p2, n_sel_el_0p2, sort_sel_mu, n_sel_mu)")
                .Define("m_tautau_vis_OS", "best_tautau_TLVs[2].M()")
                .Define("tau_1_tlv_LH", "best_tautau_TLVs[0]")
                .Define("tau_2_tlv_LH", "best_tautau_TLVs[1]")
                .Define("pT_tau1_tlv_LH", "tau_1_tlv_LH.Pt()")
                .Define("pT_tau2_tlv_LH", "tau_2_tlv_LH.Pt()")
                
                                
                # now we can get visible pTs for reco-matched 1/2s
                .Define("visDistance", "TLV_pvis_truthHadronicTaus_1.DeltaR(TLV_pvis_truthHadronicTaus_2)")
                #.Filter("visDistance > 0.4")
                .Define("tauVisRecoAngle", "AnalysisFCChh::matchTwoTruthTwoRecoAndAngles(TLV_pvis_truthHadronicTaus_1, TLV_pmis_truthHadronicTaus_1, n_charged_truthHadronicTaus_1, TLV_pvis_truthHadronicTaus_2, TLV_pmis_truthHadronicTaus_2, n_charged_truthHadronicTaus_2, tau_1_tlv_LH, tau_2_tlv_LH, 0.4)")
                #.Filter("pT_reco_matched_tau1 > 10 and pT_reco_matched_tau2 > 10")
                
                .Define("tau0_matched",      "tauVisRecoAngle[0].matched")
                .Define("tau0_pt_reco",      "tauVisRecoAngle[0].pt_reco")
                .Define("tau0_theta_rt",     "tauVisRecoAngle[0].theta_rt")
                .Define("tau0_theta_tt",     "tauVisRecoAngle[0].theta_tt")
                .Define("tau0_nprongs",      "tauVisRecoAngle[0].n_prongs_truth")
                .Define("tau0_dR_best",      "tauVisRecoAngle[0].dR_best")
                .Define("tau0_reco_idx",     "tauVisRecoAngle[0].reco_idx")
                

                # Tau 1 
                .Define("tau1_matched",      "tauVisRecoAngle[1].matched")
                .Define("tau1_pt_reco",      "tauVisRecoAngle[1].pt_reco")
                .Define("tau1_theta_rt",     "tauVisRecoAngle[1].theta_rt")
                .Define("tau1_theta_tt",     "tauVisRecoAngle[1].theta_tt")
                .Define("tau1_nprongs",      "tauVisRecoAngle[1].n_prongs_truth")
                .Define("tau1_dR_best",      "tauVisRecoAngle[1].dR_best")
                .Define("tau1_reco_idx",     "tauVisRecoAngle[1].reco_idx")
                # define parameters for masses
                .Define("dpT_tautau", "std::abs(pT_tau1_tlv_LH - pT_tau2_tlv_LH)")
                

                # smin, mT and collinear
                .Define("m_tautau_mT", "AnalysisFCChh::mT_tautau(tau_1_tlv_LH, tau_2_tlv_LH, MET_x, MET_y)")
                .Define("m_tautau_smin", "AnalysisFCChh::compute_smin(tau_1_tlv_LH, tau_2_tlv_LH, MissingET)")
                .Define("m_tautau_col", "AnalysisFCChh::ditau_mass_collinear_then_mT(tau_1_tlv_LH, tau_2_tlv_LH,  MET_x, MET_y)")
                
                .Define("best_tautau_Recos", 
                        "AnalysisFCChh::get_Htautau_vis_exclusive_recoTLV(sel_eta_tau_med, sel_el_0p2, sort_sel_mu)")
                .Define("tau_1_reco", "std::get<0>(best_tautau_Recos)")
                .Define("tau_2_reco", "std::get<1>(best_tautau_Recos)")
                .Define("isLep1", "std::get<3>(best_tautau_Recos)")
                .Define("isLep2", "std::get<4>(best_tautau_Recos)")
                .Define("tautau_candidates", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "tau_1_reco, tau_2_reco}")
                .Define("thrust", "AnalysisFCChh::thrust(sel_eta_b_med, tautau_candidates)")
                .Define("sphericity", "AnalysisFCChh::sphericity(sel_eta_b_med, tautau_candidates)")
                .Define("aplanarity", "AnalysisFCChh::aplanarity(sel_eta_b_med, tautau_candidates)")

                
                .Define("htau_tlv", "best_tautau_TLVs[2]")
                
                # when requiring two taus
                # tau track association
                .Alias("RP_trk_index", "_ReconstructedParticles_tracks.index")

                .Define("track_pt_proxy", "FCCAnalyses::ReconstructedParticle2Track::make_track_pt_proxy(ReconstructedParticles, RP_trk_index)")

                .Define("n_prongs_tau1",
                        "FCCAnalyses::ReconstructedParticle2Track::count_tau_tracks_cone_DRpt(tau_1_reco, EFlowTrack.trackStates_begin, "
                        "EFlowTrack.trackStates_end, _EFlowTrack_trackStates, track_pt_proxy, 0.2, 1)")

                .Define("n_prongs_tau2",
                        "FCCAnalyses::ReconstructedParticle2Track::count_tau_tracks_cone_DRpt(tau_2_reco, EFlowTrack.trackStates_begin, "
                        "EFlowTrack.trackStates_end, _EFlowTrack_trackStates, track_pt_proxy, 0.2, 1)")

               
                
                
                
                
        )
        dframe8 = (
                
                dframe7 
                # define some genmatched objects, how often are the two taus actually generator matched
                .Define("Ztautau", "AnalysisFCChh::getTruthZtautau(mc_particles, mc_daughters)")
                .Define("truth_Ztau1", "Ztautau[0]")
                .Define("truth_Ztau2", "Ztautau[1]")
                # construct visible tau hads from Z
                .Define("genmatch_truth_tau1", "AnalysisFCChh::find_reco_matches({truth_Ztau1}, sel_eta_tau_med, 0.3)")
                .Define("genmatch_truth_tau2", "AnalysisFCChh::find_reco_matches({truth_Ztau2}, sel_eta_tau_med, 0.3)")
                .Define("n_genmatch_truth_tau1", "FCCAnalyses::ReconstructedParticle::get_n(genmatch_truth_tau1)")
                .Define("n_genmatch_truth_tau2", "FCCAnalyses::ReconstructedParticle::get_n(genmatch_truth_tau2)")
                
                .Define("pT_tau1", "tau_1_tlv_LH.Pt()")
                .Define("pT_tau2", "tau_2_tlv_LH.Pt()")
                .Define("pT_htautau", "htau_tlv.Pt()")
                # redo with truth Higgs pT sum
                .Define("Higgs_HT", "pT_htautau + pT_h1 + pT_h2")
                # lephad + hadhad filter
                
                
                #.Define("tautau_pairs", "AnalysisFCChh::merge_pairs(tautau_pairs_unmerged)") #merge into one object to access inv masses etc
                #.Define("htau_tlv", "AnalysisFCChh::getTLV_reco(tautau_pairs[0])")
                .Define("dEta_tautau", "tau_1_tlv_LH.Eta() - tau_2_tlv_LH.Eta()") # should in principle
                .Define("dPhi_tautau", "ROOT::Math::VectorUtil::DeltaPhi(tau_1_tlv_LH, tau_2_tlv_LH)")
                .Define("dR_tautau", "ROOT::Math::VectorUtil::DeltaR(tau_1_tlv_LH, tau_2_tlv_LH)")

                # distance between b-jets, define b-jets
                .Define("h1_p1_tlv", "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[0].particle_1)")
                .Define("h1_p2_tlv", "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[0].particle_2)")
                .Define("h2_p1_tlv", "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[1].particle_1)")
                .Define("h2_p2_tlv", "AnalysisFCChh::getTLV_reco(higgs_pairs_dR[1].particle_2)")
                
                # check if higgs_pairs_dR[0].particle_1 and higgs_pairs_dR[0].particle_1 match the jets we reco-matched earlier
                # MC particles truth_b_1 and truth_b_2
                # make vector of decay products
                .Define("reco_h1_products", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_dR[0].particle_1, higgs_pairs_dR[0].particle_2}")
                .Define("reco_h2_products", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_dR[1].particle_1, higgs_pairs_dR[1].particle_2}")
                
                .Define("reco_matched_h1", "AnalysisFCChh::find_reco_matches_unique(truth_h1_products, reco_h1_products, 0.3)")
                .Define("reco_matched_h2", "AnalysisFCChh::find_reco_matches_unique(truth_h2_products, reco_h2_products, 0.3)")
                .Define("n_reco_matched_h1", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h1)")
                .Define("n_reco_matched_h2", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h2)")
                
                
                .Define("reco_h1_products_squaremass", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_squaremass[0].particle_1, higgs_pairs_squaremass[0].particle_2}")
                .Define("reco_h2_products_squaremass", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_squaremass[1].particle_1, higgs_pairs_squaremass[1].particle_2}")
                
                .Define("reco_matched_h1_squaremass", "AnalysisFCChh::find_reco_matches_unique(truth_h1_products, reco_h1_products_squaremass, 0.3)")
                .Define("reco_matched_h2_squaremass", "AnalysisFCChh::find_reco_matches_unique(truth_h2_products, reco_h2_products_squaremass, 0.3)")
                .Define("n_reco_matched_h1_squaremass", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h1_squaremass)")
                .Define("n_reco_matched_h2_squaremass", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h2_squaremass)")
                
                
                
                .Define("reco_h1_products_absmass", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_absmass[0].particle_1, higgs_pairs_absmass[0].particle_2}")
                .Define("reco_h2_products_absmass", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{ "
                        "higgs_pairs_absmass[1].particle_1, higgs_pairs_absmass[1].particle_2}")
                
                .Define("reco_matched_h1_absmass", "AnalysisFCChh::find_reco_matches_unique(truth_h1_products, reco_h1_products_absmass, 0.3)")
                .Define("reco_matched_h2_absmass", "AnalysisFCChh::find_reco_matches_unique(truth_h2_products, reco_h2_products_absmass, 0.3)")
                .Define("n_reco_matched_h1_absmass", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h1_absmass)")
                .Define("n_reco_matched_h2_absmass", "FCCAnalyses::ReconstructedParticle::get_n(reco_matched_h2_absmass)")
                
                # separation of Higgs decay products
                .Define("dEta_h1", "h1_p1_tlv.Eta() - h1_p2_tlv.Eta()") # should in principle
                .Define("dPhi_h1", "ROOT::Math::VectorUtil::DeltaPhi(h1_p1_tlv, h1_p2_tlv)")
                .Define("dR_h1",   "ROOT::Math::VectorUtil::DeltaR(h1_p1_tlv, h1_p2_tlv)")

                .Define("dEta_h2", "h2_p1_tlv.Eta() - h2_p2_tlv.Eta()")
                .Define("dPhi_h2", "ROOT::Math::VectorUtil::DeltaPhi(h2_p1_tlv, h2_p2_tlv)")
                .Define("dR_h2",   "ROOT::Math::VectorUtil::DeltaR(h2_p1_tlv, h2_p2_tlv)")
                
                # angular distance between this higgs and the MET
                .Define("MET_tlv", "AnalysisFCChh::getTLV_reco(MissingET[0])")

                .Define("dR_met_tautau",    "ROOT::Math::VectorUtil::DeltaR(htau_tlv, MET_tlv)")
                .Define("dEta_met_tautau",  "htau_tlv.Eta() - MET_tlv.Eta()")
                .Define("dPhi_met_tautau",  "ROOT::Math::VectorUtil::DeltaPhi(htau_tlv, MET_tlv)")

                # masses
                .Define("m_h1", "h1_tlv_dR.M()")
                .Define("m_h2", "h2_tlv_dR.M()")
                        
                # distance between Higgs candidates
                .Define("dR_H1H2", "ROOT::Math::VectorUtil::DeltaR(h1_tlv_dR, h2_tlv_dR)")
                .Define("dR_H1Htau", "ROOT::Math::VectorUtil::DeltaR(h1_tlv_dR, htau_tlv)")
                .Define("dR_H2Htau", "ROOT::Math::VectorUtil::DeltaR(h2_tlv_dR, htau_tlv)")

                .Define("dEta_H1H2", "h1_tlv_dR.Eta() - h2_tlv_dR.Eta()")
                .Define("dEta_H1Htau", "h2_tlv_dR.Eta() - htau_tlv.Eta()")
                .Define("dEta_H2Htau", "h1_tlv_dR.Eta() - htau_tlv.Eta()")
                
                .Define("dPhi_H1H2", "ROOT::Math::VectorUtil::DeltaPhi(h1_tlv_dR, h2_tlv_dR)")
                .Define("dPhi_H1Htau", "ROOT::Math::VectorUtil::DeltaPhi(h1_tlv_dR, htau_tlv)")
                .Define("dPhi_H2Htau", "ROOT::Math::VectorUtil::DeltaPhi(h2_tlv_dR, htau_tlv)")

                
                # event based features
                .Define("xwt",         "AnalysisFCChh::xwt(sel_eta_b_med, sel_eta_tau_med, sel_eta_untagged_jets, sel_el_0p2, sel_mu, MET_x, MET_y)")

                .Define("mTb_min", "AnalysisFCChh::mTb_min(sel_eta_b_med, MissingET[0])")   
                .Define("RMS_mjj", "AnalysisFCChh::RMS_mjj(sel_eta_b_med)")
                .Define("RMS_dR", "AnalysisFCChh::RMS_dR(sel_eta_b_med)")
                .Define("RMS_dEta", "AnalysisFCChh::RMS_deta(sel_eta_b_med)")
                
                
                # tri-Higgs kinematics
                .Define("hhh_vis_tlv_dR", "h1_tlv_dR + h2_tlv_dR + htau_tlv")
                .Define("m_hhh_vis", "hhh_vis_tlv_dR.M()")


                # tri-Higgs truth-event matching (had-had)
                .Define("isRecoMatchedHHH", "AnalysisFCChh::find_hhh_signal_match(sel_B_had_sorted, h3_tautau, sel_eta_tau_med, sel_eta_b_med, dR_match)")

                .Define("isRecoMatchedHHH_nonunique", "AnalysisFCChh::find_hhh_signal_match_non_unique(sel_B_had_sorted, h3_tautau, sel_eta_tau_med, sel_eta_b_med, dR_match)")        
                .Define("isRecoMatchedHHH_LR", "AnalysisFCChh::find_hhh_signal_match_LR(sel_B_had_sorted, h3_tautau, sel_eta_tau_med, sel_eta_b_med, bb_PF08_trackJet02, tautau_PF08_trackJet02, 0.3, 0.3, 0.8)")       

                #.Filter("n_b_jets_medium_tauprio == 4 and isLep1 == 0 and isLep2 == 0 and n_tau_jets_medium == 2 and n_visible_tauhad == 2 and n_sel_el_0p2 == 0 and n_sel_mu == 0 and OS_tau == 1")

                # and now we're back :)
                .Define("LepHad_MMC", "AnalysisFCChh::solve_ditau_MMC_METScan_angular_lephad_weighted(tau_1_tlv_LH, tau_2_tlv_LH, isLep1, isLep2, n_prongs_tau1, n_prongs_tau2, MET_x, MET_y, 40, metres_njet, 8, 50, 30, n_b_jets_medium_tauprio, n_tau_jets_medium)")
                .Define("unweighted_MMC_mass_lepHad", "AnalysisFCChh::weighted_mode_from_mw(LepHad_MMC, 400, 0, 1000, false, false)")

                #.Filter("n_tau_jets_medium == 2 and n_sel_el_0p2 == 0 and n_sel_mu == 0")
                #.Filter("n_visible_tauhad == 2")
                #.Define("LepHad_MMC_para_perp", "AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_debugCounters(tau_1_tlv_LH, tau_2_tlv_LH, isLep1, isLep2, n_prongs_tau1, n_prongs_tau2, MET_x, MET_y, 25, d_MET_x_derived, d_MET_y_derived, 4, 25, 30, n_b_jets_medium_tauprio, n_tau_jets_medium)")
                #.Define("unweighted_MMC_para_perp", "AnalysisFCChh::weighted_mode_from_mw(LepHad_MMC_para_perp, 800, 0, 400, false, false)")
                #.Define("weighted_MMC_para_perp", "AnalysisFCChh::weighted_mode_from_mw(LepHad_MMC_para_perp, 800, 0, 400, false, true)")

                
                .Define(
                "MMC_all",
                "AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_withWeights("
                "tau_1_tlv_LH, tau_2_tlv_LH, isLep1, isLep2, n_prongs_tau1, n_prongs_tau2,"
                "MET_x, MET_y, 25, d_MET_x_derived, d_MET_y_derived, 4, 25, 30,"
                "n_b_jets_medium_tauprio, n_tau_jets_medium, false, 10)"
                )
                .Define("MMC_solutions", "std::get<0>(MMC_all)")
                .Define("MMC_components", "std::get<1>(MMC_all)")
                .Define("MMC_metRatio", "AnalysisFCChh::reweight_mditau_components(MMC_components, true, false, true, true)")
                .Define("MMC_angleOnly", "AnalysisFCChh::reweight_mditau_components(MMC_components, false, true, false, false)")
                .Define("MMC_ratioOnly", "AnalysisFCChh::reweight_mditau_components(MMC_components, false, false, true, false)")
                .Define("MMC_metAngle", "AnalysisFCChh::reweight_mditau_components(MMC_components, true, true, false, false)")
                .Define("MMC_metOnly", "AnalysisFCChh::reweight_mditau_components(MMC_components, true, false, false, false)")

                .Define("MMC_full",     "AnalysisFCChh::reweight_mditau_components(MMC_components, true, true, true, true)")
                .Define("unweighted_MMC_para_perp",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_solutions, 800, 0, 400, false, false)")
                .Define("weighted_MMC_para_perp",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_solutions, 800, 0, 400, false, true)")
                .Define("weighted_MMC_atlasstyle", "AnalysisFCChh::atlas_style_mmc_mass(MMC_solutions, 1500, 0, 3000, false, 5, false)")

                .Define("metOnly_mode",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metOnly, 800, 0, 400, false, true)")
                .Define("metRatio_mode", 
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metRatio, 800, 0, 400, false, true)")
                .Define("angleOnly_mode",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_angleOnly, 800, 0, 400, false, true)")
                .Define("ratioOnly_mode",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_ratioOnly, 800, 0, 400, false, true)")
                .Define("metAngle_mode",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metAngle, 800, 0, 400, false, true)")
                .Define("full_mode",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_full, 800, 0, 400, false, true)")
                
                
                .Define(
                "MMC_all_vispTcal",
                "AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_vispTanglecalibration_weights("
                "tau_1_tlv_LH, tau_2_tlv_LH, isLep1, isLep2, n_prongs_tau1, n_prongs_tau2, "
                "MET_x, MET_y, 50, d_MET_x_derived, d_MET_y_derived, 4, 50, 30, "
                "n_b_jets_medium_tauprio, n_tau_jets_medium, false, 10)"
                )
                .Define("MMC_solutions_vispTcal", "std::get<0>(MMC_all_vispTcal)")
                .Define("MMC_components_vispTcal", "std::get<1>(MMC_all_vispTcal)")
                .Define("MMC_metRatio_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, true, false, true, true)")
                .Define("MMC_angleOnly_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, false, true, false, false)")
                .Define("MMC_ratioOnly_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, false, false, true, false)")
                .Define("MMC_metAngle_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, true, true, false, false)")
                .Define("MMC_metOnly_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, true, false, false, false)")
                .Define("MMC_full_vispTcal", "AnalysisFCChh::reweight_mditau_components(MMC_components_vispTcal, true, true, true, true)")
                
                .Define("unweighted_MMC_para_perp_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_solutions_vispTcal, 800, 0, 400, false, false)")
                .Define("weighted_MMC_para_perp_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_solutions_vispTcal, 800, 0, 400, false, true)")
                .Define("weighted_MMC_atlasstyle_vispTcal",
                        "AnalysisFCChh::atlas_style_mmc_mass(MMC_solutions_vispTcal, 1500, 0, 3000, false, 5, false)")
                .Define("metOnly_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metOnly_vispTcal, 800, 0, 400, false, true)")
                .Define("metRatio_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metRatio_vispTcal, 800, 0, 400, false, true)")
                .Define("angleOnly_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_angleOnly_vispTcal, 800, 0, 400, false, true)")
                .Define("ratioOnly_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_ratioOnly_vispTcal, 800, 0, 400, false, true)")
                .Define("metAngle_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_metAngle_vispTcal, 800, 0, 400, false, true)")
                .Define("full_mode_vispTcal",
                        "AnalysisFCChh::weighted_mode_from_mw(MMC_full_vispTcal, 800, 0, 400, false, true)")


                # now for reco-matched samples too
                .Define("LepHad_MMC_para_perp_recomatched", "AnalysisFCChh::solve_ditau_MMC_METScan_para_perp(vis1_reco_tlv, vis2_reco_tlv, false, false, n_prongs_tau1, n_prongs_tau2, gen_MET_x, gen_MET_y, 50, d_MET_x_derived, d_MET_y_derived, 4, 50, 30, n_b_jets_medium_tauprio, n_tau_jets_medium)")
                .Define("recomatch_unweighted_MMC_para_perp", "AnalysisFCChh::weighted_mode_from_mw(LepHad_MMC_para_perp_recomatched, 800, 0, 400, false, false)")
                .Define("recomatch_weighted_MMC_para_perp", "AnalysisFCChh::weighted_mode_from_mw(LepHad_MMC_para_perp_recomatched, 800, 0, 400, false, true)")
                
                # try with a MH codex version 1, masses included in tau TLVs
                # should attempt to find the correlation per-event
                .Define("MH_MMC", "AnalysisFCChh::solve_ditau_MMC_ATLAS_markov(tau_1_tlv_LH, tau_2_tlv_LH, isLep1, isLep2, n_prongs_tau1, n_prongs_tau2, MET_x, MET_y, metres_para, metres_perp, 0, 3000000, 50, n_b_jets_medium_tauprio, n_tau_jets_medium, true, true)")
                # solution set is by default weighted for MH algorithms
                .Define("mass_MH_MMC_unweighted", "AnalysisFCChh::weighted_mode_from_mw(MH_MMC, 800, 0, 400, false, false)")
                .Define("mass_MH_MMC_atlasstyle", "AnalysisFCChh::atlas_style_mmc_mass(MH_MMC, 1500, 0, 3000, false, 5, false)")

                # solid baseline with truth objects -> HAS to be solved here
                .Define("MMC_truthinfo", "AnalysisFCChh::solve_ditau_MMC_METScan_para_perp(TLV_pvis_truthHadronicTaus_1, TLV_pvis_truthHadronicTaus_2, isLep1, isLep2, n_charged_truthHadronicTaus_1, n_charged_truthHadronicTaus_2, gen_MET_x, gen_MET_y, 50, d_MET_x_derived, d_MET_y_derived, 4, 50, 30, n_b_jets_medium_tauprio, n_tau_jets_medium)")
        
        )
        

        return dframe8
        
    # Mandatory: output function, please make sure you return the branch list
    # as a python list
    def output(self):
        '''
        Output variables which will be saved to output root file.
        '''
        branchList = [
                "weight", #"please_work", "mmc_peak",
                "weight_k3m1_k4m1", "weight_k30_k4m2", "weight_k3m2_k40", "weight_k30_k4m1",
                "weight_k3m1_k40", "weight_k3m2_k4m1", "weight_k3m1_k4m2", "weight_k3m0p5_k4m1",
                "weight_k3m1p5_k4m1",
                
                "pT_truth_htautau", "pT_truth_hbb1", "pT_truth_hbb2",
                "pT_truth_tautau1", "pT_truth_bb1", "pT_truth_bb2",
                "Higgs_HT_truth",
                "dR_truth_tautau", "dR_truth_bb2", "dR_truth_bb1",
                #"constits_charge",
                "n_neutrinos_truthHadronicTaus_1", "n_neutrinos_truthHadronicTaus_2",
                "m_vis_truthHadronicTaus_1", "m_vis_truthHadronicTaus_2",
                "m_mis_truthHadronicTaus_1", "m_mis_truthHadronicTaus_2",
                "n_charged_truthHadronicTaus_1",
                "n_charged_truthHadronicTaus_2",
                "pmis_truthHadronicTaus_1",
                "pvis_truthHadronicTaus_1",
                "pvis_truthHadronicTaus_2",
                "pmis_truthHadronicTaus_2",
                "angle3D_vis_mis_tau2",
                "angle3D_vis_mis_tau1",
                #"x_visE_1",
                #"x_visE_2",
                #"x_misP_2","x_misP_1",
                
                
                "metres_njet", "metres_est",
                
                "n_reco_matched_tau",
                "n_reco_matched_b",
                "n_b_jets_medium",
                "n_b_jets_medium_tauprio",
                "n_contam_0p3",
                "n_b_tau_shared_jet_0p3",
                "n_tau_jets_medium",
                "n_tau_jet_not_b",
                "pT_reco_matched_jet_1",
                "pT_reco_matched_jet_2",
                "pT_reco_matched_jet_3",
                "pT_reco_matched_jet_4",
                "pT_reco_matched_b_jet_1",
                "is_contam_Bhad_matched_jet_1",
                "pt_reco_match_B_had_bjet_not_tau_1",
                "pT_reco_matched_b_jet_2",
                "pT_reco_matched_b_jet_3",
                "pT_reco_matched_b_jet_4",
                "pT_reco_matched_tau_tau_jet_1",
                "pT_reco_matched_tau_tau_jet_2", 
                "pT_reco_matched_tau_jet_1",
                "pT_reco_matched_tau_jet_2",
                "pT_truth_matched_h1_b1",
                "pT_truth_matched_h1_b2",
                "pT_truth_matched_h2_b1",
                "pT_truth_matched_h2_b2",
                "pT_visible_tauhad_1",
                "pT_visible_tauhad_2",
                "n_visible_tauhad",
                "n_jets_sel",
                "dphi_truth_vismis_2",
                "dphi_truth_vismis_1",
                
                "n_reco_matched_B_hadrons",
                "pt_reco_match_B_had_1",
                "pt_reco_match_B_had_2",
                "pt_reco_match_B_had_3",
                "pt_reco_match_B_had_4",
                "pt_reco_match_B_had_bjet_1",
                "pt_reco_match_B_had_bjet_2",
                "pt_reco_match_B_had_bjet_3",
                "pt_reco_match_B_had_bjet_4",
                "pT_B_had_1", 
                "pT_B_had_2",
                "pT_B_had_3",
                "pT_B_had_4",

                "pT_h1_LR", "eta_h1_LR", "phi_h1_LR", "m_h1_LR",
                "pT_h2_LR", "eta_h2_LR", "phi_h2_LR", "m_h2_LR",
                "pT_Htautau_LR", "m_Htautau_LR", "eta_Htautau_LR", "phi_Htautau_LR",
                "sigma_12", "sigma_23", "sigma_13",
                "sigma_x", "sigma_y",
                "m_hhh_vis_LR",
                "isRecoMatchedHHH_LR",
                "n_PF08_jets_sel","n_bb_tagged_08Jets","n_tautau_tagged_08Jets",
                "n_selected_b_tagged_trackJets_medium",
                "MET_x", "MET_y",
                
                "gen_MET", "gen_MET_x", "gen_MET_y", "gen_MET_phi", "gen_MET_eta",
                "d_MET", "d_MET_x", "d_MET_y", "d_MET_phi", "d_MET_eta",

                "eta_h1b1", "eta_h1b2", 
                "eta_h2b1", "eta_h2b2", 
                "eta_htau1", "eta_htau2",
                "n_sel_eta_B_hadrons_fromH_final",
                "isContam_0p1",
                "isContam_0p2",
                "isContam_0p3",
                "isContam_0p4",
                "isContam_0p5",
                "m_Htautau_MC",
                "m_Htautau_reco",
                "m_h1_recomatch",
                "m_h2_recomatch",
                
                "n_taumu_fromH",
                "n_tauel_fromH",
                "n_taulep_fromH",
                
                # reco-matching truth Higgs leptons
                "n_reco_matched_leading_el_fromH_allel",
                "n_reco_matched_leading_el_fromH_leading_el_0p1",
                "n_reco_matched_leading_el_fromH_leading_el_0p2",
                "n_reco_matched_leading_el_fromH_selel",
                
                "pT_reco_matched_leading_el_fromH_allel",
                "pT_reco_matched_leading_el_fromH_leading_el_0p1",
                # new working point
                "min_dr_signal_elec",
                "min_dr_reco_elec",
                "n_leading_el_fromH",
                #"el_all_iso_var",
                "pT_reco_matched_leading_el_fromH_selel_0p2",
                "n_reco_matched_leading_el_fromH_selel_0p2",
                "pT_leading_tauel_fromH",
                "eta_leading_tauel_fromH",
                "phi_leading_tauel_fromH",
                "phi_reco_matched_leading_el_fromH_leading_el_0p2",
                "pT_reco_matched_leading_el_fromH_leading_el_0p2",
                "n_reco_matched_leading_el_fromH_selel_0p3",
                "pT_reco_matched_leading_el_fromH_selel_0p3",
                "pT_reco_matched_leading_el_fromH_leading_el_0p3",
                "n_reco_matched_leading_el_fromH_leading_el_0p3",
                
                
                #"mu_all_iso_var",

                "pT_leading_taumu_fromH",
                "phi_leading_taumu_fromH",
                "eta_leading_taumu_fromH",
                "phi_reco_matched_leading_mu_fromH_selmu",
                
                "pT_reco_matched_leading_mu_fromH_selmu",
                "n_reco_matched_leading_mu_fromH_selmu_0p3",
                "pT_reco_matched_leading_mu_fromH_selmu_0p3",
                "pT_reco_matched_leading_mu_fromH_allmu",
                
                "n_sel_mu_0p3",
                "pT_reco_matched_leading_mu_fromH_leading_mu_0p3",
                "n_reco_matched_leading_mu_fromH_leading_mu_0p3",
                "n_reco_matched_leading_mu_fromH_selmu",
                "n_reco_matched_leading_mu_fromH_allmu",
                
                
                #hh->4b pairing efficiencies
                "n_reco_matched_h2",
                "n_reco_matched_h1",
                "n_reco_matched_h2_squaremass",
                "n_reco_matched_h1_squaremass",
                "n_reco_matched_h2_absmass",
                "n_reco_matched_h1_absmass",
                "m_h1_absmass",
                "m_h2_absmass",
                "m_h1_squaremass",
                "m_h2_squaremass",
                "m_h1",
                "m_h2",
                "n_reco_matched_B_hadrons_bjets",
                "OS_tau",
                "OS_taue",
                "OS_taumu",

                # purity of tautau selection
                "n_reco_matched_tau_tau_jet_1",
                "n_reco_matched_tau_tau_jet_2",
                "n_sel_el_0p2",
                "n_sel_mu",
                "n_prongs_tau1",
                "n_prongs_tau2",
                "metres_perp",
                "metres_para",
                "d_MET_y_derived",
                "d_MET_x_derived",
                "d_MET_y_ratio",
                "d_MET_x_ratio",
                "asym_x_y",
                "metsig_est",
                
                
                "truth_b12_z",
                "m_h1_squaremass_massordered",
                "m_h2_squaremass_massordered",
                "m_h1_absmass_massordered",
                "m_h2_absmass_massordered",
                "m_vis_tau1_reco",
                "m_vis_tau2_reco",
                "m_tautau_col",
                "m_tautau_mT",
                "m_tautau_smin",
                #"unweighted_MMC_para_perp",
                #"weighted_MMC_para_perp",
                #"full_mode",
                #"metAngle_mode",
                #"ratioOnly_mode",
                #"angleOnly_mode",
                #"metOnly_mode",
                #"metRatio_mode",
                #"weighted_MMC_atlasstyle",
                #"mass_MH_MMC_atlasstyle",
                
                "unweighted_MMC_para_perp_vispTcal",
                "weighted_MMC_para_perp_vispTcal",
                "full_mode_vispTcal",
                "metAngle_mode_vispTcal",
                "ratioOnly_mode_vispTcal",
                "angleOnly_mode_vispTcal",
                "metOnly_mode_vispTcal",
                "metRatio_mode_vispTcal",
                "m_hhh_truth",
        
                #"recomatch_weighted_MMC_para_perp",
                #"recomatch_unweighted_MMC_para_perp"
                #"mass_MH_MMC_unweighted",
                #"pT_reco_matched_tau2", "pT_reco_matched_tau1",
                #"theta_visReco_Miss_1",
                #"theta_visReco_Miss_2",
                
                # for ttbb classification, WW filter:
                "WlWlFilter",
                
                "dPhi_tautau", "dpT_tautau",
                # TRAINING / FITTING FEATURES:
                "m_hhh_vis",
                "RMS_mjj", "RMS_dR", "RMS_dEta",
                "pT_h1", "pT_h2",
                "mTb_min","dR_tautau", "xwt",
                "dR_h1", "dR_h2",
                "dPhi_met_tautau",
                "metsig_derived",
                "pT_tau2_tlv_LH","pT_tau1_tlv_LH",
                "pT_b1", "pT_b2", "pT_b3", "pT_b4",
                "aplanarity", "sphericity", "thrust",
                "MET", 
                "m_tautau_vis_OS"
                
        ]
        
        return branchList

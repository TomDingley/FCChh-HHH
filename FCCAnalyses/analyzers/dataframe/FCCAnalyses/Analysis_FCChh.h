// library with extra functions needed by custom FCC-hh analysis, such as
// HHbbZZ(llvv) analysis

#ifndef ANALYSIS_FCCHH_ANALYZERS_H
#define ANALYSIS_FCCHH_ANALYZERS_H

#include "ROOT/RVec.hxx"
#include "TLorentzVector.h"
#include "TString.h"
#include "TVector2.h"

#include "edm4hep/MCParticleData.h"
#include "edm4hep/ParticleIDData.h"
#include "edm4hep/ReconstructedParticleData.h"
#include "podio/ObjectID.h"
#include "TH1F.h"
#include "TMath.h"
#include <iostream>
#include "Math/VectorUtil.h"
#include <unordered_set>
#include <limits>
#include <cmath>
#include <algorithm>
#include <utility>
#include <TFile.h>
#include <TF1.h>
#include <Math/Vector4D.h> // is this needed over VectorUtil?
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/MCParticleData.h"
#include "edm4hep/TrackState.h"
#include "edm4hep/VertexData.h"
#include <array>
namespace AnalysisFCChh {

/// TESTER: return the transverse momenta of the input ReconstructedParticles
// ROOT::VecOps::RVec<float>
// get_pt_test(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in);

// helpers for reco particles:
TLorentzVector getTLV_reco(edm4hep::ReconstructedParticleData reco_part);
TLorentzVector getTLV_MC(edm4hep::MCParticleData MC_part);

ROOT::VecOps::RVec<edm4hep::MCParticleData>
SortMCByPt(const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in);

// struct to use for a pair of two reco particles, to make sure the correct ones
// stay together
struct RecoParticlePair {
  edm4hep::ReconstructedParticleData particle_1;
  edm4hep::ReconstructedParticleData particle_2;
  TLorentzVector merged_TLV() {
    TLorentzVector tlv_1 = getTLV_reco(particle_1);
    TLorentzVector tlv_2 = getTLV_reco(particle_2);
    return tlv_1 + tlv_2;
  }
  void sort_by_pT() {
    double pT_1 = sqrt(particle_1.momentum.x * particle_1.momentum.x +
                       particle_1.momentum.y * particle_1.momentum.y);
    double pT_2 = sqrt(particle_2.momentum.x * particle_2.momentum.x +
                       particle_2.momentum.y * particle_2.momentum.y);

    if (pT_1 >= pT_2) {
      return;
    } // nothing to do if already sorted corrected
    else {
      edm4hep::ReconstructedParticleData sublead = particle_1;

      particle_1 = particle_2;
      particle_2 = sublead;
      return;
    }
  }
};

// same for MC particle
struct MCParticlePair {
  edm4hep::MCParticleData particle_1;
  edm4hep::MCParticleData particle_2;
  TLorentzVector merged_TLV() {
    TLorentzVector tlv_1 = getTLV_MC(particle_1);
    TLorentzVector tlv_2 = getTLV_MC(particle_2);
    return tlv_1 + tlv_2;
  }
  void sort_by_pT() {
    double pT_1 = sqrt(particle_1.momentum.x * particle_1.momentum.x +
                       particle_1.momentum.y * particle_1.momentum.y);
    double pT_2 = sqrt(particle_2.momentum.x * particle_2.momentum.x +
                       particle_2.momentum.y * particle_2.momentum.y);

    if (pT_1 >= pT_2) {
      return;
    } // nothing to do if already sorted corrected
    else {
      edm4hep::MCParticleData sublead = particle_1;

      particle_1 = particle_2;
      particle_2 = sublead;
      return;
    }
  }
};

// merge the particles in such a pair into one edm4hep:RecoParticle to use with
// other functions (in a vector)
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
merge_pairs(ROOT::VecOps::RVec<RecoParticlePair> pairs);
int get_n_pairs(ROOT::VecOps::RVec<RecoParticlePair> pairs);
ROOT::VecOps::RVec<RecoParticlePair>
get_first_pair(ROOT::VecOps::RVec<RecoParticlePair>
                   pairs); // can use to get leading pair if the inputs to pair
                           // finding fct were pT sorted

// functions to separate the pair again - ONLY DOES THIS FOR THE FIRST PAIR IN
// THE VECTOR
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_first_from_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_second_from_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs);

// truth filter used to get ZZ(llvv) events from the ZZ(llvv+4l+4v) inclusive
// signal samples
bool ZZllvvFilter(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids);

int WWlvlvFilter(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
  ROOT::VecOps::RVec<podio::ObjectID> parent_ids);

// helper functions for the ZZllv truth filter:
bool isStablePhoton(edm4hep::MCParticleData truth_part);
bool isPhoton(edm4hep::MCParticleData truth_part);
bool isLep(edm4hep::MCParticleData truth_part);
bool isLightLep(edm4hep::MCParticleData truth_part);
bool isNeutrino(edm4hep::MCParticleData truth_part);
bool isTauNeutrino(edm4hep::MCParticleData truth_part);
bool isQuark(edm4hep::MCParticleData truth_part);
bool isZ(edm4hep::MCParticleData truth_part);
bool isW(edm4hep::MCParticleData truth_part);
bool isTau(edm4hep::MCParticleData truth_part);
bool isH(edm4hep::MCParticleData truth_part);
bool isb(edm4hep::MCParticleData truth_part);
bool isHadron(edm4hep::MCParticleData truth_part);
bool isTop(edm4hep::MCParticleData truth_part);
bool isGluon(edm4hep::MCParticleData truth_part);
bool isc(edm4hep::MCParticleData truth_part);
bool iss(edm4hep::MCParticleData truth_part);
bool isMuon(edm4hep::MCParticleData truth_part);
bool isElectron(edm4hep::MCParticleData truth_part);

float collinsSoper(
  TLorentzVector &y1,
  TLorentzVector &y2
);

// in the hh rest frame too
float cosThetaStar_yy_in_HH(
    const TLorentzVector& y1,
    const TLorentzVector& y2,
    const TLorentzVector& b1,
    const TLorentzVector& b2
);

ROOT::VecOps::RVec<float> minmax_dr(
    const TLorentzVector& y1,
    const TLorentzVector& y2,
    const TLorentzVector& b1,
    const TLorentzVector& b2
);


int checkZDecay(edm4hep::MCParticleData truth_Z,
                ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
int checkWDecay(edm4hep::MCParticleData truth_W,
                ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);

ROOT::VecOps::RVec<int> topChild(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts,
  const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids);


int printParticlesWithWFromTopParent(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc_particles,
  const ROOT::VecOps::RVec<ROOT::VecOps::RVec<podio::ObjectID>>& mc_parents);

int findTopDecayChannel(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids);  
int findHiggsDecayChannel(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids);

// truth level fct to get a Z->ll truth decay
ROOT::VecOps::RVec<edm4hep::MCParticleData> getTruthZtautau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &daughter_ids);

    
// find the SFOS pair of reconstructed leptons (electrons or muons)
ROOT::VecOps::RVec<RecoParticlePair>
getOSPairs(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> leptons_in);
ROOT::VecOps::RVec<RecoParticlePair> getDFOSPairs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> electrons_in,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> muons_in);
// ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// getOSPair(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> leptons_in);
ROOT::VecOps::RVec<RecoParticlePair> getBestOSPair(
    ROOT::VecOps::RVec<RecoParticlePair> electron_pairs,
    ROOT::VecOps::RVec<RecoParticlePair> muon_pairs); // closest to Z mass
ROOT::VecOps::RVec<RecoParticlePair>
getLeadingPair(ROOT::VecOps::RVec<RecoParticlePair> electron_pairs,
               ROOT::VecOps::RVec<RecoParticlePair>
                   muon_pairs); // pair with leading pT(pair)

// make a general pair, not caring about charges, e.g. the two b-jets
ROOT::VecOps::RVec<RecoParticlePair>
getPairs(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in);
ROOT::VecOps::RVec<RecoParticlePair> getPair_sublead(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in);


ROOT::VecOps::RVec<std::vector<float>> classify_taus(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
  ROOT::VecOps::RVec<podio::ObjectID> parent_ids
);



ROOT::VecOps::RVec<std::vector<float>> classify_taus2(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
  ROOT::VecOps::RVec<podio::ObjectID> parent_ids
);

ROOT::VecOps::RVec<MCParticlePair>
getPairs(ROOT::VecOps::RVec<edm4hep::MCParticleData> particles_in);

TLorentzVector get_leading4_merged(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in);

// struct declaration for storing Higgs-pair information.
struct HiggsDoubletResult {
    ROOT::VecOps::RVec<RecoParticlePair> pairs;
    ROOT::VecOps::RVec<int> used_bjet_indices; // indices into input bjets
};

HiggsDoubletResult getHiggsCandidateDoubletMassOrdered(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  TString strategy,
  float target_mass1,
  float target_mass2);

HiggsDoubletResult getHiggsCandidateDoublet(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untagged_jets,
    TString strategy, float target_mass1, float target_mass2);


HiggsDoubletResult getHiggsCandidateDoubletRandomBaseline(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets);

float xwt(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& taujets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untaggedjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  float met_px, float met_py
);

float topness(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  TLorentzVector tau1, TLorentzVector tau2,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untaggedjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  float met_px, float met_py
);


float sphericity(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
);

float thrust(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
);

float aplanarity(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
);

float mTb_min(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const edm4hep::ReconstructedParticleData& met   // MET as a RecoParticle with px,py in momentum.{x,y}
);

float RMS_mjj(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets);
float RMS_deta(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets);
float RMS_dR(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets);

float min_dr_signal(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& target_obj,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& dR1_obj,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& dR2_obj
);

bool isSignalContaminated(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_b_hadrons,
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_taus,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> selects_jets,
  float dR_truth
);

int nSignalContaminated(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_b_hadrons,
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_taus,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> selects_jets,
  float dR_truth
);

std::tuple<
  ROOT::VecOps::RVec<edm4hep::MCParticleData>,  // p_vis
  ROOT::VecOps::RVec<edm4hep::MCParticleData>>
visible_tauhad(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>&         daughter_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>&         parent_ids,
    TString type);

bool matchRecoToTruth(const RecoParticlePair& recoPair,
                      const std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>& truthPair,
                      float dR_threshold = 0.3);

std::tuple<RecoParticlePair, RecoParticlePair, int> getHiggsCandidateDoublet_truth(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untagged_jets,
  const std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>& truthH1,
  const std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>& truthH2,
  TString strategy,
  float target_mass1,
  float target_mass2);


// functions for HH pairing, used within HHH->4b2tau, HHH->4tau2b
ROOT::VecOps::RVec<RecoParticlePair> getHiggsCandidateTriplets(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets);

ROOT::VecOps::RVec<float> getHiggsMasses(
    const ROOT::VecOps::RVec<RecoParticlePair>& higgsCandidates);
    


ROOT::VecOps::RVec<int>
find_b_matched_nonH_tau_pairs_indices(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres);

// SORT OBJ COLLECTION
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> SortParticleCollection(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in);

// btags
ROOT::VecOps::RVec<bool>
getJet_tag(ROOT::VecOps::RVec<int> index,
           ROOT::VecOps::RVec<edm4hep::ParticleIDData> pid,
           ROOT::VecOps::RVec<float> values, int algoIndex);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getBhadron(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
           ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
bool hasBHadronParent(
    const edm4hep::MCParticleData& truth_part,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles);


ROOT::VecOps::RVec<edm4hep::MCParticleData> getBhadron_leptons_fromH(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
  const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
  const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids, // not used here, but kept for symmetry
  TString type);


ROOT::VecOps::RVec<edm4hep::MCParticleData>
getBhadron_final_fromH(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
    float ptMin);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getChadron(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
           ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_tagged_jets(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
                ROOT::VecOps::RVec<edm4hep::ParticleIDData> jet_tags,
                ROOT::VecOps::RVec<podio::ObjectID> jet_tags_indices,
                ROOT::VecOps::RVec<float> jet_tags_values, int algoIndex);

ROOT::VecOps::RVec<float> get_perp_para_metres(
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj
);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_untagged_jets(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
                  ROOT::VecOps::RVec<int> index,
                  ROOT::VecOps::RVec<edm4hep::ParticleIDData> pid,
                  ROOT::VecOps::RVec<float> values, int algoIndex);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> 
get_tau_tagged_not_btagged(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  // b/HF tags
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>&  jets_HF_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>&          jets_HF_tags_indices,
  const ROOT::VecOps::RVec<float>&                    jets_HF_tag_values,
  // tau tags
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>&  jets_tau_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>&          jets_tau_tags_indices,
  const ROOT::VecOps::RVec<float>&                    jets_tau_tag_values,
  int btagIndex,   // bit for the chosen b WP, e.g. 1 for "medium"
  int tauIndex);     // bit for the chosen tau WP, e.g. 1 for "medium"


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_btagged_not_tau_tagged(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  // b/HF tags
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>&  jets_HF_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>&          jets_HF_tags_indices,
  const ROOT::VecOps::RVec<float>&                    jets_HF_tag_values,
  // tau tags
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>&  jets_tau_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>&          jets_tau_tags_indices,
  const ROOT::VecOps::RVec<float>&                    jets_tau_tag_values,
  int btagIndex,   // bit for the chosen b WP, e.g. 1 for "medium"
  int tauIndex     // bit for the chosen tau WP, e.g. 1 for "medium"
);


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> get_tau_jets_exclusive(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jet_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jet_tags_indices,
  const ROOT::VecOps::RVec<float>& jet_tags_values,
  int tauIndex,
  int btagIndex);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> get_untagged_jets_exclusive(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jets_HF_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jets_HF_tags_indices,
  const ROOT::VecOps::RVec<float>& jets_HF_tag_values,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jets_tau_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jets_tau_tags_indices,
  const ROOT::VecOps::RVec<float>& jets_tau_tag_values,
  int btagIndex,
  int tauIndex);
/*
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_untagged_jets_hhh(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> all_jets,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets_2);
*/
// tau jets
ROOT::VecOps::RVec<edm4hep::MCParticleData> find_truth_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
get_tau_jets(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
                ROOT::VecOps::RVec<edm4hep::ParticleIDData> jet_tags,
                ROOT::VecOps::RVec<podio::ObjectID> jet_tags_indices,
                ROOT::VecOps::RVec<float> jet_tags_values, int algoIndex);

                    

ROOT::VecOps::RVec<edm4hep::MCParticleData>
getTruthTauHads(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getTruthTau(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
            ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
            ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getTruthTauLeps(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type);
ROOT::VecOps::RVec<edm4hep::MCParticleData> 
getTruthTauEmu(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type);




double weighted_mode_from_mw(
    const std::vector<std::pair<float,float>>& mw_log, // {mass, log_weight}
    int nbins, double xmin, double xmax,
    bool refine_peak,
    bool toWeight 

);

double atlas_style_mmc_mass(
    const std::vector<std::pair<float,float>>& mw, // {mass, weight}
    int nbins, double xmin, double xmax,
    bool smooth = false,
    int window_bins = 5,
    bool refine_peak = true
);

float log_normal(float x, float a, float b, float c, float d);
std::vector<float> pTtau_parametrisation(float pT_tau, int n_prongs);

// MH alg approach
std::vector<std::pair<double,double>>
solve_ditau_MMC_MH(
    const TLorentzVector& tau1,
    const TLorentzVector& tau2,
    double MET_x, double MET_y,
    double metres,              // event MET resolution (magnitude)
    int n_iter,
    int burn_in,
    int thin,
    double step_phi,
    double step_met,     // if <0, defaults to metres/sqrt(2)
    bool use_sigma_scaling,
    double alpha,         // only used if use_sigma_scaling==false
    uint64_t seed
);

std::tuple<
  ROOT::VecOps::RVec<TLorentzVector>,  // p_vis
  ROOT::VecOps::RVec<TLorentzVector>,  // p_mis
  ROOT::VecOps::RVec<int>,             // n_charged
  ROOT::VecOps::RVec<int>,             // n_neutral
  ROOT::VecOps::RVec<float>,            // m_vis
  ROOT::VecOps::RVec<float>,             // m_mis
  ROOT::VecOps::RVec<int>               // n_neutrinos
>
getTruthTauHadronic(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    TString type);

struct TauAngleRecord {
  bool  matched{false};
  int   truth_idx{-1};      // 0 or 1
  int   reco_idx{-1};       // 0 or 1 if matched

  float dR_best{999.f};
  float pt_reco{-1.f};

  float theta_tt{-1.f};     // angle(truth_vis, truth_miss)
  float theta_rt{-1.f};     // angle(reco_vis,  truth_miss)

  int   n_prongs_truth{-1};
};

std::array<TauAngleRecord, 2> matchTwoTruthTwoRecoAndAngles(
    const TLorentzVector& pvis_truth_0,
    const TLorentzVector& pmiss_truth_0,
    int nprongs_truth_0,
    const TLorentzVector& pvis_truth_1,
    const TLorentzVector& pmiss_truth_1,
    int nprongs_truth_1,
    const TLorentzVector& reco0,
    const TLorentzVector& reco1,
    float drCut = 0.2f
);

// isolation: select only those particles of sel_parts that are isolated by the
// given dR from the check_parts
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
sel_isolated(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> sel_parts,
             ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_parts,
             float dR_thres = 0.4);


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
select_with_mask(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objs,
                   const ROOT::VecOps::RVec<char>& mask);

                   
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
sel_by_iso_fail(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& parts,
                               const ROOT::VecOps::RVec<float>& isoVar,
                               float thr);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
sel_non_isolated(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> sel_parts,
             ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_parts,
             float dR_thres = 0.4);

// merge two four vectors into one to create a new particle (follow vector
// structure to be able to use with other RecoParticle fcts easily like get_pt
// etc.)
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> merge_parts_TLVs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
merge_parts_TLVs(ROOT::VecOps::RVec<edm4hep::MCParticleData> particle_1,
                 ROOT::VecOps::RVec<edm4hep::MCParticleData> particle_2);

// reco level quantities
// transverse masses:
ROOT::VecOps::RVec<float>
get_mT(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
       ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2);
ROOT::VecOps::RVec<float>
get_mT_new(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
           ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2);
ROOT::VecOps::RVec<float>
get_m_pseudo(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
             ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj);
ROOT::VecOps::RVec<float>
get_mT_pseudo(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
              ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj);
TLorentzVector getTLV_MET(edm4hep::ReconstructedParticleData met_object);

// stransverse mass mT2 :
// https://www.hep.phy.cam.ac.uk/~lester/mt2/#Alternatives
// ROOT::VecOps::RVec<float>
// get_mT2(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
// ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2,
// ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj);
// ROOT::VecOps::RVec<float>
// get_mT2_125(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_1, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_2, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj);

// angular distances:function can return dR, dEta, dPhi for any two fully
// reconstructed particles that have a full 4 vector
ROOT::VecOps::RVec<float> get_angularDist(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2,
    TString type = "dR");
ROOT::VecOps::RVec<float> get_angularDist_MET(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj,
    TString type = "dR");

ROOT::VecOps::RVec<float>
get_angularDist_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs,
                     TString type = "dR");
ROOT::VecOps::RVec<float>
get_angularDist_pair(ROOT::VecOps::RVec<MCParticlePair> pairs,
                     TString type = "dR");

// HT variables
ROOT::VecOps::RVec<float>
get_HT2(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
        ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2);
ROOT::VecOps::RVec<float>
get_HT_wInv(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET,
            ROOT::VecOps::RVec<RecoParticlePair> ll_pair,
            ROOT::VecOps::RVec<RecoParticlePair> bb_pair);
ROOT::VecOps::RVec<float>
get_HT_true(ROOT::VecOps::RVec<RecoParticlePair> ll_pair,
            ROOT::VecOps::RVec<RecoParticlePair> bb_pair);
ROOT::VecOps::RVec<float> get_HT2_ratio(ROOT::VecOps::RVec<float> HT2,
                                        ROOT::VecOps::RVec<float> HT_wInv);
ROOT::VecOps::RVec<float>
get_MET_significance(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET,
                     ROOT::VecOps::RVec<float> HT_true, bool doSqrt = true);

// reco mass of lepton+b-jet, to try suppress ttbar processes
ROOT::VecOps::RVec<RecoParticlePair>
make_lb_pairing(ROOT::VecOps::RVec<RecoParticlePair> lepton_pair,
                ROOT::VecOps::RVec<RecoParticlePair> bb_pair);
ROOT::VecOps::RVec<float>
get_mlb_reco(ROOT::VecOps::RVec<RecoParticlePair> lb_pairs);
ROOT::VecOps::RVec<float>
get_mlb_MET_reco(ROOT::VecOps::RVec<RecoParticlePair> lb_pairs,
                 ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET);

// for separating bbWW and bbtautau?
ROOT::VecOps::RVec<float>
get_pzeta_vis(ROOT::VecOps::RVec<RecoParticlePair> lepton_pair);
ROOT::VecOps::RVec<float>
get_pzeta_miss(ROOT::VecOps::RVec<RecoParticlePair> lepton_pair,
               ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET);
ROOT::VecOps::RVec<float> get_dzeta(ROOT::VecOps::RVec<float> pzeta_miss,
                                    ROOT::VecOps::RVec<float> pzeta_vis,
                                    float factor = 0.85);

// combine particles:
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
build_HZZ(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
          ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj);

// retrieve children from a given truth particle
ROOT::VecOps::RVec<edm4hep::MCParticleData> get_immediate_children(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids);

// Already have:
enum class TauDecay : int {
  Unknown=0, LeptonicE=1, LeptonicMu=2,
  Hadronic1p=3, Hadronic3p=4, HadronicOther=5
};

inline bool tau_is_leptonic(TauDecay d) {
  return d == TauDecay::LeptonicE || d == TauDecay::LeptonicMu;
}
inline bool tau_is_hadronic(TauDecay d) {
  return d == TauDecay::Hadronic1p || d == TauDecay::Hadronic3p || d == TauDecay::HadronicOther;
}
inline bool tau_is_e(TauDecay d) { return d == TauDecay::LeptonicE; }
inline bool tau_is_mu(TauDecay d) { return d == TauDecay::LeptonicMu; }

inline const char* tau_decay_label(TauDecay d) {
  switch (d) {
    case TauDecay::LeptonicE:    return "τ→enunu";
    case TauDecay::LeptonicMu:   return "τ→mununu";
    case TauDecay::Hadronic1p:   return "τ→had (1-prong)";
    case TauDecay::Hadronic3p:   return "τ→had (3-prong)";
    case TauDecay::HadronicOther:return "τ→had (other)";
    default:                     return "Unknown";
  }
}



// Select the truth Higgs and its decay children, depending on which particles it decays to
std::pair<ROOT::VecOps::RVec<edm4hep::MCParticleData>, 
          ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>>
get_truth_Higgs(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                TString decay = "bb");

std::pair<ROOT::VecOps::RVec<edm4hep::MCParticleData>,
          ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>>
get_truth_Higgs_6b(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids, TString /*decay*/);

// Select the truth Higgs and its decay children, depending on which particles it decays to
std::pair<ROOT::VecOps::RVec<edm4hep::MCParticleData>, 
ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>>
get_truth_Higgs_pTorder(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
      ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
      TString decay = "bb");
ROOT::VecOps::RVec<edm4hep::MCParticleData>
get_truth_Z_decay(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
                  TString decay = "ZZ");

// Filters and specifics for the bbtautau analysis:
bool isFromHadron(edm4hep::MCParticleData truth_part,
                  ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
                  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
              
int tauIsFromWhere(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);

int find_mc_matched_particle_both_PDGID(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &mc_parents,
    float dR_thres);
bool hasHiggsParent(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
bool hasZParent(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);


bool isFromHiggsDirect(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);

std::tuple<ROOT::VecOps::RVec<std::pair<int, int>>,
ROOT::VecOps::RVec<std::pair<int, int>>>
getJetPairings(
const ROOT::VecOps::RVec<edm4hep::MCParticleData>& reco_jets,
const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles);

bool isChildOfTauFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);

bool isChildOfTauFromZ(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
bool isChildOfTauHadFromZ(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);

bool isChildOfTauLepFromZ(
    const edm4hep::MCParticleData &truth_part,
    const ROOT::VecOps::RVec<podio::ObjectID> &parent_ids,
    const ROOT::VecOps::RVec<podio::ObjectID> &daughter_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &truth_particles);
    
bool isChildOfWFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
bool isChildOfZFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getLepsFromTau(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
               ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getLepsFromW(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
             ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getLepsFromZ(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
             ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getPhotonsFromH(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
                ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<int> getTruthLepLepFlavour(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getTruthEle(ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getTruthMu(ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau);

float get_mtautau_vis_bestOS(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
                              const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& e,
                              const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& mu);

                              TLorentzVector
get_Htautau_vis_exclusive_TLV(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
  int n_tau,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  int n_el,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  int n_mu);

ROOT::VecOps::RVec<TLorentzVector>
get_Htautau_vis_exclusive_TLVs(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
  int n_tau,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  int n_el,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  int n_mu);

std::tuple<
  edm4hep::ReconstructedParticleData,
  edm4hep::ReconstructedParticleData,
  TLorentzVector,
  bool,
  bool
> get_Htautau_vis_exclusive_recoTLV(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>&,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>&,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>&);

// tautau specific masses/variables
ROOT::VecOps::RVec<float> get_x_fraction(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> visible_particle,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET);
    
ROOT::VecOps::RVec<float> get_x_fraction_truth(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> visible_particle,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET);

ROOT::VecOps::RVec<float> get_mtautau_col(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> ll_pair_merged,
    float x1, 
    float x2);

ROOT::VecOps::RVec<float> get_mbbtautau_col(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> bb_pair_merged,
    ROOT::VecOps::RVec<float> mtautau_col);

float solve_ditau_system(
    TLorentzVector tau1,
    TLorentzVector tau2,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET
);

float mT_tautau(
  const TLorentzVector& tau1_vis,
  const TLorentzVector& tau2_vis,
  float MET_x,
  float MET_y
);

float compute_smin(
    TLorentzVector tau1_vis,
    TLorentzVector tau2_vis,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET
);

float ComputeMMCPeak(const std::vector<float>& mtt,
                     const std::vector<float>& logL,
                     float lo, float hi, int nbins);

float find_MMC_mass(
  const std::pair<std::vector<float>, std::vector<float>>& mmc_result,
  float lo = 0.f,
  float hi = 600.f,
  int nbins = 120);


std::pair<std::vector<double>, std::vector<double>> solve_pTnu(
  TLorentzVector tau1,
  TLorentzVector tau2,
  double MET_x,
  double MET_y
);

float GetModeFromVector(
    std::vector<float> values,
    double xmin,
    double xmax,
    int nbins);


std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_weighted(
  TLorentzVector tau1,
  TLorentzVector tau2,
  float MET_x,
  float MET_y,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  int nsteps,
  float metres,
  int nMETsig,
  int nMETsteps
);



std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_angular_weighted(
  TLorentzVector tau1,
  TLorentzVector tau2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres,
  int nMETsig,
  int nMETsteps
);


std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium
);

std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp_debugCounters(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  std::vector<std::array<float,8>>* weight_components = nullptr,
  bool diagnostic = false,
  int diag_topN = 10
);



using MMCSolutionsAndWeights = std::pair<
std::vector<std::pair<float,float>>,
std::vector<std::array<float,8>>
>;

MMCSolutionsAndWeights solve_ditau_MMC_METScan_para_perp_withWeights(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  bool diagnostic,
  int diag_topN
);

MMCSolutionsAndWeights solve_ditau_MMC_METScan_para_perp_withWeights(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  bool diagnostic = false,
  int diag_topN = 10
);

// Convenience wrapper to fetch per-solution weight breakdown (mditau, w_met, w_theta1, w_theta2, w_ratio1, w_ratio2, w_miss, w_event)
std::vector<std::array<float,8>> solve_ditau_MMC_METScan_para_perp_weights(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  bool diagnostic = false,
  int diag_topN = 10
);

// Build mditau-weight pairs from stored components with selectable factors
std::vector<std::pair<float,float>> reweight_mditau_components(
  const std::vector<std::array<float,8>>& components,
  bool use_met,
  bool use_theta,
  bool use_ratio,
  bool use_miss
);

// ATLAS TF1-based weight evaluation (angle + ratio) using MMC_params_v1_fixed.root
// Returns {w_event, w_angle1, w_angle2, w_ratio1, w_ratio2}
std::array<float,5> atlas_tf1_weights(
  const TLorentzVector& tau1_vis,
  const TLorentzVector& nu1,
  bool isLep1,
  int n_charged_tracks_1,
  bool isLead1,
  const TLorentzVector& tau2_vis,
  const TLorentzVector& nu2,
  bool isLep2,
  int n_charged_tracks_2,
  bool isLead2
);


static std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp_vispTAngleCalibration(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  std::vector<std::array<float,8>>* weight_components,
  bool diagnostic,
  int diag_topN,
  bool use_atlas_tf1
);
MMCSolutionsAndWeights solve_ditau_MMC_METScan_para_perp_vispTanglecalibration_weights(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  bool diagnostic,
  int diag_topN
);
// Full MMC MET-scan using ATLAS TF1 angle/ratio weights (optional components/diagnostics)
std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp_ATLAS_tf1(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres_x,
  float metres_y,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  std::vector<std::array<float,8>>* weight_components = nullptr,
  bool diagnostic = false,
  int diag_topN = 10
);


// ATLAS-style Markov-chain MMC (non-LFV) with para/perp MET and covphi
std::vector<std::pair<float,float>> solve_ditau_MMC_ATLAS_markov(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  float sigma_para,
  float sigma_perp,
  float met_cov_phi,
  int nIter = 100000,
  int nMass_steps = 30,
  int n_b_jets_medium = 0,
  int n_tau_jets_medium = 0,
  bool efficiency_recover = true,
  bool use_atlas_tf1 = false
);

std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp_covphi(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float sigma_para,
  float sigma_perp,
  float met_cov_phi,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium
);

std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_angular_lephad_weighted(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres,
  int nMETsig,
  int nMETsteps,
  int nMass_steps, 
  int n_b_jets_medium,
  int n_tau_jets_medium
);

std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_angular_lephad_weighted_retry(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium
);

std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_angular_lephad_weighted_rescan(
  TLorentzVector tau1,
  TLorentzVector tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  float MET_x,
  float MET_y,
  int nsteps,
  float metres,
  int nMETsig,
  int nMETsteps,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium
);


struct MMC_MH_Result {
  std::vector<float> masses; // unweighted mass samples from the chain (post burn-in, thinned)
  double accept_rate;        // fraction of proposals accepted (0..1)
};

struct MMCPoint { float m; float lo68; float hi68; };

MMCPoint mmc_from_samples_mode(const std::vector<float>& v);

MMC_MH_Result solve_ditau_MMC_MH(
  const TLorentzVector& tau1,
  const TLorentzVector& tau2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  double MET_x_meas,
  double MET_y_meas,
  double metres,         // event-level MET resolution (total); split equally to x,y
  int    n_iter,         // total MH iterations (including burn-in)
  int    burn_in,
  int    thin,
  double sigma_phi,          // proposal stddev for angles (radians)
  double sigma_met,          // proposal stddev for MET x,y; if <0, set to metres/sqrt(2)/2
  unsigned long long seed       // RNG seed (0 => seed from time)
);


MMC_MH_Result solve_ditau_MMC_MH_lephad_ATLAS(
  const TLorentzVector& tau1,
  const TLorentzVector& tau2,
  bool isLep1,
  bool isLep2,
  int n_charged_tracks_1,
  int n_charged_tracks_2,
  double MET_x_meas,
  double MET_y_meas,
  double metres,         // event-level MET resolution (total); split equally to x,y
  int    n_iter,         // total MH iterations (including burn-in)
  int    burn_in,
  int    thin,
  double sigma_phi,      // proposal stddev for angles (radians)
  double sigma_met,      // proposal stddev for MET x,y; if <0, set to metres/sqrt(2)/2
  unsigned long long seed,
  double /*sigma_mass*/  // currently unused
);


float ditau_mass_collinear_then_mT(
    const TLorentzVector& vis1,
    const TLorentzVector& vis2,
    double METx, double METy
);



struct MMCandidate {
  TLorentzVector tautau;   // reconstructed ττ 4-vector
  double weight;     // total weight = w_met * w_ang
  double w_met;      // MET Gaussian weight
  double w_ang;      // angular prior weight
  // diagnostics (optional but handy)
  double phi1, phi2;
  double pT1, pT2;
  double z1, z2; // chosen pz solutions (when multiple exist)
};


TLorentzVector get_collinearApproximationTLV(const RecoParticlePair& taupair, float x1, float x2);


ROOT::VecOps::RVec<float> get_m_dihiggs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tau_pair_merged,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> yy_pair_merged,
    float x1, 
    float x2);

bool isHHH_4b2tau(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids);

// truth matching:
//  old function that can match only 1 particle -> TO REMOVE?
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> find_reco_matched(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts_all,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_matches_unique(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_parts,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres);


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_reco_matches_unique(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles_to_match,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres);


bool find_mc_matched_particle_both(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    float dR_thres);

bool find_mc_matched_particle_both_HadronTau(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &mc_parents,
    float dR_thres);

std::pair<int, edm4hep::MCParticleData>
find_highpt_tau_and_parent(
    const edm4hep::ReconstructedParticleData& reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& mc_parents,
    float dR_thres);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_matches_both(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_set1,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_set2,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres);
  
float metres(int n_jet);


int find_hhh_signal_match(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets,
  float dR_thres);

int find_hhh_signal_match_LR(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets_R04,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_LR_bb_jets,      // NEW: bb-tagged LR jets
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_LR_tautau_jets,  // NEW: tautau-tagged LR jets
  float dR_match_b,    // e.g. 0.3–0.4
  float dR_match_tau,  // e.g. 0.2–0.3
  float Rmatch_LR      // e.g. 0.8 or 1.5
);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_bb_tagged(
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> LR_jets,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> b_tagged_track_jets);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
remove_bb_b_overlap(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_tagged_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& smallR_bjets);
  
std::pair<TLorentzVector, TLorentzVector>
find_4b_hh(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_LR_jets,
           const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& b_jets);

std::vector<std::vector<TLorentzVector>>
find_4b_hh_pairs(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& b_jets);


TLorentzVector find_tautau_h(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tautau_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_jets
);

ROOT::VecOps::RVec<float> compute_sigma(
  TLorentzVector higgs_1,
  TLorentzVector higgs_2,
  TLorentzVector higgs_3
);


bool find_hhh_signal_match_non_unique(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets,
  float dR_thres);

std::tuple<
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>,
  ROOT::VecOps::RVec<edm4hep::MCParticleData>,
  ROOT::VecOps::RVec<edm4hep::MCParticleData>>
find_reco_matches_both_withTruth(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_jets_b_nonH_tau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    const ROOT::VecOps::RVec<int>& mother,  // immediate mother index
    float dR_thres);

ROOT::VecOps::RVec<edm4hep::MCParticleData>
find_mc_taus_b_nonH_tau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    const ROOT::VecOps::RVec<int>& mother,  // immediate mother index
    float dR_thres);
// isolation criterion, delphes style. flag exclude_light_leps does not check
// for isolation of test_parts vs electrons or muons (using the mass) as seems
// to be done in FCC-hh delphes sim
ROOT::VecOps::RVec<float> get_IP_delphes(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> test_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts_all,
    float dR_min = 0.3, float pT_min = 0.5, bool exclude_light_leps = true);

ROOT::VecOps::RVec<float> get_IP_delphes_new(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& test_parts,          // e.g. muons (NoIso)
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_tracks,        // EFlowTrack
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_photons,       // EFlowPhoton
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_neutral_hadr,  // EFlowNeutralHadron
    float dR_max, float pT_min);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
filter_lightLeps(ROOT::VecOps::RVec<int> recind, ROOT::VecOps::RVec<int> mcind,
                 ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco,
                 ROOT::VecOps::RVec<edm4hep::MCParticleData> mc);

// truth MET
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getNusFromW(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
            ROOT::VecOps::RVec<podio::ObjectID> parent_ids);
ROOT::VecOps::RVec<edm4hep::MCParticleData>
getNusFromTau(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
            ROOT::VecOps::RVec<podio::ObjectID> parent_ids, ROOT::VecOps::RVec<podio::ObjectID> daughter_ids, TString tau_type);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
getTruthMETObj(ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
               ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
               TString type = "hww_only");

// for checking signal efficiencies in delphes card validation
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> find_reco_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres = 0.1);

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> find_reco_matches_new(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres);

// for checking event-wide matching in HHH->4b2tau
ROOT::VecOps::RVec<edm4hep::MCParticleData> get_H_decay_products(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids);


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_matches_no_remove(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_matches_exclusive(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts_exc,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<int> find_reco_match_indices(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_reco_matched_particle(
    edm4hep::MCParticleData truth_part_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_reco_parts,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<edm4hep::MCParticleData> find_mc_matched_particle(
    edm4hep::ReconstructedParticleData reco_part_to_match,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> check_mc_parts,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<int> find_reco_matched_index(
    edm4hep::MCParticleData truth_part_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_reco_parts,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
find_true_signal_leps_reco_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_leps_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_electrons,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_muons,
    float dR_thres = 0.1);
ROOT::VecOps::RVec<int> find_truth_to_reco_matches_indices(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_leps_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts,
    int pdg_ID, float dR_thres = 0.1);

// Minimal MMC-style result container (non-LFV)
struct MMCResult {
  double massMaxW = -1.0;
  double massWeighted = -1.0;
  TLorentzVector nu1;
  TLorentzVector nu2;
  int nSolutions = 0;
  int status = 0; // 1=found, 0=none
};

// Lightweight non-LFV MMC scan (Markov-like) using tau TLVs and MET with
// longitudinal/transverse resolutions (sigmaL parallel, sigmaP perpendicular)
MMCResult runSimpleMMCNonLFV(const edm4hep::ReconstructedParticleData &tau1,
                             const edm4hep::ReconstructedParticleData &tau2,
                             const TVector2 &met, double sigmaL, double sigmaP,
                             double metCovPhi, int nIter = 50000,
                             double dPhiMax = 0.4);

// retrieving isoVar from delphes:
//  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  get_isoVar(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  reco_parts_to_check, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  all_reco_parts);

// template function for getting vector via indices - needed to read e.g.
// UserDataCollections
template <typename T>
ROOT::VecOps::RVec<T> get(const ROOT::VecOps::RVec<int> &index,
                          const ROOT::VecOps::RVec<T> &in) {
  ROOT::VecOps::RVec<T> result;
  result.reserve(index.size());
  for (size_t i = 0; i < index.size(); ++i) {
    if (index[i] > -1)
      result.push_back(in[index[i]]);
  }
  return result;
}

} // namespace AnalysisFCChh

#endif

#include "FCCAnalyses/Analysis_FCChh.h"
// #include "FCCAnalyses/lester_mt2_bisect.h"

#include <iostream>
#include "TVector2.h"
#include <cmath>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <mutex>
#include <random>
#include <limits>
#include <cstdlib>

// EDM4hep
#include "FCCAnalyses/VertexingUtils.h"
#include "edm4hep/EDM4hepVersion.h"
#if __has_include("edm4hep/utils/bit_utils.h")
#include "edm4hep/utils/bit_utils.h"
#endif
#include <edm4hep/ReconstructedParticleData.h>
#include <edm4hep/TrackState.h>
#include "FCCAnalyses/ReconstructedParticle2Track.h"  // getRP2TRK
#include <ROOT/RVec.hxx>
#include <TLorentzVector.h>
#include <TVector2.h>
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TFile.h"
#include "TH2D.h"
#include "TRandom3.h"

using namespace AnalysisFCChh;

// truth filter helper functions:
bool AnalysisFCChh::isStablePhoton(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 22 && truth_part.generatorStatus == 1) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isPhoton(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 22) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isLep(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 11 || abs(pdg_id) == 13 || abs(pdg_id) == 15) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isLightLep(
    edm4hep::MCParticleData truth_part) // only electrons or muons, no taus
{
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 11 || abs(pdg_id) == 13) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isNeutrino(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 12 || abs(pdg_id) == 14 || abs(pdg_id) == 16) {
    return true;
  } else {
    return false;
  }
}
bool AnalysisFCChh::isTauNeutrino(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 16) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isQuark(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) >= 1 && abs(pdg_id) <= 6) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isZ(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 23) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isW(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 24) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isTau(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 15) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isH(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 25) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isb(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 5) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isHadron(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) >= 100) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isTop(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 6) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isGluon(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 21) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isc(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 4) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::iss(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 3) {
    return true;
  } else {
    return false;
  }
}

bool AnalysisFCChh::isMuon(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 13) {
    return true;
  } else {
    return false;
  }
}

// bbaa quartic analysis
float AnalysisFCChh::collinsSoper(
  TLorentzVector &y1,
  TLorentzVector &y2
) {
  float cs;
  float inv_sqrt2 = 1/std::sqrt(2);

  // prep inputs for CS-function
  float p1_u = (y1.E() + y1.Pz()) * inv_sqrt2;
  float p1_d = (y1.E() - y1.Pz()) * inv_sqrt2;

  float p2_u = (y2.E() + y2.Pz()) * inv_sqrt2;
  float p2_d = (y2.E() - y2.Pz()) * inv_sqrt2;

  float myy = (y1+y2).M();
  float pTyy = (y1+y2).Pt();

  cs = 2*(p1_u * p2_d - p1_d * p2_u) / (myy * std::sqrt(myy*myy + pTyy * pTyy));

  return cs; 
}

// angle of diphoton with beam in di-Higgs rest frame
float AnalysisFCChh::cosThetaStar_yy_in_HH(
    const TLorentzVector& y1,
    const TLorentzVector& y2,
    const TLorentzVector& b1,
    const TLorentzVector& b2
) {

  // construct individual higgs' + di-Higgs TLVs
  const TLorentzVector Hyy = y1 + y2;
  const TLorentzVector Hbb = b1 + b2;
  const TLorentzVector HH  = Hyy + Hbb;

  if (HH.E() <= 0.) return 0.f;
  if (Hyy.Vect().Mag2() == 0.) return 0.f;

  // Boost into HH rest frame
  const TVector3 beta = -HH.BoostVector();

  // Beam 4-vectors (massless proxies along +-z); normalisation irrelevant
  TLorentzVector beam1(0., 0., +1., 1.);
  TLorentzVector beam2(0., 0., -1., 1.);
  beam1.Boost(beta);
  beam2.Boost(beta);

  // Collins–Soper beam axis in HH frame
  TVector3 zCS = beam1.Vect().Unit() - beam2.Vect().Unit();
  if (zCS.Mag2() == 0.) return 0.f;
  zCS = zCS.Unit();

  // Diphoton system in HH frame
  TLorentzVector Hyy_HH = Hyy;
  Hyy_HH.Boost(beta);

  const TVector3 pHat = Hyy_HH.Vect().Unit();
  double cosTheta = pHat.Dot(zCS);

  // numerical safety
  if (cosTheta >  1.0) cosTheta =  1.0;
  if (cosTheta < -1.0) cosTheta = -1.0;

  // pp symmetric: often use |cosθ*|
  return static_cast<float>(std::abs(cosTheta));
}

// min, max dR (y,j)
// min, max dR (y,j) over {y1,y2} × {b1,b2}
ROOT::VecOps::RVec<float> AnalysisFCChh::minmax_dr(
    const TLorentzVector& y1,
    const TLorentzVector& y2,
    const TLorentzVector& b1,
    const TLorentzVector& b2
) {
  float dR_min = std::numeric_limits<float>::infinity();
  float dR_max = 0.f;

  // more aesthetically clean version possible, but this will do:
  const float dR_y1b1 = y1.DeltaR(b1);
  const float dR_y1b2 = y1.DeltaR(b2);
  const float dR_y2b1 = y2.DeltaR(b1);
  const float dR_y2b2 = y2.DeltaR(b2);

  dR_min = std::min(dR_min, dR_y1b1);
  dR_min = std::min(dR_min, dR_y1b2);
  dR_min = std::min(dR_min, dR_y2b1);
  dR_min = std::min(dR_min, dR_y2b2);

  dR_max = std::max(dR_max, dR_y1b1);
  dR_max = std::max(dR_max, dR_y1b2);
  dR_max = std::max(dR_max, dR_y2b1);
  dR_max = std::max(dR_max, dR_y2b2);

  ROOT::VecOps::RVec<float> out(2);
  out[0] = dR_min;
  out[1] = dR_max;
  return out;
}

bool AnalysisFCChh::isElectron(edm4hep::MCParticleData truth_part) {
  auto pdg_id = truth_part.PDG;
  // std::cout << "pdg id of truth part is" << pdg_id << std::endl;
  if (abs(pdg_id) == 11) {
    return true;
  } else {
    return false;
  }
}

constexpr double kTauMass = 1.77686;

double wrapPhi(double phi) {
  while (phi > TMath::Pi())
    phi -= TMath::TwoPi();
  while (phi <= -TMath::Pi())
    phi += TMath::TwoPi();
  return phi;
}

std::vector<std::pair<TLorentzVector, TLorentzVector>>
solveNuSystem(const TLorentzVector &vis1, const TLorentzVector &vis2, double mNu1,
              double mNu2, double metX, double metY, double phi1, double phi2) {
  std::vector<std::pair<TLorentzVector, TLorentzVector>> out;

  const double pTmiss2 = metY * std::cos(phi2) - metX * std::sin(phi2);
  const double dPhi = wrapPhi(phi1 - phi2);
  const int dPhiSign = dPhi >= 0 ? 1 : -1;
  if (pTmiss2 * dPhiSign < 0)
    return out;

  const double pTmiss1 = metY * std::cos(phi1) - metX * std::sin(phi1);
  if (pTmiss1 * (-dPhiSign) < 0)
    return out;

  const double sinDPhi = std::sin(dPhi);
  if (std::abs(sinDPhi) < 1e-6)
    return out;

  const double mVis1 = vis1.M();
  const double mVis2 = vis2.M();

  const double ET2v1 = vis1.E() * vis1.E() - vis1.Pz() * vis1.Pz();
  const double ET2v2 = vis2.E() * vis2.E() - vis2.Pz() * vis2.Pz();
  if (ET2v1 <= 0. || ET2v2 <= 0.)
    return out;

  const double m2noma1 = kTauMass * kTauMass - mNu1 * mNu1 - mVis1 * mVis1;
  const double m2noma2 = kTauMass * kTauMass - mNu2 * mNu2 - mVis2 * mVis2;
  const double pv1proj = vis1.Px() * std::cos(phi1) + vis1.Py() * std::sin(phi1);
  const double pv2proj = vis2.Px() * std::cos(phi2) + vis2.Py() * std::sin(phi2);
  const double pTmiss2CscDPhi = pTmiss2 / sinDPhi;
  const double pTmiss1CscDPhi = pTmiss1 / (-sinDPhi);

  const double pTn1 = pTmiss2CscDPhi;
  const double pTn2 = pTmiss1CscDPhi;
  const double pTn1sq = pTn1 * pTn1;
  const double pTn2sq = pTn2 * pTn2;

  const double discri1 =
      m2noma1 * m2noma1 + 4 * m2noma1 * pTmiss2CscDPhi * pv1proj -
      4 * (ET2v1 * (mNu1 * mNu1 + pTn1sq) - (pTn1sq * pv1proj * pv1proj));
  const double discri2 =
      m2noma2 * m2noma2 + 4 * m2noma2 * pTmiss1CscDPhi * pv2proj -
      4 * (ET2v2 * (mNu2 * mNu2 + pTn2sq) - (pTn2sq * pv2proj * pv2proj));

  if (discri1 < 0 || discri2 < 0)
    return out;

  const double Ev1 = std::sqrt(vis1.Pz() * vis1.Pz() + ET2v1);
  const double Ev2 = std::sqrt(vis2.Pz() * vis2.Pz() + ET2v2);
  const double first1 =
      (m2noma1 * vis1.Pz() + 2 * pTmiss2CscDPhi * pv1proj * vis1.Pz()) / (2 * ET2v1);
  const double second1 = std::sqrt(discri1) * Ev1 / (2 * ET2v1);
  const double first2 =
      (m2noma2 * vis2.Pz() + 2 * pTmiss1CscDPhi * pv2proj * vis2.Pz()) / (2 * ET2v2);
  const double second2 = std::sqrt(discri2) * Ev2 / (2 * ET2v2);

  auto tryAdd = [&](double pn1Z, double pn2Z) {
    const double cond1 = m2noma1 + 2 * pTmiss2CscDPhi * pv1proj + 2 * pn1Z * vis1.Pz();
    const double cond2 = m2noma2 + 2 * pTmiss1CscDPhi * pv2proj + 2 * pn2Z * vis2.Pz();
    if (cond1 <= 0 || cond2 <= 0)
      return;
    TLorentzVector nu1, nu2;
    const double e1 = std::sqrt(pTn1sq + pn1Z * pn1Z + mNu1 * mNu1);
    const double e2 = std::sqrt(pTn2sq + pn2Z * pn2Z + mNu2 * mNu2);
    nu1.SetPxPyPzE(pTn1 * std::cos(phi1), pTn1 * std::sin(phi1), pn1Z, e1);
    nu2.SetPxPyPzE(pTn2 * std::cos(phi2), pTn2 * std::sin(phi2), pn2Z, e2);
    out.emplace_back(nu1, nu2);
  };

  tryAdd(first1 + second1, first2 + second2);
  tryAdd(first1 - second1, first2 + second2);
  tryAdd(first1 + second1, first2 - second2);
  tryAdd(first1 - second1, first2 - second2);

  return out;
}


AnalysisFCChh::MMCResult AnalysisFCChh::runSimpleMMCNonLFV(
    const edm4hep::ReconstructedParticleData &tau1,
    const edm4hep::ReconstructedParticleData &tau2, const TVector2 &met,
    double sigmaL, double sigmaP, double metCovPhi, int nIter, double dPhiMax) {

  MMCResult result;
  const TLorentzVector tlv1 = getTLV_reco(tau1);
  const TLorentzVector tlv2 = getTLV_reco(tau2);

  TRandom3 rng(12345);
  double bestWeight = -1.;
  double weightedMassSum = 0.;
  double weightSum = 0.;

  for (int i = 0; i < nIter; ++i) {
    const double metL = rng.Gaus(0., sigmaL);
    const double metP = rng.Gaus(0., sigmaP);
    const double deltaX = metL * std::cos(metCovPhi) - metP * std::sin(metCovPhi);
    const double deltaY = metL * std::sin(metCovPhi) + metP * std::cos(metCovPhi);
    const double propMetX = met.X() + deltaX;
    const double propMetY = met.Y() + deltaY;

    const double phi1 = rng.Uniform(tlv1.Phi() - dPhiMax, tlv1.Phi() + dPhiMax);
    const double phi2 = rng.Uniform(tlv2.Phi() - dPhiMax, tlv2.Phi() + dPhiMax);

    const auto solutions =
        solveNuSystem(tlv1, tlv2, 0.0, 0.0, propMetX, propMetY, phi1, phi2);
    if (solutions.empty())
      continue;

    const double metWeight =
        std::exp(-0.5 * (metL * metL / (sigmaL * sigmaL) + metP * metP / (sigmaP * sigmaP)));

    for (const auto &sol : solutions) {
      const TLorentzVector tauFull1 = tlv1 + sol.first;
      const TLorentzVector tauFull2 = tlv2 + sol.second;
      const double mass = (tauFull1 + tauFull2).M();

      result.nSolutions++;
      weightSum += metWeight;
      weightedMassSum += metWeight * mass;
      if (metWeight > bestWeight) {
        bestWeight = metWeight;
        result.massMaxW = mass;
        result.nu1 = sol.first;
        result.nu2 = sol.second;
      }
    }
  }

  if (result.nSolutions > 0 && weightSum > 0) {
    result.massWeighted = weightedMassSum / weightSum;
    result.status = 1;
  }

  return result;
}



// helpers
inline bool isBHadronPDG(int pdgabs) {
  return (pdgabs >= 500 && pdgabs < 600) || (pdgabs >= 5000 && pdgabs < 6000);
}

bool hasBHadronDaughter(const edm4hep::MCParticleData& p,
                        const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
                        const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts) {
  for (int i = p.daughters_begin; i < p.daughters_end; ++i) {
    const auto& d = parts.at(daughter_ids.at(i).index);
    if (isBHadronPDG(std::abs(d.PDG))) return true;
  }
  return false;
}


// check if a truth particle came from a hadron decay, needed to veto taus that
// come from b-meson decays in the bbtautau samples
bool AnalysisFCChh::isFromHadron(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually onle 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    //std::cout << "Found parent of the tau as:" << parent.PDG << std::endl;
    if (abs(parent.PDG) >= 100) {
      return true;
    }
  }
  return false;
}



int AnalysisFCChh::tauIsFromWhere(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;
  // return tau parent PDG
  int PDG;
  // loop over all parents (usually onle 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    std::cout << "Found parent of the tau as:" << parent.PDG << std::endl;
    
    PDG = abs(parent.PDG);
    return PDG;
    
  }
  return -999;
}

// check if a truth particle had a Higgs as a parent somewhere up the chain
bool AnalysisFCChh::hasHiggsParent(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually only 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    // std::cout << "Found parent of the tau as:" << parent.PDG << std::endl;
    if (isH(parent)) {
      return true;
    }
    return hasHiggsParent(parent, parent_ids, truth_particles);
  }

  return false;
}


bool hasBHadronParentIdx(
    int mcIndex,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    std::vector<char>& visited   // 0/1 flags
) {
  if (mcIndex < 0 || mcIndex >= (int)truth_particles.size()) return false;

  // break cycles
  if (visited[mcIndex]) return false;
  visited[mcIndex] = 1;

  const auto& truth_part = truth_particles[mcIndex];

  const int first_parent_index = truth_part.parents_begin;
  const int last_parent_index  = truth_part.parents_end;

  for (int parent_i = first_parent_index; parent_i < last_parent_index; ++parent_i) {
    const int parent_MC_index = parent_ids[parent_i].index;
    const auto& parent = truth_particles[parent_MC_index];
    const int pdgabs = std::abs(parent.PDG);

    if (isBHadronPDG(pdgabs)) {
      return true;
    }

    if (hasBHadronParentIdx(parent_MC_index, parent_ids, truth_particles, visited)) {
      return true;
    }
  }

  return false;
}


bool AnalysisFCChh::hasBHadronParent(
    const edm4hep::MCParticleData& truth_part,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles) {

  // Find index of this particle in the truth_particles vector
  const auto* base = &truth_particles[0];
  const int mcIndex = int(&truth_part - base);

  std::vector<char> visited(truth_particles.size(), 0);
  return hasBHadronParentIdx(mcIndex, parent_ids, truth_particles, visited);
}

// check if a truth particle had a Higgs as a parent somewhere up the chain
bool AnalysisFCChh::hasZParent(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually only 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    if (isZ(parent)) {
      std::cout << "Found Z parent of the tau as:" << parent.PDG << std::endl;
      std::cout << "Found PDG parent " << parent.PDG << " with daughter of PDG ID " << truth_part.PDG << std::endl;
      return true;
    }
    return hasZParent(parent, parent_ids, truth_particles);
  }

  return false;
}


// check if the immediate parent of a particle is a Higgs
bool AnalysisFCChh::isFromHiggsDirect(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually only 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    std::cout << "Found parent of the tau as:" << parent.PDG << std::endl;
    if (isH(parent)) {
      return true;
    }
  }

  return false;
}


// Generalized function to find correct and incorrect Higgs pairings for HHH->4b2tau
std::tuple<ROOT::VecOps::RVec<std::pair<int, int>>, ROOT::VecOps::RVec<std::pair<int, int>>>
AnalysisFCChh::getJetPairings(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& reco_jets,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles) {

  ROOT::VecOps::RVec<std::pair<int, int>> true_pairs;
  ROOT::VecOps::RVec<std::pair<int, int>> false_pairs;

  int Njets = reco_jets.size();
  for (int i = 0; i < Njets; i++) {
    for (int j = i + 1; j < Njets; j++) {
      auto jet1_parent_idx = reco_jets[i].parents_begin;
      auto jet2_parent_idx = reco_jets[j].parents_begin;

      auto jet1_parent = truth_particles.at(parent_ids.at(jet1_parent_idx).index);
      auto jet2_parent = truth_particles.at(parent_ids.at(jet2_parent_idx).index);

      bool same_Higgs = (jet1_parent.PDG == 25) &&
                        (jet2_parent.PDG == 25) &&
                        (parent_ids.at(jet1_parent_idx).index == parent_ids.at(jet2_parent_idx).index);

      if (same_Higgs)
        true_pairs.push_back({i, j});
      else
        false_pairs.push_back({i, j});
    }
  }

  return {true_pairs, false_pairs};
}


// check if a particle came from a tau that itself came from a Higgs, and not
// ever from a hadron
bool AnalysisFCChh::isChildOfTauFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {
  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually onle 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    if (isTau(parent)) {
      // veto taus from b-decays
      if (isFromHadron(parent, parent_ids, truth_particles)) {
        return false;
      }
      if (hasHiggsParent(parent, parent_ids, truth_particles)) {
        return true;
      }
    }
  }
  return false;
}

bool AnalysisFCChh::isChildOfTauFromZ(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {
  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually onle 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    if (isTau(parent)) {
      // veto taus from b-decays
      if (isFromHadron(parent, parent_ids, truth_particles)) {
        return false;
      }
      if (hasZParent(parent, parent_ids, truth_particles)) {
        return true;
      }
    }
  }
  return false;
}

bool AnalysisFCChh::isChildOfTauHadFromZ(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  // Loop over parents of this particle
  for (int parent_i = truth_part.parents_begin; parent_i < truth_part.parents_end; parent_i++) {
    auto parent_MC_index = parent_ids.at(parent_i).index;
    const auto &parent = truth_particles.at(parent_MC_index);

    // Check if parent is a tau
    if (!isTau(parent)) continue;

    // Exclude taus from b-hadron decays
    if (isFromHadron(parent, parent_ids, truth_particles)) continue;

    // Require Z as the origin
    if (!hasZParent(parent, parent_ids, truth_particles)) continue;

    // Now require that this tau is hadronic
    bool hasHadronicDaughter = false;
    for (int ch_i = parent.daughters_begin; ch_i < parent.daughters_end; ch_i++) {
      auto daughter_index = daughter_ids.at(ch_i).index;
      auto daughter = truth_particles.at(daughter_index);
      if (isHadron(daughter)) {
        hasHadronicDaughter = true;
        break;
      }
    }

    if (hasHadronicDaughter) {
      return true;
    }
  }

  return false;
}

bool AnalysisFCChh::isChildOfTauLepFromZ(
    const edm4hep::MCParticleData &truth_part,
    const ROOT::VecOps::RVec<podio::ObjectID> &parent_ids,
    const ROOT::VecOps::RVec<podio::ObjectID> &daughter_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &truth_particles) {

  // Only check e or mu
  if (!isLightLep(truth_part)) return false;

  // Loop over parents of this particle
  for (int parent_i = truth_part.parents_begin; parent_i < truth_part.parents_end; ++parent_i) {
    int parent_idx = parent_ids.at(parent_i).index;
    const auto &parent = truth_particles.at(parent_idx);
    std::cout << "start of ischildlep " << std::endl;
    // Must be a tau
    if (!isTau(parent)) continue;
    std::cout << "is a tau " << std::endl;

    // Exclude taus from b-decays
    if (isFromHadron(parent, parent_ids, truth_particles)) continue;
    std::cout << "not from B " << std::endl;

    // Require that the tau has a Z as parent
    if (!hasZParent(parent, parent_ids, truth_particles)) continue;
    std::cout << "have we made it? must be a Z->tautau, tau->nulepnu " << std::endl;


   
    return true;  // Valid light lepton from tau from Z, and tau decayed leptonically
    
  }

  return false;
}

// check if a particle came from a Z that itself came from a Higgs, and not ever
// from a hadron
bool AnalysisFCChh::isChildOfZFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {
  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually onle 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    if (isZ(parent)) {
      // veto taus from b-decays
      if (isFromHadron(parent, parent_ids, truth_particles)) {
        return false;
      }
      if (hasHiggsParent(parent, parent_ids, truth_particles)) {
        return true;
      }
    }
  }
  return false;
}

// check if a particle came from a W that itself came from a Higgs, and not ever
// from a hadron
bool AnalysisFCChh::isChildOfWFromHiggs(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {
  // cannot use the podio getParents fct, need to implement manually:
  auto first_parent_index = truth_part.parents_begin;
  auto last_parent_index = truth_part.parents_end;

  // loop over all parents (usually only 1, but sometimes more for reasons not
  // understood?):
  for (int parent_i = first_parent_index; parent_i < last_parent_index;
       parent_i++) {
    // first get the index from the parent
    auto parent_MC_index = parent_ids.at(parent_i).index;

    // then go back to the original vector of MCParticles
    auto parent = truth_particles.at(parent_MC_index);

    if (isW(parent)) {
      // veto Ws from b-decays
      if (isFromHadron(parent, parent_ids, truth_particles)) {
        return false;
      }
      if (hasHiggsParent(parent, parent_ids, truth_particles)) {
        return true;
      }
    }
  }
  return false;
}

// check what type the Z decay is: to ll or vv
int AnalysisFCChh::checkZDecay(
    edm4hep::MCParticleData truth_Z,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  auto first_child_index = truth_Z.daughters_begin;
  auto last_child_index = truth_Z.daughters_end;

  if (last_child_index - first_child_index != 2) {
    std::cout << "Error in checkZDecay! Found more or fewer than exactly 2 "
                 "daughters of a Z boson - this is not expected by code. Need "
                 "to implement a solution still!"
              << std::endl;
    return 0;
  }

  // now get the indices in the daughters vector
  auto child_1_MC_index = daughter_ids.at(first_child_index).index;
  auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

  // std::cout << "Daughters run from: " << child_1_MC_index << " to " <<
  // child_2_MC_index << std::endl;

  // then go back to the original vector of MCParticles
  auto child_1 = truth_particles.at(child_1_MC_index);
  auto child_2 = truth_particles.at(child_2_MC_index);

  if (isLep(child_1) && isLep(child_2)) {
    return 1;
  } else if (isNeutrino(child_1) && isNeutrino(child_2)) {
    return 2;
  } else {
    std::cout << "Found different decay of Z boson than 2 leptons (e or mu), "
                 "neutrinos or taus! Please check."
              << std::endl;
    return 0;
  }
}

// check what type the W decay is: to lv or qq
int AnalysisFCChh::checkWDecay(
    edm4hep::MCParticleData truth_W,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles) {

  auto first_child_index = truth_W.daughters_begin;
  auto last_child_index = truth_W.daughters_end;
  auto children_size = last_child_index - first_child_index;

  if (children_size < 1) {
    std::cout
        << "Error in checkWDecay! Checking W with no daughters, returning 0."
        << std::endl;
    return 0;
  }

  // have at least 1 child -> if its also a W, continue the chain -> skip the
  // intermediate Ws, and those that radiated photon
  auto child_1_index = daughter_ids.at(first_child_index).index;
  auto child_1 = truth_particles.at(child_1_index);

  // if the child of the W is also a W, use that one
  if (isW(child_1)) {
    return checkWDecay(child_1, daughter_ids, truth_particles);
  }

  if (children_size != 2) {
    std::cout << "Error in checkWDecay! Found unexpected W decay, please check."
              << std::endl;

    return 0;
  }

  // now get the indices in the daughters vector
  //  auto child_1_MC_index = daughter_ids.at(first_child_index).index;
  auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

  // std::cout << "Daughters run from: " << child_1_MC_index << " to " <<
  // child_2_MC_index << std::endl;

  // then go back to the original vector of MCParticles
  //  auto child_1 = truth_particles.at(child_1_MC_index);
  auto child_2 = truth_particles.at(child_2_MC_index);

  // debug
  //  std::cout << "Checking W decay: PDGID 1 = " << child_1.PDG << " and PDGID
  //  2 = " <<  child_2.PDG << std::endl;
  // scheme: is e/mu: 1, tau:2, had: 4
  // scheme: e/μ = 1, τ = 2, hadronic = 4
  if ((isLightLep(child_1) && isNeutrino(child_2)) ||
      (isNeutrino(child_1) && isLightLep(child_2))) {
    return 1; // W → eν or μν
  }
  else if ((isTau(child_1) && isNeutrino(child_2)) ||
          (isNeutrino(child_1) && isTau(child_2))) {
    return 2; // W → τν
  }
  else if (isQuark(child_1) && isQuark(child_2)) {
    return 4; // W → qq′
  }
  else {
    std::cout << "Found different decay of W boson than lv or qq! Please check."
              << std::endl;
    std::cout << "PDGID 1 = " << child_1.PDG << " and PDGID 2 = " << child_2.PDG
              << std::endl;
    return -9;
  }
}


bool hasTopChild(const edm4hep::MCParticleData& p,
                 const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
                 const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts)
{
  for (int off = p.daughters_begin; off < p.daughters_end; ++off) {
    const int idx = daughter_ids.at(off).index;
    if (idx < 0 || idx >= (int)parts.size()) continue;
    if (std::abs(parts.at(idx).PDG) == 6) return true;
  }
  return false;
}

int findFinalTopIndex(int startIdx,
                      const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts,
                      const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids)
{
  auto inRange = [&](int idx){ return idx >= 0 && idx < (int)parts.size(); };

  int idx = startIdx;
  std::set<int> visited;
  int depth = 0;
  const int maxDepth = 20;

  while (true) {
    // protection
    if (visited.count(idx) || depth++ > maxDepth) {
      std::cerr << "Warning: loop or too-deep chain in top decay (idx=" << idx << ")\n";
      break;
    }
    visited.insert(idx);

    bool foundChild = false;
    for (int off = parts[idx].daughters_begin; off < parts[idx].daughters_end; ++off) {
      int cidx = daughter_ids.at(off).index;
      if (!inRange(cidx)) continue;
      if (std::abs(parts[cidx].PDG) == 6 && cidx != idx) {
        idx = cidx;
        foundChild = true;
        break;
      }
    }
    if (!foundChild) break; // reached final top
  }
  return idx;
}


ROOT::VecOps::RVec<int> AnalysisFCChh::topChild(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts,
  const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids)
{
  ROOT::VecOps::RVec<int> pdgs; // store all child PDG codes

  auto inRange = [&](int idx){ return idx >= 0 && idx < (int)parts.size(); };

  for (size_t i = 0; i < parts.size(); ++i) {
    const auto& p = parts[i];
    if (std::abs(p.PDG) != 6) continue; // only top quarks

    std::cout << "Top " << i << " daughters [" 
              << p.daughters_begin << ", " << p.daughters_end << ")\n";

    for (int off = p.daughters_begin; off < p.daughters_end; ++off) {
      int cidx = daughter_ids.at(off).index;
      
      if (!inRange(cidx)) continue;
      const auto& c = parts[cidx];
      
      pdgs.push_back(c.PDG);
      std::cout << "  Child index " << cidx 
                << " PDG " << c.PDG << std::endl;
    }
  }

  return pdgs;
}


// check top decay: use to see fraction of dileptionic ttbar events at pre-sel
int AnalysisFCChh::findTopDecayChannel(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids) {

  // std::cout << "Checking truth ttbar decay type" << std::endl;

  std::vector<edm4hep::MCParticleData> top_list; // do i need this?
  int ttbar_decay_type = 1;

  for (auto &truth_part : truth_particles) {
    if (isTop(truth_part)) {
      // check what children the top has:
      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;

      auto children_size = last_child_index - first_child_index;

      // skip intermediate tops that just have another top as children
      if (last_child_index - first_child_index != 2) {
        continue;
      }

      // get the pdg ids of the children
      // now get the indices in the daughters vector
      auto child_1_MC_index = daughter_ids.at(first_child_index).index;
      auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

      // then go back to the original vector of MCParticles
      auto child_1 = truth_particles.at(child_1_MC_index);
      auto child_2 = truth_particles.at(child_2_MC_index);

      std::cout << "PDG ID of first child: " << child_1.PDG <<  std::endl;
      std::cout << "PDG ID of second child: " << child_2.PDG <<  std::endl;

      // skip the case where a top radiated a gluon
      if (isTop(child_1) || isTop(child_2)) {
        std::cout << "continuing because we've found top in decay chain " << std::endl;
        continue;
      }

      // also need to skip the gluon child
      if (isGluon(child_1) || isGluon(child_2)) {
        std::cout << "continuing because we've found a gluon in decay chain " << std::endl;
        continue;
      }

      int w_decay_type = 0;

      if (isW(child_1) && !isW(child_2)) {
        w_decay_type = checkWDecay(child_1, daughter_ids, truth_particles);
      }

      else if (isW(child_2)) {
        w_decay_type = checkWDecay(child_2, daughter_ids, truth_particles);
      }

      else {
        std::cout << "Warning! Found two or Ws from top decay. Please check. "
                     "Skipping top."
                  << std::endl;
        continue;
      }

      std::cout << "Code for W decay is = " << w_decay_type << std::endl;

      // codes from W decays are: 1 = W->lv and 2 W->qq, 0 for error
      // ttbar code is the code W1*W2 so that its 0 in case of any error
      // ttbar codes: 1=dileptonic, 2=semileptonic, 3=hadronic
      ttbar_decay_type *= w_decay_type;

      top_list.push_back(truth_part);
    }
  }
  std::cout << "Number of tops in event :" << top_list.size()<< std::endl;
  // std::cout << "Code for ttbar decay is = " << ttbar_decay_type << std::endl;
  return ttbar_decay_type;
}



// check Higgs decay: use to see if can improve stats for single Higgs bkg with
// exclusive samples
int AnalysisFCChh::findHiggsDecayChannel(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids) {

  // std::cout << "Checking truth Higgs decay type" << std::endl;

  // std::vector<edm4hep::MCParticleData> higgs_list; // FOR DEBUG

  int higgs_decay_type = 0;

  for (auto &truth_part : truth_particles) {
    if (isH(truth_part)) {
      // check what children the top has:
      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;

      auto children_size = last_child_index - first_child_index;

      // skip intermediate tops that just have another Higgs as children
      if (last_child_index - first_child_index != 2) {
        continue;
      }

      // higgs_list.push_back(truth_part);

      // get the pdg ids of the children
      // now get the indices in the daughters vector
      auto child_1_MC_index = daughter_ids.at(first_child_index).index;
      auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

      // then go back to the original vector of MCParticles
      auto child_1 = truth_particles.at(child_1_MC_index);
      auto child_2 = truth_particles.at(child_2_MC_index);

      // std::cout << "Found Higgs with two children:" << std::endl;
      // std::cout << "PDG ID of first child: " << child_1.PDG <<  std::endl;
      // std::cout << "PDG ID of second child: " << child_2.PDG <<  std::endl;

      // to check
      // //skip the case where a top radiated a gluon
      // if (isTop(child_1) || isTop(child_2)){
      // 	continue;
      // }

      // Higgs decay types:
      //  1: Hbb, 2: HWW, 3: Hgg, 4: Htautau, 5: Hcc, 6:HZZ, 7:Hyy, 8:HZy,
      //  9:Hmumu, 10:Hss

      if (isb(child_1) && isb(child_2)) {
        higgs_decay_type = 1;
      }

      else if (isW(child_1) && isW(child_2)) {
        higgs_decay_type = 2;
      }

      else if (isGluon(child_1) && isGluon(child_2)) {
        higgs_decay_type = 3;
      }

      else if (isTau(child_1) && isTau(child_2)) {
        higgs_decay_type = 4;
      }

      else if (isc(child_1) && isc(child_2)) {
        higgs_decay_type = 5;
      }

      else if (isZ(child_1) && isZ(child_2)) {
        higgs_decay_type = 6;
      }

      else if (isPhoton(child_1) && isPhoton(child_2)) {
        higgs_decay_type = 7;
      }

      else if ((isZ(child_1) && isPhoton(child_2)) ||
               (isZ(child_2) && isPhoton(child_1))) {
        higgs_decay_type = 8;
      }

      else if (isMuon(child_1) && isMuon(child_2)) {
        higgs_decay_type = 9;
      }

      else if (iss(child_1) && iss(child_2)) {
        higgs_decay_type = 10;
      }

      else {
        std::cout << "Warning! Found unkown decay of Higgs!" << std::endl;
        std::cout << "Pdg ids are " << child_1.PDG << " , " << child_2.PDG
                  << std::endl;
        continue;
      }
    }
  }
  // std::cout << "Number of higgs in event :" << higgs_list.size()<< std::endl;
  // std::cout << "Code for higgs decay is = " << higgs_decay_type << std::endl;
  return higgs_decay_type;
}

// truth filter used to get ZZ(llvv) events from the ZZ(llvv+4l+4v) inclusive
// signal samples
bool AnalysisFCChh::ZZllvvFilter(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids) {
  // first scan through the truth particles to find Z bosons
  std::vector<edm4hep::MCParticleData> z_list;
  for (auto &truth_part : truth_particles) {
    if (isZ(truth_part)) {
      z_list.push_back(truth_part);
    }
    // Tau veto:
    if (isTau(truth_part)) {
      return false;
    }
  }

  // check how many Zs are in event and build the flag:
  //  std::cout << "Number of Zs" << z_list.size() << std::endl;
  if (z_list.size() == 2) {
    int z1_decay = checkZDecay(z_list.at(0), daughter_ids, truth_particles);
    int z2_decay = checkZDecay(z_list.at(1), daughter_ids, truth_particles);

    int zz_decay_flag = z1_decay + z2_decay;

    // flags are Z(ll) =1 and Z(vv) =2, so flag for llvv is =3 (4l=2, 4v=4)
    if (zz_decay_flag == 3) {
      return true;
    } else {
      return false;
    }

  } else {
    return false;
  }
}

// truth filter used to get WW(lvlv) events from the inclusive bbWW samples
int AnalysisFCChh::WWlvlvFilter(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {
  // first scan through the truth particles to find Z bosons

  auto isHardProcess = [](const edm4hep::MCParticleData& p){
    const int s = std::abs(p.generatorStatus);   // Pythia8: negative if decayed
    //std::cout << "Generator status " << s;
    return (s >= 21 && s <= 29);                 // hardest subprocess
  };

  std::vector<edm4hep::MCParticleData> w_list;
  for (auto &truth_part : truth_particles) {
    if (isW(truth_part) && isHardProcess(truth_part)) {
      w_list.push_back(truth_part);
    }
    // Tau veto: - actually probably doesnt work as intended, to revise!!
    //  if  (isTau(truth_part)){
    //  return false;
    // }
  }
  //if (w_list.size() != 2) {
  //  std::cout << " found how many Ws? " << w_list.size() << std::endl; 
  //}
  if (w_list.size() == 2) {
    int w1_decay = checkWDecay(w_list.at(0), daughter_ids, truth_particles);
    int w2_decay = checkWDecay(w_list.at(1), daughter_ids, truth_particles);

    int ww_decay_flag = w1_decay + w2_decay;

    // flags are W(lv) =1 and W(qq) =2, so flag for lvlvv is =2 (lvqq=3, 4q=4)
    return ww_decay_flag;

  } else {
    return -9;
  }
}


// find a Z->ll decay on truth level
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthZtautau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &daughter_ids) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> tau_list;

  for (auto &truth_part : truth_particles) {
    if (!isZ(truth_part)) continue;

    std::vector<edm4hep::MCParticleData> daughters;
    for (int i = truth_part.daughters_begin; i < truth_part.daughters_end; ++i) {
      auto &cand = truth_particles[daughter_ids[i].index];
      daughters.push_back(cand);
    }

    if (daughters.size() == 2 &&
        std::abs(daughters[0].PDG) == 15 &&
        std::abs(daughters[1].PDG) == 15) {

      tau_list.push_back(daughters[0]);
      tau_list.push_back(daughters[1]);
    }
  }

  return tau_list;
}
// helper functions for reco particles:
TLorentzVector
AnalysisFCChh::getTLV_reco(edm4hep::ReconstructedParticleData reco_part) {
  TLorentzVector tlv;
  tlv.SetXYZM(reco_part.momentum.x, reco_part.momentum.y, reco_part.momentum.z,
              reco_part.mass);
  return tlv;
}

// build a MET four momentum:
TLorentzVector
AnalysisFCChh::getTLV_MET(edm4hep::ReconstructedParticleData met_object) {
  TLorentzVector tlv;
  float met_pt = sqrt(met_object.momentum.x * met_object.momentum.x +
                      met_object.momentum.y * met_object.momentum.y);
  tlv.SetPxPyPzE(met_object.momentum.x, met_object.momentum.y, 0., met_pt);

  // debug:
  //  std::cout << "Set MET 4-vector with pT = " << tlv.Pt() << " px = " <<
  //  tlv.Px() << " , py = " << tlv.Py() << " , pz = " << tlv.Pz() << " , E = "
  //  << tlv.E()  << " and m = " << tlv.M() << std::endl;

  return tlv;
}

// truth particles
TLorentzVector AnalysisFCChh::getTLV_MC(edm4hep::MCParticleData MC_part) {
  TLorentzVector tlv;
  tlv.SetXYZM(MC_part.momentum.x, MC_part.momentum.y, MC_part.momentum.z,
              MC_part.mass);
  return tlv;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::merge_pairs(ROOT::VecOps::RVec<RecoParticlePair> pairs) {
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> merged_pairs;

  for (auto &pair : pairs) {
    TLorentzVector pair_tlv = pair.merged_TLV();

    // //build a edm4hep reco particle from the  pair:
    edm4hep::ReconstructedParticleData pair_particle;
    pair_particle.momentum.x = pair_tlv.Px();
    pair_particle.momentum.y = pair_tlv.Py();
    pair_particle.momentum.z = pair_tlv.Pz();
    pair_particle.mass = pair_tlv.M();

    merged_pairs.push_back(pair_particle);
  }

  return merged_pairs;
}

// select only the first pair in a vector (and retun as vector with size 1,
// format needed for the rdf)
ROOT::VecOps::RVec<RecoParticlePair>
AnalysisFCChh::get_first_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs) {
  ROOT::VecOps::RVec<RecoParticlePair> first_pair;

  if (pairs.size()) {
    first_pair.push_back(pairs.at(0));
  }

  return first_pair;
}

// split the pair again: return only the first particle or second particle in
// the pairs - needed for getting eg. pT etc of the selected DFOS pair
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_first_from_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> first_particle;

  if (pairs.size()) {
    // sort by pT first:
    pairs.at(0).sort_by_pT();
    first_particle.push_back(pairs.at(0).particle_1);
  }

  return first_particle;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_second_from_pair(
    ROOT::VecOps::RVec<RecoParticlePair> pairs) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> second_particle;

  if (pairs.size()) {
    pairs.at(0).sort_by_pT();
    second_particle.push_back(pairs.at(0).particle_2);
  }

  return second_particle;
}

// count pairs
int AnalysisFCChh::get_n_pairs(ROOT::VecOps::RVec<RecoParticlePair> pairs) {
  return pairs.size();
}

// correct function to get jets with a certain tag (can b-tag, c-tag) - check
// delphes card of sample for which taggers are used
ROOT::VecOps::RVec<bool>
AnalysisFCChh::getJet_tag(ROOT::VecOps::RVec<int> index,
                          ROOT::VecOps::RVec<edm4hep::ParticleIDData> pid,
                          ROOT::VecOps::RVec<float> values, int algoIndex) {
  ROOT::VecOps::RVec<bool> result;
  for (size_t i = 0; i < index.size(); ++i) {
    auto v =
        static_cast<unsigned>(values.at(pid.at(index.at(i)).parameters_begin));

    result.push_back((v & (1u << algoIndex)) == (1u << algoIndex));
  }
  return result;
}

// return the list of c hadrons
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getChadron(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> c_had_list;
  for (auto &truth_part : truth_particles) {
    int num = int(abs(truth_part.PDG));
    if ((num > 400 && num < 500) || (num > 4000 && num < 5000)) {
      c_had_list.push_back(truth_part);
    }
  }
  return c_had_list;
}

// return the list of b hadrons
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getBhadron(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> b_had_list;
  for (auto &truth_part : truth_particles) {
    int num = int(abs(truth_part.PDG));
    if ((num > 500 && num < 600) || (num > 5000 && num < 6000)) {
      //	int npos3, npos4;
      //	for(int i=0; i<3; i++){
      //		npos3 = num%10;
      //		num = num/10;
      //	}
      //        num = int(abs(truth_part.PDG));
      //        for(int i=0; i<4; i++){
      //                npos4 = num%10;
      //                num = num/10;
      //	}
      //	if (npos3==5 || npos4==5){
      
      // check also if from Higgs to count only from Higgs ones
      if ( !hasHiggsParent(truth_part, parent_ids, truth_particles) ){
      	std::cout << "Not from Higgs, PDG? " << num << std::endl;
        continue;
      }

      b_had_list.push_back(truth_part);
    }
  }

  return b_had_list;
}





// returns B hadrons with (a) a Higgs ancestor, (b) no B-hadron daughters,
// and (c) optional pT cut
ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::getBhadron_final_fromH(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
    float ptMin) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> out;
  out.reserve(truth_particles.size()/50); // heuristic

  for (const auto& p : truth_particles) {
    const int pdgabs = std::abs(p.PDG);
    if (!isBHadronPDG(pdgabs)) continue;

    // ancestor must include a Higgs (recursive)
    if (!hasHiggsParent(p, parent_ids, truth_particles)) continue;

    // keep only "final" B hadrons (no B-hadron in daughters)
    if (hasBHadronDaughter(p, daughter_ids, truth_particles)) continue;

    // optional pT threshold
    const float pt = std::hypot(p.momentum.x, p.momentum.y);
    if (pt < ptMin) continue;

    out.emplace_back(p);
  }
  return out;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::getBhadron_leptons_fromH(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids, // not used here, but kept for symmetry
    TString type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> out;
  out.reserve(truth_particles.size()/50); // heuristic
  std::cout << "Searching for a " << type << std::endl;
  for (const auto& p : truth_particles) {
    const int pdgabs = std::abs(p.PDG);

    // 1) must be a "light lepton"
    if (type.Contains("el")) {
      if (!isElectron(p)) continue;
    } else if (type.Contains("mu")) {
      if (!isMuon(p)) continue;
    } else {
      std::cout << "Please configure a type! Defaulting to electrons" << std::endl;
      if (!isElectron(p)) continue;
    }
    
  
    // print the lepton
    std::cout << "Found electron of PDG ID: " << pdgabs << std::endl;

    // 2) ancestor must include a B-hadron 
    if (!hasBHadronParent(p, parent_ids, truth_particles)) continue;

    std::cout << "Found light lepton with B-hadron parent!" << std::endl;

    // 3) ancestor must include a Higgs as well
    if (!hasHiggsParent(p, parent_ids, truth_particles)) continue;
    std::cout << "Found light lepton with B-hadron parent and a Higgs parent!" << std::endl;

    out.emplace_back(p);
  }

  return out;
}

// Function that returns 1 if a HHH event truth-matches to 4b2tau
bool AnalysisFCChh::isHHH_4b2tau(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids)
{
  int n_b = 0;
  int n_tau = 0;

  for (size_t i = 0; i < truth_particles.size(); ++i) {
    const auto& part = truth_particles[i];

    // Only select Higgs bosons
    if (!isH(part)) continue;

    // Now, find the *immediate* daughters of the Higgs
    int first_daughter = part.daughters_begin;
    int last_daughter = part.daughters_end;
    // Safety check
    if (first_daughter < 0 || last_daughter < 0) continue;
    if (first_daughter >= (int)daughter_ids.size() || last_daughter > (int)daughter_ids.size()) continue;
    if (first_daughter >= last_daughter) continue; // no daughters

    for (int j = first_daughter; j < last_daughter; ++j) {
      auto daughter_index = daughter_ids[j].index;
      if (daughter_index >= truth_particles.size()) continue; // safety check

      const auto& daughter = truth_particles[daughter_index];

      // Only count immediate daughters
      if (isb(daughter)) {
          n_b++;
      } else if (isTau(daughter)) {
          n_tau++;
      }
    }
  }

  // Accept if exactly 4 b-quarks and 2 taus from immediate Higgs decays
  return (n_b == 4 && n_tau == 2);
}


// function that returns the list of all *immediate* Higgs decay products in an event
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::get_H_decay_products(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids)
{
  ROOT::VecOps::RVec<edm4hep::MCParticleData> selected_particles;

  for (size_t i = 0; i < truth_particles.size(); ++i) {
      const auto& part = truth_particles[i];

      if (!isH(part)) continue; // Only loop over Higgses

      int first_daughter = part.daughters_begin;
      int last_daughter = part.daughters_end;

      // Safety checks
      if (first_daughter < 0 || last_daughter < 0) continue;
      if (first_daughter >= (int)daughter_ids.size() || last_daughter > (int)daughter_ids.size()) continue;
      if (first_daughter >= last_daughter) continue; // No daughters

      for (int j = first_daughter; j < last_daughter; ++j) {
          int daughter_index = daughter_ids[j].index;
          if (daughter_index < 0 || daughter_index >= (int)truth_particles.size()) continue;

          const auto& daughter = truth_particles[daughter_index];

          // Only select b or tau daughters
          if (isb(daughter) || isTau(daughter)) {
              selected_particles.push_back(daughter);
          }
      }
  }

  return selected_particles;
}

inline double get_sigma_rel_from_map(double pt, double abs_eta)
{
  // Static so the file and hist are loaded only once.
  static bool   initialized = false;
  static TFile* f_maps      = nullptr;
  static TH2D*  h_sigma_rel = nullptr;

  if (!initialized)
  {
    initialized = true;
    // Adjust path/name to whatever you actually use
    f_maps = TFile::Open("/data/atlas/users/dingleyt/FCChh/FCCAnalyses/jet_response_maps.root", "READ");
    if (!f_maps || f_maps->IsZombie()) {
      std::cerr << "[get_sigma_rel_from_map] ERROR: cannot open jet_response_maps.root\n";
      h_sigma_rel = nullptr;
      return 0.0;
    }

    // Combined relative pT resolution map from your Python script
    h_sigma_rel = dynamic_cast<TH2D*>(f_maps->Get("all_sigma_rel"));
    if (!h_sigma_rel) {
      std::cerr << "[get_sigma_rel_from_map] ERROR: histogram all_sigma_rel not found in jet_response_maps.root\n";
    }
  }

  if (!h_sigma_rel) {
    // Fail gracefully: no information, no resolution
    return 0.0;
  }

  int binx = h_sigma_rel->GetXaxis()->FindBin(pt);
  int biny = h_sigma_rel->GetYaxis()->FindBin(abs_eta);

  double sigma_rel = h_sigma_rel->GetBinContent(binx, biny);
  if (!std::isfinite(sigma_rel) || sigma_rel <= 0.0) {
    return 0.0;
  }
  return sigma_rel;
}

// function to compute the perpendicular and parallel components of the MET

ROOT::VecOps::RVec<float> AnalysisFCChh::get_perp_para_metres(
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj
)
{
  ROOT::VecOps::RVec<float> out;
  out.reserve(4);  // now returning 4 components

  if (MET_obj.empty()) {
    out.push_back(0.f); // para
    out.push_back(0.f); // perp
    out.push_back(0.f); // x
    out.push_back(0.f); // y
    return out;
  }

  // MET direction for parallel / perpendicular
  TLorentzVector met_tlv = getTLV_MET(MET_obj[0]);
  const double phi_met = met_tlv.Phi();

  double sum_sigma_par2  = 0.0;
  double sum_sigma_perp2 = 0.0;
  double sum_sigma_x2    = 0.0;
  double sum_sigma_y2    = 0.0;

  for (const auto& j : jets) {
    TLorentzVector j_tlv = getTLV_reco(j);
    const double pt  = j_tlv.Pt();
    const double eta = j_tlv.Eta();
    const double phi = j_tlv.Phi();

    if (pt <= 0.0)
      continue;

    // get relative pT resolution from map
    const double abs_eta = std::fabs(eta);
    const double sigma_rel = get_sigma_rel_from_map(pt, abs_eta);
    if (sigma_rel <= 0.0)
      continue;

    // Absolute pT resolution for this jet
    const double sigma_pt = sigma_rel * pt;

    // ---- parallel / perpendicular to MET ----
    const double dphi = TVector2::Phi_mpi_pi(phi - phi_met);
    const double c_par = std::cos(dphi);
    const double s_perp = std::sin(dphi);

    const double sigma_par  = sigma_pt * c_par;
    const double sigma_perp = sigma_pt * s_perp;

    sum_sigma_par2  += sigma_par  * sigma_par;
    sum_sigma_perp2 += sigma_perp * sigma_perp;

    // ---- x / y in the lab frame ----
    // Use jet φ directly w.r.t. x-axis
    const double c_x = std::cos(phi);
    const double s_y = std::sin(phi);

    const double sigma_px = sigma_pt * c_x;
    const double sigma_py = sigma_pt * s_y;

    sum_sigma_x2 += sigma_px * sigma_px;
    sum_sigma_y2 += sigma_py * sigma_py;
  }

  const double sigma_parallel = std::sqrt(sum_sigma_par2);
  const double sigma_perp     = std::sqrt(sum_sigma_perp2);
  const double sigma_x        = std::sqrt(sum_sigma_x2);
  const double sigma_y        = std::sqrt(sum_sigma_y2);

  out.push_back(static_cast<float>(sigma_parallel)); // [0]
  out.push_back(static_cast<float>(sigma_perp));     // [1]
  out.push_back(static_cast<float>(sigma_x));        // [2]
  out.push_back(static_cast<float>(sigma_y));        // [3]

  return out;
}



// rewrite of functions to get tagged jets to work with updated edm4hep,
// https://github.com/key4hep/EDM4hep/pull/268
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_tagged_jets(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
    ROOT::VecOps::RVec<edm4hep::ParticleIDData> jet_tags,
    ROOT::VecOps::RVec<podio::ObjectID> jet_tags_indices,
    ROOT::VecOps::RVec<float> jet_tags_values, int algoIndex) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets;

  // make sure we have the right collections: every tag should have exactly one
  // jet index
  assert(jet_tags.size() == jet_tags_indices.size());

  for (size_t jet_tags_i = 0; jet_tags_i < jet_tags.size(); ++jet_tags_i) {

    const auto tag = static_cast<unsigned>(
        jet_tags_values[jet_tags[jet_tags_i].parameters_begin]);

    if (tag & (1 << algoIndex)) {
      tagged_jets.push_back(jets[jet_tags_indices[jet_tags_i].index]);
    }
  }
  return tagged_jets;
}

// return the full jets rather than the list of tags
//  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  AnalysisFCChh::get_tagged_jets(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  jets, ROOT::VecOps::RVec<int> index,
//  ROOT::VecOps::RVec<edm4hep::ParticleIDData> pid, ROOT::VecOps::RVec<float>
//  tag_values, int algoIndex){

// 	ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>  tagged_jets;

// 	// std::cout << "running AnalysisFCChh::get_tagged_jets() on jet
// collection of size" << jets.size() << std::endl;

// 	for (size_t iJet = 0; iJet < jets.size(); ++iJet){
// 	// for (auto & jet : jets){
// 		// std::cout << jet.particles_begin << " to " <<
// jet.particles_end << std::endl;
// 		// get the jet particle id index for the jet
// 		const auto jetIDIndex = index[jets[iJet].particles_begin];
// 		// std::cout << "jet index = " << jetIDIndex << std::endl;
// 		const auto jetID = pid[jetIDIndex];
// 		// get the tag value
// 		const auto tag =
// static_cast<unsigned>(tag_values[jetID.parameters_begin]);
// 		// std::cout << "Tag = " << tag << std::endl;
// 		// check if the tag satisfies what we want
//     	if (tag & (1 << algoIndex)) {
//       		tagged_jets.push_back(jets[iJet]);
//     	}
// 	}

// 	return tagged_jets;
// }

// same for tau tags: they are second entry in the
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_tau_jets(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
    ROOT::VecOps::RVec<edm4hep::ParticleIDData> jet_tags,
    ROOT::VecOps::RVec<podio::ObjectID> jet_tags_indices,
    ROOT::VecOps::RVec<float> jet_tags_values, int algoIndex) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets;

  // Ensure each tag corresponds to exactly one jet index
  assert(jet_tags.size() == jet_tags_indices.size());

  for (size_t jet_tags_i = 0; jet_tags_i < jet_tags.size(); ++jet_tags_i) {

    // Retrieve the tag value associated with the current tag
    const auto tag = static_cast<unsigned>(
        jet_tags_values[jet_tags[jet_tags_i].parameters_begin]);

    // Check if the tag satisfies the desired condition
    if (tag & (1 << algoIndex)) {
      tagged_jets.push_back(jets[jet_tags_indices[jet_tags_i].index]);
    }
  }

  return tagged_jets;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_untagged_jets_exclusive(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jets_HF_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jets_HF_tags_indices,
  const ROOT::VecOps::RVec<float>& jets_HF_tag_values,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jets_tau_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jets_tau_tags_indices,
  const ROOT::VecOps::RVec<float>& jets_tau_tag_values,
  int btagIndex,
  int tauIndex)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> untagged_jets;

  // Create lookup tables: jet index -> tag
  std::unordered_map<int, unsigned> btag_map;
  for (size_t i = 0; i < jets_HF_tags.size(); ++i) {
    int jet_index = jets_HF_tags_indices[i].index;
    unsigned tag = static_cast<unsigned>(jets_HF_tag_values[jets_HF_tags[i].parameters_begin]);
    btag_map[jet_index] = tag;
  }

  std::unordered_map<int, unsigned> tautag_map;
  for (size_t i = 0; i < jets_tau_tags.size(); ++i) {
    int jet_index = jets_tau_tags_indices[i].index;
    unsigned tag = static_cast<unsigned>(jets_tau_tag_values[jets_tau_tags[i].parameters_begin]);
    tautag_map[jet_index] = tag;
  }

  // Loop over all jets
  for (size_t i = 0; i < jets.size(); ++i) {
    bool is_btagged = (btag_map.count(i) > 0) && (btag_map[i] & (1 << btagIndex));
    bool is_tautagged = (tautag_map.count(i) > 0) && (tautag_map[i] & (1 << tauIndex));

    if (!is_btagged && !is_tautagged) {
      untagged_jets.push_back(jets[i]);
    }
  }

  return untagged_jets;
}

// Return b-tagged jets that are NOT tau-tagged
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_btagged_not_tau_tagged(
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
) {
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  const size_t N = jets.size();
  if (N == 0) return out;

  // mask for b/tau
  const unsigned bmask   = (btagIndex >= 0) ? (1u << btagIndex) : 0u;
  const unsigned taumask = (tauIndex  >= 0) ? (1u << tauIndex)  : 0u;

  // 1) Tau veto per jet
  std::vector<unsigned char> tau_veto(N, 0);
  for (size_t i = 0; i < jets_tau_tags.size(); ++i) {
    const int j = jets_tau_tags_indices[i].index;
    if (j < 0 || static_cast<size_t>(j) >= N) continue;
    const int p = jets_tau_tags[i].parameters_begin;
    if (p < 0 || static_cast<size_t>(p) >= jets_tau_tag_values.size()) continue;
    const unsigned bits = static_cast<unsigned>(std::lrint(jets_tau_tag_values[p]));
    const bool is_tau = (tauIndex >= 0) ? ((bits & taumask) != 0u)
                                        : (bits != 0u);  // any tau bit
    if (is_tau) tau_veto[j] = 1;
  }

  // 2) Collect b-tagged jets that are not tau-tagged (no duplicates)
  std::vector<unsigned char> added(N, 0);
  out.reserve(N);
  for (size_t i = 0; i < jets_HF_tags.size(); ++i) {
    const int j = jets_HF_tags_indices[i].index;
    if (j < 0 || static_cast<size_t>(j) >= N) continue;
    if (tau_veto[j]) continue;              // tau priority: skip
    const int p = jets_HF_tags[i].parameters_begin;
    if (p < 0 || static_cast<size_t>(p) >= jets_HF_tag_values.size()) continue;
    const unsigned bits = static_cast<unsigned>(std::lrint(jets_HF_tag_values[p]));
    if ((bits & bmask) == 0u) continue;
    if (!added[j]) {                        // avoid pushing same jet twice
      out.push_back(jets[j]);
      added[j] = 1;
    }
  }
  return out;

}

// for isolation
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::select_with_mask(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objs,
                   const ROOT::VecOps::RVec<char>& mask) {
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  out.reserve(objs.size());
  for (size_t i = 0; i < objs.size(); ++i) {
    if (mask[i]) out.push_back(objs[i]);
  }
  return out;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_tau_tagged_not_btagged(
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
) {
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  const size_t N = jets.size();
  if (N == 0) return out;

  // Accumulate bitfields per jet (handles multiple tag rows per jet)
  std::vector<unsigned> bbits(N, 0u), taubits(N, 0u);

  // Build b/HF bits per jet
  for (size_t i = 0; i < jets_HF_tags.size(); ++i) {
    const int j = jets_HF_tags_indices[i].index;
    if (j < 0 || static_cast<size_t>(j) >= N) continue;
    const int p = jets_HF_tags[i].parameters_begin;
    if (p < 0 || static_cast<size_t>(p) >= jets_HF_tag_values.size()) continue;
    const unsigned bits = static_cast<unsigned>(jets_HF_tag_values[p]);
    bbits[j] |= bits;
  }

  // Build tau bits per jet
  for (size_t i = 0; i < jets_tau_tags.size(); ++i) {
    const int j = jets_tau_tags_indices[i].index;
    if (j < 0 || static_cast<size_t>(j) >= N) continue;
    const int p = jets_tau_tags[i].parameters_begin;
    if (p < 0 || static_cast<size_t>(p) >= jets_tau_tag_values.size()) continue;
    const unsigned bits = static_cast<unsigned>(jets_tau_tag_values[p]);
    taubits[j] |= bits;
  }

  // Masks for the requested working points
  const unsigned bmask  = (btagIndex  >= 0) ? (1u << btagIndex)  : 0u;
  const unsigned taumask= (tauIndex   >= 0) ? (1u << tauIndex)   : 0u;

  // Tau-tagged AND NOT b-tagged
  out.reserve(N);
  for (size_t j = 0; j < N; ++j) {
    const bool is_b   = (bbits[j]   & bmask)   != 0u;
    const bool is_tau = (taubits[j] & taumask) != 0u;
    if (is_tau && !is_b) out.push_back(jets[j]);
  }

  return out;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_tau_jets_exclusive(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& jets,
  const ROOT::VecOps::RVec<edm4hep::ParticleIDData>& jet_tags,
  const ROOT::VecOps::RVec<podio::ObjectID>& jet_tags_indices,
  const ROOT::VecOps::RVec<float>& jet_tags_values,
  int tauIndex,
  int btagIndex)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tau_jets;

  assert(jet_tags.size() == jet_tags_indices.size());

  for (size_t i = 0; i < jet_tags.size(); ++i) {
    const auto tag = static_cast<unsigned>(jet_tags_values[jet_tags[i].parameters_begin]);

    bool is_tau = tag & (1 << tauIndex);
    bool is_b = tag & (1 << btagIndex);

    if (is_tau && !is_b) {
      const auto& jet = jets[jet_tags_indices[i].index];
      tau_jets.push_back(jet);
    }
  }

  return tau_jets;
}

// complementary: return the all jets that do not have the requested tag
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_untagged_jets(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets,
    ROOT::VecOps::RVec<int> index,
    ROOT::VecOps::RVec<edm4hep::ParticleIDData> pid,
    ROOT::VecOps::RVec<float> tag_values, int algoIndex) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> untagged_jets;

  // std::cout << "running AnalysisFCChh::get_tagged_jets() on jet collection of
  // size" << jets.size() << std::endl;

  for (size_t iJet = 0; iJet < jets.size(); ++iJet) {
    // for (auto & jet : jets){
    // std::cout << jet.particleIDs_begin << " to " << jet.particleIDs_end <<
    // std::endl; get the jet particle id index for the jet
    const auto jetIDIndex = index[jets[iJet].particles_begin];
    // std::cout << "jet index = " << jetIDIndex << std::endl;
    const auto jetID = pid[jetIDIndex];
    // get the tag value
    const auto tag = static_cast<unsigned>(tag_values[jetID.parameters_begin]);
    // std::cout << "Tag = " << tag << std::endl;
    // check if the tag satisfies what we want
    if (!(tag & (1 << algoIndex))) {
      untagged_jets.push_back(jets[iJet]);
    }
  }

  return untagged_jets;
}

/*
// get untagged jets, neither b nor tau-tagged
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::get_untagged_jets_hhh(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> all_jets,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tagged_jets_2) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> untagged;

  for (const auto& jet : all_jets) {
    bool is_tagged = false;
    for (const auto& tagged : tagged_jets_1) {
      if (jet.core.index == tagged.core.index) {
        is_tagged = true;
        break;
      }
    }
    if (!is_tagged) {
      for (const auto& tagged : tagged_jets_2) {
        if (jet.core.index == tagged.core.index) {
          is_tagged = true;
          break;
        }
      }
    }
    if (!is_tagged) {
      untagged.push_back(jet);
    }
  }
  return untagged;
}
  */
// select objects that are isolated from the other objects with given dR
// threshold
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::sel_isolated(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> sel_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_parts,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> isolated_parts;

  for (auto &sel_part : sel_parts) {
    bool is_isolated = true;
    TLorentzVector sel_part_tlv = getTLV_reco(sel_part);
    // std::cout << "TLV found with pT() =" << sel_part_tlv.Pt() << std::endl;

    // loop over all particles to check against and see if any are within the dR
    // threshold
    for (auto &check_part : check_parts) {
      TLorentzVector check_part_tlv = getTLV_reco(check_part);
      float dR_val = sel_part_tlv.DeltaR(check_part_tlv);

      if (dR_val <= dR_thres) {
        is_isolated = false;
      }

      check_part_tlv.Clear();
    }

    sel_part_tlv.Clear();

    if (is_isolated) {
      isolated_parts.push_back(sel_part);
    }
  }

  return isolated_parts;
}

// Keep particles with isoVar >= thr (i.e. "non-isolated" by isoVar)
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::sel_by_iso_fail(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& parts,
                               const ROOT::VecOps::RVec<float>& isoVar,
                               float thr)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  const auto n = parts.size();
  if (isoVar.size() != n) return out; // size guard
  out.reserve(n);
  for (size_t i = 0; i < n; ++i)
    if (isoVar[i] >= thr) out.push_back(parts[i]);
  return out;
}


ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::sel_non_isolated(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> sel_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_parts,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> non_isolated_parts;

  for (auto &sel_part : sel_parts) {
    bool is_isolated = true;
    TLorentzVector sel_part_tlv = getTLV_reco(sel_part);
    // std::cout << "TLV found with pT() =" << sel_part_tlv.Pt() << std::endl;

    // loop over all particles to check against and see if any are within the dR
    // threshold
    for (auto &check_part : check_parts) {
      TLorentzVector check_part_tlv = getTLV_reco(check_part);
      float dR_val = sel_part_tlv.DeltaR(check_part_tlv);

      if (dR_val <= dR_thres) {
        is_isolated = false;
      }

      check_part_tlv.Clear();
    }

    sel_part_tlv.Clear();

    if (!is_isolated) {
      non_isolated_parts.push_back(sel_part);
    }
  }

  return non_isolated_parts;
}
// find the lepton pair that likely originates from a Z decay:
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getOSPairs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> leptons_in) {

  ROOT::VecOps::RVec<RecoParticlePair> OS_pairs;

  // need at least 2 leptons in the input
  if (leptons_in.size() < 2) {
    return OS_pairs;
  }

  // separate the leptons by charges
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> leptons_pos;
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> leptons_neg;

  for (auto &lep : leptons_in) {
    auto charge = lep.charge;
    if (charge > 0) {
      leptons_pos.push_back(lep);
    } else if (charge < 0) {
      leptons_neg.push_back(lep);
    }

    else {
      std::cout << "Error in function  AnalysisFCChh::getOSPair() - found "
                   "neutral particle! Function is supposed to be used for "
                   "electrons or muons only."
                << std::endl;
    }
  }

  // std::cout << "Found leptons: " << leptons_pos.size() << " pos, " <<
  // leptons_neg.size() << " neg." << std::endl;

  // check charges: if don't have one of each, cannot build OS pair and if is
  // only one of each there is no ambiguity
  if (leptons_pos.size() < 1 || leptons_neg.size() < 1) {
    return OS_pairs;
  }

  // std::cout << "Have enough leptons to build pair!" << std::endl;

  // build all possible pairs
  // TLorentzVector OS_pair_tlv;

  for (auto &lep_pos : leptons_pos) {
    // sum up the momentum components to get the TLV of the OS pair: first the
    // positive one
    //  TLorentzVector lep_pos_tlv = getTLV_reco(lep_pos);

    for (auto &lep_neg : leptons_neg) {
      // TLorentzVector lep_neg_tlv = getTLV_reco(lep_neg);
      // TLorentzVector OS_pair_tlv = lep_pos_tlv+lep_neg_tlv;

      // //build a edm4hep rco particle from the os pair:
      // edm4hep::ReconstructedParticleData OS_pair;
      // OS_pair.momentum.x = OS_pair_tlv.Px();
      // OS_pair.momentum.y = OS_pair_tlv.Py();
      // OS_pair.momentum.z = OS_pair_tlv.Pz();
      // OS_pair.mass = OS_pair_tlv.M();

      // new code: do not merge the pair but store them separately
      RecoParticlePair OS_pair;
      OS_pair.particle_1 = lep_pos;
      OS_pair.particle_2 = lep_neg;

      OS_pairs.push_back(OS_pair);
    }
  }

  // FOR DEBUG?
  //  if (OS_pairs.size() > 1){
  //  	std::cout << "Number of possible OS pairs: " << OS_pairs.size() <<
  //  std::endl; 	std::cout << "Build from: " << leptons_pos.size() << "
  //  pos, "
  //  << leptons_neg.size() << " neg." << std::endl;
  //  }

  return OS_pairs;
}

// pick the pair that is closest to Z mass:
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getBestOSPair(
    ROOT::VecOps::RVec<RecoParticlePair> electron_pairs,
    ROOT::VecOps::RVec<RecoParticlePair> muon_pairs) {

  ROOT::VecOps::RVec<RecoParticlePair> best_pair;

  // std::cout << "N_elec_pairs = " << electron_pairs.size() << ", N_muon_pairs
  // = " << muon_pairs.size() << std::endl;

  // check if any pairs in input:
  if (electron_pairs.size() == 0 && muon_pairs.size() == 0) {
    return best_pair;
  }

  // if only one pair in input, return that one:
  else if (electron_pairs.size() == 1 && muon_pairs.size() == 0) {
    best_pair.push_back(electron_pairs.at(0));
    return best_pair;
  }

  else if (electron_pairs.size() == 0 && muon_pairs.size() == 1) {
    best_pair.push_back(muon_pairs.at(0));
    return best_pair;
  }

  // if there are mor options, pick the one that is closest to Z mass

  const double Z_mass = 91.1876;

  // make a vector with both electron and muons pairs in it:
  ROOT::VecOps::RVec<RecoParticlePair> all_pairs;
  for (auto &elec_pair : electron_pairs) {
    all_pairs.push_back(elec_pair);
  }
  for (auto &muon_pair : muon_pairs) {
    all_pairs.push_back(muon_pair);
  }

  // from Clement's main code: use std::sort on the mass difference
  auto resonancesort = [&](RecoParticlePair i, RecoParticlePair j) {
    return (abs(Z_mass - i.merged_TLV().M()) <
            abs(Z_mass - j.merged_TLV().M()));
  };
  // auto resonancesort = [&] (edm4hep::ReconstructedParticleData i
  // ,edm4hep::ReconstructedParticleData j) { return (abs( Z_mass
  // -i.mass)<abs(Z_mass-j.mass)); };
  std::sort(all_pairs.begin(), all_pairs.end(), resonancesort);

  // first one should be the closest one
  best_pair.push_back(all_pairs.at(0));

  return best_pair;
}

// for the bbWW SF analysis: pick the pair that is leading in pTll
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getLeadingPair(
    ROOT::VecOps::RVec<RecoParticlePair> electron_pairs,
    ROOT::VecOps::RVec<RecoParticlePair> muon_pairs) {

  ROOT::VecOps::RVec<RecoParticlePair> best_pair;

  // std::cout << "N_elec_pairs = " << electron_pairs.size() << ", N_muon_pairs
  // = " << muon_pairs.size() << std::endl;

  // check if any pairs in input:
  if (electron_pairs.size() == 0 && muon_pairs.size() == 0) {
    return best_pair;
  }

  // if only one pair in input, return that one:
  else if (electron_pairs.size() == 1 && muon_pairs.size() == 0) {
    best_pair.push_back(electron_pairs.at(0));
    return best_pair;
  }

  else if (electron_pairs.size() == 0 && muon_pairs.size() == 1) {
    best_pair.push_back(muon_pairs.at(0));
    return best_pair;
  }

  // have at least one of each
  // make a vector with both electron and muons pairs in it:
  ROOT::VecOps::RVec<RecoParticlePair> all_pairs;
  for (auto &elec_pair : electron_pairs) {
    all_pairs.push_back(elec_pair);
  }
  for (auto &muon_pair : muon_pairs) {
    all_pairs.push_back(muon_pair);
  }

  // take the combined pT to sort
  auto pTll_sort = [&](RecoParticlePair i, RecoParticlePair j) {
    return (abs(i.merged_TLV().Pt()) > abs(j.merged_TLV().Pt()));
  };
  std::sort(all_pairs.begin(), all_pairs.end(), pTll_sort);

  best_pair.push_back(all_pairs.at(0));

  return best_pair;
}

// build all possible emu OS combinations, for eg tautau and ww analysis
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getDFOSPairs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> electrons_in,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> muons_in) {

  ROOT::VecOps::RVec<RecoParticlePair> DFOS_pairs;

  // need at least 2 leptons in the input
  if (electrons_in.size() < 1 || muons_in.size() < 1) {
    return DFOS_pairs;
  }

  // sort the vectors by size, so that the first pair is always the leading
  auto sort_by_pT = [&](edm4hep::ReconstructedParticleData part_i,
                        edm4hep::ReconstructedParticleData part_j) {
    return (getTLV_reco(part_i).Pt() > getTLV_reco(part_j).Pt());
  };
  std::sort(electrons_in.begin(), electrons_in.end(), sort_by_pT);
  std::sort(muons_in.begin(), muons_in.end(), sort_by_pT);

  // loop over the electrons and make a pair if a muons with opposite charge is
  // found

  for (auto &elec : electrons_in) {
    for (auto &muon : muons_in) {
      auto total_charge = elec.charge + muon.charge;
      if (total_charge == 0) {
        // std::cout << "found DFOS pair!" << std::endl;
        RecoParticlePair DFOS_pair;
        DFOS_pair.particle_1 = elec;
        DFOS_pair.particle_2 = muon;

        DFOS_pairs.push_back(DFOS_pair);
      }
    }
  }

  // debug
  //  if (DFOS_pairs.size() > 1){
  //  	std::cout << "Number of possible DFOS pairs: " << DFOS_pairs.size() <<
  //  std::endl; 	std::cout << "Build from: " << electrons_in.size() << "
  //  electrons, " << muons_in.size() << " muons" << std::endl;
  //  }

  return DFOS_pairs;
}

// SortParticleCollection
//
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::SortParticleCollection(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in) {
  if (particles_in.size() < 2) {
    return particles_in;
  } else {
    auto sort_by_pT = [&](edm4hep::ReconstructedParticleData part_i,
                          edm4hep::ReconstructedParticleData part_j) {
      return (getTLV_reco(part_i).Pt() > getTLV_reco(part_j).Pt());
    };
    std::sort(particles_in.begin(), particles_in.end(), sort_by_pT);
    return particles_in;
  }
}


#include "TRandom3.h" // or TRandom.h, depending on your setup

AnalysisFCChh::HiggsDoubletResult
AnalysisFCChh::getHiggsCandidateDoubletRandomBaseline(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets)
{
  HiggsDoubletResult out;
  out.pairs.clear();
  out.used_bjet_indices.clear();

  // Need at least 4 b-jets
  if (bjets.size() < 4) {
    return out; // empty result
  }

  // We will only use the first 4 b-jets as in your main algorithm
  // Indices: 0,1,2,3

  // Three unique pairings of {0,1,2,3} into 2 disjoint pairs:
  //  (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
  const int pairings[3][4] = {
    {0, 1, 2, 3},
    {0, 2, 1, 3},
    {0, 3, 1, 2}
  };

  // Choose one of the three pairings uniformly at random
  // gRandom->Integer(3) returns 0,1,2 with equal probability
  const int p = gRandom->Integer(3);

  const int i = pairings[p][0];
  const int j = pairings[p][1];
  const int k = pairings[p][2];
  const int l = pairings[p][3];

  RecoParticlePair H1 = {bjets[i], bjets[j]};
  RecoParticlePair H2 = {bjets[k], bjets[l]};

  // Build TLVs to get masses
  TLorentzVector h1 = getTLV_reco(H1.particle_1) + getTLV_reco(H1.particle_2);
  TLorentzVector h2 = getTLV_reco(H2.particle_1) + getTLV_reco(H2.particle_2);

  const float m1 = h1.M();
  const float m2 = h2.M();
  const float pt1 = h1.Pt();
  const float pt2 = h2.Pt();

  // Mass-order the two Higgs candidates so that:
  //   out.pairs[0] = leading in mass
  //   out.pairs[1] = subleading in mass
  std::pair<int,int> idxH1{i, j};
  std::pair<int,int> idxH2{k, l};

  if (pt1 >= pt2) {
    out.pairs = {H1, H2};
  } else {
    out.pairs = {H2, H1};
    std::swap(idxH1, idxH2);
  }

  // Record which b-jets were used (global indices into the original bjet RVec)
  out.used_bjet_indices = {
    idxH1.first, idxH1.second,
    idxH2.first, idxH2.second
  };

  return out;
}


AnalysisFCChh::HiggsDoubletResult
AnalysisFCChh::getHiggsCandidateDoubletMassOrdered(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  TString strategy,
  float target_mass1,
  float target_mass2)
{
  HiggsDoubletResult out;
  out.pairs.clear();
  out.used_bjet_indices.clear();

  RecoParticlePair bestH1, bestH2;
  std::pair<int,int> bestIdxH1{-1,-1};
  std::pair<int,int> bestIdxH2{-1,-1};
  float bestScore = std::numeric_limits<float>::max();

  if (bjets.size() < 4) {
    // Not enough b-jets: return empty result
    return out;
  }

  // Symmetrised mass-score: we take the best assignment of (m1,m2) to (target_mass1, target_mass2)
  auto score_mass = [&](float m1, float m2)->float {
    const float s1_lin  = std::fabs(m1 - target_mass1) + std::fabs(m2 - target_mass2);
    const float s2_lin  = std::fabs(m1 - target_mass2) + std::fabs(m2 - target_mass1);
    const float s1_sq   = std::pow(m1 - target_mass1, 2.f) + std::pow(m2 - target_mass2, 2.f);
    const float s2_sq   = std::pow(m1 - target_mass2, 2.f) + std::pow(m2 - target_mass1, 2.f);

    if (strategy == "linearmass")  return std::min(s1_lin, s2_lin);
    if (strategy == "squaremass")  return std::min(s1_sq,  s2_sq);
    return std::numeric_limits<float>::infinity();
  };

  auto score_dRmax = [&](const TLorentzVector& a1, const TLorentzVector& a2,
                         const TLorentzVector& b1, const TLorentzVector& b2)->float {
    return std::max(a1.DeltaR(a2), b1.DeltaR(b2));
  };

  // Three unique pairings of 4 objects {0,1,2,3} into 2 disjoint pairs:
  //  (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
  const int pairings[3][4] = {
    {0, 1, 2, 3},
    {0, 2, 1, 3},
    {0, 3, 1, 2}
  };

  for (int p = 0; p < 3; ++p) {
    const int i = pairings[p][0];
    const int j = pairings[p][1];
    const int k = pairings[p][2];
    const int l = pairings[p][3];

    TLorentzVector tlv_i = getTLV_reco(bjets[i]);
    TLorentzVector tlv_j = getTLV_reco(bjets[j]);
    TLorentzVector tlv_k = getTLV_reco(bjets[k]);
    TLorentzVector tlv_l = getTLV_reco(bjets[l]);

    const TLorentzVector h1 = tlv_i + tlv_j;
    const TLorentzVector h2 = tlv_k + tlv_l;

    const float m_h1 = h1.M();
    const float m_h2 = h2.M();

    float score = 0.f;
    if (strategy == "dRminmax") {
      score = score_dRmax(tlv_i, tlv_j, tlv_k, tlv_l);
    } else {
      score = score_mass(m_h1, m_h2);
    }

    if (score < bestScore) {
      bestScore = score;
      bestH1 = {bjets[i], bjets[j]};
      bestH2 = {bjets[k], bjets[l]};
      bestIdxH1 = {i, j};
      bestIdxH2 = {k, l};
    }
  }

  if (bestScore < std::numeric_limits<float>::max()) {
    // Now order the two Higgs candidates by MASS, not pT
    TLorentzVector h1 = getTLV_reco(bestH1.particle_1) + getTLV_reco(bestH1.particle_2);
    TLorentzVector h2 = getTLV_reco(bestH2.particle_1) + getTLV_reco(bestH2.particle_2);

    const float m1 = h1.M();
    const float m2 = h2.M();

    if (m1 >= m2) {
      // h1 is leading in mass
      out.pairs = {bestH1, bestH2};
    } else {
      // h2 is leading in mass
      out.pairs = {bestH2, bestH1};
      std::swap(bestIdxH1, bestIdxH2);
    }

    out.used_bjet_indices = {
      bestIdxH1.first, bestIdxH1.second,
      bestIdxH2.first, bestIdxH2.second
    };
  }

  return out;
}

AnalysisFCChh::HiggsDoubletResult AnalysisFCChh::getHiggsCandidateDoublet(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untagged_jets,
  TString strategy,
  float target_mass1,
  float target_mass2)
{
  HiggsDoubletResult out;
  out.pairs.clear();
  out.used_bjet_indices.clear();

  RecoParticlePair bestH1, bestH2;
  std::pair<int,int> bestIdxH1{-1,-1};
  std::pair<int,int> bestIdxH2{-1,-1};
  float bestScore = std::numeric_limits<float>::max();

  auto score_mass   = [&](float m1, float m2)->float {
    if (strategy == "linearmass")
      return std::fabs(m1 - target_mass1) + std::fabs(m2 - target_mass2);
    if (strategy == "squaremass")
      return std::pow(m1 - target_mass1, 2.f) + std::pow(m2 - target_mass2, 2.f);
    return std::numeric_limits<float>::infinity();
  };
  auto score_dRmax = [&](const TLorentzVector& a1, const TLorentzVector& a2,
                         const TLorentzVector& b1, const TLorentzVector& b2)->float {
    return std::max(a1.DeltaR(a2), b1.DeltaR(b2));
  };

  // --------------------------
  // Case 1: Use 4 leading b-jets
  // --------------------------
  if (bjets.size() >= 4) {
    for (size_t i = 0; i < 4; ++i) {
      TLorentzVector tlv_i = getTLV_reco(bjets[i]);
      for (size_t j = i + 1; j < 4; ++j) {
        TLorentzVector tlv_j = getTLV_reco(bjets[j]);
        const float m_h1 = (tlv_i + tlv_j).M();

        for (size_t k = 0; k < 4; ++k) {
          if (k == i || k == j) continue;
          TLorentzVector tlv_k = getTLV_reco(bjets[k]);

          for (size_t l = k + 1; l < 4; ++l) {
            if (l == i || l == j) continue;
            TLorentzVector tlv_l = getTLV_reco(bjets[l]);
            const float m_h2 = (tlv_k + tlv_l).M();

            float score = 0.f;
            if (strategy == "dRminmax") {
              score = score_dRmax(tlv_i, tlv_j, tlv_k, tlv_l);
            } else {
              score = score_mass(m_h1, m_h2);
            }

            if (score < bestScore) {
              bestScore = score;
              bestH1 = {bjets[i], bjets[j]};
              bestH2 = {bjets[k], bjets[l]};
              bestIdxH1 = {static_cast<int>(i), static_cast<int>(j)};
              bestIdxH2 = {static_cast<int>(k), static_cast<int>(l)};
            }
          }
        }
      }
    }

    if (bestScore < std::numeric_limits<float>::max()) {
      TLorentzVector h1 = getTLV_reco(bestH1.particle_1) + getTLV_reco(bestH1.particle_2);
      TLorentzVector h2 = getTLV_reco(bestH2.particle_1) + getTLV_reco(bestH2.particle_2);

      // order by pT (first = leading)
      if (h1.Pt() >= h2.Pt()) {
        out.pairs = {bestH1, bestH2};
      } else {
        out.pairs = {bestH2, bestH1};
        std::swap(bestIdxH1, bestIdxH2);
      }
      out.used_bjet_indices = {bestIdxH1.first, bestIdxH1.second, bestIdxH2.first, bestIdxH2.second};
      return out;
    }
  }

  // --------------------------
  // Case 2: 3 b-jets — boosted Higgs among them?
  // --------------------------
  if (bjets.size() == 3) {
    for (size_t i = 0; i < 3; ++i) {
      TLorentzVector fat = getTLV_reco(bjets[i]);
      if (fat.M() > 100.0) {
        const size_t j1 = (i + 1) % 3;
        const size_t j2 = (i + 2) % 3;

        RecoParticlePair h_res   = {bjets[j1], bjets[j2]};
        RecoParticlePair h_boost = {bjets[i],  bjets[i] }; // same jet twice to encode "fat" Higgs

        TLorentzVector p_res = getTLV_reco(h_res.particle_1) + getTLV_reco(h_res.particle_2);

        if (p_res.Pt() >= fat.Pt()) out.pairs = {h_res, h_boost};
        else                        out.pairs = {h_boost, h_res};

        out.used_bjet_indices = {static_cast<int>(i), static_cast<int>(j1), static_cast<int>(j2)};
        return out;
      }
    }
  }

  // --------------------------
  // Case 3: 3 b-jets + 1 untagged jet 
  // --------------------------
  if (bjets.size() == 3 && !untagged_jets.empty()) {
    for (size_t u = 0; u < untagged_jets.size(); ++u) {
      ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> combo = {
        bjets[0], bjets[1], bjets[2], untagged_jets[u]
      };

      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
          TLorentzVector tlv_i = getTLV_reco(combo[i]);
          TLorentzVector tlv_j = getTLV_reco(combo[j]);
          const float m_h1 = (tlv_i + tlv_j).M();

          for (size_t k = 0; k < 4; ++k) {
            if (k == i || k == j) continue;

            for (size_t l = k + 1; l < 4; ++l) {
              if (l == i || l == j) continue;

              TLorentzVector tlv_k = getTLV_reco(combo[k]);
              TLorentzVector tlv_l = getTLV_reco(combo[l]);
              const float m_h2 = (tlv_k + tlv_l).M();

              float score = 0.f;
              if (strategy == "dRminmax") {
                score = score_dRmax(tlv_i, tlv_j, tlv_k, tlv_l);
              } else {
                score = score_mass(m_h1, m_h2);
              }

              if (score < bestScore) {
                bestScore = score;
                bestH1 = {combo[i], combo[j]};
                bestH2 = {combo[k], combo[l]};
                bestIdxH1 = {static_cast<int>(i), static_cast<int>(j)};
                bestIdxH2 = {static_cast<int>(k), static_cast<int>(l)};
              }
            }
          }
        }
      }
    }
  }

  // 
  if (bestScore < std::numeric_limits<float>::max()) {
    TLorentzVector h1 = getTLV_reco(bestH1.particle_1) + getTLV_reco(bestH1.particle_2);
    TLorentzVector h2 = getTLV_reco(bestH2.particle_1) + getTLV_reco(bestH2.particle_2);

    if (h1.Pt() >= h2.Pt()) {
      out.pairs = {bestH1, bestH2};
    } else {
      out.pairs = {bestH2, bestH1};
      std::swap(bestIdxH1, bestIdxH2);
    }

    auto push_if_b = [&](int idxLocal){
      if (0 <= idxLocal && idxLocal <= 2) out.used_bjet_indices.push_back(idxLocal);
    };
    push_if_b(bestIdxH1.first);
    push_if_b(bestIdxH1.second);
    push_if_b(bestIdxH2.first);
    push_if_b(bestIdxH2.second);

    return out;
  }

  // Nothing found: return empty
  return out;
}


// bb-matching function
bool AnalysisFCChh::matchRecoToTruth(const RecoParticlePair& recoPair,
                                     const std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>& truthPair,
                                     float dR_threshold) 
{
  TLorentzVector t1 = getTLV_MC(truthPair.first);
  TLorentzVector t2 = getTLV_MC(truthPair.second);

  TLorentzVector r1 = getTLV_reco(recoPair.particle_1);
  TLorentzVector r2 = getTLV_reco(recoPair.particle_2);

  bool match1 = (r1.DeltaR(t1) < dR_threshold || r1.DeltaR(t2) < dR_threshold);
  bool match2 = (r2.DeltaR(t1) < dR_threshold || r2.DeltaR(t2) < dR_threshold);

  return match1 && match2;
}

// build all pairs from the input particles -> this returns the pair made of pT
// leading particles!!!
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getPairs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in) {

  ROOT::VecOps::RVec<RecoParticlePair> pairs;

  // need at least 2 particles in the input
  if (particles_in.size() < 2) {
    return pairs;
  }

  // else sort them by pT, and take the only the leading pair
  else {
    auto sort_by_pT = [&](edm4hep::ReconstructedParticleData part_i,
                          edm4hep::ReconstructedParticleData part_j) {
      return (getTLV_reco(part_i).Pt() > getTLV_reco(part_j).Pt());
    };
    std::sort(particles_in.begin(), particles_in.end(), sort_by_pT);

    // old method
    //  TLorentzVector tlv_1 = getTLV_reco(particles_in.at(0));
    //  TLorentzVector tlv_2 = getTLV_reco(particles_in.at(1));

    // TLorentzVector tlv_pair = tlv_1+tlv_2;

    // edm4hep::ReconstructedParticleData pair;
    // pair.momentum.x = tlv_pair.Px();
    // pair.momentum.y = tlv_pair.Py();
    // pair.momentum.z = tlv_pair.Pz();
    // pair.mass = tlv_pair.M();

    // new method, dont merge the pair
    RecoParticlePair pair;
    pair.particle_1 = particles_in.at(0);
    pair.particle_2 = particles_in.at(1);

    pairs.push_back(pair);
  }

  return pairs;
}

// same for MC particle
ROOT::VecOps::RVec<MCParticlePair> AnalysisFCChh::getPairs(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> particles_in) {

  ROOT::VecOps::RVec<MCParticlePair> pairs;

  // need at least 2 particles in the input
  if (particles_in.size() < 2) {
    return pairs;
  }

  // else sort them by pT, and take the only the leading pair
  else {
    auto sort_by_pT = [&](edm4hep::MCParticleData part_i,
                          edm4hep::MCParticleData part_j) {
      return (getTLV_MC(part_i).Pt() > getTLV_MC(part_j).Pt());
    };
    std::sort(particles_in.begin(), particles_in.end(), sort_by_pT);

    // new method, dont merge the pair
    MCParticlePair pair;
    pair.particle_1 = particles_in.at(0);
    pair.particle_2 = particles_in.at(1);

    pairs.push_back(pair);
  }

  return pairs;
}

// make the subleading pair, ie. from particles 3 and 4 in pT order
ROOT::VecOps::RVec<RecoParticlePair> AnalysisFCChh::getPair_sublead(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particles_in) {

  ROOT::VecOps::RVec<RecoParticlePair> pairs;

  // need at least 2 particles in the input
  if (particles_in.size() < 4) {
    return pairs;
  }

  // else sort them by pT, and take the only the subleading pair
  else {
    auto sort_by_pT = [&](edm4hep::ReconstructedParticleData part_i,
                          edm4hep::ReconstructedParticleData part_j) {
      return (getTLV_reco(part_i).Pt() > getTLV_reco(part_j).Pt());
    };
    std::sort(particles_in.begin(), particles_in.end(), sort_by_pT);

    // new method, dont merge the pair
    RecoParticlePair pair;
    pair.particle_1 = particles_in.at(2);
    pair.particle_2 = particles_in.at(3);

    pairs.push_back(pair);
  }

  return pairs;
}


#include <TLorentzVector.h>
#include <TDatabasePDG.h>
#include <iostream>
#include <stack>
#include <limits>


// Collect stable daughters recursively
void collect_final_daughters(int idx,
                             const ROOT::VecOps::RVec<podio::ObjectID>& daughters,
                             const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts,
                             std::vector<int>& out) {
  const auto& p = parts[idx];
  if (p.daughters_end <= p.daughters_begin) {
    // no daughters -> stable final state
    out.push_back(idx);
    return;
  }
  for (int i = p.daughters_begin; i < p.daughters_end; ++i) {
    int di = daughters.at(i).index;
    if (di >= 0 && (size_t)di < parts.size()) {
      collect_final_daughters(di, daughters, parts, out);
    }
  }
}




ROOT::VecOps::RVec<std::vector<float>> AnalysisFCChh::classify_taus(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
  ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
  ROOT::VecOps::RVec<podio::ObjectID> parent_ids
) {
  ROOT::VecOps::RVec<std::vector<float>> tau_stuff;
  tau_stuff.reserve(truth_particles.size());

  for (size_t itau = 0; itau < truth_particles.size(); ++itau) {
    const auto& tau = truth_particles[itau];

    if (std::abs(tau.PDG) != 15) continue;
    if (!hasZParent(tau, parent_ids, truth_particles) && (!hasHiggsParent(tau, parent_ids, truth_particles))) continue;

    std::cout << "\n==== Tau idx " << itau << " (pdg=" << tau.PDG << ") ====\n";
    //std::cout << "\n==== Tau pT " << tau.Pt() << ") ====\n";

    TLorentzVector p_vis(0,0,0,0);
    TLorentzVector p_invis(0,0,0,0);

    int n_neutrino = 0;
    int isHad = 0; // 0=unknown, 1=hadronic, 2=leptonic
    int n_charged_hadrons = 0;
    int n_neutral_hadrons = 0;
    std::vector<int> final_daus;
    collect_final_daughters(itau, daughter_ids, truth_particles, final_daus);

    for (int di : final_daus) {    
      const auto& d = truth_particles[di];
      std::cout << "  Daughter idx=" << di << " PDG=" << d.PDG 
                << " charge=" << d.charge << "\n";

      const double px = static_cast<double>(d.momentum.x);
      const double py = static_cast<double>(d.momentum.y);
      const double pz = static_cast<double>(d.momentum.z);
      const double m  = static_cast<double>(d.mass);       // or PDG mass if you prefer
      const double e  = std::sqrt(px*px + py*py + pz*pz + m*m);

      TLorentzVector p_d; p_d.SetPxPyPzE(px, py, pz, e);

      if (isNeutrino(d)) {
        p_invis += p_d;
        ++n_neutrino;
        std::cout << "    -> Neutrino added\n";
      } else if (!isNeutrino(d)) {
          std::cout << " adding particle to p " << d.PDG;
          p_vis += p_d;
          if (isLightLep(d)) {
            std::cout << " mass of " << d.PDG << "lepton: " << d.mass;
          }
          if (isHadron(d)) {
            if (std::abs(d.charge) > 0) ++n_charged_hadrons;
            else                        ++n_neutral_hadrons;
            // seeing any hadron ⇒ hadronic decay
            isHad = 1;
          } else if (isLightLep(d)) {
            // only set leptonic if we haven't already seen hadrons
            if (isHad == 0) isHad = 2;
          } else {
            // neither hadron nor e/μ; keep current isHad unless still unknown
            if (isHad == 0) isHad = -9; // optional "unknown"
          }
        }

        
      }
    

    std::cout << "  Summary for tau idx " << itau << ":\n"
              << "    n_neutrino = " << n_neutrino << "\n"
              << "    n_charged_hadrons = " << n_charged_hadrons << "\n"
              << "    n_neutral_hadrons = " << n_neutral_hadrons << "\n"
              << "    m_vis             = " << p_vis.M() << "\n"
              << "    isHad = " << isHad << "\n";
    if (p_vis.M() < 0) {
      std::cout << "is this mass different to the lepton mass directly? " << p_vis.M();
    } 

    // Build features
    const float p_vis_mag   = static_cast<float>(p_vis.P());
    const float p_invis_mag = static_cast<float>(p_invis.P());
    const float truth_m_vis = static_cast<float>(p_vis.M());

    const float theta_vis_invis =
      (p_vis.P() > 0.0 && p_invis.P() > 0.0)
        ? static_cast<float>(p_vis.Angle(p_invis.Vect()))
        : -99.0f;

    const bool has_vis   = (p_vis.P()   > 0.0);
    const bool has_invis = (p_invis.P() > 0.0);

    std::vector<float> features;
    features.reserve(16);
    features.push_back(p_vis_mag);
    features.push_back(p_invis_mag);
    features.push_back(theta_vis_invis);
    features.push_back(static_cast<float>(n_neutrino));
    features.push_back(static_cast<float>(isHad));
    features.push_back(truth_m_vis);

    features.push_back(static_cast<float>(p_vis.Px()));
    features.push_back(static_cast<float>(p_vis.Py()));
    features.push_back(has_vis   ? static_cast<float>(p_vis.Eta()) : NAN);
    features.push_back(has_vis   ? static_cast<float>(p_vis.Phi()) : NAN);

    features.push_back(static_cast<float>(p_invis.Px()));
    features.push_back(static_cast<float>(p_invis.Py()));
    features.push_back(has_invis ? static_cast<float>(p_invis.Eta()) : NAN);
    features.push_back(has_invis ? static_cast<float>(p_invis.Phi()) : NAN);

    features.push_back(static_cast<float>(n_charged_hadrons));
    features.push_back(static_cast<float>(n_neutral_hadrons));

    tau_stuff.emplace_back(std::move(features));
  }

  return tau_stuff;
}

// calculate the transverse mass ob two objects: massless approximation?
ROOT::VecOps::RVec<float> AnalysisFCChh::get_mT(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj) {

  ROOT::VecOps::RVec<float> mT_vector;

  // if one of the input particles is empty, just fill a default value of -999
  // as mT
  if (Z_ll_pair.size() < 1 || MET_obj.size() < 1) {
    mT_vector.push_back(-999.);
    return mT_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  auto Z_ll = Z_ll_pair.at(0);
  auto MET = MET_obj.at(0);

  // Z_ll is fully reconstructed and regular 4 vector
  TLorentzVector tlv_Zll = getTLV_reco(Z_ll);
  float pT_ll = tlv_Zll.Pt();
  TVector3 vec_pT_ll;
  vec_pT_ll.SetXYZ(Z_ll.momentum.x, Z_ll.momentum.y, 0.);

  // for MET take the components separately: absolute MET pt and the x and y
  // component in a vector:
  TLorentzVector tlv_met = getTLV_MET(MET);
  float pT_met = tlv_met.Pt();
  TVector3 vec_pT_met;
  vec_pT_met.SetXYZ(MET.momentum.x, MET.momentum.y, 0.);

  float mT = sqrt(2. * pT_ll * pT_met *
                  (1 - cos(abs(vec_pT_ll.DeltaPhi(vec_pT_met)))));

  mT_vector.push_back(mT);

  // std::cout << "Debug mT: mT with old func = " << mT << std::endl;

  return mT_vector;
}

// different definition -> in tests it agreed 100% with previous def, keep for
// reference
ROOT::VecOps::RVec<float> AnalysisFCChh::get_mT_new(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> vis_part,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj) {

  ROOT::VecOps::RVec<float> mT_vector;

  // if one of the input particles is empty, just fill a default value of -999
  // as mT
  if (vis_part.size() < 1 || MET_obj.size() < 1) {
    mT_vector.push_back(-999.);
    return mT_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  auto visible_particle = vis_part.at(0);
  auto MET = MET_obj.at(0);

  TLorentzVector tlv_vis = getTLV_reco(visible_particle);
  float pT_vis = tlv_vis.Pt();
  TVector3 vec_pT_vis;
  vec_pT_vis.SetXYZ(visible_particle.momentum.x, visible_particle.momentum.y,
                    0.);

  // for MET take the components separately: absolute MET pt and the x and y
  // component in a vector:
  TLorentzVector tlv_met = getTLV_MET(MET);
  float pT_met = tlv_met.Pt();
  TVector3 vec_pT_met;
  vec_pT_met.SetXYZ(MET.momentum.x, MET.momentum.y, 0.);

  float mt_term1 = (pT_vis + pT_met) * (pT_vis + pT_met);
  float mt_term2 = (vec_pT_vis + vec_pT_met).Mag2();

  float mT = sqrt(mt_term1 - mt_term2);

  mT_vector.push_back(mT);

  // std::cout << "Debug mT: mT with new func = " << mT << std::endl;

  return mT_vector;
}

// pseudo-invariant mass - see CMS paper, PHYS. REV. D 102, 032003 (2020)
ROOT::VecOps::RVec<float> AnalysisFCChh::get_m_pseudo(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj) {

  ROOT::VecOps::RVec<float> m_pseudo_vector;

  // if one of the input particles is empty, just fill a default value of -999
  // as mT
  if (Z_ll_pair.size() < 1 || MET_obj.size() < 1) {
    m_pseudo_vector.push_back(-999.);
    return m_pseudo_vector;
  }

  TLorentzVector tlv_Zll = getTLV_reco(Z_ll_pair.at(0));
  TLorentzVector tlv_MET = getTLV_MET(MET_obj.at(0));

  TLorentzVector tlv_H_pseudo = tlv_Zll + tlv_MET;

  m_pseudo_vector.push_back(tlv_H_pseudo.M());

  return m_pseudo_vector;
}

// pseudo-transverse mass - see CMS paper, PHYS. REV. D 102, 032003 (2020)
ROOT::VecOps::RVec<float> AnalysisFCChh::get_mT_pseudo(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj) {

  ROOT::VecOps::RVec<float> m_pseudo_vector;

  // if one of the input particles is empty, just fill a default value of -999
  // as mT
  if (Z_ll_pair.size() < 1 || MET_obj.size() < 1) {
    m_pseudo_vector.push_back(-999.);
    return m_pseudo_vector;
  }

  TLorentzVector tlv_Zll = getTLV_reco(Z_ll_pair.at(0));
  TLorentzVector tlv_MET = getTLV_MET(MET_obj.at(0));

  TLorentzVector tlv_H_pseudo = tlv_Zll + tlv_MET;

  m_pseudo_vector.push_back(sqrt(tlv_H_pseudo.E() * tlv_H_pseudo.E() -
                                 tlv_H_pseudo.Pz() * tlv_H_pseudo.Pz()));

  return m_pseudo_vector;
}

// try the stransverse mass as defined in arXiv:1411.4312
// ROOT::VecOps::RVec<float>
// AnalysisFCChh::get_mT2(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_1, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_2, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj){

// 	asymm_mt2_lester_bisect::disableCopyrightMessage();

// 	ROOT::VecOps::RVec<float> m_strans_vector;

// 	//if one of the input particles is empty, just fill a default value of
// -999 as mT 	if (particle_1.size() < 1 || particle_2.size() < 1||
// MET_obj.size() < 1 ){ 		m_strans_vector.push_back(-999.);
// return m_strans_vector;
// 	}

// 	TLorentzVector tlv_vis1 = getTLV_reco(particle_1.at(0));
// 	TLorentzVector tlv_vis2 = getTLV_reco(particle_2.at(0));
// 	TLorentzVector tlv_met = getTLV_MET(MET_obj.at(0));

// 	// std::cout << "Part1 : Compare TLV w. direct for px:" << tlv_vis1.Px()
// << " vs " << particle_1.at(0).momentum.x << std::endl;
// 	// std::cout << "Part1 : Compare TLV w. direct for py:" << tlv_vis1.Py()
// << " vs " << particle_1.at(0).momentum.y << std::endl;

// 	// std::cout << "Part2 : Compare TLV w. direct for px:" << tlv_vis2.Px()
// << " vs " << particle_2.at(0).momentum.x << std::endl;
// 	// std::cout << "Part2 : Compare TLV w. direct for px:" << tlv_vis2.Py()
// << " vs " << particle_2.at(0).momentum.y << std::endl;

// 	// std::cout << "MET : Compare TLV w. direct for px:" << tlv_met.Px() <<
// " vs " << MET_obj.at(0).momentum.x << std::endl;
// 	// std::cout << "MET : Compare TLV w. direct for px:" << tlv_met.Py() <<
// " vs " << MET_obj.at(0).momentum.y << std::endl;

// 	double MT2 =  asymm_mt2_lester_bisect::get_mT2(
//            tlv_vis1.M(), tlv_vis1.Px(), tlv_vis1.Py(),
//            tlv_vis2.M(), tlv_vis2.Px(), tlv_vis2.Py(),
//            tlv_met.Px(), tlv_met.Py(),
//            0., 0.);

// 	// define our inputs:

// 	m_strans_vector.push_back(MT2);

// 	return m_strans_vector;
// }

// // stransverse mass but fixing the two visible to higgs masses
// ROOT::VecOps::RVec<float>
// AnalysisFCChh::get_mT2_125(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_1, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
// particle_2, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj){

// 	asymm_mt2_lester_bisect::disableCopyrightMessage();

// 	ROOT::VecOps::RVec<float> m_strans_vector;

// 	//if one of the input particles is empty, just fill a default value of
// -999 as mT 	if (particle_1.size() < 1 || particle_2.size() < 1||
// MET_obj.size() < 1 ){ 		m_strans_vector.push_back(-999.);
// return m_strans_vector;
// 	}

// 	TLorentzVector tlv_vis1 = getTLV_reco(particle_1.at(0));
// 	TLorentzVector tlv_vis2 = getTLV_reco(particle_2.at(0));
// 	TLorentzVector tlv_met = getTLV_MET(MET_obj.at(0));

// 	double MT2 =  asymm_mt2_lester_bisect::get_mT2(
//            125., tlv_vis1.Px(), tlv_vis1.Py(),
//            125., tlv_vis2.Px(), tlv_vis2.Py(),
//            tlv_met.Px(), tlv_met.Py(),
//            0., 0.);

// 	// define our inputs:

// 	m_strans_vector.push_back(MT2);

// 	return m_strans_vector;
// }

// HT2 variable as in ATLAS bblvlv analysis
ROOT::VecOps::RVec<float> AnalysisFCChh::get_HT2(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2) {

  ROOT::VecOps::RVec<float> HT2_vector;

  if (particle_1.size() < 1 || particle_2.size() < 1) {
    HT2_vector.push_back(-999.);
    return HT2_vector;
  }

  TLorentzVector tlv_1 = getTLV_reco(particle_1.at(0));
  TLorentzVector tlv_2 = getTLV_reco(particle_2.at(0));

  // scalar sum
  float HT2 = tlv_1.Pt() + tlv_2.Pt();
  HT2_vector.push_back(HT2);
  return HT2_vector;
}

// HT_w_inv = scalar sum of all pT from objects of a HH->bblvlv decay as used in
// ATLAS paper for the HT ratio
ROOT::VecOps::RVec<float> AnalysisFCChh::get_HT_wInv(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET,
    ROOT::VecOps::RVec<RecoParticlePair> ll_pair,
    ROOT::VecOps::RVec<RecoParticlePair> bb_pair) {

  ROOT::VecOps::RVec<float> HT_wInv_vector;

  if (MET.size() < 1 || ll_pair.size() < 1 || bb_pair.size() < 1) {
    HT_wInv_vector.push_back(-999.);
    return HT_wInv_vector;
  }

  // if all objects are there, get the first entry in vector always (should be
  // leading) and take the pTs
  float MET_pT = getTLV_MET(MET.at(0)).Pt();

  float lep1_pT = getTLV_reco(ll_pair.at(0).particle_1).Pt();
  float lep2_pT = getTLV_reco(ll_pair.at(0).particle_2).Pt();

  float b1_pT = getTLV_reco(bb_pair.at(0).particle_1).Pt();
  float b2_pT = getTLV_reco(bb_pair.at(0).particle_2).Pt();

  // and sum ..
  float HT_w_inv = MET_pT + lep1_pT + lep2_pT + b1_pT + b2_pT;
  HT_wInv_vector.push_back(HT_w_inv);
  return HT_wInv_vector;
}

// get the true HT = scalar sum of only the visible objects, here the bs and the
// leptons (true HT in contrast to the HT with the MET)
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_HT_true(ROOT::VecOps::RVec<RecoParticlePair> ll_pair,
                           ROOT::VecOps::RVec<RecoParticlePair> bb_pair) {

  ROOT::VecOps::RVec<float> HT_wInv_vector;

  if (ll_pair.size() < 1 || bb_pair.size() < 1) {
    HT_wInv_vector.push_back(-999.);
    return HT_wInv_vector;
  }

  // if all objects are there, get the first entry in vector always (should be
  // leading) and take the pTs
  float lep1_pT = getTLV_reco(ll_pair.at(0).particle_1).Pt();
  float lep2_pT = getTLV_reco(ll_pair.at(0).particle_2).Pt();

  float b1_pT = getTLV_reco(bb_pair.at(0).particle_1).Pt();
  float b2_pT = getTLV_reco(bb_pair.at(0).particle_2).Pt();

  // and sum ..
  float HT_w_inv = lep1_pT + lep2_pT + b1_pT + b2_pT;
  HT_wInv_vector.push_back(HT_w_inv);
  return HT_wInv_vector;
}

// construct ratio of HT2 and HT_w_inv
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_HT2_ratio(ROOT::VecOps::RVec<float> HT2,
                             ROOT::VecOps::RVec<float> HT_wInv) {

  ROOT::VecOps::RVec<float> HT2_ratio_vector;

  if (HT2.size() < 1 || HT_wInv.size() < 1) {
    HT2_ratio_vector.push_back(-999.);
    return HT2_ratio_vector;
  }

  float HT2_ratio = HT2.at(0) / HT_wInv.at(0);
  HT2_ratio_vector.push_back(HT2_ratio);
  return HT2_ratio_vector;
}

// construct met signifcance as ratio of MET pt and true HT
ROOT::VecOps::RVec<float> AnalysisFCChh::get_MET_significance(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET,
    ROOT::VecOps::RVec<float> HT_true, bool doSqrt) {

  ROOT::VecOps::RVec<float> MET_sig_vector;

  if (MET.size() < 1 || HT_true.size() < 1) {
    MET_sig_vector.push_back(-999.);
    return MET_sig_vector;
  }

  float MET_pt = getTLV_MET(MET.at(0)).Pt();

  if (doSqrt) {
    MET_sig_vector.push_back(MET_pt / sqrt(HT_true.at(0)));
  } else {
    MET_sig_vector.push_back(MET_pt / HT_true.at(0));
  }

  return MET_sig_vector;
}

// helper function which merges two particles into one using the TLVs
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::merge_parts_TLVs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2) {
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if one of the input particles is empty, return an empty vector
  if (particle_1.size() < 1 || particle_2.size() < 1) {
    // std::cout << "Warning in AnalysisFCChh::merge_parts_TLVs - one input
    // vector is empty, returning an empty vector." << std::endl;
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_reco(particle_1.at(0));
  TLorentzVector tlv_2 = getTLV_reco(particle_2.at(0));

  TLorentzVector tlv_merged = tlv_1 + tlv_2;

  edm4hep::ReconstructedParticleData particle_merged;
  particle_merged.momentum.x = tlv_merged.Px();
  particle_merged.momentum.y = tlv_merged.Py();
  particle_merged.momentum.z = tlv_merged.Pz();
  particle_merged.mass = tlv_merged.M();

  out_vector.push_back(particle_merged);

  return out_vector;
}

// same as above, overloaded for MCParticles
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::merge_parts_TLVs(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> particle_2) {
  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_vector;

  // if one of the input particles is empty, return an empty vector
  if (particle_1.size() < 1 || particle_2.size() < 1) {
    // std::cout << "Warning in AnalysisFCChh::merge_parts_TLVs - one input
    // vector is empty, returning an empty vector." << std::endl;
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_MC(particle_1.at(0));
  TLorentzVector tlv_2 = getTLV_MC(particle_2.at(0));

  TLorentzVector tlv_merged = tlv_1 + tlv_2;

  edm4hep::MCParticleData particle_merged;
  particle_merged.momentum.x = tlv_merged.Px();
  particle_merged.momentum.y = tlv_merged.Py();
  particle_merged.momentum.z = tlv_merged.Pz();
  particle_merged.mass = tlv_merged.M();

  out_vector.push_back(particle_merged);

  return out_vector;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::SortMCByPt(const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in) {
  auto out = in;
  std::stable_sort(out.begin(), out.end(),
                   [](const edm4hep::MCParticleData& a,
                      const edm4hep::MCParticleData& b){
                     return getTLV_MC(a).Pt() > getTLV_MC(b).Pt();
                   });
  return out;
}

// combine one lepton with one b-jet each, in case of ttbar events this should
// reconstruct the visible top

// find lb pairs with smallest average
ROOT::VecOps::RVec<RecoParticlePair>
AnalysisFCChh::make_lb_pairing(ROOT::VecOps::RVec<RecoParticlePair> lepton_pair,
                               ROOT::VecOps::RVec<RecoParticlePair> bb_pair) {

  ROOT::VecOps::RVec<RecoParticlePair> out_vector;
  RecoParticlePair lb_pair_1;
  RecoParticlePair lb_pair_2;

  // if one of the input particles is empty, return an empty vector
  if (lepton_pair.size() < 1 || bb_pair.size() < 1) {
    return out_vector;
  }

  // take the separate particles
  TLorentzVector tlv_lepton_1 = getTLV_reco(lepton_pair.at(0).particle_1);
  TLorentzVector tlv_lepton_2 = getTLV_reco(lepton_pair.at(0).particle_2);

  TLorentzVector tlv_bjet_1 = getTLV_reco(bb_pair.at(0).particle_1);
  TLorentzVector tlv_bjet_2 = getTLV_reco(bb_pair.at(0).particle_2);

  // then make the two possible combinations:
  TLorentzVector tlv_l1_b1 = tlv_lepton_1 + tlv_bjet_1;
  TLorentzVector tlv_l2_b2 = tlv_lepton_2 + tlv_bjet_2;

  TLorentzVector tlv_l1_b2 = tlv_lepton_1 + tlv_bjet_2;
  TLorentzVector tlv_l2_b1 = tlv_lepton_2 + tlv_bjet_1;

  // calculate the average invariant masses for the two combinations:
  float mlb_comb1 = (tlv_l1_b1.M() + tlv_l2_b2.M()) / 2.;
  float mlb_comb2 = (tlv_l1_b2.M() + tlv_l2_b1.M()) / 2.;

  // std::cout << "Mlb_comb1: " << mlb_comb1 << std::endl;
  // std::cout << "Mlb_comb2: " << mlb_comb2 << std::endl;

  // the combination with minimum mlb is the one we pick
  if (mlb_comb1 < mlb_comb2) {
    lb_pair_1.particle_1 = lepton_pair.at(0).particle_1;
    lb_pair_1.particle_2 = bb_pair.at(0).particle_1;

    lb_pair_2.particle_1 = lepton_pair.at(0).particle_2;
    lb_pair_2.particle_2 = bb_pair.at(0).particle_2;

  }

  else {

    lb_pair_1.particle_1 = lepton_pair.at(0).particle_1;
    lb_pair_1.particle_2 = bb_pair.at(0).particle_2;

    lb_pair_2.particle_1 = lepton_pair.at(0).particle_2;
    lb_pair_2.particle_2 = bb_pair.at(0).particle_1;
  }

  out_vector.push_back(lb_pair_1);
  out_vector.push_back(lb_pair_2);

  return out_vector;
}

// rather inefficienct, but now get the actually value of mlb again
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_mlb_reco(ROOT::VecOps::RVec<RecoParticlePair> lb_pairs) {

  ROOT::VecOps::RVec<float> out_vector;

  // there should be two pairs
  if (lb_pairs.size() < 2) {
    return out_vector;
  }

  TLorentzVector tlv_pair1 = lb_pairs.at(0).merged_TLV();
  TLorentzVector tlv_pair2 = lb_pairs.at(1).merged_TLV();

  float mlb_reco = (tlv_pair1.M() + tlv_pair2.M()) / 2.;

  out_vector.push_back(mlb_reco);
  // std::cout << "Mlb : " << mlb_reco << std::endl;

  return out_vector;
}

// do trhe same thing and also add met into it
ROOT::VecOps::RVec<float> AnalysisFCChh::get_mlb_MET_reco(
    ROOT::VecOps::RVec<RecoParticlePair> lb_pairs,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET) {

  ROOT::VecOps::RVec<float> out_vector;

  // there should be two pairs and one met
  if (lb_pairs.size() < 2 || MET.size() < 1) {
    return out_vector;
  }

  TLorentzVector tlv_pair1 = lb_pairs.at(0).merged_TLV();
  TLorentzVector tlv_pair2 = lb_pairs.at(1).merged_TLV();
  TLorentzVector tlv_MET = getTLV_MET(MET.at(0));

  float mlb_reco = (tlv_pair1 + tlv_pair2 + tlv_MET).M() / 2.;

  out_vector.push_back(mlb_reco);
  // std::cout << "Mlb : " << mlb_reco << std::endl;

  return out_vector;
}

// calculate the pzetas, following CMS tautau analyses:
// https://github.com/cardinia/DesyTauAnalysesUL/blob/dev/Common/interface/functions.h#L792-L841

ROOT::VecOps::RVec<float>
AnalysisFCChh::get_pzeta_vis(ROOT::VecOps::RVec<RecoParticlePair> lepton_pair) {

  ROOT::VecOps::RVec<float> out_vector;

  // there should be one lepton pair
  if (lepton_pair.size() < 1) {
    return out_vector;
  }

  // get the tlvs of the leptons:
  TLorentzVector tlv_lepton_1 = getTLV_reco(lepton_pair.at(0).particle_1);
  TLorentzVector tlv_lepton_2 = getTLV_reco(lepton_pair.at(0).particle_2);

  // normalize the pT vectors of the leptons to their magnitudes -> make unit
  // vectors, split in x and y components
  float vec_unit_lep1_x = tlv_lepton_1.Px() / tlv_lepton_1.Pt();
  float vec_unit_lep1_y = tlv_lepton_1.Py() / tlv_lepton_1.Pt();

  float vec_unit_lep2_x = tlv_lepton_2.Px() / tlv_lepton_2.Pt();
  float vec_unit_lep2_y = tlv_lepton_2.Py() / tlv_lepton_2.Pt();

  // the sum of the two unit vectors is the bisector
  float zx = vec_unit_lep1_x + vec_unit_lep2_x;
  float zy = vec_unit_lep1_y + vec_unit_lep2_y;

  // normalize with magnitude again?
  float modz = sqrt(zx * zx + zy * zy);
  zx = zx / modz;
  zy = zy / modz;

  // build the projection of pTll onto this bisector
  float vis_ll_x = tlv_lepton_1.Px() + tlv_lepton_2.Px();
  float vis_ll_y = tlv_lepton_1.Py() + tlv_lepton_2.Py();

  float pzeta_vis = zx * vis_ll_x + zy * vis_ll_y;

  out_vector.push_back(pzeta_vis);

  // std::cout << "pzeta_vis = " << pzeta_vis << std::endl;

  return out_vector;
}

// for pzeta_miss do the same for bisector, but then project etmiss onot it:
ROOT::VecOps::RVec<float> AnalysisFCChh::get_pzeta_miss(
    ROOT::VecOps::RVec<RecoParticlePair> lepton_pair,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET) {

  ROOT::VecOps::RVec<float> out_vector;

  // there should be one lepton pair and one MET
  if (lepton_pair.size() < 1 || MET.size() < 1) {
    return out_vector;
  }

  // get the tlvs of the leptons:
  TLorentzVector tlv_lepton_1 = getTLV_reco(lepton_pair.at(0).particle_1);
  TLorentzVector tlv_lepton_2 = getTLV_reco(lepton_pair.at(0).particle_2);
  TLorentzVector tlv_MET = getTLV_MET(MET.at(0));

  // normalize the pT vectors of the leptons to their magnitudes -> make unit
  // vectors, split in x and y components
  float vec_unit_lep1_x = tlv_lepton_1.Px() / tlv_lepton_1.Pt();
  float vec_unit_lep1_y = tlv_lepton_1.Py() / tlv_lepton_1.Pt();

  float vec_unit_lep2_x = tlv_lepton_2.Px() / tlv_lepton_2.Pt();
  float vec_unit_lep2_y = tlv_lepton_2.Py() / tlv_lepton_2.Pt();

  // the sum of the two unit vectors is the bisector
  float zx = vec_unit_lep1_x + vec_unit_lep2_x;
  float zy = vec_unit_lep1_y + vec_unit_lep2_y;

  // normalize with magnitude again?
  float modz = sqrt(zx * zx + zy * zy);
  zx = zx / modz;
  zy = zy / modz;

  // build the projection of MET onto this bisector
  float pzeta_miss = zx * tlv_MET.Pt() * cos(tlv_MET.Phi()) +
                     zy * tlv_MET.Pt() * sin(tlv_MET.Phi());

  out_vector.push_back(pzeta_miss);

  // std::cout << "pzeta_miss = " << pzeta_miss << std::endl;

  return out_vector;
}

// combine the two with a factor applied: CMS tautau uses 0.85
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_dzeta(ROOT::VecOps::RVec<float> pzeta_miss,
                         ROOT::VecOps::RVec<float> pzeta_vis, float factor) {

  ROOT::VecOps::RVec<float> out_vector;

  // there should be one pzeta each
  if (pzeta_miss.size() < 1 || pzeta_vis.size() < 1) {
    return out_vector;
  }

  out_vector.push_back(pzeta_miss.at(0) - factor * pzeta_vis.at(0));

  // std::cout << "dzeta = " << pzeta_miss.at(0) - factor*pzeta_vis.at(0) <<
  // std::endl;

  return out_vector;
}

// combine MET with the Zll pair into the HZZ candidate
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> AnalysisFCChh::build_HZZ(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> Z_ll_pair,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if one of the input particles is empty, return an empty vector and a
  // warning
  if (Z_ll_pair.size() < 1 || MET_obj.size() < 1) {
    // std::cout << "Warning in AnalysisFCChh::build_HZZ - one input vector is
    // empty, returning an empty vector." << std::endl;
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_reco(Z_ll_pair.at(0));
  TLorentzVector tlv_2 = getTLV_MET(MET_obj.at(0));

  TLorentzVector tlv_merged = tlv_1 + tlv_2;

  edm4hep::ReconstructedParticleData particle_merged;
  particle_merged.momentum.x = tlv_merged.Px();
  particle_merged.momentum.y = tlv_merged.Py();
  particle_merged.momentum.z = tlv_merged.Pz();
  particle_merged.mass = tlv_merged.M();

  out_vector.push_back(particle_merged);

  return out_vector;
}

// get dR between two objects:
ROOT::VecOps::RVec<float> AnalysisFCChh::get_angularDist(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_2,
    TString type) {

  ROOT::VecOps::RVec<float> out_vector;

  // if one of the input particles is empty, fill default value
  if (particle_1.size() < 1 || particle_2.size() < 1) {
    out_vector.push_back(-999.);
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_reco(particle_1.at(0));
  TLorentzVector tlv_2 = getTLV_reco(particle_2.at(0));

  if (type.Contains("dR")) {
    out_vector.push_back(tlv_1.DeltaR(tlv_2));
  }

  else if (type.Contains("dEta")) {
    out_vector.push_back(abs(tlv_1.Eta() - tlv_2.Eta()));
  }

  else if (type.Contains("dPhi")) {
    out_vector.push_back(tlv_1.DeltaPhi(tlv_2));
  }

  else {
    std::cout
        << " Error in AnalysisFCChh::get_angularDist - requested unknown type "
        << type << "Returning default of -999." << std::endl;
    out_vector.push_back(-999.);
  }

  return out_vector;
}

// get angular distances between MET and an object:
ROOT::VecOps::RVec<float> AnalysisFCChh::get_angularDist_MET(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> particle_1,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET_obj,
    TString type) {

  ROOT::VecOps::RVec<float> out_vector;

  // if one of the input particles is empty, fill default value
  if (particle_1.size() < 1 || MET_obj.size() < 1) {
    out_vector.push_back(-999.);
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_reco(particle_1.at(0));
  TLorentzVector tlv_2 = getTLV_MET(MET_obj.at(0));

  if (type.Contains("dR")) {
    out_vector.push_back(tlv_1.DeltaR(tlv_2));
  }

  else if (type.Contains("dEta")) {
    out_vector.push_back(abs(tlv_1.Eta() - tlv_2.Eta()));
  }

  else if (type.Contains("dPhi")) {
    out_vector.push_back(tlv_1.DeltaPhi(tlv_2));
  }

  else {
    std::cout
        << " Error in AnalysisFCChh::get_angularDist - requested unknown type "
        << type << "Returning default of -999." << std::endl;
    out_vector.push_back(-999.);
  }

  return out_vector;
}

// get angular distances between the two particles in a pair:
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_angularDist_pair(ROOT::VecOps::RVec<RecoParticlePair> pairs,
                                    TString type) {

  ROOT::VecOps::RVec<float> out_vector;

  // if input pairs is empty, fill default value
  if (pairs.size() < 1) {
    out_vector.push_back(-999.);
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_reco(pairs.at(0).particle_1);
  TLorentzVector tlv_2 = getTLV_reco(pairs.at(0).particle_2);

  if (type.Contains("dR")) {
    out_vector.push_back(tlv_1.DeltaR(tlv_2));
  }

  else if (type.Contains("dEta")) {
    out_vector.push_back(abs(tlv_1.Eta() - tlv_2.Eta()));
  }

  else if (type.Contains("dPhi")) {
    out_vector.push_back(tlv_1.DeltaPhi(tlv_2));
  }

  else {
    std::cout
        << " Error in AnalysisFCChh::get_angularDist - requested unknown type "
        << type << "Returning default of -999." << std::endl;
    out_vector.push_back(-999.);
  }

  return out_vector;
}


// get angular distances between the two particles in a pair: MC particles
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_angularDist_pair(ROOT::VecOps::RVec<MCParticlePair> pairs,
                                    TString type) {

  ROOT::VecOps::RVec<float> out_vector;

  // if input pairs is empty, fill default value
  if (pairs.size() < 1) {
    out_vector.push_back(-999.);
    return out_vector;
  }

  // else, for now, just take the first of each, should be the "best" one (by
  // user input) - flexibility to use all combinations is there, to be
  // implemented if needed
  TLorentzVector tlv_1 = getTLV_MC(pairs.at(0).particle_1);
  TLorentzVector tlv_2 = getTLV_MC(pairs.at(0).particle_2);

  if (type.Contains("dR")) {
    out_vector.push_back(tlv_1.DeltaR(tlv_2));
  }

  else if (type.Contains("dEta")) {
    out_vector.push_back(abs(tlv_1.Eta() - tlv_2.Eta()));
  }

  else if (type.Contains("dPhi")) {
    out_vector.push_back(tlv_1.DeltaPhi(tlv_2));
  }

  else {
    std::cout
        << " Error in AnalysisFCChh::get_angularDist - requested unknown type "
        << type << "Returning default of -999." << std::endl;
    out_vector.push_back(-999.);
  }

  return out_vector;
}

// function which returns the immediate children of a truth particle
ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::get_immediate_children(
    edm4hep::MCParticleData truth_part,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> child_list;
  auto first_child_index = truth_part.daughters_begin;
  auto last_child_index = truth_part.daughters_end;
  auto children_size = last_child_index - first_child_index;

  std::cout << "children size: " << children_size << std::endl;

  for (int child_i = 0; child_i < children_size; child_i++) {
    auto child_i_index = daughter_ids.at(first_child_index + child_i).index;
    auto child = truth_particles.at(child_i_index);
    std::cout << "PDG ID of child number " << child_i << " : " << child.PDG;
    // <<  std::endl;
    child_list.push_back(child);
  }

  return child_list;
}

// function which finds truth higgs in the MC particles and selects the one that
// decays according to requested type (to ZZ or bb here)
std::pair<
  ROOT::VecOps::RVec<edm4hep::MCParticleData>,
  ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>
>
AnalysisFCChh::get_truth_Higgs(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    TString decay)
{
  using namespace ROOT::VecOps;

  // Helper lambdas
  auto pt = [](const edm4hep::MCParticleData& p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };

  enum class Cat { BB, TAUTAU, OTHER };

  struct HiggsCand {
    edm4hep::MCParticleData H;
    edm4hep::MCParticleData d1; // pt-ordered
    edm4hep::MCParticleData d2; // pt-ordered
    Cat cat;
  };

  RVec<HiggsCand> cands;
  cands.reserve(8);

  // ---------------------------
  // Collect all Higgs with 2 daughters (store daughter pair regardless of type)
  // ---------------------------
  for (auto &truth_part : truth_particles) {
    if (!isH(truth_part)) continue;

    const auto first_child_index = truth_part.daughters_begin;
    const auto last_child_index  = truth_part.daughters_end;
    if (last_child_index - first_child_index != 2) continue;

    const auto child_1_MC_index = daughter_ids.at(first_child_index).index;
    const auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

    auto child_1 = truth_particles.at(child_1_MC_index);
    auto child_2 = truth_particles.at(child_2_MC_index);

    // pt-order decay products
    if (pt(child_2) > pt(child_1)) std::swap(child_1, child_2);

    Cat cat = Cat::OTHER;
    if (isb(child_1) && isb(child_2)) {
      cat = Cat::BB;
    } else if (isTau(child_1) && isTau(child_2)) {
      cat = Cat::TAUTAU;
    }

    cands.push_back({truth_part, child_1, child_2, cat});
  }

  // Sort candidates by Higgs pT (descending) via indices
  auto sort_by_Hpt = [&](const auto &a, const auto &b) { return pt(a.H) > pt(b.H); };
  auto all_idx = Argsort(cands, sort_by_Hpt);

  // Separate indices by category, already in pT order
  RVec<size_t> bb_idx, tautau_idx, other_idx;
  bb_idx.reserve(all_idx.size());
  tautau_idx.reserve(all_idx.size());
  other_idx.reserve(all_idx.size());

  for (auto i : all_idx) {
    if      (cands[i].cat == Cat::BB)     bb_idx.push_back(i);
    else if (cands[i].cat == Cat::TAUTAU) tautau_idx.push_back(i);
    else                                  other_idx.push_back(i);
  }

  // ---------------------------
  // Selection:
  //   - take up to 2 H->bb (leading pT)
  //   - take 3rd Higgs: prefer leading H->tautau, else leading remaining Higgs of any type
  // ---------------------------
  RVec<edm4hep::MCParticleData> selected_higgs;
  RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>> selected_pairs;

  selected_higgs.reserve(3);
  selected_pairs.reserve(3);

  // Track used indices to avoid duplicates
  auto used = RVec<char>(cands.size(), 0);

  for (size_t k = 0; k < std::min<size_t>(2, bb_idx.size()); ++k) {
    const auto i = bb_idx[k];
    used[i] = 1;
    selected_higgs.push_back(cands[i].H);
    selected_pairs.emplace_back(cands[i].d1, cands[i].d2);
  }

  auto push_if_unused = [&](size_t i) {
    if (i >= used.size() || used[i]) return false;
    used[i] = 1;
    selected_higgs.push_back(cands[i].H);
    selected_pairs.emplace_back(cands[i].d1, cands[i].d2);
    return true;
  };

  // Third Higgs: prefer tautau
  bool got_third = false;
  if (!tautau_idx.empty()) {
    got_third = push_if_unused(tautau_idx[0]);
  }

  // Otherwise: take highest-pT remaining Higgs of any category
  if (!got_third) {
    for (auto i : all_idx) {
      if (push_if_unused(i)) { got_third = true; break; }
    }
  }

  return {selected_higgs, selected_pairs};
}

std::pair<ROOT::VecOps::RVec<edm4hep::MCParticleData>,
          ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>>
AnalysisFCChh::get_truth_Higgs_6b(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids, TString /*decay*/) {

  using namespace ROOT::VecOps;

  // Collect all H->bb candidates
  RVec<edm4hep::MCParticleData> higgs_bb;
  RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>> bb_pairs;

  for (auto &truth_part : truth_particles) {
    if (!isH(truth_part)) continue;

    auto first_child_index = truth_part.daughters_begin;
    auto last_child_index  = truth_part.daughters_end;
    if (last_child_index - first_child_index != 2) continue;

    auto child_1_MC_index = daughter_ids.at(first_child_index).index;
    auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;
    const auto &child_1 = truth_particles.at(child_1_MC_index);
    const auto &child_2 = truth_particles.at(child_2_MC_index);

    higgs_bb.push_back(truth_part);
    bb_pairs.emplace_back(child_1, child_2);
    
  }

  // Sort H->bb candidates by pT (descending)
  auto pt = [](const edm4hep::MCParticleData &p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };

  // Build indices and sort them to avoid copying big objects
  std::vector<size_t> idx(higgs_bb.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](size_t i, size_t j){ return pt(higgs_bb[i]) > pt(higgs_bb[j]); });

  // Select up to three H->bb (for HHH->6b)
  RVec<edm4hep::MCParticleData> selected_higgs;
  RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>> selected_pairs;

  const size_t n_sel = std::min<size_t>(3, idx.size());
  for (size_t k = 0; k < n_sel; ++k) {
    selected_higgs.push_back(higgs_bb[idx[k]]);
    selected_pairs.push_back(bb_pairs[idx[k]]);
  }

  return {selected_higgs, selected_pairs};
}

// now a function that pT-orders all three truth Higgs
std::pair<ROOT::VecOps::RVec<edm4hep::MCParticleData>,
          ROOT::VecOps::RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>>>
AnalysisFCChh::get_truth_Higgs_pTorder(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids, TString decay) {

  using namespace ROOT::VecOps;

  // Collect all valid Higgs candidates and daughter pairs
  RVec<edm4hep::MCParticleData> higgs_all;
  RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>> all_pairs;

  for (auto &truth_part : truth_particles) {
    if (!isH(truth_part)) continue;

    auto first_child_index = truth_part.daughters_begin;
    auto last_child_index = truth_part.daughters_end;
    if (last_child_index - first_child_index != 2) continue;

    auto child_1_MC_index = daughter_ids.at(first_child_index).index;
    auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;
    auto child_1 = truth_particles.at(child_1_MC_index);
    auto child_2 = truth_particles.at(child_2_MC_index);

    if ((decay == "bb" && isb(child_1) && isb(child_2)) ||
        (decay == "tautau" && isTau(child_1) && isTau(child_2)) ||
        (decay == "any" && ((isb(child_1) && isb(child_2)) || (isTau(child_1) && isTau(child_2))))) {
      higgs_all.push_back(truth_part);
      all_pairs.emplace_back(child_1, child_2);
    }
  }

  // Sort by Higgs pT
  auto pt = [](const edm4hep::MCParticleData &p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };

  auto indices = Argsort(higgs_all, [&](const auto &a, const auto &b) {
    return pt(a) > pt(b);
  });

  RVec<edm4hep::MCParticleData> selected_higgs;
  RVec<std::pair<edm4hep::MCParticleData, edm4hep::MCParticleData>> selected_pairs;

  // Return top 3 Higgses (or fewer if less found)
  for (size_t i = 0; i < std::min(size_t(3), indices.size()); ++i) {
    selected_higgs.push_back(higgs_all[indices[i]]);
    selected_pairs.push_back(all_pairs[indices[i]]);
  }

  return {selected_higgs, selected_pairs};
}

// Same for getting a truth Z (->bb)
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::get_truth_Z_decay(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids, TString decay) {
  ROOT::VecOps::RVec<edm4hep::MCParticleData> Z_list;

  // std::cout << "looking for higgs .." << std::endl;

  for (auto &truth_part : truth_particles) {
    if (isZ(truth_part)) {
      // check into which particles the Higgs decays:

      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;

      // std::cout << "Found Z with children with indices " << first_child_index
      // << " , " << last_child_index << std::endl; std::cout << "number of Z
      // daughters:" << last_child_index - first_child_index << std::endl;

      // skip the intermediate Z that only lead to another Z
      if (last_child_index - first_child_index != 2) {
        continue;
      }

      // now get the indices in the daughters vector
      auto child_1_MC_index = daughter_ids.at(first_child_index).index;
      auto child_2_MC_index = daughter_ids.at(last_child_index - 1).index;

      // then go back to the original vector of MCParticles
      auto child_1 = truth_particles.at(child_1_MC_index);
      auto child_2 = truth_particles.at(child_2_MC_index);

      // std::cout << "PDG ID of first child: " << child_1.PDG <<  std::endl;
      // std::cout << "PDG ID of second child: " << child_2.PDG <<  std::endl;

      if (decay.Contains("bb") && isb(child_1) && isb(child_2)) {
        Z_list.push_back(truth_part);
      }
    }
  }

  return Z_list;
}

// get the truth flavour of the leptons from taus
ROOT::VecOps::RVec<int> AnalysisFCChh::getTruthLepLepFlavour(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau) {

  ROOT::VecOps::RVec<int> results_vec;

  if (leps_from_tau.size() != 2) {
    std::cout
        << "Error - running getTruthLepLepFlavour on event which doesn't have "
           "exactly two leptons from taus. This isnt the intended usage."
        << std::endl;
    return results_vec;
  }

  auto pdg_1 = leps_from_tau.at(0).PDG;
  auto pdg_2 = leps_from_tau.at(1).PDG;

  if (abs(pdg_1) == 11 && abs(pdg_2) == 11) {
    results_vec.push_back(0);
  }

  else if (abs(pdg_1) == 13 && abs(pdg_2) == 13) {
    results_vec.push_back(1);
  }

  else if ((abs(pdg_1) == 11 && abs(pdg_2) == 13) ||
           (abs(pdg_1) == 13 && abs(pdg_2) == 11)) {
    results_vec.push_back(2);
  }

  // option for taus, needed for checking bbWW
  else if (abs(pdg_1) == 15 || abs(pdg_2) == 15) {
    results_vec.push_back(3);
  }

  else {
    std::cout << "Error - found leptons from taus that are neither electrons "
                 "nor muons"
              << std::endl;
    results_vec.push_back(-999);
  }

  return results_vec;
}

// take the vector with truth leptons from taus  and pick out the electron or
// muon
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthEle(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> results_vec;

  for (auto &truth_lep : leps_from_tau) {

    // std::cout << "PDG ID " << abs(truth_lep.PDG) << std::endl;

    // electrons
    if (abs(truth_lep.PDG) == 11) {
      results_vec.push_back(truth_lep);
    }
  }
  return results_vec;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthMu(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_from_tau) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> results_vec;

  for (auto &truth_lep : leps_from_tau) {

    // muons
    if (abs(truth_lep.PDG) == 13) {
      results_vec.push_back(truth_lep);
    }
  }
  return results_vec;
}

// find the light leptons (e or mu) that originate from a tau decay (which comes
// from a higgs, and not a b-meson) using the truth info -> to use as filter for
// emu tautau evts
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getLepsFromTau(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {
  // test by simply counting first:
  //  int counter = 0;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_list;

  // loop over all truth particles and find light leptons from taus that came
  // from higgs (the direction tau->light lepton as child appears to be missing
  // in the tautau samples)
  for (auto &truth_part : truth_particles) {
    if (isLightLep(truth_part)) {
      bool from_tau_higgs =
          isChildOfTauFromHiggs(truth_part, parent_ids, truth_particles);
      if (from_tau_higgs) {
        // counter+=1;
        leps_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return leps_list;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthTau(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> tau_list;
  for (auto &truth_part : truth_particles) {
    bool flagchildren = false;
    if (isTau(truth_part)) {

      // check also if from Higgs to count only from Higgs ones
      if (type.Contains("from_higgs") &&
          !isFromHiggsDirect(
              truth_part, parent_ids,
              truth_particles)) { //&& isFromHadron(truth_part, parent_ids,
                                  // truth_particles) ) {
        continue;
      }
      // select both from higgs or from hadrons
      if (type.Contains("higgshad") &&
          !isFromHiggsDirect(truth_part, parent_ids, truth_particles) &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }

      tau_list.push_back(truth_part);
    }
  }
  //}

  return tau_list;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthTauLeps(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> tau_list;
  bool flagchildren = false;
  for (auto &truth_part : truth_particles) {
    flagchildren = false;
    if (isTau(truth_part)) {

      // check also if from Higgs to count only from Higgs ones
      if (type.Contains("from_higgs") &&
          !isFromHiggsDirect(
              truth_part, parent_ids,
              truth_particles)) { //&& isFromHadron(truth_part, parent_ids,
                                  // truth_particles) ) {
        continue;
      }
      if (type.Contains("from_Z") && !hasZParent(truth_part, parent_ids, truth_particles)) {
        continue;
      }
      // select both from higgs or from hadrons
      if (type.Contains("higgshad") &&
          !isFromHiggsDirect(truth_part, parent_ids, truth_particles) &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }
      bool isItself = true;
      while (isItself) {
        auto first_child_index = truth_part.daughters_begin;
        auto last_child_index = truth_part.daughters_end;
        // auto child_1_MC_index = daughter_ids.at(first_child_index).index;
        // auto hild_2_MC_index = daughter_ids.at(last_child_index-1).index;
        for (int child_i = first_child_index; child_i < last_child_index;
             child_i++) {
          auto child = truth_particles.at(daughter_ids.at(child_i).index);
          if (abs(child.PDG) == 15) {
            isItself = true;
            truth_part = child;
            break;
          } else {
            isItself = false;
          }
        }
      }

      // std::cout << " found tau with children" << std::endl;
      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;
      for (int ch_i = first_child_index; ch_i < last_child_index; ch_i++) {
        auto ch = truth_particles.at(daughter_ids.at(ch_i).index);
        std::cout << "Child ID: " << ch.PDG << std::endl;
        if (isLep(ch)) {
          std::cout << " Is leptonic" << std::endl;
          flagchildren = true;
          break;
        }
      }
      if (flagchildren) {
        tau_list.push_back(truth_part);
      }
    }
  }

  return tau_list;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthTauEmu(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> tau_list;
  bool flagchildren = false;
  for (auto &truth_part : truth_particles) {
    flagchildren = false;
    if (isTau(truth_part)) {

      // check also if from Higgs to count only from Higgs ones
      if (type.Contains("from_higgs") &&
          !isFromHiggsDirect(
              truth_part, parent_ids,
              truth_particles)) { //&& isFromHadron(truth_part, parent_ids,
                                  // truth_particles) ) {
        continue;
      }
      // select both from higgs or from hadrons
      if (type.Contains("higgshad") &&
          !isFromHiggsDirect(truth_part, parent_ids, truth_particles) &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }

      // select both from higgs or from hadrons
      if (type.Contains("hadonly") &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }
      bool isItself = true;
      while (isItself) {
        auto first_child_index = truth_part.daughters_begin;
        auto last_child_index = truth_part.daughters_end;
        // auto child_1_MC_index = daughter_ids.at(first_child_index).index;
        // auto hild_2_MC_index = daughter_ids.at(last_child_index-1).index;
        for (int child_i = first_child_index; child_i < last_child_index;
             child_i++) {
          auto child = truth_particles.at(daughter_ids.at(child_i).index);
          if (abs(child.PDG) == 15) {
            isItself = true;
            truth_part = child;
            break;
          } else {
            isItself = false;
          }
        }
      }
      edm4hep::MCParticleData tau_child{};  // NEW

      // std::cout << " found tau with children" << std::endl;
      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;
      for (int ch_i = first_child_index; ch_i < last_child_index; ch_i++) {
        auto ch = truth_particles.at(daughter_ids.at(ch_i).index);
        std::cout << "Child ID: " << ch.PDG << std::endl;
        if (type.Contains("mu") && isMuon(ch)){
          flagchildren = true;
          tau_child = ch;

          break;
        } else {
            if ((type.Contains("el") && isElectron(ch))) {
              flagchildren = true;
              tau_child = ch;
              break;
            if ((type.Contains("mu") && isMuon(ch))) {
              flagchildren = true;
              tau_child = ch;
              break;

            }
        }
      }
        
        
      }
      if (flagchildren) {
        tau_list.push_back(tau_child);
      }
    }
  }

  return tau_list;
}

// Return tuple of 5 RVecs: (pT_vis, pT_mis, n_charged, n_neutral, m_vis)
std::tuple<
  ROOT::VecOps::RVec<TLorentzVector>,  // p_vis
  ROOT::VecOps::RVec<TLorentzVector>,  // p_mis
  ROOT::VecOps::RVec<int>,             // n_charged
  ROOT::VecOps::RVec<int>,             // n_neutral
  ROOT::VecOps::RVec<float>,            // m_vis
  ROOT::VecOps::RVec<float>,            // m_mis
  ROOT::VecOps::RVec<int>               // n_neutrinos
>
AnalysisFCChh::getTruthTauHadronic(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& daughter_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    TString type)
{
  ROOT::VecOps::RVec<TLorentzVector> p_vis_list;
  ROOT::VecOps::RVec<TLorentzVector> p_mis_list;
  ROOT::VecOps::RVec<int> n_charged_list;
  ROOT::VecOps::RVec<int> n_neutral_list;
  ROOT::VecOps::RVec<float> m_vis_list;
  ROOT::VecOps::RVec<float> m_mis_list;
  ROOT::VecOps::RVec<float> n_neutrinos_list;

  for (auto& truth_part : truth_particles) {
    if (!isTau(truth_part)) continue;

    // Parent-type filters
    if (type.Contains("from_higgs") && !isFromHiggsDirect(truth_part, parent_ids, truth_particles))
      {
        continue;
      }

    if (type.Contains("from_Z") && !hasZParent(truth_part, parent_ids, truth_particles)) {
      continue;
    }

    if (type.Contains("higgshad") && !isFromHiggsDirect(truth_part, parent_ids, truth_particles) && !isFromHadron(truth_part, parent_ids, truth_particles)) {
      continue;
    }

    // Descend to final tau (last one in decay chain)
    auto tau = truth_part;
    bool descend = true;
    while (descend) {
      descend = false;
      for (int i = tau.daughters_begin; i < tau.daughters_end; i++) {
        auto child = truth_particles.at(daughter_ids.at(i).index);
        if (std::abs(child.PDG) == 15) {
          tau = child;
          descend = true;
          break;
        }
      }
    }

    // Build visible and invisible 4-vectors
    TLorentzVector p_vis(0, 0, 0, 0);
    TLorentzVector p_mis(0, 0, 0, 0);
    int n_charged = 0, n_neutral = 0;
    int n_neutrinos = 0;
    for (int i = tau.daughters_begin; i < tau.daughters_end; i++) {
      auto child = truth_particles.at(daughter_ids.at(i).index);
      if (isTau(child)) {
        std::cout << "skipping if we find another tau" << std::endl;
        continue;
      }
      TLorentzVector p4 = getTLV_MC(child);

      if (isNeutrino(child)) {
        p_mis += p4;
        n_neutrinos++;
      } else {
        p_vis += p4;
        if (!isLightLep(child)) {
          if (std::abs(child.charge) > 0) {
            n_charged++;
          }
          else if (child.charge == 0) {
            n_neutral++;
          }
          else {
            std::cout << "Child is non-existent?" << std::endl;
          }
        }
      }
    }

    // Store results
    p_vis_list.push_back(p_vis);
    p_mis_list.push_back(p_mis);
    n_charged_list.push_back(n_charged);
    n_neutral_list.push_back(n_neutral);
    n_neutrinos_list.push_back(n_neutrinos);
    float m_vis = p_vis.M();
    
    if (m_vis > 0) {
      m_vis_list.push_back(p_vis.M());
    } else {
      m_vis_list.push_back(-9);
    }

    // also for the lep-had cases
    float m_mis = p_mis.M();
    
    if (n_neutrinos != 1) {
      if (m_mis > 0) {
        m_mis_list.push_back(p_mis.M());
      } else {
        std::cout << " why are we having a mass less than 0? Mass: " << m_mis << std::endl;
        m_mis_list.push_back(-9);
      }
    } 
    // debugging check that we have a tau genuinely, and have obtained all decay products appropriately
    TLorentzVector tau_tlv = p_vis + p_mis;
    std::cout << "Mass of the tau " << tau_tlv.M() << " matches the tau? " << std::endl;
     
  }
  return std::make_tuple(p_vis_list, p_mis_list, n_charged_list, n_neutral_list, m_vis_list, m_mis_list, n_neutrinos_list);
}

// 3D angle computation
inline float safeAngle3D(const TLorentzVector& a, const TLorentzVector& b) {
    const TVector3 av = a.Vect();
    const TVector3 bv = b.Vect();
    if (av.Mag2() <= 0.0 || bv.Mag2() <= 0.0) return -1.f;
    return static_cast<float>(av.Angle(bv)); // radians
  }

inline AnalysisFCChh::TauAngleRecord makeBaseRecord(
    int tidx, int npr, const TLorentzVector& pvis_t, const TLorentzVector& pmiss_t) {
  AnalysisFCChh::TauAngleRecord r;
  r.truth_idx = tidx;
  r.n_prongs_truth = npr;
  r.theta_tt = safeAngle3D(pvis_t, pmiss_t);
  return r;
}

std::array<AnalysisFCChh::TauAngleRecord, 2>
AnalysisFCChh::matchTwoTruthTwoRecoAndAngles(
    const TLorentzVector& pvis_truth_0,
    const TLorentzVector& pmiss_truth_0,
    int nprongs_truth_0,
    const TLorentzVector& pvis_truth_1,
    const TLorentzVector& pmiss_truth_1,
    int nprongs_truth_1,
    const TLorentzVector& reco0,
    const TLorentzVector& reco1,
    float drCut
) {
  std::array<TauAngleRecord, 2> out = {
      makeBaseRecord(0, nprongs_truth_0, pvis_truth_0, pmiss_truth_0),
      makeBaseRecord(1, nprongs_truth_1, pvis_truth_1, pmiss_truth_1)
  };

  // ΔR matrix between truth visible axes and reco candidates
  const float dR00 = pvis_truth_0.DeltaR(reco0);
  const float dR01 = pvis_truth_0.DeltaR(reco1);
  const float dR10 = pvis_truth_1.DeltaR(reco0);
  const float dR11 = pvis_truth_1.DeltaR(reco1);

  // Two possible 1-1 assignments; pick the smaller total ΔR
  const float sumA = dR00 + dR11; // (0->0, 1->1)
  const float sumB = dR01 + dR10; // (0->1, 1->0)
  const bool useA = (sumA <= sumB);

  auto applyMatch = [&](int truthIdx, int recoIdx, float dR) {
    TauAngleRecord& r = out.at(truthIdx);

    const TLorentzVector& pmiss_t = (truthIdx == 0) ? pmiss_truth_0 : pmiss_truth_1;
    const TLorentzVector& reco    = (recoIdx  == 0) ? reco0        : reco1;

    r.dR_best = dR;

    if (dR >= drCut) {
      r.matched  = false;
      r.reco_idx = -1;
      r.pt_reco  = -1.f;
      r.theta_rt = -1.f;
      return;
    }

    r.matched  = true;
    r.reco_idx = recoIdx;
    r.pt_reco  = static_cast<float>(reco.Pt());
    r.theta_rt = safeAngle3D(reco, pmiss_t); // reco visible vs truth missing
  };

  if (useA) {
    applyMatch(0, 0, dR00);
    applyMatch(1, 1, dR11);
  } else {
    applyMatch(0, 1, dR01);
    applyMatch(1, 0, dR10);
  }

  return out;
}


inline bool safeHasZParent(
    const edm4hep::MCParticleData& p,
    const ROOT::VecOps::RVec<podio::ObjectID>& parent_ids,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& parts)
{
  const auto nParents = static_cast<int>(parent_ids.size());
  const auto nParts   = static_cast<int>(parts.size());

  auto inParentRange = [&](int a, int b) {
    return (0 <= a) && (a <= b) && (b <= nParents);
  };

  // Seed with p's parents (bounds-checked)
  std::vector<int> stack;
  if (!inParentRange(p.parents_begin, p.parents_end)) return false;
  for (int i = p.parents_begin; i < p.parents_end; ++i) {
    const int idx = parent_ids[i].index;
    if (0 <= idx && idx < nParts) stack.push_back(idx);
  }

  std::unordered_set<int> seen;
  const int maxSteps = std::max(2*nParts, 100); // cycle/bug guard

  int steps = 0;
  while (!stack.empty() && steps++ < maxSteps) {
    int idx = stack.back(); stack.pop_back();
    if (!seen.insert(idx).second) continue;

    const auto& par = parts[idx];
    if (std::abs(par.PDG) == 23) return true;

    if (!inParentRange(par.parents_begin, par.parents_end)) continue;
    for (int j = par.parents_begin; j < par.parents_end; ++j) {
      const int pid = parent_ids[j].index;
      if (0 <= pid && pid < nParts) stack.push_back(pid);
    }
  }
  return false;
}


// classiying ttZ / ttH


// find truth (hadronic) taus, to check the tau veto
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getTruthTauHads(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> tau_list;
  bool flagchildren = false;
  for (auto &truth_part : truth_particles) {
    flagchildren = false;
    if (isTau(truth_part)) {

      // check also if from Higgs to count only from Higgs ones
      if (type.Contains("from_higgs") &&
          !isFromHiggsDirect(
              truth_part, parent_ids,
              truth_particles)) { //&& isFromHadron(truth_part, parent_ids,
                                  // truth_particles) ) {
        continue;
      }
      if (type.Contains("from_Z") && !safeHasZParent(truth_part, parent_ids, truth_particles)) {
        continue;
      }
      
      // select both from higgs or from hadrons
      if (type.Contains("higgshad") &&
          !isFromHiggsDirect(truth_part, parent_ids, truth_particles) &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }

      if (type.Contains("higgs_not_had") &&
          isFromHiggsDirect(truth_part, parent_ids, truth_particles) &&
          !isFromHadron(truth_part, parent_ids, truth_particles)) {
        // continue;
        std::cout << " found tau neither from Higgs or from Had" << std::endl;
        auto first_parent_index = truth_part.parents_begin;
        auto last_parent_index = truth_part.parents_end;
        for (int parent_i = first_parent_index; parent_i < last_parent_index;
             parent_i++) {
          auto parent_MC_index = parent_ids.at(parent_i).index;
          auto parent = truth_particles.at(parent_MC_index);
          std::cout << "Parent PDG:" << std::endl;
          std::cout << parent.PDG << std::endl;
        }
        continue;
      }
      bool isItself = true;
      while (isItself) {
        auto first_child_index = truth_part.daughters_begin;
        auto last_child_index = truth_part.daughters_end;
        // auto child_1_MC_index = daughter_ids.at(first_child_index).index;
        // auto hild_2_MC_index = daughter_ids.at(last_child_index-1).index;
        for (int child_i = first_child_index; child_i < last_child_index;
             child_i++) {
          auto child = truth_particles.at(daughter_ids.at(child_i).index);
          if (abs(child.PDG) == 15) {
            isItself = true;
            truth_part = child;
            break;
          } else {
            isItself = false;
          }
        }
      }

      // std::cout << " found tau with children" << std::endl;
      auto first_child_index = truth_part.daughters_begin;
      auto last_child_index = truth_part.daughters_end;
      for (int ch_i = first_child_index; ch_i < last_child_index; ch_i++) {
        auto ch = truth_particles.at(daughter_ids.at(ch_i).index);
        std::cout << "Child ID: " << ch.PDG << std::endl;
        if (isHadron(ch)) {
          std::cout << " Is hadronic" << std::endl;
          flagchildren = true;
          break;
        }
      }
      if (flagchildren) {
        tau_list.push_back(truth_part);
      }
    }
  }

  return tau_list;
}

float AnalysisFCChh::metres(int n_jet)
{
  // implemented function derived from Z+jets (sigma(MET) = sqrt(a^2 + b^2n))
  return std::sqrt(5.26*5.26 + 4.06*4.06*n_jet);
}

std::tuple<
  ROOT::VecOps::RVec<edm4hep::MCParticleData>,  // p_vis
  ROOT::VecOps::RVec<edm4hep::MCParticleData> // p_mis
>
AnalysisFCChh::visible_tauhad(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>&         daughter_ids,
    const ROOT::VecOps::RVec<podio::ObjectID>&         parent_ids,
    TString type)
{
  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_vis;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_mis;

  // find hadronic taus from H/Z (from_higgs, from_Z)
  auto tau_list = AnalysisFCChh::getTruthTauHads(
      truth_particles, daughter_ids, parent_ids, type);

  
  if (tau_list.size() < 1) {
    return std::make_tuple(out_vis, out_mis);
  }
    
  // For each hadronic tau: sum visible daughters
  for (const auto& tau : tau_list) {
    int n_nu = 0;
    // Walk the final tau daughters
    const int first_child = tau.daughters_begin;
    const int last_child  = tau.daughters_end;

    TLorentzVector p_vis(0,0,0,0);
    TLorentzVector p_mis(0,0,0,0);

    for (int ch_i = first_child; ch_i < last_child; ++ch_i) {
      const auto child_idx = daughter_ids.at(ch_i).index;
      if (child_idx < 0 || child_idx >= (int)truth_particles.size()) continue;

      const auto& d = truth_particles[child_idx];
      const double px = d.momentum.x;
      const double py = d.momentum.y;
      const double pz = d.momentum.z;
      const double m  = d.mass;
      const double e  = std::sqrt(px*px + py*py + pz*pz + m*m);
      TLorentzVector p4; p4.SetPxPyPzE(px, py, pz, e);
      if (isNeutrino(d)) {
        n_nu++;
        p_mis += p4;
      } else {
        p_vis += p4;
      }
    }


    if (n_nu > 1) {
      std::cout << "We're in a leptonic decay of N_nu = " << n_nu << std::endl;
    }

    if (p_vis.P() <= 0.0) continue; // nothing visible, skip

    edm4hep::MCParticleData visTau = {};
    visTau.PDG    = tau.PDG;                        
    visTau.charge = tau.charge;                     
    visTau.mass   = static_cast<float>(p_vis.M()); 
    visTau.momentum.x = static_cast<float>(p_vis.Px());
    visTau.momentum.y = static_cast<float>(p_vis.Py());
    visTau.momentum.z = static_cast<float>(p_vis.Pz());
    visTau.vertex     = tau.vertex;                 

    edm4hep::MCParticleData misTau = {};
    misTau.PDG    = tau.PDG;                        
    misTau.charge = tau.charge;                     
    misTau.mass   = static_cast<float>(p_mis.M());  
    misTau.momentum.x = static_cast<float>(p_mis.Px());
    misTau.momentum.y = static_cast<float>(p_mis.Py());
    misTau.momentum.z = static_cast<float>(p_mis.Pz());
    misTau.vertex     = tau.vertex;   

    misTau.parents_begin   = misTau.parents_end   = 0;
    misTau.daughters_begin = misTau.daughters_end = 0;

    //out.emplace_back(ROOT::VecOps::RVec<edm4hep::MCParticleData>{visTau, misTau});
    out_vis.push_back(visTau);
    out_mis.push_back(misTau);
  }

  return std::make_tuple(out_vis, out_mis);
}

// function to find the minimum distance between electron and reco-matched signal objects
float AnalysisFCChh::min_dr_signal(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& target_obj,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& dR1_obj,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& dR2_obj
)
{
  // No target: return a sentinel
  if (target_obj.empty()) {
    return 999.f;
  }

  const auto& tar_obj = target_obj[0];
  TLorentzVector tar_obj_tlv = getTLV_reco(tar_obj);

  float mindR = 999.f;

  // First collection
  for (const auto& obji : dR1_obj) {
    TLorentzVector obji_tlv = getTLV_reco(obji);
    const float dRi = obji_tlv.DeltaR(tar_obj_tlv);
    if (dRi < mindR) {
      mindR = dRi;
    }
  }

  // Second collection
  for (const auto& obji : dR2_obj) {
    TLorentzVector obji_tlv = getTLV_reco(obji);
    const float dRi = obji_tlv.DeltaR(tar_obj_tlv);
    if (dRi < mindR) {
      mindR = dRi;
    }
  }

  return mindR;
}



bool AnalysisFCChh::isSignalContaminated(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_b_hadrons,
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_taus,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> selected_jets,
  float dR_truth
)
{
  bool isContam = false;

  for (auto jet : selected_jets) {
    TLorentzVector jet_tlv = getTLV_reco(jet);
    // now loop over both sets of truth particles, make TLVs and look at separations
    for (auto b : truth_b_hadrons) {
      TLorentzVector b_tlv = getTLV_MC(b);
      float dr_b_jet = jet_tlv.DeltaR(b_tlv);
      
      for (auto tau : truth_taus) {
        TLorentzVector tau_tlv = getTLV_MC(tau);
        float dr_tau_jet = jet_tlv.DeltaR(tau_tlv);
        if ((dr_tau_jet < dR_truth) & (dr_b_jet < dR_truth)) {
          isContam = true;
          return isContam;
        }

      }
    }
  }
  return isContam;


}


int AnalysisFCChh::nSignalContaminated(
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_b_hadrons,
  ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_taus,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> selected_jets,
  float dR_truth
)
{
  int nContam = 0;

  for (auto jet : selected_jets) {
    TLorentzVector jet_tlv = getTLV_reco(jet);
    // now loop over both sets of truth particles, make TLVs and look at separations
    for (auto b : truth_b_hadrons) {
      TLorentzVector b_tlv = getTLV_MC(b);
      float dr_b_jet = jet_tlv.DeltaR(b_tlv);
      
      for (auto tau : truth_taus) {
        TLorentzVector tau_tlv = getTLV_MC(tau);
        float dr_tau_jet = jet_tlv.DeltaR(tau_tlv);
        if ((dr_tau_jet < dR_truth) & (dr_b_jet < dR_truth)) {
          nContam++;
        }

      }
    }
  }
  return nContam;


}
// find leptons (including taus?) that came from a H->WW decay
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getLepsFromW(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {
  // test by simply counting first:
  //  int counter = 0;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_list;

  // loop over all truth particles and find light leptons from taus that came
  // from higgs (the direction tau->light lepton as child appears to be missing
  // in the tautau samples)
  for (auto &truth_part : truth_particles) {
    if (isLep(truth_part)) { // switch to isLightLep for tau veto!
      bool from_W_higgs =
          isChildOfWFromHiggs(truth_part, parent_ids, truth_particles);
      if (from_W_higgs) {
        // counter+=1;
        leps_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return leps_list;
}

// find leptons (including taus?) that came from a H->ZZ decay
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getLepsFromZ(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {
  // test by simply counting first:
  //  int counter = 0;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> leps_list;

  // loop over all truth particles and find light leptons from taus that came
  // from higgs (the direction tau->light lepton as child appears to be missing
  // in the tautau samples)
  for (auto &truth_part : truth_particles) {
    if (isLep(truth_part)) { // switch to isLightLep for tau veto!
      bool from_Z_higgs =
          isChildOfZFromHiggs(truth_part, parent_ids, truth_particles);
      if (from_Z_higgs) {
        // counter+=1;
        leps_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return leps_list;
}

// find photons that came from a H->yy decay
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getPhotonsFromH(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {
  ROOT::VecOps::RVec<edm4hep::MCParticleData> gamma_list;

  // loop over all truth particles and find stable photons that do not come
  // (directly) from a hadron decay
  for (auto &truth_part : truth_particles) {
    if (isStablePhoton(truth_part)) {
      bool from_higgs = hasHiggsParent(truth_part, parent_ids, truth_particles);
      if (isFromHadron(truth_part, parent_ids, truth_particles)) {
        from_higgs = false;
      }
      if (from_higgs) {
        gamma_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return gamma_list;
}

// momentum fraction x for tau decays
ROOT::VecOps::RVec<float> AnalysisFCChh::get_x_fraction(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> visible_particle,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET) {
  ROOT::VecOps::RVec<float> results_vec;

  if (visible_particle.size() < 1 || MET.size() < 1) {
    results_vec.push_back(-999.);
    return results_vec;
  }

  // get the components of the calculation
  TLorentzVector met_tlv = getTLV_reco(MET.at(0));
  // TLorentzVector met_tlv = getTLV_MET(MET.at(0));
  TLorentzVector vis_tlv = getTLV_reco(visible_particle.at(0));

  // float x_fraction = vis_tlv.Pt()/(vis_tlv.Pt()+met_tlv.Pt()); // try scalar
  // sum
  float x_fraction =
      vis_tlv.Pt() / (vis_tlv + met_tlv).Pt(); // vector sum makes more sense?

  // std::cout << " Debug m_col: pT_vis : " << vis_tlv.Pt() << std::endl;
  // std::cout << " Debug m_col: pT_miss : " << met_tlv.Pt() << std::endl;
  // std::cout << " Debug m_col: x with scalar sum: " << x_fraction <<
  // std::endl; std::cout << " Debug m_col: x with vector sum: " <<
  // vis_tlv.Pt()/(vis_tlv+met_tlv).Pt() << std::endl;

  results_vec.push_back(x_fraction);
  return results_vec;
}


// You already have this somewhere:
// TLorentzVector getTLV_reco(const edm4hep::ReconstructedParticleData& p);

inline bool isOS(const edm4hep::ReconstructedParticleData& a,
                  const edm4hep::ReconstructedParticleData& b) {
  return (static_cast<int>(a.charge) * static_cast<int>(b.charge)) < 0;
}

inline float mass2(const edm4hep::ReconstructedParticleData& a,
                    const edm4hep::ReconstructedParticleData& b) {
  TLorentzVector A = AnalysisFCChh::getTLV_reco(a);
  TLorentzVector B = AnalysisFCChh::getTLV_reco(b);
  return static_cast<float>((A + B).M());
}

inline float sumPt2(const edm4hep::ReconstructedParticleData& a,
                    const edm4hep::ReconstructedParticleData& b) {
  TLorentzVector A = AnalysisFCChh::getTLV_reco(a);
  TLorentzVector B = AnalysisFCChh::getTLV_reco(b);
  return static_cast<float>(A.Pt() + B.Pt());
}

float AnalysisFCChh::get_mtautau_vis_bestOS(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
                              const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& e,
                              const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& mu)
{
  const int nTau = (int)tauJets.size();
  const int nE   = (int)e.size();
  const int nMu  = (int)mu.size();
  const int Ntot = nTau + nE + nMu;

  if (Ntot < 2) return -999.f;

  struct Cand { float score; float mass; };
  std::vector<Cand> cands;
  cands.reserve(nTau*(nTau-1)/2 + nTau*(nE+nMu));

  auto consider_pair = [&](const edm4hep::ReconstructedParticleData& a,
                            const edm4hep::ReconstructedParticleData& b,
                            bool requireOS) {
    if (requireOS && !isOS(a,b)) return;
    cands.push_back({sumPt2(a,b), mass2(a,b)});
  };

  const bool requireOS = (Ntot > 2);

  // tau–tau
  for (int i = 0; i < nTau; ++i)
    for (int j = i+1; j < nTau; ++j)
      consider_pair(tauJets[i], tauJets[j], requireOS);

  // e–tau
  for (int ie = 0; ie < nE; ++ie)
    for (int it = 0; it < nTau; ++it)
      consider_pair(e[ie], tauJets[it], requireOS);

  // mu–tau
  for (int im = 0; im < nMu; ++im)
    for (int it = 0; it < nTau; ++it)
      consider_pair(mu[im], tauJets[it], requireOS);

  if (!cands.empty()) {
    // choose the candidate with the largest pT sum
    const Cand* best = &cands[0];
    for (const auto& c : cands) if (c.score > best->score) best = &c;
    return best->mass;
  }

  // If exactly two objects but none of the above added a candidate (e.g. same-sign tau–tau),
  // allow the 2-body mass if it's an allowed pair type (no OS requirement for Ntot==2).
  if (Ntot == 2) {
    if (nTau == 2) return mass2(tauJets[0], tauJets[1]);
    if (nTau == 1 && nE == 1) return mass2(tauJets[0], e[0]);
    if (nTau == 1 && nMu == 1) return mass2(tauJets[0], mu[0]);
  }

  return -999.f;
}

// define function to return TLorentzVector of Htautau
TLorentzVector
AnalysisFCChh::get_Htautau_vis_exclusive_TLV(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
  int n_tau,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  int n_el,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  int n_mu)
{
  const int nTau = (int)tauJets.size();
  const int nE   = (int)electrons.size();
  const int nMu  = (int)muons.size();

  auto pt = [](const edm4hep::ReconstructedParticleData& p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };
  auto isOS = [](const edm4hep::ReconstructedParticleData& a,
                 const edm4hep::ReconstructedParticleData& b) {
    return a.charge * b.charge < 0;
  };

  // ---- 1) lep–had: exactly one tau, exactly one e (and no mu), OS(τ,e)
  if (n_tau == 1 && n_el == 1 && n_mu == 0) {
    if (isOS(tauJets[0], electrons[0])) {
      return getTLV_reco(tauJets[0]) + getTLV_reco(electrons[0]);
    }
    return TLorentzVector();
  }

  // ---- 2) lep–had: exactly one tau, exactly one mu (and no e), OS(τ,μ)
  if (n_tau == 1 && n_mu == 1 && n_el == 0) {
    if (isOS(tauJets[0], muons[0])) {
      return getTLV_reco(tauJets[0]) + getTLV_reco(muons[0]);
    }
    return TLorentzVector(); 
  }

  // ---- 3) had–had: no leptons, ≥2 taus. Pick OS pair with largest ΣpT.
  if (n_el == 0 && n_mu == 0 && n_tau == 2) {
    float bestScore = -1.f;
    TLorentzVector bestP4; // default 0-vector

    for (int i = 0; i < nTau; ++i) {
      for (int j = i + 1; j < nTau; ++j) {
        const auto& a = tauJets[i];
        const auto& b = tauJets[j];
        if (!isOS(a, b)) continue;
        const float score = pt(a) + pt(b);
        if (score > bestScore) {
          bestScore = score;
          bestP4 = getTLV_reco(a) + getTLV_reco(b);
        }
      }
    }
    return bestP4; // if no OS pair, this is 0-vector
  }

  // ---- None of the exclusive categories matched
  return TLorentzVector(); // 0-vector sentinel
}


// Return [cand_lead, cand_sublead, Htautau_vis] (three TLorentzVectors).
// If no valid exclusive category/pair is found, returns {0,0,0}.
ROOT::VecOps::RVec<TLorentzVector>
AnalysisFCChh::get_Htautau_vis_exclusive_TLVs(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
  int n_tau,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  int n_el,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  int n_mu)
{

  auto pt = [](const edm4hep::ReconstructedParticleData& p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };
  auto isOS = [](const edm4hep::ReconstructedParticleData& a,
                 const edm4hep::ReconstructedParticleData& b) {
    return a.charge * b.charge < 0;
  };

  auto orderByPt = [](const TLorentzVector& a, const TLorentzVector& b) {
    return (a.Pt() >= b.Pt()) ? std::make_pair(a, b) : std::make_pair(b, a);
  };

  // 1) lep–had: exactly one tau, one electron, no mu; OS(tau,e)
  if (n_tau == 1 && n_el == 1 && n_mu == 0) {
    if (isOS(tauJets[0], electrons[0])) {
      const TLorentzVector p4a = getTLV_reco(tauJets[0]);
      const TLorentzVector p4b = getTLV_reco(electrons[0]);
      //auto [lead, sub] = orderByPt(p4a, p4b);
      return {p4a, p4b, p4a + p4b};
    }
    return ROOT::VecOps::RVec<TLorentzVector>(3); // {0,0,0}
  }

  // 2) lep–had: exactly one tau, one mu, no e; OS(τ,μ)
  if (n_tau == 1 && n_mu == 1 && n_el == 0) {
    if (isOS(tauJets[0], muons[0])) {
      const TLorentzVector p4a = getTLV_reco(tauJets[0]);
      const TLorentzVector p4b = getTLV_reco(muons[0]);
      //auto [lead, sub] = orderByPt(p4a, p4b);
      return {p4a, p4b, p4a + p4b};
    }
    return ROOT::VecOps::RVec<TLorentzVector>(3);
  }

  // 3) had–had: no leptons, exactly two taus
  if (n_el == 0 && n_mu == 0 && n_tau == 2) {
    // assume exactly two taus: tauJets[0], tauJets[1]
    const auto& tau1 = tauJets[0];
    const auto& tau2 = tauJets[1];

    if (!isOS(tau1, tau2)) {
        // return empty if not opposite sign
        return ROOT::VecOps::RVec<TLorentzVector>(3);
    }

    const TLorentzVector p4a = getTLV_reco(tau1);
    const TLorentzVector p4b = getTLV_reco(tau2);

    auto [lead, sub] = orderByPt(p4a, p4b);

    return ROOT::VecOps::RVec<TLorentzVector>{lead, sub, lead + sub};
  }

  // None matched exclusively → return zeroed triplet
  return ROOT::VecOps::RVec<TLorentzVector>(3);
}


std::tuple<
  edm4hep::ReconstructedParticleData, // lead constituent
  edm4hep::ReconstructedParticleData, // sublead constituent
  TLorentzVector,                      // visible H→ττ
  bool, // return if tau1 is a lepton or not
  bool // return if tau2 is a lepton or not
> AnalysisFCChh::get_Htautau_vis_exclusive_recoTLV(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tauJets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons)
{
  const int nTau = static_cast<int>(tauJets.size());
  const int nE   = static_cast<int>(electrons.size());
  const int nMu  = static_cast<int>(muons.size());

  auto pt = [](const edm4hep::ReconstructedParticleData& p) {
    return std::hypot(p.momentum.x, p.momentum.y);
  };
  auto isOS = [](const edm4hep::ReconstructedParticleData& a,
                 const edm4hep::ReconstructedParticleData& b) {
    return a.charge * b.charge < 0;
  };
  auto orderByPt = [&](const edm4hep::ReconstructedParticleData& a,
                       const edm4hep::ReconstructedParticleData& b) {
    return (pt(a) >= pt(b)) ? std::make_pair(a, b) : std::make_pair(b, a);
  };
  auto makeVisH = [&](const edm4hep::ReconstructedParticleData& a,
                      const edm4hep::ReconstructedParticleData& b) {
    // Prefer using your existing builder to ensure consistent calibration
    const TLorentzVector p4a = getTLV_reco(a);
    const TLorentzVector p4b = getTLV_reco(b);
    return p4a + p4b;
  };

  // 1) lep–had: exactly one tau, one electron, no mu
  if (nTau == 1 && nE == 1 && nMu == 0) {
    if (isOS(tauJets[0], electrons[0])) {
      auto lead = tauJets[0];
      auto sub = electrons[0];
      auto visH = makeVisH(lead, sub);
      
      return {lead, sub, visH, false, true};
    }
    return { {}, {}, {}, false, false };
  }

  // 2) lep–had: exactly one tau, one mu, no e
  if (nTau == 1 && nMu == 1 && nE == 0) {
    if (isOS(tauJets[0], muons[0])) {
      auto lead = tauJets[0];
      auto sub = muons[0];
      auto visH = makeVisH(lead, sub);
      
      return {lead, sub, visH, false, true};
    }
    return { {}, {}, {}, false, false };
  }

  // 3) had–had: no leptons, at least two taus
  if (nE == 0 && nMu == 0 && nTau >= 2) {
    float bestScore = -1.f;
    int   bestI = -1, bestJ = -1;
    for (int i = 0; i < nTau; ++i) {
      for (int j = i + 1; j < nTau; ++j) {
        if (!isOS(tauJets[i], tauJets[j])) continue;
        const float score = pt(tauJets[i]) + pt(tauJets[j]);
        if (score > bestScore) { bestScore = score; bestI = i; bestJ = j; }
      }
    }
    if (bestI >= 0) {
      auto [lead, sub] = orderByPt(tauJets[bestI], tauJets[bestJ]);
      auto visH = makeVisH(lead, sub);
      return {lead, sub, visH, false, false};
    }
    return { {}, {}, {}, false, false };
  }

  // None matched exclusively
  return { {}, {}, {}, false, false };
}

/// Solve the ditau system using the collinear approximation:
///   p_tau1 = p_vis1 / x1 ,  p_tau2 = p_vis2 / x2 , 0 < x1,x2 < 1
///   MET_T = (1/x1 - 1) * p_vis1_T + (1/x2 - 1) * p_vis2_T
/// Returns the collinear mass M_col = (p_tau1 + p_tau2).M() if physical; else -1.f.
float AnalysisFCChh::solve_ditau_system(
    TLorentzVector tau1,
    TLorentzVector tau2,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET
) {
  using std::abs;

  // Sum MET 2-vector from provided ReconstructedParticleData collection.
  // (Assumes MET entries store missing pT as a single or few pseudo-particles.)
  double met_px = 0.0, met_py = 0.0;
  for (const auto& p : MET) {
    met_px += p.momentum.x;
    met_py += p.momentum.y;
  }

  // Visible tau transverse components
  const double px1 = tau1.Px(), py1 = tau1.Py();
  const double px2 = tau2.Px(), py2 = tau2.Py();

  // Build linear system for u1 = 1/x1, u2 = 1/x2:
  // [ px1  px2 ] [u1] = [ met_px + px1 + px2 ]
  // [ py1  py2 ] [u2]   [ met_py + py1 + py2 ]
  const double c1 = met_px + px1 + px2;
  const double c2 = met_py + py1 + py2;

  const double D = px1 * py2 - py1 * px2;  // 2x2 determinant
  constexpr double det_eps = 1e-9;
  if (abs(D) < det_eps) {
    return -5.f;  // nearly parallel or ill-conditioned -> no physical solution
  }

  const double u1 = ( c1 * py2 - c2 * px2) / D;  // 1/x1
  const double u2 = (-c1 * py1 + c2 * px1) / D;  // 1/x2

  // Guard against unphysical / numerically bad solutions
  constexpr double frac_eps = 1e-9;
  if (u1 <= 0.0 || u2 <= 0.0) return -1.f;

  const double x1 = 1.0 / u1;
  const double x2 = 1.0 / u2;

  // require 0 < x < 1 (strict). Loosen with eps to be robust to rounding.
  if (!(x1 > frac_eps && x1 < 1.0 - frac_eps && x2 > frac_eps && x2 < 1.0 - frac_eps)) {
    return -5.f;
  }

  // Scale visible 4-vectors to get tau 4-vectors in the collinear approximation
  TLorentzVector tau1_col = tau1 * (1.0 / x1);
  TLorentzVector tau2_col = tau2 * (1.0 / x2);

  const double mcol = (tau1_col + tau2_col).M();
  if (!std::isfinite(mcol) || mcol <= 0.0) return -5.f;

  return static_cast<float>(mcol);
}

// tranverse mass
float AnalysisFCChh::mT_tautau(
  const TLorentzVector& tau1_vis,
  const TLorentzVector& tau2_vis,
  float MET_x,
  float MET_y
){
  const TLorentzVector v = tau1_vis + tau2_vis;
  const TVector2 pT_vis(v.Px(), v.Py());
  const TVector2 MET(MET_x, MET_y);

  const double mvis2   = std::max(0.0, v.M2());
  const double pTvis   = pT_vis.Mod();
  const double ETvis   = std::sqrt(mvis2 + pTvis*pTvis);
  const double METmag  = MET.Mod();
  const double dot     = pT_vis.X()*MET.X() + pT_vis.Y()*MET.Y();

  double mT2 = mvis2 + 2.0*(ETvis*METmag - dot);
  if (mT2 < 0.0) mT2 = 0.0;
  const double mT = std::sqrt(mT2);

  if (!std::isfinite(mT) || mT <= 0.0) return -1.f;
  return static_cast<float>(mT);
}


float AnalysisFCChh::compute_smin(
    TLorentzVector tau1_vis,
    TLorentzVector tau2_vis,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET
)
{
  float results_vec;

  TLorentzVector met_tlv  = getTLV_reco(MET.at(0));

  float min_mass = 1e12;

  float met_x = met_tlv.Px();
  float met_y = met_tlv.Py();

  for (float alpha = 0.01f; alpha < 1.0f; alpha += 0.05f) {
    float nu1_px = alpha * met_x;
    float nu1_py = alpha * met_y;
    float nu2_px = (1.0f - alpha) * met_x;
    float nu2_py = (1.0f - alpha) * met_y;

    for (float pz1 = -500.f; pz1 <= 500.f; pz1 += 20.f) {
      for (float pz2 = -500.f; pz2 <= 500.f; pz2 += 20.f) {

        TLorentzVector nu1, nu2;
        nu1.SetPxPyPzE(nu1_px, nu1_py, pz1, std::sqrt(nu1_px*nu1_px + nu1_py*nu1_py + pz1*pz1));
        nu2.SetPxPyPzE(nu2_px, nu2_py, pz2, std::sqrt(nu2_px*nu2_px + nu2_py*nu2_py + pz2*pz2));

        TLorentzVector tau1_total = tau1_vis + nu1;
        TLorentzVector tau2_total = tau2_vis + nu2;

        float m_total = (tau1_total + tau2_total).M();

        if (m_total < min_mass)
          min_mass = m_total;
      }
    }
  }
  return min_mass;
}

std::vector<float> linspace(float start, float stop, int num) {
  std::vector<float> result;
  if (num <= 1) {
    result.push_back(start);
    return result;
  }
  float step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) {
    result.push_back(start + i * step);
  }
  return result;
}




std::pair<std::vector<double>, std::vector<double>> AnalysisFCChh::solve_pTnu(
  TLorentzVector tau1,
  TLorentzVector tau2,
  double MET_x,
  double MET_y
)
{
  // solve system of equations for pT(nu1/2) assuming MET == p_Tnu1+2
  // has to assume a position of neutrinos
  std::vector<double> pT_nu1, pT_nu2;
  double dphi = 0.1;
  int Nphi = static_cast<int>(std::ceil(2*M_PI / dphi));
  for (int i = 0; i < Nphi; ++i) {
        const double phi1 = -M_PI + (i + 0.5) * dphi;
        for (int j = 0; j < Nphi; ++j) {
          const double phi2 = -M_PI + (j + 0.5) * dphi;
          double csc_dphi = 1/std::sin(phi2 - phi1);
          double pT1 = csc_dphi * ( MET_x * std::sin(phi2) - MET_y*std::cos(phi2) );
          double pT2 = -csc_dphi * ( MET_x * std::sin(phi1) - MET_y*std::cos(phi1) );

          pT_nu1.push_back(pT1);
          pT_nu2.push_back(pT2);

        }
      }
  return {std::move(pT_nu1), std::move(pT_nu2)};

}




constexpr double M_TAU = 1.77686; // GeV

  // Solve neutrino pz for a given side from (p_vis + p_nu)^2 = m_tau^2
  // Inputs: fixed ν px,py via (pT,phi); m_nu=0
inline std::vector<double> solve_neutrino_pz_single(const TLorentzVector& tau_vis,
                                                    double phi_nu, double pT_nu)
{
  const double px = pT_nu * std::cos(phi_nu);
  const double py = pT_nu * std::sin(phi_nu);
  const double pT2 = px*px + py*py;

  const double Ev  = tau_vis.E();
  const double mv2 = tau_vis.M();
  const double pvx = tau_vis.Px();
  const double pvy = tau_vis.Py();
  const double pvz = tau_vis.Pz();

  // From (Ev*En - pv·pn) = (m_tau^2 - m_vis^2)/2  with En = sqrt(pT^2 + pz^2)
  const double A = 0.5*(M_TAU*M_TAU - mv2) + pvx*px + pvy*py;

  // Quadratic in pz: α pz^2 + β pz + γ = 0
  const double alpha = std::max(1e-12, Ev*Ev - pvz*pvz);  // > 0 for physical Ev
  const double beta  = -2.0 * A * pvz;
  const double gamma = Ev*Ev * pT2 - A*A;

  const double disc = beta*beta - 4.0*alpha*gamma;
  if (disc < 0.0) return {};

  const double sd    = std::sqrt(std::max(0.0, disc));
  const double denom = 2.0 * alpha;

  const double z1 = (-beta + sd) / denom;
  const double z2 = (-beta - sd) / denom;

  std::vector<double> out;
  if (std::isfinite(z1)) out.push_back(z1);
  if (std::isfinite(z2) && std::abs(z2 - z1) > 1e-12) out.push_back(z2);
  return out;
}





float AnalysisFCChh::GetModeFromVector(
    std::vector<float> values,
    double xmin,
    double xmax,
    int nbins)
{
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();

  // Create a temporary histogram in memory (no ROOT file writing)
  TH1D hist("temp_hist", "", nbins, xmin, xmax);
  hist.SetDirectory(nullptr);  // Avoid attaching to gDirectory

  // Fill with data
  for (const auto& v : values) {
    if (v >= xmin && v <= xmax)
      hist.Fill(v);
  }

  // If histogram is empty
  if (hist.GetEntries() == 0)
    return std::numeric_limits<float>::quiet_NaN();

  // Find the bin with maximum content
  int maxBin = hist.GetMaximumBin();
  float mode = hist.GetBinCenter(maxBin);
  return mode;
}


float AnalysisFCChh::ComputeMMCPeak(const std::vector<float>& mtt,
                     const std::vector<float>& logL,
                     float lo, float hi, int nbins)
{
  if (mtt.empty() || logL.empty() || mtt.size() != logL.size())
    return -1.f;

  // Select points in window and finite
  std::vector<float> msel; msel.reserve(mtt.size());
  std::vector<float> lsel; lsel.reserve(logL.size());
  for (size_t i = 0; i < mtt.size(); ++i) {
    const float m = mtt[i];
    const float L = logL[i];
    if (std::isfinite(m) && std::isfinite(L) && m > lo && m < hi) {
      msel.push_back(m);
      lsel.push_back(L);
    }
  }
  if (msel.empty()) return -1.f;

  // Stable weights: w = exp(logL - max_logL)
  float maxL = *std::max_element(lsel.begin(), lsel.end());
  std::vector<double> counts(nbins, 0.0);
  const double width = (hi - lo) / nbins;

  for (size_t i = 0; i < msel.size(); ++i) {
    const double w = std::exp(static_cast<double>(lsel[i] - maxL));
    int bin = static_cast<int>((msel[i] - lo) / width);
    if (bin >= 0 && bin < nbins) counts[bin] += w;
  }

  auto it = std::max_element(counts.begin(), counts.end());
  if (it == counts.end() || *it <= 0.0) return -1.f;

  int best = static_cast<int>(std::distance(counts.begin(), it));
  const double bin_lo  = lo + best * width;
  const double bin_hi  = bin_lo + width;
  const double centre  = 0.5 * (bin_lo + bin_hi);
  return static_cast<float>(centre);
}


float AnalysisFCChh::find_MMC_mass(
  const std::pair<std::vector<float>, std::vector<float>>& mmc_result,
  float lo,
  float hi,
  int nbins)
{
  const auto& masses = mmc_result.first;
  const auto& logL   = mmc_result.second;
  return ComputeMMCPeak(masses, logL, lo, hi, nbins);
}


TLorentzVector get_leading4_merged(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in, int n) {
  // Sort indices by descending pt
  std::vector<std::pair<float, size_t>> pt_indices;
  for (size_t i = 0; i < in.size(); ++i) {
    float pt = std::sqrt(in[i].momentum.x * in[i].momentum.x +
                         in[i].momentum.y * in[i].momentum.y);
    pt_indices.emplace_back(pt, i);
  }
  std::sort(pt_indices.begin(), pt_indices.end(),
            [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
              return a.first > b.first;
            });

  // Build TLorentzVector from the leading n particles
  TLorentzVector sum;
  for (size_t i = 0; i < pt_indices.size() && i < n; ++i) {
    const auto& p = in[pt_indices[i].second];
    TLorentzVector vec(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);
    sum += vec;
  }

  return sum;
}


TLorentzVector AnalysisFCChh::get_collinearApproximationTLV(const RecoParticlePair& taupair, float x1, float x2)
{
  // Sanity check
  if (x1 <= 0.f || x2 <= 0.f) {
    // return an "empty" TLV if something is wrong
    return TLorentzVector();
  }

  TLorentzVector tau1_vis = getTLV_reco(taupair.particle_1);
  TLorentzVector tau2_vis = getTLV_reco(taupair.particle_2);

  // Rescale their 3-momenta by 1/x1 and 1/x2
  TVector3 tau1_vec = tau1_vis.Vect() * (1.f / x1);
  TVector3 tau2_vec = tau2_vis.Vect() * (1.f / x2);

  // Rebuild four-vectors assuming the mass is negligible (set E = |p|)
  TLorentzVector tau1_col;
  tau1_col.SetVectM(tau1_vec, tau1_vis.M());

  TLorentzVector tau2_col;
  tau2_col.SetVectM(tau2_vec, tau2_vis.M());

  // Sum the two rescaled taus
  return tau1_col + tau2_col;
}

// 19th October MMC
// function that returns solutions for neutrinos
inline std::vector<float> solve_pz_quadratic(const TLorentzVector& tau_vis,
                                                    float phi_nu, float pT_nu, int n_charged_tracks)
{
  /* implementing solution to pz quadratic
  starting from (p_vis + p_nu)^2 = m_tau^2 with m_nu = 0:
    m_tau^2 = m_vis^2 + 2 (E_v* E_nu - p_vis*p_nu),
  and E_nu = sqrt(pT^2 + pz^2), we can rearrange to
    E_v * sqrt(pT^2 + pz^2) = B + p_vz * pz,
  where
    B = 0.5 * (m_tau^2 - m_vis^2 - m_miss^2) + p_vx * px + p_vy * py.

  Squaring yields a quadratic in pz:
    (E_v^2 - p_vz^2) * pz^2 - 2 * B * p_vz * pz + (E_v^2 * pT^2 - B^2) = 0.

  coefficients are:
    a = E_v^2 - p_vz^2 
    b = -2 * B * p_vz,
    c = E_v^2 * pT^2 - B^2.

  Solutions:
      pz = (-b ± sqrt(b^2 - 4ac)) / (2a)
  */

  // now, define variables:
  const float px = pT_nu * std::cos(phi_nu);
  const float py = pT_nu * std::sin(phi_nu);
  const float pT2 = px*px + py*py;

  const float Ev  = tau_vis.E();
  const float pvx = tau_vis.Px();
  const float pvy = tau_vis.Py();
  const float pvz = tau_vis.Pz();
  //const float mvis2 = tau_vis.M2(); 
  // input the visible mass

  // default fallback:
  const float mvis2 = 0.8*0.8;

  if (n_charged_tracks <= 2) {
    const float mvis2 = 0.8 * 0.8;
  } else {
    const float mvis2 = 1.2 * 1.2;
  } 

  // std::cout << "Check the visible mass of tau " << mvis2 << std::endl;          
  const float mtau2 = M_TAU * M_TAU;         

  // Ev*sqrt(pT2+pz^2) - (pvx*px + pvy*py + pvz*pz) = (mtau2 - mvis2)/2
  const float B = 0.5*(mtau2 - mvis2) + pvx*px + pvy*py;

  // (Ev^2 - pvz^2) pz^2 - 2 B pvz pz + (Ev^2 pT2 - B^2) = 0
  const float a = Ev*Ev - pvz*pvz;
  const float b = -2.0 * pvz * B;
  const float c = Ev*Ev * pT2 - B*B;

  std::vector<float> out;

  // Handle near-linear case to avoid
  const float eps_a = 1e-12;
  if (std::abs(a) < eps_a) {
    if (std::abs(b) < 1e-18) return out;  
    const float pz = -c / b;
    if (std::isfinite(pz)) out.push_back(pz);
    return out;
  }

  float disc = b*b - 4.0*a*c;
  // return null solutions for unphysical mtautaus
  if (disc < 0) {
    return out;
  }

  const float sqrt_disc = std::sqrt(disc);
  const float denom = 2.0*a;

  const float pz1 = (-b + sqrt_disc) / denom;
  const float pz2 = (-b - sqrt_disc) / denom;

  if (std::isfinite(pz1)) out.push_back(pz1);
  if (std::isfinite(pz2) && std::abs(pz2 - pz1) > 1e-12) out.push_back(pz2);

  return out;
}



// ---------- constants & utils ----------
constexpr float INV_SQRT_2PI = 0.39894228040143267794f;
static constexpr float MMC_THETA_MAX = 0.3f; // must match the fit

inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float safe_log(float x) { return std::log(std::max(x, 1e-6f)); }

inline float sigmoid(float z) {
  if (z >= 0.f) { float ez = std::exp(-z); return 1.f / (1.f + ez); }
  float ez = std::exp(z); return ez / (1.f + ez);
}

// ---------- PDFs ----------
inline float gauss(float theta, float mu, float sigma) {
  const float s = std::max(sigma, 1e-6f);
  const float z = (theta - mu) / s;
  return INV_SQRT_2PI * std::exp(-0.5f * z * z) / s;
}

inline float moyal(float theta, float mu, float sigma) {
  const float s = std::max(sigma, 1e-6f);
  const float z = (theta - mu) / s;
  const float emz = std::exp(clampf(-z, -80.f, 80.f));   // guard exp(-z)
  const float expo = -0.5f * (z + emz);
  return INV_SQRT_2PI * std::exp(clampf(expo, -80.f, 50.f)) / s;
}

// 16-pt Gauss–Legendre nodes/weights on [-1,1]
static inline void gauss_legendre_16(double x[16], double w[16]) {
  static const double X[8] = {
    0.09501250983763744, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
    0.7554044083550030,  0.8656312023878318, 0.9445750230732326, 0.9894009349916499
  };
  static const double W[8] = {
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.09515851168249278, 0.06225352393864789, 0.02715245941175409
  };
  for (int i=0;i<8;++i) { x[i] = -X[7-i]; x[15-i] = X[7-i]; w[i] = W[7-i]; w[15-i] = W[7-i]; }
}

// ---------- parameter maps ----------
struct MixCoeffs {
  float b0, b1, b2;  // logit(pi) = b0 + b1 ln p + b2 (ln p)^2
  float c0, c1;      // mu_G = exp(c0 + c1 ln p)
  float d0, d1;      // sig_G = exp(d0 + d1 ln p)
  float e0, e1;      // mu_L = exp(e0 + e1 ln p)
  float f0, f1;      // sig_L = exp(f0 + f1 ln p)
};

// 1-prong (standard binning, thesis)
static const MixCoeffs COEFFS_1PRONG{
  0.033999102f, -0.37195870f,  0.09936033f,
  0.39291218f,  -0.86139226f,
 -0.43344867f,  -0.89916545f,
  1.00960356f,  -0.93863595f,
 -0.91546136f,  -0.77372742f
};

// 1-prong (standard binning, thesis)
static const MixCoeffs COEFFS_1PRONG_THESIS{
  -8.24133814f,  3.52463730f, -0.36326607f,
   0.88118047f, -0.98673942f,
  -0.03353991f, -0.98704956f,
   1.08335148f, -0.98403806f,
  -0.89737468f, -0.92882386f
};


// 3-prong (standard binning, thesis)
static const MixCoeffs COEFFS_3PRONG{
 -10.7172809f,   5.09002542f, -0.58232820f,
  -0.07445920f, -0.82593048f,
  -0.59968162f, -0.89220572f,
   1.45588231f, -1.11874896f,
  -0.55780400f, -0.84524834f
};


// 3-prong (standard binning, thesis)
static const MixCoeffs COEFFS_3PRONG_THESIS{
  -0.63506705f, -0.64607196f,  0.06512885f,
  -0.30908537f, -0.96600377f,
  -1.73227246f, -0.70392651f,
   1.01741435f, -1.06118480f,
  -0.02095602f, -1.09303131f
};


// leptonic (standard binning, thesis)
static const MixCoeffs COEFFS_LEPTONIC_THESIS{
  -2.39685225f,  0.93643581f, -0.07741001f,
   0.70537906f, -0.97452647f,
  -0.12344593f, -0.98113762f,
   0.82681874f, -0.91963804f,
  -0.67769345f, -0.98603278f
};

inline void params(float p_tau, const MixCoeffs& K,
                   float& pi, float& mu_g, float& sig_g, float& mu_l, float& sig_l)
{
  const float lnp = safe_log(p_tau);
  const float t = K.b0 + K.b1 * lnp + K.b2 * lnp * lnp;
  pi    = clampf(sigmoid(t), 1e-5f, 1.f - 1e-5f);
  mu_g  = std::exp(clampf(K.c0 + K.c1 * lnp, -20.f, 10.f));
  sig_g = std::exp(clampf(K.d0 + K.d1 * lnp, -20.f, 10.f));
  mu_l  = std::exp(clampf(K.e0 + K.e1 * lnp, -20.f, 10.f));
  sig_l = std::exp(clampf(K.f0 + K.f1 * lnp, -20.f, 10.f));
  sig_g = std::max(sig_g, 1e-6f);
  sig_l = std::max(sig_l, 1e-6f);
}

// mixture density on R
inline float theta_pdf_mixture(float theta, float p_tau, const MixCoeffs& K) {
  float pi, mg, sg, ml, sl;
  params(p_tau, K, pi, mg, sg, ml, sl);
  const float g = gauss(theta, mg, sg);
  const float l = moyal(theta, ml, sl);
  return pi * g + (1.f - pi) * l;
}

inline float theta_pdf_mixture_R(float theta, float p_tau, const MixCoeffs& K) {
  return theta_pdf_mixture(theta, p_tau, K);
}

inline float mixture_Z_0_thetaMax(float p_tau, const MixCoeffs& K, float theta_max) {
  double x[16], w[16]; gauss_legendre_16(x, w);
  const double a = 0.0, b = static_cast<double>(theta_max);
  const double c1 = 0.5*(b - a), c2 = 0.5*(b + a);
  double acc = 0.0;
  for (int i=0;i<16;++i) {
    const double t = c1 * x[i] + c2;    // map [-1,1] → [0,theta_max]
    acc += w[i] * static_cast<double>(theta_pdf_mixture_R(static_cast<float>(t), p_tau, K));
  }
  const double Z = c1 * acc;
  return static_cast<float>(std::max(Z, 1e-20)); // floor
}

// Proper pdf on [0, theta_max]
inline float theta_pdf_mixture_trunc(float theta, float p_tau, const MixCoeffs& K, float theta_max) {
  if (theta < 0.f || theta > theta_max) return 0.f;
  const float den = mixture_Z_0_thetaMax(p_tau, K, theta_max);
  return theta_pdf_mixture_R(theta, p_tau, K) / den;
}

// new theta parametrisation, hopefully simpler
// log-normal distribution
// ---- Log-normal MMC model (smooth vs pT) ----

struct Params { double m0, m1, s0, s1, c0, c1; };

static constexpr double theta_max_lognorm = 0.125; // rad

// 0‑prong (lep)
static constexpr Params P_LEP{
  -3.1427396862566863,  -0.23064074685568436,
  -0.1444890870687475,   0.1181859898431094,
  -13.812019296033004,   0.012060181926359414
};

// 1‑prong
static constexpr Params P_1P{
  -2.5618981829580747,  -0.23200455632017555,
    0.29105380432423994,  0.004788320942230688,
  -13.81585340322015,   -0.0014075505863843064
};

// 3‑prong
static constexpr Params P_3P{
  -4.514775603668456,    0.22546005420743231,
  -0.3428670208948243,   0.18177782151813016,
  -13.818297545084299,  -0.00882179217482995
};

inline const Params& params_for_nprongs(int nprongs) {
  if (nprongs <= 2 && nprongs >= 0) return P_1P;
  if (nprongs >= 3) return P_3P;
  return P_LEP;
}

inline double mu(double pt, const Params& p) {
  return p.m0 + p.m1 * std::log(std::max(pt, 1e-6));
}
inline double sigma(double pt, const Params& p) {
  return std::exp(std::min(std::max(p.s0 + p.s1 * std::log(std::max(pt, 1e-6)), -10.0), 5.0));
}
inline double shift_c(double pt, const Params& p) {
  return std::exp(std::min(std::max(p.c0 + p.c1 * std::log(std::max(pt, 1e-6)), -20.0), 5.0));
}

inline double lognormal_cdf(double theta, double mu_v, double sig_v, double c_v) {
  const double x = theta + c_v;
  if (x <= 0.0) return 0.0;
  const double z = (std::log(x) - mu_v) / (sig_v * std::sqrt(2.0));
  return 0.5 * (1.0 + std::erf(z));
}

inline double lognormal_pdf_trunc(double theta, double pt, int nprongs) {
  const Params& p = params_for_nprongs(nprongs);
  const double mu_v  = mu(pt, p);
  const double sig_v = sigma(pt, p);
  const double c_v   = shift_c(pt, p);
  const double x = theta + c_v;
  if (x <= 0.0 || theta < 0.0 || theta > theta_max_lognorm) return 0.0;

  const double z = (std::log(x) - mu_v) / sig_v;
  const double logpdf = -std::log(x * sig_v * std::sqrt(2.0 * M_PI)) - 0.5 * z * z;

  const double cdf_lo = lognormal_cdf(0.0, mu_v, sig_v, c_v);
  const double cdf_hi = lognormal_cdf(theta_max_lognorm, mu_v, sig_v, c_v);
  const double norm = std::max(cdf_hi - cdf_lo, 1e-12);

  return std::exp(logpdf) / norm;
}


double AnalysisFCChh::weighted_mode_from_mw(
    const std::vector<std::pair<float,float>>& mw_log, // {mass, log_weight}
    int nbins, double xmin, double xmax,
    bool refine_peak,
    bool toWeight     
)
{
  if (nbins <= 0 || !(xmax > xmin)) 
      return std::numeric_limits<double>::quiet_NaN();

  double max_logw = -std::numeric_limits<double>::infinity();
  if (toWeight) {
    for (const auto& [m, lw] : mw_log) {
      if (!std::isfinite(m) || !std::isfinite(lw)) continue;
      if (m < xmin || m >= xmax) continue;
      if (lw > max_logw) max_logw = lw;
    }
    if (!std::isfinite(max_logw))
        return std::numeric_limits<double>::quiet_NaN();
  }

  TH1D h("h_mmmc", ";m_{#tau#tau} [GeV];weighted entries", nbins, xmin, xmax);
  h.SetDirectory(nullptr);

  for (const auto& [m, lw] : mw_log) {
    if (!std::isfinite(m) || !std::isfinite(lw)) continue;
    if (m < xmin || m >= xmax) continue;

    double w = 1.0;
    if (toWeight) {
      w = std::exp(static_cast<double>(lw) - max_logw);
      if (!(w > 0.0)) continue;
    }

    h.Fill(m, w);
  }

  const int ibin_max = h.GetMaximumBin();
  if (ibin_max <= 0) return std::numeric_limits<double>::quiet_NaN();

  double mode = h.GetBinCenter(ibin_max);

  // optional parabolic interpolation
  if (refine_peak && ibin_max > 1 && ibin_max < nbins) {
    const double yL = h.GetBinContent(ibin_max - 1);
    const double yC = h.GetBinContent(ibin_max);
    const double yR = h.GetBinContent(ibin_max + 1);
    const double denom = (yL - 2.0*yC + yR);
    if (std::abs(denom) > 1e-12) {
      const double dx   = h.GetBinWidth(ibin_max);
      const double frac = 0.5 * (yL - yR) / denom;
      const double frac_clamped = std::max(-0.5, std::min(0.5, frac));
      mode = h.GetBinCenter(ibin_max) + frac_clamped * dx;
    }
  }

  return mode;
}

double AnalysisFCChh::atlas_style_mmc_mass(
    const std::vector<std::pair<float,float>>& mw, // {mass, weight}
    int nbins, double xmin, double xmax,
    bool smooth,
    int window_bins,
    bool refine_peak)
{
  if (nbins <= 0 || !(xmax > xmin) || window_bins < 1)
    return std::numeric_limits<double>::quiet_NaN();

  TH1F hW("h_mmmc_w", ";m_{#tau#tau} [GeV];weighted entries", nbins, xmin, xmax);
  TH1F hU("h_mmmc_u", ";m_{#tau#tau} [GeV];entries", nbins, xmin, xmax);
  hW.SetDirectory(nullptr);
  hU.SetDirectory(nullptr);

  for (const auto& [m, w] : mw) {
    if (!std::isfinite(m) || !std::isfinite(w)) continue;
    if (m < xmin || m >= xmax) continue;
    if (w <= 0.0) continue;
    hW.Fill(m, w);
    hU.Fill(m, 1.0);
  }

  if (smooth) {
    hW.Smooth();
    hU.Smooth();
  }

  double best_sum = -1.0;
  int best_bin = -1;
  const int nb = hW.GetNbinsX();
  const int win = std::min(window_bins, nb);
  for (int b = 1; b <= nb - win + 1; ++b) {
    double s = 0.0;
    for (int k = 0; k < win; ++k) s += hW.GetBinContent(b + k);
    if (s > best_sum) { best_sum = s; best_bin = b + win/2; }
  }
  if (best_bin < 1) return std::numeric_limits<double>::quiet_NaN();

  double mode = hW.GetBinCenter(best_bin);
  if (refine_peak && best_bin > 1 && best_bin < nb) {
    const double yL = hW.GetBinContent(best_bin - 1);
    const double yC = hW.GetBinContent(best_bin);
    const double yR = hW.GetBinContent(best_bin + 1);
    const double denom = (yL - 2.0*yC + yR);
    if (std::abs(denom) > 1e-12) {
      const double dx   = hW.GetBinWidth(best_bin);
      const double frac = 0.5 * (yL - yR) / denom;
      const double frac_clamped = std::max(-0.5, std::min(0.5, frac));
      mode = hW.GetBinCenter(best_bin) + frac_clamped * dx;
    }
  }

  return mode;
}

// new function for computing weighted solutions to tautau system
std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_angular_weighted(
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
)
{ 

  std::vector<std::pair<float,float>> mditau_solutions;

  // using nstep, define granularity of phi grid scan
  const float dphi = 2.0 * M_PI / static_cast<float>(nsteps);

  // met scanning, dividing by sqrt(2) to go from event -> x/y cmpt met resolutions 
  // extracted correlation(x,y) ~ 0, hence this is safe to do. 
  const float isqrt2 = 1 / std::sqrt(2);
  const float metres_xy = metres * isqrt2;
  const float inv_metres_xy2 = 1/(metres_xy*metres_xy);
  const float half = nMETsig*metres_xy;
  const float dmet = (2*half)/(nMETsteps-1);

  /* 
  MMC implementation that scans over azimuthal angles of each tau 
  with a square grid around MET_x/y - Gauss weighting to penalise large deviations.
  */
  for (int i = 0; i < nsteps; ++i) {
      float phi1 = -M_PI + (i + 0.5) * dphi;
      for (int j = 0; j < nsteps; ++j) {
        float phi2 = -M_PI + (j + 0.5) * dphi;
        // define csc dphi for use in each step in for-loop
        float csc_dphi = 1 / (std::sin(phi2 - phi1));

        // also now perform MET-scan in grid around measured x/y values
        for (int k = 0; k < nMETsteps; ++k) {
          float sMET_x = MET_x - nMETsig * metres_xy + k * dmet;
          for (int l = 0; l < nMETsteps; ++l) {
            float sMET_y = MET_y - nMETsig * metres_xy + l * dmet;
            
            // compute pTs of neutrinos
            float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
            float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );

            // using these solutions, solve each quadratic
            std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
            std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);

            
            // we have four solutions, store all for processing later
            // since we store all solutions, pre-compute MET weight (logged, taking out constant terms)
            const float dx = sMET_x - MET_x;
            const float dy = sMET_y - MET_y;
          
            // log w_MET = -0.5 * (dx^2/σx^2 + dy^2/σy^2)
            const double log_w_met = -0.5 * (dx*dx * inv_metres_xy2 + dy*dy * inv_metres_xy2);
            const double w_met     = exp(log_w_met); // linear MET weight

            for (float z1 : pz1) {
              // neutrino 1 TLV from (pT1, phi1, z1)
              const float px1 = pT1 * std::cos(phi1);
              const float py1 = pT1 * std::sin(phi1);
              const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); 
              TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

              for (float z2 : pz2) {
                // neutrino 2 TLV from (pT2, phi2, z2)
                const float px2 = pT2 * std::cos(phi2);
                const float py2 = pT2 * std::sin(phi2);
                const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); 
                TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                // full tau four-vectors
                TLorentzVector tau_full_1 = tau1 + nu1;
                TLorentzVector tau_full_2 = tau2 + nu2;
                
                // angular kinematic weight 
                const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());

                // require tau 4-momenta
                const float p_tau1 = tau_full_1.P();
                const float p_tau2 = tau_full_2.P();

              
                const float THETA_MAX = 0.3f; 

                const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                const float w_event  = w_theta1 * w_theta2 * w_met;

                // di-tau invariant mass
                const float mditau = (tau_full_1 + tau_full_2).M();

                // store every solution 
                mditau_solutions.push_back({mditau, w_event});
            }
          }

          }

        }
        
      }
  }

  return mditau_solutions;

}



// new function needed to solve pz quadratic with missing mass scan for leptonic leg
inline std::vector<float> solve_pz_quadratic_tau(const TLorentzVector& tau_vis,
                                                 float phi_nu,
                                                 float pT_nu,
                                                 bool isLep,
                                                 int n_charged_tracks = 1,
                                                 float m_nu_eff = 0.0)
{
  /*
    General quadratic solver for tau -> vis + missing system.

    Start with 4-vector eqn:
      (p_vis + p_missing)^2 = m_tau^2, with p_missing^2 = m_missing^2, 
      simplifies for hadronic system with a singular neutrino -> m_nu == 0

    Write dot-product of missing and visible 3-momenta as sum of pTmissing dot pTvisible + pzmissing * pzvisible
    Isolate sqrt, solve for pz
    C = (m_tau^2 - m_vis^2 - m_miss^2)/2 - pTmissing dot pTvisible

    Quadratic becomes:
    pz^2 (pzvisible^2 - E_visible^2) + pz (2Cpzvisible) + (C^2 - E_visible^2p_Tmissing^2 - Evisible^2m_missing^2xw)

    => E_v * sqrt(pT^2 + pz^2 + m_nu^2)
       = K + p_vx*px + p_vy*py + p_vz*pz,
       where K = 0.5 * (m_tau^2 - m_vis^2 - m_nu^2).

    Squaring gives:
       (E_v^2 - p_vz^2) pz^2
       - 2 p_vz (K + p_vx*px + p_vy*py) pz
       + [E_v^2 (pT^2 + m_nu^2) - (K + p_vx*px + p_vy*py)^2] = 0.

    - For hadronic τ: isLep = false, m_nu_eff = 0.
      Choose m_vis ≈ 0.8 GeV (1-prong) or 1.2 GeV (3-prong).
    - For leptonic τ: isLep = true,
      m_vis = m_lep (0.0005 or 0.105 GeV), m_nu_eff typically 0.3–0.8 GeV.
  */

  const float px = pT_nu * std::cos(phi_nu);
  const float py = pT_nu * std::sin(phi_nu);
  const float pT2 = px*px + py*py;

  const float Ev  = tau_vis.E();
  const float pvx = tau_vis.Px();
  const float pvy = tau_vis.Py();
  const float pvz = tau_vis.Pz();

  float mvis2 = tau_vis.M2();
  // check here that the code is producing taus of more or less the correct mass
  //std::cout << "Checking mass of tau " << std::sqrt(mvis2) << std::endl; 

  
  const float mtau2 = M_TAU * M_TAU;
  // Override visible mass where appropriate
  if (!isLep) {
    if (n_charged_tracks <= 2) {
      mvis2 = 0.8f * 0.8f;   // 1-prong
    } else {
      mvis2 = 1.2f * 1.2f;   // 3-prong
    }
    // also hard-code that the missing mass here as zero:
    m_nu_eff = 0.0;
  }
  const float mnu2  = m_nu_eff * m_nu_eff;

  const float K = 0.5f * (mtau2 - mvis2 - mnu2);
  const float B_T = pvx*px + pvy*py;
  // adapting to my own solution set
  const float C = 0.5 * (mtau2 - mvis2 - mnu2) + pvx*px + pvy*py;

  // Quadratic coefficients
  const float a = Ev*Ev - pvz*pvz;
  const float b = -2.0f * pvz * C; 
  const float c = Ev*Ev * (pT2 + mnu2) - C*C;

  std::vector<float> out;

  const float eps_a = 1e-12f;
  if (std::abs(a) < eps_a) {
    // if a == 0 and b == 0, no solutions
    if (std::abs(b) < 1e-18f) return out;
    // if a close to zero, is a linear system and soln:
    const float pz = -c / b;
    std::cout << " Are we ever solving a linear system?" << std::endl;
    if (std::isfinite(pz)) out.push_back(pz);
    return out;
  }

  const float disc = b*b - 4.0f*a*c;
  if (disc < 0) return out; // no physical solution

  const float sqrt_disc = std::sqrt(disc);
  const float denom = 2.0f * a;

  const float pz1 = (-b + sqrt_disc) / denom;
  const float pz2 = (-b - sqrt_disc) / denom;

  // if pz1 finite, store in solution set
  if (std::isfinite(pz1)) out.push_back(pz1);
  // if pz2 finite and distinct from pz1, store in solution set alongside pz1
  if (std::isfinite(pz2) && std::fabs(pz2 - pz1) > 1e-12f) out.push_back(pz2);

  return out;
}

inline float ditau_mass_collinear_or_vis(
  const TLorentzVector& tau1,
  const TLorentzVector& tau2,
  float MET_x,
  float MET_y
)
{
  // Visible ditau mass (fallback)
  const double m_vis = (tau1 + tau2).M();

  // Transverse components of visible taus
  const double v1x = tau1.Px();
  const double v1y = tau1.Py();
  const double v2x = tau2.Px();
  const double v2y = tau2.Py();

  const double metx = static_cast<double>(MET_x);
  const double mety = static_cast<double>(MET_y);

  // Solve MET = α1 * v1T + α2 * v2T
  const double det = v1x * v2y - v1y * v2x;
  if (std::fabs(det) < 1e-6) {
    // Nearly collinear taus in φ → ill-conditioned
    return static_cast<float>(m_vis);
  }

  const double inv_det = 1.0 / det;
  const double alpha1  = ( metx * v2y - mety * v2x) * inv_det;
  const double alpha2  = (-metx * v1y + mety * v1x) * inv_det;

  // Require physical solution: neutrinos along tau directions, positive scaling
  if (alpha1 < 0.0 || alpha2 < 0.0) {
    return static_cast<float>(m_vis);
  }

  // Collinear tau four-vectors: p_tau = (1+α) * p_vis
  TLorentzVector tau1_full = tau1;
  TLorentzVector tau2_full = tau2;
  tau1_full *= (1.0 + alpha1);
  tau2_full *= (1.0 + alpha2);

  const double m_coll = (tau1_full + tau2_full).M();

  // If something goes crazy (NaN / inf), still fall back to visible
  if (!std::isfinite(m_coll) || m_coll <= 0.0) {
    return static_cast<float>(m_vis);
  }

  return static_cast<float>(m_coll);
}


std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_angular_lephad_weighted(
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
)
{ 

  std::vector<std::pair<float,float>> mditau_solutions;

  // not processing leplep: exit early
  if (isLep1 && isLep2) {
    // do not run if di-leptonic
    return mditau_solutions;
  }
  
  // for had-had we can increase the scanning steps and MET-width
  if (!isLep1 && !isLep2) {
    nMETsteps *= 2;
    nsteps *= 2;
    nMETsig = 10;
  }

  // also apply preselection here, otherwise we spend too long computing configurations that are nonsense
  if ((n_b_jets_medium != 4) || (n_tau_jets_medium < 1) || (n_tau_jets_medium > 2)) {
    return mditau_solutions;
  }

  auto wrapToPi = [](float a) -> float {
    // robust (-pi, pi] wrapping
    a = std::fmod(a + M_PI, 2.f*M_PI);
    if (a < 0.f) a += 2.f*M_PI;
    return a - M_PI;
  };
    
  // using nstep, define granularity of phi grid scan
  const float dphi = 2.0 * M_PI / static_cast<float>(nsteps);

  // met scanning, dividing by sqrt(2) to go from event -> x/y cmpt met resolutions 
  // extracted correlation(x,y) ~ 0, hence this is safe to do. 
  const float isqrt2 = 1 / std::sqrt(2);
  const float metres_xy = metres * isqrt2;
  const float inv_metres_xy2 = 1/(metres_xy*metres_xy);
  const float half = nMETsig*metres_xy;
  const float dmet = (2*half)/(nMETsteps-1);
  const float m_tau =  1.77686;

  const float PHI_WIN = 0.4f;                    // requested window
  const float phi1_c  = tau1.Phi();              // visible tau1 azimuth
  const float phi2_c  = tau2.Phi();              // visible tau2 azimuth
  const float dphi1   = (2.f*PHI_WIN) / static_cast<float>(nsteps);
  const float dphi2   = (2.f*PHI_WIN) / static_cast<float>(nsteps);

  // quick function for mass weighting
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau*mTau;
    const float x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2; // ~ 1 - mMiss^2/mt2
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;    // outside physical (approx)
    // Michel prior (V-A) with Jacobian to m_miss;
    const float w_x = x*x*(3.f - 2.f*x);
    const float jac = 2.f*mMiss/mt2;
    return w_x * jac;
  };

  /* 
  MMC implementation that scans over azimuthal angles of each tau 
  with a square grid around MET_x/y - Gauss weighting to penalise large deviations
  */

  for (int i = 0; i < nsteps; i++) {
      const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f)*dphi1);
      for (int j = 0; j < nsteps; j++) {
        const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f)*dphi2);
        // define csc dphi for use in each step in for-loop
        const float sin12 = std::sin(phi2 - phi1);
        if (std::fabs(sin12) < 1e-4f) continue;
        float csc_dphi = 1 / sin12;

        // also now perform MET-scan in grid around measured x/y values
        for (int k = 0; k < nMETsteps; k++) {
          float sMET_x = MET_x - nMETsig * metres_xy + k * dmet;
          for (int l = 0; l < nMETsteps; l++) {
            float sMET_y = MET_y - nMETsig * metres_xy + l * dmet;

            // all of this is doable before placing assumptions on the missing mass
            // since we store all solutions, pre-compute MET weight
            const float dx = sMET_x - MET_x;
            const float dy = sMET_y - MET_y;
          
            // log w_MET = -0.5 * (dx^2/σx^2 + dy^2/σy^2)
            const double log_w_met = -0.5 * (dx*dx * inv_metres_xy2 + dy*dy * inv_metres_xy2);
            const double w_met     = exp(log_w_met); // linear MET weight
            
            // compute pTs of neutrinos
            float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
            float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );
            if (pT1 <= 0.f || pT2 <= 0.f) continue;     // enforce pT > 0 in this window

            
            // now if we have a leptonic leg, need to scan over missing mass
            if (!isLep1 && !isLep2) {
              // using these solutions, solve each quadratic
              std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
              std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);
              for (float z1 : pz1) {
                // neutrino 1 TLV from (pT1, phi1, z1)
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); 
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  // neutrino 2 TLV from (pT2, phi2, z2)
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); 
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  // full tau four-vectors
                  TLorentzVector tau_full_1 = tau1 + nu1;
                  TLorentzVector tau_full_2 = tau2 + nu2;
                  
                  // angular kinematic weight 
                  const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());

                  // require tau 4-momenta
                  const float p_tau1 = tau_full_1.P();
                  const float p_tau2 = tau_full_2.P();

                
                  const float THETA_MAX = 0.3f; 

                  const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                  const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                  const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  const float w_event  = w_theta1 * w_theta2 * w_met;

                  // di-tau invariant mass
                  const float mditau = (tau_full_1 + tau_full_2).M();

                  // store every solution 
                  mditau_solutions.push_back({mditau, w_event});
                }
              }
              
            } else if (isLep1) {
                // m_miss in [0, m_tau - m_vis]; here m_vis ~ m(ℓ) ~ small; use midpoint sampling
                const float mLep = tau1.M();                         // visible lepton mass (≈0 or 0.105)
                const float mMax = std::max(0.f, m_tau - mLep);
                const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);

                for (int t = 0; t < nMass_steps; ++t) {
                  const float m_scan = (t + 0.5f)*dm;               // midpoint
                  if (m_scan <= 0.f) continue;

                  // leptonic prior weight for this m_scan
                  const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                  if (w_miss <= 0.f) continue;

                  std::vector<float> pz1 = solve_pz_quadratic_tau(tau1, phi1, pT1, isLep1,
                                                                  n_charged_tracks_1, m_scan);
                  for (float z1 : pz1) {
                    const float px1 = pT1 * std::cos(phi1);
                    const float py1 = pT1 * std::sin(phi1);
                    const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                    for (float z2 : pz2) {
                      const float px2 = pT2 * std::cos(phi2);
                      const float py2 = pT2 * std::sin(phi2);
                      const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                      TLorentzVector tau_full_1 = tau1 + nu1;
                      TLorentzVector tau_full_2 = tau2 + nu2;

                      const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                      const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                      const float p_tau1     = tau_full_1.P();
                      const float p_tau2     = tau_full_2.P();

                      const float THETA_MAX = 0.3f;
                      const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                      const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                      const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                      const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                      const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; // <-- include prior

                      const float mditau   = (tau_full_1 + tau_full_2).M();
                      mditau_solutions.push_back({mditau, w_event});
                    }
                  }
                }
              } else if (isLep2) {
                  // m_miss in [0, m_tau - m_vis]; here m_vis ~ m(ℓ) ~ small; use midpoint sampling
                  const float mLep = tau2.M();                         // visible lepton mass (≈0 or 0.105)
                  const float mMax = std::max(0.f, m_tau - mLep);
                  const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                  std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);

                  for (int t = 0; t < nMass_steps; ++t) {
                    const float m_scan = (t + 0.5f)*dm;               // midpoint
                    if (m_scan <= 0.f) continue;

                    // leptonic prior weight for this m_scan
                    const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                    if (w_miss <= 0.f) continue;

                    std::vector<float> pz2 = solve_pz_quadratic_tau(tau2, phi2, pT2, isLep2,
                                                                    n_charged_tracks_2, m_scan);
                    for (float z1 : pz1) {
                      const float px1 = pT1 * std::cos(phi1);
                      const float py1 = pT1 * std::sin(phi1);
                      const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                      TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                      for (float z2 : pz2) {
                        const float px2 = pT2 * std::cos(phi2);
                        const float py2 = pT2 * std::sin(phi2);
                        const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                        TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                        TLorentzVector tau_full_1 = tau1 + nu1;
                        TLorentzVector tau_full_2 = tau2 + nu2;

                        const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                        const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                        const float p_tau1     = tau_full_1.P();
                        const float p_tau2     = tau_full_2.P();

                        const float THETA_MAX = 0.3f;
                        const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                        const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                        const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                        const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                        const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; // <-- include prior

                        const float mditau   = (tau_full_1 + tau_full_2).M();
                        mditau_solutions.push_back({mditau, w_event});
                      }
                    }
                  }
                } else {
            // should never make it here..
            std::cout << "Neither? Please check configuration." << std::endl;
          }

          }

        }
        
      }
      // If we found any solutions, don't retry
      if (!mditau_solutions.empty()) break;
  }

  // --- Fallback if MMC scan produced no solutions ---
  if (mditau_solutions.empty()) {
    float m_fb = ditau_mass_collinear_or_vis(tau1, tau2, MET_x, MET_y);
    // Always positive for m_vis at least, but be defensive
    if (m_fb > 0.f && std::isfinite(m_fb)) {
      mditau_solutions.emplace_back(m_fb, 1.0f);
    }
  }

  return mditau_solutions;
}


std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_para_perp(
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
)
{ 
  std::vector<std::pair<float,float>> mditau_solutions;

  // not processing leplep: exit early
  if (isLep1 && isLep2) {
    // do not run if di-leptonic
    return mditau_solutions;
  }
  
  // for had-had we can increase the scanning steps and MET-width
  if (!isLep1 && !isLep2) {
    nMETsteps *= 2;
    nsteps *= 2;
    nMETsig = 10;
  }

  // also apply preselection here, otherwise we spend too long computing configurations that are nonsense
  //if ((n_b_jets_medium != 4) || (n_tau_jets_medium < 1) || (n_tau_jets_medium > 2)) {
  //  return mditau_solutions;
  //}

  auto wrapToPi = [](float a) -> float {
    // robust (-pi, pi] wrapping
    a = std::fmod(a + M_PI, 2.f*M_PI);
    if (a < 0.f) a += 2.f*M_PI;
    return a - M_PI;
  };
    
  // using nstep, define granularity of phi grid scan
  const float dphi = 2.0 * M_PI / static_cast<float>(nsteps);

  // met scanning, dividing by sqrt(2) to go from event -> x/y cmpt met resolutions 
  // extracted correlation(x,y) ~ 0, hence this is safe to do. 
  const float isqrt2 = 1 / std::sqrt(2);
  const float inv_metres_x2 = 1/(metres_x*metres_x);
  const float inv_metres_y2 = 1/(metres_y*metres_y);

  const float halfx = nMETsig*metres_x;
  const float halfy = nMETsig*metres_y;

  const float dmetx = (2*halfx)/(nMETsteps-1);
  const float dmety = (2*halfy)/(nMETsteps-1);

  const float m_tau =  1.77686;

  const float PHI_WIN = 0.4f;                    // requested window
  const float phi1_c  = tau1.Phi();              // visible tau1 azimuth
  const float phi2_c  = tau2.Phi();              // visible tau2 azimuth
  const float dphi1   = (2.f*PHI_WIN) / static_cast<float>(nsteps);
  const float dphi2   = (2.f*PHI_WIN) / static_cast<float>(nsteps);

  // quick function for mass weighting
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau*mTau;
    const float x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2; 
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;    

    const float w_x = x*x*(3.f - 2.f*x);
    const float jac = 2.f*mMiss/mt2;
    return w_x * jac;
  };

  /* 
  MMC implementation that scans over azimuthal angles of each tau 
  with a square grid around MET_x/y - Gauss weighting to penalise large deviations
  */

  for (int i = 0; i < nsteps; i++) {
      const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f)*dphi1);
      for (int j = 0; j < nsteps; j++) {
        const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f)*dphi2);
        // define csc dphi for use in each step in for-loop
        const float sin12 = std::sin(phi2 - phi1);
        if (std::fabs(sin12) < 1e-4f) continue;
        float csc_dphi = 1 / sin12;

        // also now perform MET-scan in grid around measured x/y values
        for (int k = 0; k < nMETsteps; k++) {
          float sMET_x = MET_x - nMETsig * metres_x + k * dmetx;
          for (int l = 0; l < nMETsteps; l++) {
            float sMET_y = MET_y - nMETsig * metres_y + l * dmety;

            // all of this is doable before placing assumptions on the missing mass
            // since we store all solutions, pre-compute MET weight
            const float dx = sMET_x - MET_x;
            const float dy = sMET_y - MET_y;
          
            // log w_MET = -0.5 * (dx^2/σx^2 + dy^2/σy^2)
            const double log_w_met = -0.5 * (dx*dx * inv_metres_x2 + dy*dy * inv_metres_y2);
            const double w_met     = exp(log_w_met); // linear MET weight
            
            // compute pTs of neutrinos
            float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
            float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );
            if (pT1 <= 0.f || pT2 <= 0.f) continue;     // enforce pT > 0 in this window

            
            // now if we have a leptonic leg, need to scan over missing mass
            if (!isLep1 && !isLep2) {
              // using these solutions, solve each quadratic
              std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
              std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);
              for (float z1 : pz1) {
                // neutrino 1 TLV from (pT1, phi1, z1)
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); 
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  // neutrino 2 TLV from (pT2, phi2, z2)
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); 
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  // full tau four-vectors
                  TLorentzVector tau_full_1 = tau1 + nu1;
                  TLorentzVector tau_full_2 = tau2 + nu2;
                  
                  // angular kinematic weight 
                  const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());

                  // require tau 4-momenta
                  const float p_tau1 = tau_full_1.P();
                  const float p_tau2 = tau_full_2.P();

                
                  const float THETA_MAX = 0.3f; 

                  const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;

                  const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  const float w_event  = w_theta1 * w_theta2 * w_met;

                  // di-tau invariant mass
                  const float mditau = (tau_full_1 + tau_full_2).M();

                  // store every solution 
                  mditau_solutions.push_back({mditau, w_event});
                }
              }
              
            } else if (isLep1) {
                // m_miss in [0, m_tau - m_vis]; 
                const float mLep = tau1.M();  // for leptonic use the visible mass                 
                const float mMax = std::max(0.f, m_tau - mLep);
                const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);

                for (int t = 0; t < nMass_steps; ++t) {
                  const float m_scan = (t + 0.5f)*dm;               // midpoint
                  if (m_scan <= 0.f) continue;

                  // leptonic prior weight for this m_scan
                  const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                  if (w_miss <= 0.f) continue;

                  std::vector<float> pz1 = solve_pz_quadratic_tau(tau1, phi1, pT1, isLep1,
                                                                  n_charged_tracks_1, m_scan);
                  for (float z1 : pz1) {
                    const float px1 = pT1 * std::cos(phi1);
                    const float py1 = pT1 * std::sin(phi1);
                    const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1 + m_scan*m_scan);
                    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                    for (float z2 : pz2) {
                      const float px2 = pT2 * std::cos(phi2);
                      const float py2 = pT2 * std::sin(phi2);
                      const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                      TLorentzVector tau_full_1 = tau1 + nu1;
                      TLorentzVector tau_full_2 = tau2 + nu2;

                      const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                      const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                      const float p_tau1     = tau_full_1.P();
                      const float p_tau2     = tau_full_2.P();

                      const float THETA_MAX = 0.3f;
                      const MixCoeffs& K1 = COEFFS_LEPTONIC_THESIS;
                      const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;

                      const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, 0.5);
                      const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                      const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; // <-- include prior

                      const float mditau   = (tau_full_1 + tau_full_2).M();
                      mditau_solutions.push_back({mditau, w_event});
                    }
                  }
                }
              } else if (isLep2) {
                  // m_miss in [0, m_tau - m_vis]; here m_vis ~ m(ℓ) ~ small; use midpoint sampling
                  const float mLep = tau2.M();                         // visible lepton mass (≈0 or 0.105)
                  const float mMax = std::max(0.f, m_tau - mLep);
                  const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                  std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);

                  for (int t = 0; t < nMass_steps; ++t) {
                    const float m_scan = (t + 0.5f)*dm;               // midpoint
                    if (m_scan <= 0.f) continue;

                    // leptonic prior weight for this m_scan
                    const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                    if (w_miss <= 0.f) continue;

                    std::vector<float> pz2 = solve_pz_quadratic_tau(tau2, phi2, pT2, isLep2,
                                                                    n_charged_tracks_2, m_scan);
                    for (float z1 : pz1) {
                      const float px1 = pT1 * std::cos(phi1);
                      const float py1 = pT1 * std::sin(phi1);
                      const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                      TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                      for (float z2 : pz2) {
                        const float px2 = pT2 * std::cos(phi2);
                        const float py2 = pT2 * std::sin(phi2);
                        const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                        TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                        TLorentzVector tau_full_1 = tau1 + nu1;
                        TLorentzVector tau_full_2 = tau2 + nu2;

                        const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                        const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                        const float p_tau1     = tau_full_1.P();
                        const float p_tau2     = tau_full_2.P();

                        const float THETA_MAX = 0.3f;
                        const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                        const MixCoeffs& K2 = COEFFS_LEPTONIC_THESIS;

                        const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                        const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, 0.5);

                        const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; 

                        const float mditau   = (tau_full_1 + tau_full_2).M();
                        mditau_solutions.push_back({mditau, w_event});
                      }
                    }
                  }
                } else {
            // should never make it here..
            std::cout << "Neither? Please check configuration." << std::endl;
          }

          }

        }
        
      }
    }
  // If we found any solutions, don't retry
  // make sure this isn't closing a loop too early!
  //if (!mditau_solutions.empty()) break;

  // --- Fallback if MMC scan produced no solutions ---
  //if (mditau_solutions.empty()) {
  //  float m_fb = ditau_mass_collinear_or_vis(tau1, tau2, MET_x, MET_y);
    // Always positive for m_vis at least, but be defensive
  //  if (m_fb > 0.f && std::isfinite(m_fb)) {
  //    mditau_solutions.emplace_back(m_fb, 1.0f);
  //  }
  //}

  return mditau_solutions;
}



std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_covphi(
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
)
{ 
  std::vector<std::pair<float,float>> mditau_solutions;

  if (isLep1 && isLep2) {
    return mditau_solutions;
  }

  if (!isLep1 && !isLep2) {
    nMETsteps *= 2;
    nsteps *= 2;
    nMETsig = 10;
  }

  auto wrapToPi = [](float a) -> float {
    a = std::fmod(a + M_PI, 2.f*M_PI);
    if (a < 0.f) a += 2.f*M_PI;
    return a - M_PI;
  };

  // Apply ATLAS visible mass convention to tau inputs
  auto setVisMass = [](const TLorentzVector& tlv, bool isLep, int nprong) {
    TLorentzVector out;
    double mvis = tlv.M();
    if (!isLep) {
      mvis = (nprong <= 2) ? 0.8 : 1.2;
    }
    out.SetPtEtaPhiM(tlv.Pt(), tlv.Eta(), tlv.Phi(), mvis);
    return out;
  };
  TLorentzVector vis1 = setVisMass(tau1, isLep1, n_charged_tracks_1);
  TLorentzVector vis2 = setVisMass(tau2, isLep2, n_charged_tracks_2);

  const float phi1_c  = vis1.Phi();
  const float phi2_c  = vis2.Phi();
  const float PHI_WIN = 0.4f;
  float dphi1   = (2.f*PHI_WIN) / static_cast<float>(nsteps);
  float dphi2   = (2.f*PHI_WIN) / static_cast<float>(nsteps);

  const float m_tau = 1.77686f;

  const float halfL = nMETsig * sigma_para;
  const float halfP = nMETsig * sigma_perp;
  const float dL = (2 * halfL) / (nMETsteps - 1);
  const float dP = (2 * halfP) / (nMETsteps - 1);
  const float inv_sigmaL2 = 1.f / (sigma_para * sigma_para);
  const float inv_sigmaP2 = 1.f / (sigma_perp * sigma_perp);
  const float cphi = std::cos(met_cov_phi);
  const float sphi = std::sin(met_cov_phi);

  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau*mTau;
    const float x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2;
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;
    const float w_x = x*x*(3.f - 2.f*x);
    const float jac = 2.f*mMiss/mt2;
    return w_x * jac;
  };

  for (int i = 0; i < nsteps; i++) {
      const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f)*dphi1);
      for (int j = 0; j < nsteps; j++) {
        const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f)*dphi2);

        const float sin12 = std::sin(phi2 - phi1);
        if (std::fabs(sin12) < 1e-4f) continue;
        const float csc_dphi = 1.f / sin12;

        for (int k = 0; k < nMETsteps; k++) {
          const float deltaL = -halfL + k * dL;
          for (int l = 0; l < nMETsteps; l++) {
            const float deltaP = -halfP + l * dP;

            const float sMET_x = MET_x + deltaL * cphi - deltaP * sphi;
            const float sMET_y = MET_y + deltaL * sphi + deltaP * cphi;

            const float w_met = std::exp(-0.5f * (deltaL*deltaL * inv_sigmaL2 + deltaP*deltaP * inv_sigmaP2));

            const float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
            const float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );
            if (pT1 <= 0.f || pT2 <= 0.f) continue;

            if (!isLep1 && !isLep2) {
              std::vector<float> pz1 = solve_pz_quadratic(vis1, phi1, pT1, n_charged_tracks_1);
              std::vector<float> pz2 = solve_pz_quadratic(vis2, phi2, pT2, n_charged_tracks_2);
              for (float z1 : pz1) {
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); 
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); 
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = vis1 + nu1;
                  TLorentzVector tau_full_2 = vis2 + nu2;

                  const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                  const float p_tau1 = tau_full_1.P();
                  const float p_tau2 = tau_full_2.P();

                  const float THETA_MAX = 0.3f; 
                  const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;

                  const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  const float w_event  = w_theta1 * w_theta2 * w_met;

                  const float mditau = (tau_full_1 + tau_full_2).M();
                  mditau_solutions.push_back({mditau, w_event});
                }
              }
              
            } else if (isLep1) {
                const float mLep = vis1.M();                 
                const float mMax = std::max(0.f, m_tau - mLep);
                const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                std::vector<float> pz2 = solve_pz_quadratic(vis2, phi2, pT2, n_charged_tracks_2);

                for (int t = 0; t < nMass_steps; ++t) {
                  const float m_scan = (t + 0.5f)*dm;
                  if (m_scan <= 0.f) continue;

                  const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                  if (w_miss <= 0.f) continue;

                  std::vector<float> pz1 = solve_pz_quadratic_tau(vis1, phi1, pT1, isLep1,
                                                                  n_charged_tracks_1, m_scan);
                  for (float z1 : pz1) {
                    const float px1 = pT1 * std::cos(phi1);
                    const float py1 = pT1 * std::sin(phi1);
                    const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                    for (float z2 : pz2) {
                      const float px2 = pT2 * std::cos(phi2);
                      const float py2 = pT2 * std::sin(phi2);
                      const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                      TLorentzVector tau_full_1 = vis1 + nu1;
                      TLorentzVector tau_full_2 = vis2 + nu2;

                      const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                      const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                      const float p_tau1     = tau_full_1.P();
                      const float p_tau2     = tau_full_2.P();

                      const float THETA_MAX = 0.3f;
                      const MixCoeffs& K1 = COEFFS_LEPTONIC_THESIS;
                      const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;

                      const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, 0.5);
                      const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                      const float w_event  = w_theta1 * w_theta2 * w_met * w_miss;

                      const float mditau   = (tau_full_1 + tau_full_2).M();
                      mditau_solutions.push_back({mditau, w_event});
                    }
                  }
                }
              } else if (isLep2) {
                  const float mLep = vis2.M();
                  const float mMax = std::max(0.f, m_tau - mLep);
                  const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                  std::vector<float> pz1 = solve_pz_quadratic(vis1, phi1, pT1, n_charged_tracks_1);

                  for (int t = 0; t < nMass_steps; ++t) {
                    const float m_scan = (t + 0.5f)*dm;
                    if (m_scan <= 0.f) continue;

                    const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                    if (w_miss <= 0.f) continue;

                    std::vector<float> pz2 = solve_pz_quadratic_tau(vis2, phi2, pT2, isLep2,
                                                                    n_charged_tracks_2, m_scan);
                    for (float z1 : pz1) {
                      const float px1 = pT1 * std::cos(phi1);
                      const float py1 = pT1 * std::sin(phi1);
                      const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                      TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                      for (float z2 : pz2) {
                        const float px2 = pT2 * std::cos(phi2);
                        const float py2 = pT2 * std::sin(phi2);
                        const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                        TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                        TLorentzVector tau_full_1 = vis1 + nu1;
                        TLorentzVector tau_full_2 = vis2 + nu2;

                        const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                        const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                        const float p_tau1     = tau_full_1.P();
                        const float p_tau2     = tau_full_2.P();

                        const float THETA_MAX = 0.3f;
                        const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                        const MixCoeffs& K2 = COEFFS_LEPTONIC_THESIS;

                        const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                        const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, 0.5);

                        const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; 

                        const float mditau   = (tau_full_1 + tau_full_2).M();
                        mditau_solutions.push_back({mditau, w_event});
                      }
                    }
                  }
                } else {
            std::cout << "Neither? Please check configuration." << std::endl;
          }

          }

        }
        
      }
    }

  return mditau_solutions;
}




static std::vector<std::pair<float,float>> solve_ditau_MMC_METScan_para_perp_debugCounters_impl(
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
)
{
  std::vector<std::pair<float,float>> mditau_solutions;
  struct DiagEntry {
    float w_event, mditau;
    float theta1, theta2;
    float denom1, denom2;
    float nuP1, nuP2;
    float pTau1, pTau2;
    float w_theta1, w_theta2;
    float w_ratio1, w_ratio2;
    float w_met, w_miss;
  };
  std::vector<DiagEntry> diag_entries;
  if (diagnostic) diag_entries.reserve(4096);

  // ------------------------------------------------------------
  // Debug counters
  struct MMCScanStats {
    uint64_t n_phi_pairs         = 0;
    uint64_t n_sin12_small       = 0;
    uint64_t n_met_points        = 0;
    uint64_t n_pt_negative       = 0;
    uint64_t n_hadhad_nodes      = 0;
    uint64_t n_lephad_nodes      = 0;
    uint64_t n_pz1_empty         = 0;
    uint64_t n_pz2_empty         = 0;
    uint64_t n_mass_points       = 0;
    uint64_t n_mscan_invalid     = 0;
    uint64_t n_wmiss_zero        = 0;
    uint64_t n_theta_zero        = 0;
    uint64_t n_solutions         = 0;
  } stats;

  auto printStats = [&](const MMCScanStats& s) {
    std::cout
      << "[MMC debug] phi_pairs="   << s.n_phi_pairs
      << " sin12_small="           << s.n_sin12_small
      << " met_points="            << s.n_met_points
      << " pt_negative="           << s.n_pt_negative
      << " hadhad_nodes="          << s.n_hadhad_nodes
      << " lephad_nodes="          << s.n_lephad_nodes
      << " pz1_empty="             << s.n_pz1_empty
      << " pz2_empty="             << s.n_pz2_empty
      << " mass_points="           << s.n_mass_points
      << " mscan_invalid="         << s.n_mscan_invalid
      << " wmiss_zero="            << s.n_wmiss_zero
      << " theta_zero="            << s.n_theta_zero
      << " solutions="             << s.n_solutions
      << std::endl;
  };
  // ------------------------------------------------------------

  // not processing leplep: exit early
  if (isLep1 && isLep2) {
    return mditau_solutions;
  }

  // for had-had we can increase scanning steps and MET-width
  if (!isLep1 && !isLep2) {
    nMETsteps *= 2;
    nsteps    *= 2;
    nMETsig    = 4;
  }

  (void)n_b_jets_medium;
  (void)n_tau_jets_medium;

  auto wrapToPi = [](float a) -> float {
    a = std::fmod(a + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
    if (a < 0.f) a += 2.f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
  };

  /* 
  ------------------------------------------------------------------
  IMPORTANT FIX: visible hadronic mass poor representation of true underlying tau visible mass
  Must therefore assign most-probable mass for each prong-set, 0.8(1.2) for 1(3) prongs.
  Increases baseline had-had efficiency from ~60% to ~95% (without increasing MET scan window)
  If altered you may see significantly worse acceptance, where the algorithm gets 'stuck' in the quadratic solvers
  ------------------------------------------------------------------ 
  */
  auto applyHadVisMassHyp = [](const TLorentzVector& v, int n_charged_tracks) -> TLorentzVector {
    const float m_hyp = (n_charged_tracks <= 2) ? 0.8f : 1.2f; 
    TLorentzVector out;
    // still using all measured visible components, just with an altered mass hypothesis
    out.SetPtEtaPhiM(v.Pt(), v.Eta(), v.Phi(), m_hyp);
    return out;
  };

  // Only apply to hadronic legs. For leptonic legs, keep lepton mass as is
  TLorentzVector tau1_use = tau1;
  // just for sanity checks, see lepton mass
  if (isLep1) std::cout << "Mass of lepton as stored in used tau: " << tau1_use.M() << std::endl;

  TLorentzVector tau2_use = tau2;
  if (!isLep1) tau1_use = applyHadVisMassHyp(tau1, n_charged_tracks_1);
  if (!isLep2) tau2_use = applyHadVisMassHyp(tau2, n_charged_tracks_2);
  // ------------------------------------------------------------------

  // met scanning (x/y) resolutions, important for penalising large deviations
  const float inv_metres_x2 = 1.f / (metres_x * metres_x);
  const float inv_metres_y2 = 1.f / (metres_y * metres_y);

  const float halfx = nMETsig * metres_x;
  const float halfy = nMETsig * metres_y;


  // define step-width based on nsteps in grid with specific met x/y resolution
  // before para/perp calibration this is identical, now with object-based MET-res
  // these should now differ
  const float dmetx = (2.f * halfx) / (nMETsteps - 1);
  const float dmety = (2.f * halfy) / (nMETsteps - 1);

  const float m_tau = 1.77686f;

  const float PHI_WIN = 0.4f;
  const float phi1_c  = tau1_use.Phi();
  const float phi2_c  = tau2_use.Phi();
  const float dphi1   = (2.f * PHI_WIN) / static_cast<float>(nsteps);
  const float dphi2   = (2.f * PHI_WIN) / static_cast<float>(nsteps);

  // leptonic missing-mass prior (Michel mapped to m_miss)
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau * mTau;
    const float x   = (mt2 + mLep * mLep - mMiss * mMiss) / mt2;
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;
    const float w_x = x * x * (3.f - 2.f * x);
    const float jac = 2.f * mMiss / mt2;
    return w_x * jac;
  };

  // ------------------------------------------------------------------
  // Empirical beta-PDF weights for missing momentum fraction R = |p_mis| / |p_tau|
  // Parameters taken from x_misP_fits_beta.csv (leg-specific).
  // TODO: check leptonic branch (p0)
  struct BetaParams { float a; float b; };
  const BetaParams beta_leg1_p1{1.07885234f, 1.63434510f};
  const BetaParams beta_leg1_p3{1.24681582f, 3.68901032f};
  const BetaParams beta_leg1_p0{1.96933834f, 1.11499697f}; 
  const BetaParams beta_leg2_p1{1.07475452f, 1.69395393f};
  const BetaParams beta_leg2_p3{1.24888073f, 3.70853247f};
  const BetaParams beta_leg2_p0{2.02570375f, 1.11420266f}; 

  auto beta_pdf = [](float x, const BetaParams& p) -> float {
    if (!(x > 0.f && x < 1.f)) return 0.f;
    const double a = p.a;
    const double b = p.b;
    const double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double lnpdf = (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x) - lnB;
    return static_cast<float>(std::exp(lnpdf));
  };

  auto pick_beta = [&](int ntracks, bool isLep, int leg) -> BetaParams {
    if (isLep) {
     if (leg == 1) {
      return beta_leg1_p0; // 0 here corresponds to 0 truth charged hadrons
     } else {
      return beta_leg2_p0;
     } 
    }
      // no ratio weight on leptonic leg -> why?
    if (leg == 1) {
      if (ntracks >= 3) return beta_leg1_p3;
      if (ntracks >= 1) return beta_leg1_p1;
      return beta_leg1_p0;
    } else {
      if (ntracks >= 3) return beta_leg2_p3;
      if (ntracks >= 1) return beta_leg2_p1;
      return beta_leg2_p0;
    }
  };

  for (int i = 0; i < nsteps; i++) {
    const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f) * dphi1);

    for (int j = 0; j < nsteps; j++) {
      const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f) * dphi2);

      stats.n_phi_pairs++;

      const float sin12 = std::sin(phi2 - phi1);
      if (std::fabs(sin12) < 1e-4f) {
        stats.n_sin12_small++;
        continue;
      }
      const float csc_dphi = 1.f / sin12;

      for (int k = 0; k < nMETsteps; k++) {
        const float sMET_x = MET_x - nMETsig * metres_x + k * dmetx;

        for (int l = 0; l < nMETsteps; l++) {
          const float sMET_y = MET_y - nMETsig * metres_y + l * dmety;

          stats.n_met_points++;

          const float dx = sMET_x - MET_x;
          const float dy = sMET_y - MET_y;
          const double log_w_met = -0.5 * (dx * dx * inv_metres_x2 + dy * dy * inv_metres_y2);
          const double w_met = std::exp(log_w_met);

          // Solve pT1,pT2 from MET decomposition along phi1,phi2
          float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y * std::cos(phi2) );
          float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y * std::cos(phi1) );

          // Physical requirement: magnitudes positive -> makes wedge in MET / pT space
          if (pT1 <= 0.f || pT2 <= 0.f) {
            stats.n_pt_negative++;
            continue;
          }

          // ---------------------------
          // Had-had
          if (!isLep1 && !isLep2) {
            stats.n_hadhad_nodes++;

            // IMPORTANT: use tau*_use with mass hypotheses to avoid inefficient solvers
            std::vector<float> pz1 = solve_pz_quadratic_tau(tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, 0.0);
            std::vector<float> pz2 = solve_pz_quadratic_tau(tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, 0.0);

            if (pz1.empty()) { stats.n_pz1_empty++; continue; }
            if (pz2.empty()) { stats.n_pz2_empty++; continue; }

            for (float z1 : pz1) {
              const float px1 = pT1 * std::cos(phi1);
              const float py1 = pT1 * std::sin(phi1);
              const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); // no need to worry about mass hypothesis -> is 0 for hadhad
              TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

              for (float z2 : pz2) {
                const float px2 = pT2 * std::cos(phi2);
                const float py2 = pT2 * std::sin(phi2);
                const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); // no need to worry about mass hypothesis -> is 0 for hadhad
                TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                TLorentzVector tau_full_1 = tau1_use + nu1;
                TLorentzVector tau_full_2 = tau2_use + nu2;

                const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                const float p_tau1     = tau_full_1.P();
                const float p_tau2     = tau_full_2.P();

                float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                if (use_atlas_tf1) {
                  const bool lead1 = tau1_use.Pt() >= tau2_use.Pt();
                  const bool lead2 = !lead1;
                  auto w = atlas_tf1_weights(
                    tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                    tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                  );
                  w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                  if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                } else {
                  const float THETA_MAX = 0.3f;
                  const MixCoeffs& K1 =
                    (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  const MixCoeffs& K2 =
                    (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                  const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                  const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                  w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                  w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                  if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                }
                const float w_miss = 1.f;
                // also check when we're using the ATLAS weighting:
                //std::cout << "checking the derived weights from the ATLAS implementation. w_met: " << w_met <<
                //" w_miss: " << w_miss << " w_theta1: " << w_theta1 << " w_theta2: " << w_theta2 << 
                //" w_ratio1: " << w_ratio1 << " w_ratio1: " << w_ratio1 << std::endl;
                const float w_event = use_atlas_tf1
                  ? static_cast<float>(w_met * w_ratio1 * w_ratio2 * w_theta1 * w_theta2)
                  : static_cast<float>(w_theta1 * w_theta2 * w_met * w_ratio1 * w_ratio2);
                const float mditau  = (tau_full_1 + tau_full_2).M();
                if (diagnostic) {
                  const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                  const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                  diag_entries.push_back(
                    {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                     static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                     p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                     static_cast<float>(w_met), w_miss}
                  );
                }
                if (weight_components) {
                  weight_components->push_back(
                    {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                     static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                  );
                }

                mditau_solutions.push_back({mditau, w_event});
                stats.n_solutions++;
              }
            }

          // ---------------------------
          // Lep-had (leg1 leptonic)
          } else if (isLep1) {
            stats.n_lephad_nodes++;

            const float mLep = tau1_use.M(); // leptonic leg unchanged
            const float mMax = std::max(0.f, m_tau - mLep);
            const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;
            if (dm <= 0.f) { stats.n_mscan_invalid++; continue; }

            std::vector<float> pz2 = solve_pz_quadratic_tau(tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, 0.0);
            if (pz2.empty()) { stats.n_pz2_empty++; continue; }

            for (int t = 0; t < nMass_steps; ++t) {
              stats.n_mass_points++;

              const float m_scan = (t + 0.5f) * dm;
              if (m_scan <= 0.f) { stats.n_mscan_invalid++; continue; }

              const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
              if (w_miss <= 0.f) { stats.n_wmiss_zero++; continue; }

              std::vector<float> pz1 = solve_pz_quadratic_tau(
                tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, m_scan
              );
              if (pz1.empty()) { stats.n_pz1_empty++; continue; }

              for (float z1 : pz1) {
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1 + m_scan*m_scan); // need missing mass included for leptonic legs
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = tau1_use + nu1;
                  TLorentzVector tau_full_2 = tau2_use + nu2;

                  const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                  const float p_tau1     = tau_full_1.P();
                  const float p_tau2     = tau_full_2.P();

                  float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                  if (use_atlas_tf1) {
                    const bool lead1 = true; // leptonic leg treated as leading by construction
                    const bool lead2 = false;
                    auto w = atlas_tf1_weights(
                      tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                      tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                    );
                    w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                    if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                  } else {
                    const float THETA_MAX = 0.3f;
                    const MixCoeffs& K1 = COEFFS_LEPTONIC_THESIS;
                    const MixCoeffs& K2 =
                      (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                    w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, 0.5f);
                    w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                    if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                    const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                    const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                    w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                    w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                    if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                  }

                  const float w_event = use_atlas_tf1
                    ? static_cast<float>(w_met * w_miss * w_theta1 * w_theta2 * w_ratio1 * w_ratio2)
                    : static_cast<float>(w_theta1 * w_theta2 * w_met * w_miss * w_ratio1 * w_ratio2);
                  const float mditau  = (tau_full_1 + tau_full_2).M();
                  if (diagnostic) {
                    const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                    const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                    diag_entries.push_back(
                      {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                       static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                       p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                       static_cast<float>(w_met), w_miss}
                    );
                  }
                  if (weight_components) {
                    weight_components->push_back(
                      {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                       static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                    );
                  }

                  mditau_solutions.push_back({mditau, w_event});
                  stats.n_solutions++;
                }
              }
            }

          // ---------------------------
          // Had-lep (leg2 leptonic)
          } else if (isLep2) {
            stats.n_lephad_nodes++;

            const float mLep = tau2_use.M();
            const float mMax = std::max(0.f, m_tau - mLep);
            // can check the lepton measured mass:
            //std::cout << "mass of the reconstructed leptons " << mLep << std::endl;
            const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;
            if (dm <= 0.f) { stats.n_mscan_invalid++; continue; }

            std::vector<float> pz1 = solve_pz_quadratic_tau(tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, 0.0);
            if (pz1.empty()) { stats.n_pz1_empty++; continue; }

            for (int t = 0; t < nMass_steps; ++t) {
              stats.n_mass_points++;

              const float m_scan = (t + 0.5f) * dm;
              if (m_scan <= 0.f) { stats.n_mscan_invalid++; continue; }

              const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
              if (w_miss <= 0.f) { stats.n_wmiss_zero++; continue; }

              std::vector<float> pz2 = solve_pz_quadratic_tau(
                tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, m_scan
              );
              if (pz2.empty()) { stats.n_pz2_empty++; continue; }

              for (float z1 : pz1) {
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2 + m_scan*m_scan);
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = tau1_use + nu1;
                  TLorentzVector tau_full_2 = tau2_use + nu2;

                  const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                  const float p_tau1     = tau_full_1.P();
                  const float p_tau2     = tau_full_2.P();

                  float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                  if (use_atlas_tf1) {
                    const bool lead1 = false;
                    const bool lead2 = true;
                    auto w = atlas_tf1_weights(
                      tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                      tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                    );
                    w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                    if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                  } else {
                    const float THETA_MAX = 0.3f;
                    const MixCoeffs& K1 =
                      (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                    const MixCoeffs& K2 = COEFFS_LEPTONIC_THESIS;
                    w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                    w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, 0.5f);
                    if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                    const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                    const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                    w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                    w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                    if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                  }
                  
                  const float w_event = use_atlas_tf1
                    ? static_cast<float>(w_met * w_miss * w_theta1 * w_theta2 * w_ratio1 * w_ratio2)
                    : static_cast<float>(w_theta1 * w_theta2 * w_met * w_miss * w_ratio1 * w_ratio2);
                  const float mditau  = (tau_full_1 + tau_full_2).M();
                  if (diagnostic) {
                    const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                    const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                    diag_entries.push_back(
                      {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                       static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                       p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                       static_cast<float>(w_met), w_miss}
                    );
                  }
                  if (weight_components) {
                    weight_components->push_back(
                      {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                       static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                    );
                  }

                  mditau_solutions.push_back({mditau, w_event});
                  stats.n_solutions++;
                }
              }
            }

          } else {
            std::cout << "[MMC debug] Neither lep/had? Check configuration." << std::endl;
          }

        } // MET_y
      }   // MET_x
    }     // phi2
  }       // phi1

  // Debug summary for problematic events
  if (mditau_solutions.empty()) {
    std::cout << "[MMC debug] NO SOLUTIONS"
              << " isLep1=" << isLep1 << " isLep2=" << isLep2
              << " Tau1(raw): pT=" << tau1.Pt() << " eta=" << tau1.Eta() << " phi=" << tau1.Phi()
              << " E=" << tau1.E() << " M=" << tau1.M()
              << " Tau1(use): pT=" << tau1_use.Pt() << " eta=" << tau1_use.Eta() << " phi=" << tau1_use.Phi()
              << " E=" << tau1_use.E() << " M=" << tau1_use.M()
              << " Tau2(raw): pT=" << tau2.Pt() << " eta=" << tau2.Eta() << " phi=" << tau2.Phi()
              << " E=" << tau2.E() << " M=" << tau2.M()
              << " Tau2(use): pT=" << tau2_use.Pt() << " eta=" << tau2_use.Eta() << " phi=" << tau2_use.Phi()
              << " E=" << tau2_use.E() << " M=" << tau2_use.M()
              << " MET=(" << MET_x << "," << MET_y << ")"
              << " phi_vis=(" << tau1_use.Phi() << "," << tau2_use.Phi() << ")"
              << " nsteps=" << nsteps << " nMETsteps=" << nMETsteps
              << " nMass_steps=" << nMass_steps
              << std::endl;
    printStats(stats);
  } else if (mditau_solutions.size() < 20) {
    std::cout << "[MMC debug] FEW SOLUTIONS: " << mditau_solutions.size() << std::endl;
    printStats(stats);
  }

  if (diagnostic && !diag_entries.empty()) {
    std::sort(diag_entries.begin(), diag_entries.end(),
              [](const DiagEntry& a, const DiagEntry& b) { return a.w_event > b.w_event; });
    const int nPrint = std::min<int>(diag_topN, diag_entries.size());
    std::cout << "[MMC diagnostic] Top " << nPrint << " solutions by weight (w_event)" << std::endl;
    for (int i = 0; i < nPrint; ++i) {
      const auto& d = diag_entries[i];
      std::cout << "  #" << i
                << " mditau=" << d.mditau
                << " w_event=" << d.w_event
                << " w_met=" << d.w_met
                << " w_theta=(" << d.w_theta1 << "," << d.w_theta2 << ")"
                << " w_ratio=(" << d.w_ratio1 << "," << d.w_ratio2 << ")"
                << " w_miss=" << d.w_miss
                << " theta_3D=(" << d.theta1 << "," << d.theta2 << ")"
                << " denom=(" << d.denom1 << "," << d.denom2 << ")"
                << " nuP=(" << d.nuP1 << "," << d.nuP2 << ")"
                << " p_tau=(" << d.pTau1 << "," << d.pTau2 << ")"
                << std::endl;
    }
  }

  return mditau_solutions;
}

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
)
{
  std::vector<std::pair<float,float>> mditau_solutions;
  struct DiagEntry {
    float w_event, mditau;
    float theta1, theta2;
    float denom1, denom2;
    float nuP1, nuP2;
    float pTau1, pTau2;
    float w_theta1, w_theta2;
    float w_ratio1, w_ratio2;
    float w_met, w_miss;
  };
  std::vector<DiagEntry> diag_entries;
  if (diagnostic) diag_entries.reserve(4096);

  // ------------------------------------------------------------
  // Debug counters
  struct MMCScanStats {
    uint64_t n_phi_pairs         = 0;
    uint64_t n_sin12_small       = 0;
    uint64_t n_met_points        = 0;
    uint64_t n_pt_negative       = 0;
    uint64_t n_hadhad_nodes      = 0;
    uint64_t n_lephad_nodes      = 0;
    uint64_t n_pz1_empty         = 0;
    uint64_t n_pz2_empty         = 0;
    uint64_t n_mass_points       = 0;
    uint64_t n_mscan_invalid     = 0;
    uint64_t n_wmiss_zero        = 0;
    uint64_t n_theta_zero        = 0;
    uint64_t n_solutions         = 0;
  } stats;

  auto printStats = [&](const MMCScanStats& s) {
    std::cout
      << "[MMC debug] phi_pairs="   << s.n_phi_pairs
      << " sin12_small="           << s.n_sin12_small
      << " met_points="            << s.n_met_points
      << " pt_negative="           << s.n_pt_negative
      << " hadhad_nodes="          << s.n_hadhad_nodes
      << " lephad_nodes="          << s.n_lephad_nodes
      << " pz1_empty="             << s.n_pz1_empty
      << " pz2_empty="             << s.n_pz2_empty
      << " mass_points="           << s.n_mass_points
      << " mscan_invalid="         << s.n_mscan_invalid
      << " wmiss_zero="            << s.n_wmiss_zero
      << " theta_zero="            << s.n_theta_zero
      << " solutions="             << s.n_solutions
      << std::endl;
  };
  // ------------------------------------------------------------

  // not processing leplep: exit early
  if (isLep1 && isLep2) {
    return mditau_solutions;
  }

  // for had-had we can increase scanning steps and MET-width
  //if (!isLep1 && !isLep2) {
  //  nMETsteps *= 2;
  //  nsteps    *= 2;
  //  nMETsig    = 4;
  //}

  (void)n_b_jets_medium;
  (void)n_tau_jets_medium;

  auto wrapToPi = [](float a) -> float {
    a = std::fmod(a + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
    if (a < 0.f) a += 2.f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
  };

  /* 
  ------------------------------------------------------------------
  IMPORTANT FIX: visible hadronic mass poor representation of true underlying tau visible mass
  Must therefore assign most-probable mass for each prong-set, 0.8(1.2) for 1(3) prongs.
  Increases baseline had-had efficiency from ~60% to ~95% (without increasing MET scan window)
  If altered you may see significantly worse acceptance, where the algorithm gets 'stuck' in the quadratic solvers
  ------------------------------------------------------------------ 
  */
  auto applyHadVisMassHyp = [](const TLorentzVector& v, int n_charged_tracks) -> TLorentzVector {
    const float m_hyp = (n_charged_tracks <= 2) ? 0.8f : 1.2f; 
    TLorentzVector out;
    // still using all measured visible components, just with an altered mass hypothesis
    out.SetPtEtaPhiM(v.Pt(), v.Eta(), v.Phi(), m_hyp);
    return out;
  };

  // Only apply to hadronic legs. For leptonic legs, keep lepton mass as is
  TLorentzVector tau1_use = tau1;
  // just for sanity checks, see lepton mass
  if (isLep1) std::cout << "Mass of lepton as stored in used tau: " << tau1_use.M() << std::endl;

  TLorentzVector tau2_use = tau2;
  if (!isLep1) tau1_use = applyHadVisMassHyp(tau1, n_charged_tracks_1);
  if (!isLep2) tau2_use = applyHadVisMassHyp(tau2, n_charged_tracks_2);
  // ------------------------------------------------------------------

  // met scanning (x/y) resolutions, important for penalising large deviations
  const float inv_metres_x2 = 1.f / (metres_x * metres_x);
  const float inv_metres_y2 = 1.f / (metres_y * metres_y);

  const float halfx = nMETsig * metres_x;
  const float halfy = nMETsig * metres_y;

  // define step-width based on nsteps in grid with specific met x/y resolution
  // before para/perp calibration this is identical, now with object-based MET-res
  // these should now differ
  const float dmetx = (2.f * halfx) / (nMETsteps - 1);
  const float dmety = (2.f * halfy) / (nMETsteps - 1);

  const float m_tau = 1.77686f;

  const float PHI_WIN = 0.4f;
  const float phi1_c  = tau1_use.Phi();
  const float phi2_c  = tau2_use.Phi();
  const float dphi1   = (2.f * PHI_WIN) / static_cast<float>(nsteps);
  const float dphi2   = (2.f * PHI_WIN) / static_cast<float>(nsteps);

  // leptonic missing-mass prior (Michel mapped to m_miss)
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau * mTau;
    const float x   = (mt2 + mLep * mLep - mMiss * mMiss) / mt2;
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;
    const float w_x = x * x * (3.f - 2.f * x);
    const float jac = 2.f * mMiss / mt2;
    return w_x * jac;
  };

  // ------------------------------------------------------------------
  // Empirical beta-PDF weights for missing momentum fraction R = |p_mis| / |p_tau|
  // Parameters taken from x_misP_fits_beta
  // TODO: check leptonic branch (p0)
  struct BetaParams { float a; float b; };
  const BetaParams beta_leg1_p1{1.07885234f, 1.63434510f};
  const BetaParams beta_leg1_p3{1.24681582f, 3.68901032f};
  const BetaParams beta_leg1_p0{1.96933834f, 1.11499697f}; 
  const BetaParams beta_leg2_p1{1.07475452f, 1.69395393f};
  const BetaParams beta_leg2_p3{1.24888073f, 3.70853247f};
  const BetaParams beta_leg2_p0{2.02570375f, 1.11420266f}; 

  auto beta_pdf = [](float x, const BetaParams& p) -> float {
    if (!(x > 0.f && x < 1.f)) return 0.f;
    const double a = p.a;
    const double b = p.b;
    const double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double lnpdf = (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x) - lnB;
    return static_cast<float>(std::exp(lnpdf));
  };

  auto pick_beta = [&](int ntracks, bool isLep, int leg) -> BetaParams {
    if (isLep) {
     if (leg == 1) {
      return beta_leg1_p0; // 0 here corresponds to 0 truth charged hadrons
     } else {
      return beta_leg2_p0;
     } 
    }
      // no ratio weight on leptonic leg -> why?
    if (leg == 1) {
      if (ntracks >= 3) return beta_leg1_p3;
      if (ntracks >= 1) return beta_leg1_p1;
      return beta_leg1_p0;
    } else {
      if (ntracks >= 3) return beta_leg2_p3;
      if (ntracks >= 1) return beta_leg2_p1;
      return beta_leg2_p0;
    }
  };

  for (int i = 0; i < nsteps; i++) {
    const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f) * dphi1);

    for (int j = 0; j < nsteps; j++) {
      const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f) * dphi2);

      stats.n_phi_pairs++;

      const float sin12 = std::sin(phi2 - phi1);
      if (std::fabs(sin12) < 1e-4f) {
        stats.n_sin12_small++;
        continue;
      }
      const float csc_dphi = 1.f / sin12;

      for (int k = 0; k < nMETsteps; k++) {
        const float sMET_x = MET_x - nMETsig * metres_x + k * dmetx;

        for (int l = 0; l < nMETsteps; l++) {
          const float sMET_y = MET_y - nMETsig * metres_y + l * dmety;

          stats.n_met_points++;

          const float dx = sMET_x - MET_x;
          const float dy = sMET_y - MET_y;
          const double log_w_met = -0.5 * (dx * dx * inv_metres_x2 + dy * dy * inv_metres_y2);
          const double w_met = std::exp(log_w_met);

          // Solve pT1,pT2 from MET decomposition along phi1,phi2
          float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y * std::cos(phi2) );
          float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y * std::cos(phi1) );

          // Physical requirement: magnitudes positive -> makes wedge in MET / pT space 
          // i.e. force MET to lie in between the two proposed neutrino directions in the transverse plane
          if (pT1 <= 0.f || pT2 <= 0.f) {
            stats.n_pt_negative++;
            continue;
          }

          // ---------------------------
          // Had-had
          if (!isLep1 && !isLep2) {
            stats.n_hadhad_nodes++;

            // IMPORTANT: use tau*_use with mass hypotheses to avoid inefficient solvers
            std::vector<float> pz1 = solve_pz_quadratic_tau(tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, 0.0);
            std::vector<float> pz2 = solve_pz_quadratic_tau(tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, 0.0);

            if (pz1.empty()) { stats.n_pz1_empty++; continue; }
            if (pz2.empty()) { stats.n_pz2_empty++; continue; }

            for (float z1 : pz1) {
              const float px1 = pT1 * std::cos(phi1);
              const float py1 = pT1 * std::sin(phi1);
              const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); // no need to worry about mass hypothesis -> is 0 for hadhad
              TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

              for (float z2 : pz2) {
                const float px2 = pT2 * std::cos(phi2);
                const float py2 = pT2 * std::sin(phi2);
                const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); // no need to worry about mass hypothesis -> is 0 for hadhad
                TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                TLorentzVector tau_full_1 = tau1_use + nu1;
                TLorentzVector tau_full_2 = tau2_use + nu2;

                const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                const float p_tau1     = tau_full_1.P();
                const float p_tau2     = tau_full_2.P();

                float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                if (use_atlas_tf1) {
                  const bool lead1 = tau1_use.Pt() >= tau2_use.Pt();
                  const bool lead2 = !lead1;
                  auto w = atlas_tf1_weights(
                    tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                    tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                  );
                  w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                  if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                } else {
                  const float THETA_MAX = 0.3f;
                  const MixCoeffs& K1 =
                    (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  const MixCoeffs& K2 =
                    (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                  //w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  //w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  w_theta1 = lognormal_pdf_trunc(theta_3D_1, tau1_use.Pt(), n_charged_tracks_1);
                  w_theta2 = lognormal_pdf_trunc(theta_3D_2, tau2_use.Pt(), n_charged_tracks_2);

                  if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                  const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                  const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                  w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                  w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                  if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                }
                const float w_miss = 1.f;
                // also check when we're using the ATLAS weighting:
                //std::cout << "checking the derived weights from the ATLAS implementation. w_met: " << w_met <<
                //" w_miss: " << w_miss << " w_theta1: " << w_theta1 << " w_theta2: " << w_theta2 << 
                //" w_ratio1: " << w_ratio1 << " w_ratio1: " << w_ratio1 << std::endl;
                const float w_event = use_atlas_tf1
                  ? static_cast<float>(w_met * w_ratio1 * w_ratio2 * w_theta1 * w_theta2)
                  : static_cast<float>(w_theta1 * w_theta2 * w_met * w_ratio1 * w_ratio2);
                const float mditau  = (tau_full_1 + tau_full_2).M();
                if (diagnostic) {
                  const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                  const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                  diag_entries.push_back(
                    {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                     static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                     p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                     static_cast<float>(w_met), w_miss}
                  );
                }
                if (weight_components) {
                  weight_components->push_back(
                    {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                     static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                  );
                }

                mditau_solutions.push_back({mditau, w_event});
                stats.n_solutions++;
              }
            }

          // ---------------------------
          // Lep-had (leg1 leptonic)
          } else if (isLep1) {
            stats.n_lephad_nodes++;

            const float mLep = tau1_use.M(); // leptonic leg unchanged
            const float mMax = std::max(0.f, m_tau - mLep);
            const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;
            if (dm <= 0.f) { stats.n_mscan_invalid++; continue; }

            std::vector<float> pz2 = solve_pz_quadratic_tau(tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, 0.0);
            if (pz2.empty()) { stats.n_pz2_empty++; continue; }

            for (int t = 0; t < nMass_steps; ++t) {
              stats.n_mass_points++;

              const float m_scan = (t + 0.5f) * dm;
              if (m_scan <= 0.f) { stats.n_mscan_invalid++; continue; }

              const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
              if (w_miss <= 0.f) { stats.n_wmiss_zero++; continue; }

              std::vector<float> pz1 = solve_pz_quadratic_tau(
                tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, m_scan
              );
              if (pz1.empty()) { stats.n_pz1_empty++; continue; }

              for (float z1 : pz1) {
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1 + m_scan*m_scan); // need missing mass included for leptonic legs
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = tau1_use + nu1;
                  TLorentzVector tau_full_2 = tau2_use + nu2;

                  const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                  const float p_tau1     = tau_full_1.P();
                  const float p_tau2     = tau_full_2.P();

                  float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                  if (use_atlas_tf1) {
                    const bool lead1 = true; // leptonic leg treated as leading by construction
                    const bool lead2 = false;
                    auto w = atlas_tf1_weights(
                      tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                      tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                    );
                    w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                    if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                  } else {
                    
                    w_theta1 = lognormal_pdf_trunc(theta_3D_1, tau1_use.Pt(), -1);
                    w_theta2 = lognormal_pdf_trunc(theta_3D_2, tau2_use.Pt(), n_charged_tracks_2);

                    if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                    const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                    const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                    w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                    w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                    if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                  }

                  const float w_event = use_atlas_tf1
                    ? static_cast<float>(w_met * w_miss * w_theta1 * w_theta2 * w_ratio1 * w_ratio2)
                    : static_cast<float>(w_theta1 * w_theta2 * w_met * w_miss * w_ratio1 * w_ratio2);
                  const float mditau  = (tau_full_1 + tau_full_2).M();
                  if (diagnostic) {
                    const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                    const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                    diag_entries.push_back(
                      {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                       static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                       p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                       static_cast<float>(w_met), w_miss}
                    );
                  }
                  if (weight_components) {
                    weight_components->push_back(
                      {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                       static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                    );
                  }

                  mditau_solutions.push_back({mditau, w_event});
                  stats.n_solutions++;
                }
              }
            }

          // ---------------------------
          // Had-lep (leg2 leptonic)
          } else if (isLep2) {
            stats.n_lephad_nodes++;

            const float mLep = tau2_use.M();
            const float mMax = std::max(0.f, m_tau - mLep);
            // can check the lepton measured mass:
            //std::cout << "mass of the reconstructed leptons " << mLep << std::endl;
            const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;
            if (dm <= 0.f) { stats.n_mscan_invalid++; continue; }

            std::vector<float> pz1 = solve_pz_quadratic_tau(tau1_use, phi1, pT1, isLep1, n_charged_tracks_1, 0.0);
            if (pz1.empty()) { stats.n_pz1_empty++; continue; }

            for (int t = 0; t < nMass_steps; ++t) {
              stats.n_mass_points++;

              const float m_scan = (t + 0.5f) * dm;
              if (m_scan <= 0.f) { stats.n_mscan_invalid++; continue; }

              const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
              if (w_miss <= 0.f) { stats.n_wmiss_zero++; continue; }

              std::vector<float> pz2 = solve_pz_quadratic_tau(
                tau2_use, phi2, pT2, isLep2, n_charged_tracks_2, m_scan
              );
              if (pz2.empty()) { stats.n_pz2_empty++; continue; }

              for (float z1 : pz1) {
                const float px1 = pT1 * std::cos(phi1);
                const float py1 = pT1 * std::sin(phi1);
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                for (float z2 : pz2) {
                  const float px2 = pT2 * std::cos(phi2);
                  const float py2 = pT2 * std::sin(phi2);
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2 + m_scan*m_scan);
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = tau1_use + nu1;
                  TLorentzVector tau_full_2 = tau2_use + nu2;

                  const float theta_3D_1 = tau1_use.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2_use.Vect().Angle(nu2.Vect());
                  const float p_tau1     = tau_full_1.P();
                  const float p_tau2     = tau_full_2.P();

                  float w_theta1=0.f, w_theta2=0.f, w_ratio1=0.f, w_ratio2=0.f;
                  if (use_atlas_tf1) {
                    const bool lead1 = false;
                    const bool lead2 = true;
                    auto w = atlas_tf1_weights(
                      tau1_use, nu1, isLep1, n_charged_tracks_1, lead1,
                      tau2_use, nu2, isLep2, n_charged_tracks_2, lead2
                    );
                    w_theta1 = w[1]; w_theta2 = w[2]; w_ratio1 = w[3]; w_ratio2 = w[4];
                    if (w[0] <= 0.f) { stats.n_theta_zero++; continue; }
                  } else {
                    w_theta1 = lognormal_pdf_trunc(theta_3D_1, tau1_use.Pt(), n_charged_tracks_1);
                    w_theta2 = lognormal_pdf_trunc(theta_3D_2, tau2_use.Pt(), -1);

                    if (w_theta1 <= 0.f || w_theta2 <= 0.f) { stats.n_theta_zero++; continue; }
                    const float r1 = (p_tau1 > 0.f) ? (nu1.P() / p_tau1) : 0.f;
                    const float r2 = (p_tau2 > 0.f) ? (nu2.P() / p_tau2) : 0.f;
                    w_ratio1 = beta_pdf(r1, pick_beta(n_charged_tracks_1, isLep1, 1));
                    w_ratio2 = beta_pdf(r2, pick_beta(n_charged_tracks_2, isLep2, 2));
                    if (w_ratio1 <= 0.f || w_ratio2 <= 0.f) { stats.n_theta_zero++; continue; }
                  }
                  
                  const float w_event = use_atlas_tf1
                    ? static_cast<float>(w_met * w_miss * w_theta1 * w_theta2 * w_ratio1 * w_ratio2)
                    : static_cast<float>(w_theta1 * w_theta2 * w_met * w_miss * w_ratio1 * w_ratio2);
                  const float mditau  = (tau_full_1 + tau_full_2).M();
                  if (diagnostic) {
                    const float denom1 = tau1_use.E() - tau1_use.P() * std::cos(theta_3D_1);
                    const float denom2 = tau2_use.E() - tau2_use.P() * std::cos(theta_3D_2);
                    diag_entries.push_back(
                      {w_event, mditau, theta_3D_1, theta_3D_2, denom1, denom2,
                       static_cast<float>(nu1.P()), static_cast<float>(nu2.P()),
                       p_tau1, p_tau2, w_theta1, w_theta2, w_ratio1, w_ratio2,
                       static_cast<float>(w_met), w_miss}
                    );
                  }
                  if (weight_components) {
                    weight_components->push_back(
                      {mditau, static_cast<float>(w_met), static_cast<float>(w_theta1), static_cast<float>(w_theta2),
                       static_cast<float>(w_ratio1), static_cast<float>(w_ratio2), static_cast<float>(w_miss), static_cast<float>(w_event)}
                    );
                  }

                  mditau_solutions.push_back({mditau, w_event});
                  stats.n_solutions++;
                }
              }
            }

          } else {
            std::cout << "[MMC debug] Neither lep/had? Check configuration." << std::endl;
          }

        } // MET_y
      }   // MET_x
    }     // phi2
  }       // phi1

  // Debug summary for problematic events
  if (mditau_solutions.empty()) {
    std::cout << "[MMC debug] NO SOLUTIONS"
              << " isLep1=" << isLep1 << " isLep2=" << isLep2
              << " Tau1(raw): pT=" << tau1.Pt() << " eta=" << tau1.Eta() << " phi=" << tau1.Phi()
              << " E=" << tau1.E() << " M=" << tau1.M()
              << " Tau1(use): pT=" << tau1_use.Pt() << " eta=" << tau1_use.Eta() << " phi=" << tau1_use.Phi()
              << " E=" << tau1_use.E() << " M=" << tau1_use.M()
              << " Tau2(raw): pT=" << tau2.Pt() << " eta=" << tau2.Eta() << " phi=" << tau2.Phi()
              << " E=" << tau2.E() << " M=" << tau2.M()
              << " Tau2(use): pT=" << tau2_use.Pt() << " eta=" << tau2_use.Eta() << " phi=" << tau2_use.Phi()
              << " E=" << tau2_use.E() << " M=" << tau2_use.M()
              << " MET=(" << MET_x << "," << MET_y << ")"
              << " phi_vis=(" << tau1_use.Phi() << "," << tau2_use.Phi() << ")"
              << " nsteps=" << nsteps << " nMETsteps=" << nMETsteps
              << " nMass_steps=" << nMass_steps
              << std::endl;
    printStats(stats);
  } else if (mditau_solutions.size() < 20) {
    std::cout << "[MMC debug] FEW SOLUTIONS: " << mditau_solutions.size() << std::endl;
    printStats(stats);
  }

  if (diagnostic && !diag_entries.empty()) {
    std::sort(diag_entries.begin(), diag_entries.end(),
              [](const DiagEntry& a, const DiagEntry& b) { return a.w_event > b.w_event; });
    const int nPrint = std::min<int>(diag_topN, diag_entries.size());
    std::cout << "[MMC diagnostic] Top " << nPrint << " solutions by weight (w_event)" << std::endl;
    for (int i = 0; i < nPrint; ++i) {
      const auto& d = diag_entries[i];
      std::cout << "  #" << i
                << " mditau=" << d.mditau
                << " w_event=" << d.w_event
                << " w_met=" << d.w_met
                << " w_theta=(" << d.w_theta1 << "," << d.w_theta2 << ")"
                << " w_ratio=(" << d.w_ratio1 << "," << d.w_ratio2 << ")"
                << " w_miss=" << d.w_miss
                << " theta_3D=(" << d.theta1 << "," << d.theta2 << ")"
                << " denom=(" << d.denom1 << "," << d.denom2 << ")"
                << " nuP=(" << d.nuP1 << "," << d.nuP2 << ")"
                << " p_tau=(" << d.pTau1 << "," << d.pTau2 << ")"
                << std::endl;
    }
  }

  return mditau_solutions;
}

std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_debugCounters(
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
  int diag_topN
) {
  return solve_ditau_MMC_METScan_para_perp_debugCounters_impl(
    tau1, tau2, isLep1, isLep2, n_charged_tracks_1, n_charged_tracks_2,
    MET_x, MET_y, nsteps, metres_x, metres_y, nMETsig, nMETsteps, nMass_steps,
    n_b_jets_medium, n_tau_jets_medium, weight_components, diagnostic, diag_topN, /*use_atlas_tf1=*/false
  );
}

MMCSolutionsAndWeights AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_vispTanglecalibration_weights(
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
) {
  std::vector<std::array<float,8>> weight_components;
  auto solutions = ::solve_ditau_MMC_METScan_para_perp_vispTAngleCalibration(
    tau1, tau2, isLep1, isLep2, n_charged_tracks_1, n_charged_tracks_2,
    MET_x, MET_y, nsteps, metres_x, metres_y, nMETsig, nMETsteps, nMass_steps,
    n_b_jets_medium, n_tau_jets_medium, &weight_components, diagnostic, diag_topN, /*use_atlas_tf1=*/false
  );
  return {std::move(solutions), std::move(weight_components)};
}

std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_ATLAS_tf1(
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
  int diag_topN
) {
  return solve_ditau_MMC_METScan_para_perp_debugCounters_impl(
    tau1, tau2, isLep1, isLep2, n_charged_tracks_1, n_charged_tracks_2,
    MET_x, MET_y, nsteps, metres_x, metres_y, nMETsig, nMETsteps, nMass_steps,
    n_b_jets_medium, n_tau_jets_medium, weight_components, diagnostic, diag_topN, /*use_atlas_tf1=*/false
  );
}


MMCSolutionsAndWeights AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_withWeights(
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
)
{
  std::vector<std::array<float,8>> weight_components;
  auto solutions = solve_ditau_MMC_METScan_para_perp_debugCounters(
    tau1, tau2, isLep1, isLep2, n_charged_tracks_1, n_charged_tracks_2,
    MET_x, MET_y, nsteps, metres_x, metres_y, nMETsig, nMETsteps, nMass_steps,
    n_b_jets_medium, n_tau_jets_medium, &weight_components, diagnostic, diag_topN
  );
  return {std::move(solutions), std::move(weight_components)};
}

std::vector<std::array<float,8>> AnalysisFCChh::solve_ditau_MMC_METScan_para_perp_weights(
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
)
{
  std::vector<std::array<float,8>> weight_components;
  solve_ditau_MMC_METScan_para_perp_debugCounters(
    tau1, tau2, isLep1, isLep2, n_charged_tracks_1, n_charged_tracks_2,
    MET_x, MET_y, nsteps, metres_x, metres_y, nMETsig, nMETsteps, nMass_steps,
    n_b_jets_medium, n_tau_jets_medium, &weight_components, diagnostic, diag_topN
  );
  return weight_components;
}

std::vector<std::pair<float,float>> AnalysisFCChh::reweight_mditau_components(
  const std::vector<std::array<float,8>>& components,
  bool use_met,
  bool use_theta,
  bool use_ratio,
  bool use_miss
)
{
  std::vector<std::pair<float,float>> out;
  out.reserve(components.size());

  for (const auto& c : components) {
    float w = 1.f;
    if (use_met)   w *= c[1];
    if (use_theta) w *= c[2] * c[3];
    if (use_ratio) w *= c[4] * c[5];
    if (use_miss)  w *= c[6];
    out.push_back({c[0], w});
  }
  return out;
}

// ------------------------------------------------------------
// ATLAS TF1-based angle/ratio weights (MMC2019 calibration)
// ------------------------------------------------------------
namespace {
struct AtlasTF1Store {
  std::unique_ptr<TFile> file;
  TF1* angle_lep[4]    = {nullptr,nullptr,nullptr,nullptr};
  TF1* angle_1p0n[4]   = {nullptr,nullptr,nullptr,nullptr};
  TF1* angle_3p0n[4]   = {nullptr,nullptr,nullptr,nullptr};
  TF1* ratio_lep       = nullptr;
  TF1* ratio_lead_1p0n = nullptr;
  TF1* ratio_sub_1p0n  = nullptr;
  TF1* ratio_lead_3p0n = nullptr;
  TF1* ratio_sub_3p0n  = nullptr;
};

TF1* get_tf1(TFile* f, const char* path) {
  if (!f) return nullptr;
  TObject* obj = f->Get(path);
  return dynamic_cast<TF1*>(obj);
}

AtlasTF1Store& atlas_tf1_store() {
  static AtlasTF1Store store;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    const std::string calib_path = "MMC/MMC_params_v1_fixed.root";
    store.file.reset(TFile::Open(calib_path.c_str(), "READ"));
    if (!store.file || store.file->IsZombie()) {
      std::cerr << "[atlas_tf1] Failed to open calibration file: " << calib_path << std::endl;
      store.file.reset();
      return;
    }
    // Angle params (pT-dependent) for leptonic and hadronic (1p0n, 3p0n)
    const char* angle_lep_paths[4]  = {
      "MMC2019MC16/lep/tauNuAngle3d/pTdep-par0",
      "MMC2019MC16/lep/tauNuAngle3d/pTdep-par1",
      "MMC2019MC16/lep/tauNuAngle3d/pTdep-par2",
      "MMC2019MC16/lep/tauNuAngle3d/pTdep-par3"
    };
    const char* angle_1p0n_paths[4] = {
      "MMC2019MC16/had/tauNuAngle3d-1p0n/pTdep-par0",
      "MMC2019MC16/had/tauNuAngle3d-1p0n/pTdep-par1",
      "MMC2019MC16/had/tauNuAngle3d-1p0n/pTdep-par2",
      "MMC2019MC16/had/tauNuAngle3d-1p0n/pTdep-par3"
    };
    const char* angle_3p0n_paths[4] = {
      "MMC2019MC16/had/tauNuAngle3d-3p0n/pTdep-par0",
      "MMC2019MC16/had/tauNuAngle3d-3p0n/pTdep-par1",
      "MMC2019MC16/had/tauNuAngle3d-3p0n/pTdep-par2",
      "MMC2019MC16/had/tauNuAngle3d-3p0n/pTdep-par3"
    };
    for (int i=0;i<4;++i) {
      store.angle_lep[i]  = get_tf1(store.file.get(), angle_lep_paths[i]);
      store.angle_1p0n[i] = get_tf1(store.file.get(), angle_1p0n_paths[i]);
      store.angle_3p0n[i] = get_tf1(store.file.get(), angle_3p0n_paths[i]);
    }

    // Ratio TF1s (direct fit) for lead/sublead hadronic and leptonic
    store.ratio_lep       = get_tf1(store.file.get(), "MMC2019MC16/lep/nuTauRatio/fit");
    store.ratio_lead_1p0n = get_tf1(store.file.get(), "MMC2019MC16/had/nuTauRatioLead-1p0n/fit");
    store.ratio_sub_1p0n  = get_tf1(store.file.get(), "MMC2019MC16/had/nuTauRatioSublead-1p0n/fit");
    store.ratio_lead_3p0n = get_tf1(store.file.get(), "MMC2019MC16/had/nuTauRatioLead-3p0n/fit");
    store.ratio_sub_3p0n  = get_tf1(store.file.get(), "MMC2019MC16/had/nuTauRatioSublead-3p0n/fit");
  });
  return store;
}

float atlas_angle_weight_tf1(float pt_vis, float theta, bool isLep, int nprong) {
  const auto& s = atlas_tf1_store();
  TF1* pars[4] = {nullptr,nullptr,nullptr,nullptr};
  if (isLep) {
    std::copy(std::begin(s.angle_lep), std::end(s.angle_lep), pars);
  } else if (nprong <= 2) {
    std::copy(std::begin(s.angle_1p0n), std::end(s.angle_1p0n), pars);
  } else {
    std::copy(std::begin(s.angle_3p0n), std::end(s.angle_3p0n), pars);
  }
  if (!pars[0] || !pars[1] || !pars[2] || !pars[3]) return 0.f;
  const double p0 = pars[0]->Eval(pt_vis);
  const double p1 = pars[1]->Eval(pt_vis);
  const double p2 = pars[2]->Eval(pt_vis);
  const double p3 = pars[3]->Eval(pt_vis);
  const double arg = (theta + p3) / std::max(p1, 1e-6);
  const double val = p0 * std::exp(-p2 * std::pow(std::log(arg), 2.0));
  return static_cast<float>(std::max(0.0, val));
}

float atlas_ratio_weight_tf1(float ratio, bool isLep, int nprong, bool isLead) {
  const auto& s = atlas_tf1_store();
  TF1* f = nullptr;
  if (isLep) {
    f = s.ratio_lep;
  } else if (nprong <= 2) {
    f = isLead ? s.ratio_lead_1p0n : s.ratio_sub_1p0n;
  } else {
    f = isLead ? s.ratio_lead_3p0n : s.ratio_sub_3p0n;
  }
  if (!f) return 0.f;
  const double val = f->Eval(ratio);
  return static_cast<float>(std::max(0.0, val));
}
} // namespace

std::array<float,5> AnalysisFCChh::atlas_tf1_weights(
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
) {
  const float theta1 = static_cast<float>(tau1_vis.Vect().Angle(nu1.Vect()));
  const float theta2 = static_cast<float>(tau2_vis.Vect().Angle(nu2.Vect()));
  const float pt1    = static_cast<float>(tau1_vis.Pt());
  const float pt2    = static_cast<float>(tau2_vis.Pt());
  const float R1     = (tau1_vis.P() > 0.0) ? static_cast<float>(nu1.P() / tau1_vis.P()) : 0.f;
  const float R2     = (tau2_vis.P() > 0.0) ? static_cast<float>(nu2.P() / tau2_vis.P()) : 0.f;

  const float w_ang1 = atlas_angle_weight_tf1(pt1, theta1, isLep1, n_charged_tracks_1);
  const float w_ang2 = atlas_angle_weight_tf1(pt2, theta2, isLep2, n_charged_tracks_2);
  const float w_rat1 = atlas_ratio_weight_tf1(R1, isLep1, n_charged_tracks_1, isLead1);
  const float w_rat2 = atlas_ratio_weight_tf1(R2, isLep2, n_charged_tracks_2, isLead2);

  const float w_event = w_ang1 * w_ang2 * w_rat1 * w_rat2;
  return {w_event, w_ang1, w_ang2, w_rat1, w_rat2};
}


std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_ATLAS_markov(
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
  int nIter,
  int nMass_steps,
  int n_b_jets_medium,
  int n_tau_jets_medium,
  bool efficiency_recover,
  bool use_atlas_tf1
)
{
  std::vector<std::pair<float,float>> mditau_solutions;

  auto wrapToPi = [](double a) -> double {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a - M_PI;
  };

  auto setVisMass = [](const TLorentzVector& tlv, bool isLep, int nprong) {
    TLorentzVector out;
    double mvis = tlv.M();
    if (!isLep) {
      mvis = (nprong <= 2) ? 0.8 : 1.2;
    }
    out.SetPtEtaPhiM(tlv.Pt(), tlv.Eta(), tlv.Phi(), mvis);
    return out;
  };

  TLorentzVector vis1 = setVisMass(tau1, isLep1, n_charged_tracks_1);
  TLorentzVector vis2 = setVisMass(tau2, isLep2, n_charged_tracks_2);

  // channel-dependent MET window (sigma factors) mimicking ATLAS defaults
  int nsigma = 5;
  if (!isLep1 && !isLep2) nsigma = 5;

  const double phi1_c = vis1.Phi();
  const double phi2_c = vis2.Phi();
  // hard coded dphi derived from MC truth dphi(vis,mis)
  const double dPhiMax = 0.4;
  const double phi1_min = phi1_c - dPhiMax;
  const double phi1_max = phi1_c + dPhiMax;
  const double phi2_min = phi2_c - dPhiMax;
  const double phi2_max = phi2_c + dPhiMax;

  const double m_tau = 1.77686;
  const double mnu1_max = isLep1 ? std::max(0.0, m_tau - vis1.M()) : 0.0;
  const double mnu2_max = isLep2 ? std::max(0.0, m_tau - vis2.M()) : 0.0;

  TRandom3 rng(12345);

  // michel-prior, solved lepton decay kinematics mapped onto masses of leptons
  auto w_lep_miss = [](double mMiss, double mTau, double mLep)->double {
    const double mt2 = mTau*mTau;
    const double x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2;
    if (x <= 0.0 || x >= 1.0 || mMiss <= 0.0) return 0.0;
    const double w_x = x*x*(3.0 - 2.0*x);
    const double jac = 2.0*mMiss/mt2;
    return w_x * jac;
  };

  // proposals (similar to DiTauMassTools defaults)
  const double metRangeL = nsigma * sigma_para;
  const double metRangeP = nsigma * sigma_perp;
  double mEtL = 0.0, mEtP = 0.0, phi1 = phi1_c, phi2 = phi2_c, mnu1 = 0.0, mnu2 = 0.0;
  double propMetStep = metRangeP / 30.0;
  double propPhiStep = 0.04;
  double propMnuStep1 = (mnu1_max > 0.0) ? mnu1_max / 10.0 : 0.0;
  double propMnuStep2 = (mnu2_max > 0.0) ? mnu2_max / 10.0 : 0.0;

  const double cphi = std::cos(met_cov_phi);
  const double sphi = std::sin(met_cov_phi);
  const double inv_sigmaL2 = 1.0 / (sigma_para * sigma_para);
  const double inv_sigmaP2 = 1.0 / (sigma_perp * sigma_perp);

  bool fullScan = true;
  double prevProb = 0.0;
  int accepted = 0;
  int tried = 0;
  const int adaptFreq = 1000;
  const double propScaleUp = 1.2;
  const double propScaleDown = 0.8;
  const double minMetStep = std::min(sigma_para, sigma_perp) / 200.0;
  const double maxMetStep = std::max(sigma_para, sigma_perp);

  for (int iter = 0; iter < nIter; ++iter) {
    ++tried;
    if (fullScan) {
      mEtL = rng.Uniform(-metRangeL, metRangeL);
      mEtP = rng.Uniform(-metRangeP, metRangeP);
      phi1 = rng.Uniform(phi1_min, phi1_max);
      phi2 = rng.Uniform(phi2_min, phi2_max);
      if (isLep1) mnu1 = rng.Uniform(0.0, mnu1_max);
      if (isLep2) mnu2 = rng.Uniform(0.0, mnu2_max);
    } else {
      mEtL = rng.Gaus(mEtL, propMetStep);
      mEtP = rng.Gaus(mEtP, propMetStep);
      phi1 = rng.Gaus(phi1, propPhiStep);
      phi2 = rng.Gaus(phi2, propPhiStep);
      if (isLep1) mnu1 = rng.Gaus(mnu1, propMnuStep1);
      if (isLep2) mnu2 = rng.Gaus(mnu2, propMnuStep2);
    }

    // enforce ranges
    if (phi1 < phi1_min || phi1 > phi1_max) continue;
    if (phi2 < phi2_min || phi2 > phi2_max) continue;
    if (std::abs(mEtL) > metRangeL || std::abs(mEtP) > metRangeP) continue;
    if (mnu1 < 0 || mnu1 > mnu1_max) continue;
    if (mnu2 < 0 || mnu2 > mnu2_max) continue;

    // build smeared MET
    const double dX = mEtL * cphi - mEtP * sphi;
    const double dY = mEtL * sphi + mEtP * cphi;
    const double sMET_x = MET_x + dX;
    const double sMET_y = MET_y + dY;

    const double sin12 = std::sin(phi2 - phi1);
    if (std::abs(sin12) < 1e-6) continue;
    const double csc_dphi = 1.0 / sin12;

    const double pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
    const double pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );
    if (pT1 <= 0.0 || pT2 <= 0.0) continue;

    double totalProb = 0.0;
    std::vector<std::pair<float,float>> iterSolutions;

    auto weight_MET = std::exp(-0.5 * (mEtL*mEtL * inv_sigmaL2 + mEtP*mEtP * inv_sigmaP2));

    if (!isLep1 && !isLep2) {
      std::vector<float> pz1 = solve_pz_quadratic(vis1, phi1, pT1, n_charged_tracks_1);
      std::vector<float> pz2 = solve_pz_quadratic(vis2, phi2, pT2, n_charged_tracks_2);
      for (float z1 : pz1) {
        TLorentzVector nu1;
        const double px1 = pT1 * std::cos(phi1);
        const double py1 = pT1 * std::sin(phi1);
        const double E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
        nu1.SetPxPyPzE(px1, py1, z1, E1);

        for (float z2 : pz2) {
          const double px2 = pT2 * std::cos(phi2);
          const double py2 = pT2 * std::sin(phi2);
          const double E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
          TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

          TLorentzVector tau_full_1 = vis1 + nu1;
          TLorentzVector tau_full_2 = vis2 + nu2;

          const double theta1 = vis1.Vect().Angle(nu1.Vect());
          const double theta2 = vis2.Vect().Angle(nu2.Vect());
          const double p_tau1 = tau_full_1.P();
          const double p_tau2 = tau_full_2.P();

          double w_event = 0.0;
          if (use_atlas_tf1) {
            const bool lead1 = vis1.Pt() >= vis2.Pt();
            const bool lead2 = !lead1;
            auto w = atlas_tf1_weights(
              vis1, nu1, isLep1, n_charged_tracks_1, lead1,
              vis2, nu2, isLep2, n_charged_tracks_2, lead2
            );
            w_event = w[0] * weight_MET;
          } else {
            const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
            const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
            const double w_theta1 = theta_pdf_mixture_trunc(theta1, p_tau1, K1, 0.3);
            const double w_theta2 = theta_pdf_mixture_trunc(theta2, p_tau2, K2, 0.3);
            w_event = w_theta1 * w_theta2 * weight_MET;
          }
          if (w_event <= 0) continue;
          totalProb += w_event;
          iterSolutions.push_back({static_cast<float>((tau_full_1 + tau_full_2).M()), static_cast<float>(w_event)});
        }
      }
    } else {
      // leptonic leg(s)
      const double dm1 = (isLep1 && nMass_steps>0) ? (mnu1_max / nMass_steps) : 0.0;
      const double dm2 = (isLep2 && nMass_steps>0) ? (mnu2_max / nMass_steps) : 0.0;

      std::vector<float> pz1had = solve_pz_quadratic(vis1, phi1, pT1, n_charged_tracks_1);
      std::vector<float> pz2had = solve_pz_quadratic(vis2, phi2, pT2, n_charged_tracks_2);

      if (isLep1) {
        for (int im = 0; im < nMass_steps; ++im) {
          const double mscan = (im + 0.5) * dm1;
          if (mscan <= 0) continue;
          const double w_miss = w_lep_miss(mscan, m_tau, vis1.M());
          if (w_miss <= 0) continue;
          std::vector<float> pz1 = solve_pz_quadratic_tau(vis1, phi1, pT1, isLep1, n_charged_tracks_1, mscan);
          for (float z1 : pz1) {
            const double px1 = pT1 * std::cos(phi1);
            const double py1 = pT1 * std::sin(phi1);
            const double E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1 + mscan*mscan);
            TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

            for (float z2 : pz2had) {
              const double px2 = pT2 * std::cos(phi2);
              const double py2 = pT2 * std::sin(phi2);
              const double E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
              TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

              TLorentzVector tau_full_1 = vis1 + nu1;
              TLorentzVector tau_full_2 = vis2 + nu2;

              const double theta1 = vis1.Vect().Angle(nu1.Vect());
              const double theta2 = vis2.Vect().Angle(nu2.Vect());
              const double p_tau1 = tau_full_1.P();
              const double p_tau2 = tau_full_2.P();

              double w_event = 0.0;
              if (use_atlas_tf1) {
                const bool lead1 = true;  // leptonic leg as lead
                const bool lead2 = false;
                auto w = atlas_tf1_weights(
                  vis1, nu1, isLep1, n_charged_tracks_1, lead1,
                  vis2, nu2, isLep2, n_charged_tracks_2, lead2
                );
                w_event = w[0] * weight_MET * w_miss;
              } else {
                const MixCoeffs& K1 = COEFFS_LEPTONIC_THESIS;
                const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                const double w_theta1 = theta_pdf_mixture_trunc(theta1, p_tau1, K1, 0.5);
                const double w_theta2 = theta_pdf_mixture_trunc(theta2, p_tau2, K2, 0.3);
                w_event  = w_theta1 * w_theta2 * weight_MET * w_miss;
              }
              if (w_event <= 0) continue;
              totalProb += w_event;
              iterSolutions.push_back({static_cast<float>((tau_full_1 + tau_full_2).M()), static_cast<float>(w_event)});
            }
          }
        }
      } else if (isLep2) {
        for (int im = 0; im < nMass_steps; ++im) {
          const double mscan = (im + 0.5) * dm2;
          if (mscan <= 0) continue;
          const double w_miss = w_lep_miss(mscan, m_tau, vis2.M());
          if (w_miss <= 0) continue;
          std::vector<float> pz2 = solve_pz_quadratic_tau(vis2, phi2, pT2, isLep2, n_charged_tracks_2, mscan);
          for (float z2 : pz2) {
            const double px2 = pT2 * std::cos(phi2);
            const double py2 = pT2 * std::sin(phi2);
            const double E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2 + mscan*mscan);
            TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

            for (float z1 : pz1had) {
              const double px1 = pT1 * std::cos(phi1);
              const double py1 = pT1 * std::sin(phi1);
              const double E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
              TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

              TLorentzVector tau_full_1 = vis1 + nu1;
              TLorentzVector tau_full_2 = vis2 + nu2;

              const double theta1 = vis1.Vect().Angle(nu1.Vect());
              const double theta2 = vis2.Vect().Angle(nu2.Vect());
              const double p_tau1 = tau_full_1.P();
              const double p_tau2 = tau_full_2.P();

              double w_event = 0.0;
              if (use_atlas_tf1) {
                const bool lead1 = false;
                const bool lead2 = true; // leptonic leg as lead
                auto w = atlas_tf1_weights(
                  vis1, nu1, isLep1, n_charged_tracks_1, lead1,
                  vis2, nu2, isLep2, n_charged_tracks_2, lead2
                );
                w_event = w[0] * weight_MET * w_miss;
              } else {
                const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG_THESIS : COEFFS_3PRONG_THESIS;
                const MixCoeffs& K2 = COEFFS_LEPTONIC_THESIS;
                const double w_theta1 = theta_pdf_mixture_trunc(theta1, p_tau1, K1, 0.3);
                const double w_theta2 = theta_pdf_mixture_trunc(theta2, p_tau2, K2, 0.5);
                w_event  = w_theta1 * w_theta2 * weight_MET * w_miss;
              }
              if (w_event <= 0) continue;
              totalProb += w_event;
              iterSolutions.push_back({static_cast<float>((tau_full_1 + tau_full_2).M()), static_cast<float>(w_event)});
            }
          }
        }
      }
    }

    if (fullScan) {
      if (totalProb > 0) {
        fullScan = false;
        prevProb = totalProb;
      }
      continue;
    }

    // Metropolis step
    if (totalProb <= 0) continue;
    double acceptProb = (prevProb > 0) ? std::min(1.0, totalProb / prevProb) : 1.0;
    double u = rng.Uniform();
    if (u <= acceptProb) {
      prevProb = totalProb;
      mditau_solutions.insert(mditau_solutions.end(), iterSolutions.begin(), iterSolutions.end());
      ++accepted;
    }

    // adaptive proposals every adaptFreq iterations
    if (tried % adaptFreq == 0 && tried > 0) {
      double accRate = (tried > 0) ? static_cast<double>(accepted) / static_cast<double>(tried) : 0.0;
      if (accRate < 0.15) {
        propMetStep = std::max(minMetStep, propMetStep * propScaleDown);
        propPhiStep = std::max(0.005, propPhiStep * propScaleDown);
        if (isLep1) propMnuStep1 = std::max(mnu1_max / 200.0, propMnuStep1 * propScaleDown);
        if (isLep2) propMnuStep2 = std::max(mnu2_max / 200.0, propMnuStep2 * propScaleDown);
      } else if (accRate > 0.45) {
        propMetStep = std::min(maxMetStep, propMetStep * propScaleUp);
        propPhiStep = std::min(0.2, propPhiStep * propScaleUp);
        if (isLep1) propMnuStep1 = std::min(mnu1_max, propMnuStep1 * propScaleUp);
        if (isLep2) propMnuStep2 = std::min(mnu2_max, propMnuStep2 * propScaleUp);
      }
      tried = 0;
      accepted = 0;
    }
  }

  // simple efficiency-recovery rerun with larger MET window if no solutions
  if (efficiency_recover && mditau_solutions.empty()) {
    // widen MET sigmas (x2) and rerun once
    auto rerun = solve_ditau_MMC_ATLAS_markov(
      tau1, tau2, isLep1, isLep2,
      n_charged_tracks_1, n_charged_tracks_2,
      MET_x, MET_y,
      sigma_para * 2.0, sigma_perp * 2.0, met_cov_phi,
      nIter, nMass_steps, n_b_jets_medium, n_tau_jets_medium, false);
    mditau_solutions.insert(mditau_solutions.end(), rerun.begin(), rerun.end());
  }

  return mditau_solutions;
}


std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_angular_lephad_weighted_rescan(
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
)
{ 
  std::vector<std::pair<float,float>> mditau_solutions;

  // not processing leplep: exit early
  if (isLep1 && isLep2) {
    // do not run if di-leptonic
    return mditau_solutions;
  }
  
  // for had-had we can increase the scanning steps and MET-width
  if (!isLep1 && !isLep2) {
    nMETsteps *= 2;
    nsteps *= 2;
    nMETsig = 6;
  }

  // also apply preselection here, otherwise we spend too long computing configurations that are nonsense
  if ((n_b_jets_medium != 4) || (n_tau_jets_medium < 1) || (n_tau_jets_medium > 2)) {
    return mditau_solutions;
  }

  auto wrapToPi = [](float a) -> float {
    // robust (-pi, pi] wrapping
    a = std::fmod(a + M_PI, 2.f*M_PI);
    if (a < 0.f) a += 2.f*M_PI;
    return a - M_PI;
  };
    
  // using nstep, define granularity of phi grid scan
  const float dphi = 2.0 * M_PI / static_cast<float>(nsteps);

  // met scanning, dividing by sqrt(2) to go from event -> x/y cmpt met resolutions 
  // extracted correlation(x,y) ~ 0, hence this is safe to do. 
  const float isqrt2 = 1 / std::sqrt(2);
  const float metres_xy = metres * isqrt2;
  const float inv_metres_xy2 = 1/(metres_xy*metres_xy);
  float half = nMETsig*metres_xy;
  float dmet = (2*half)/(nMETsteps-1);
  const float m_tau =  1.77686;

  const float PHI_WIN = 0.4f;                    // requested window
  const float phi1_c  = tau1.Phi();              // visible tau1 azimuth
  const float phi2_c  = tau2.Phi();              // visible tau2 azimuth
  float dphi1   = (2.f*PHI_WIN) / static_cast<float>(nsteps);
  float dphi2   = (2.f*PHI_WIN) / static_cast<float>(nsteps);

  // quick function for mass weighting
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau*mTau;
    const float x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2; // ~ 1 - mMiss^2/mt2
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;    // outside physical (approx)
    // Michel prior (V-A) with Jacobian to m_miss; normalization irrelevant
    const float w_x = x*x*(3.f - 2.f*x);
    const float jac = 2.f*mMiss/mt2;
    return w_x * jac;
  };

  /* 
  MMC implementation that scans over azimuthal angles of each tau 
  with a square grid around MET_x/y - Gauss weighting to penalise large deviations
  */
  // define number of attempts (should a solution not exist)
  const int n_attempts = 2;
  const int nMETSig_retry = 10;
  const float step_scan_factor = 1.5f;

  for (int a = 0; a < n_attempts; a++) {

    // if we're on the second try, rescale the nsteps:
    if (a != 0) {
      nsteps = static_cast<int>(std::ceil(nsteps * step_scan_factor));
      nMETsig = nMETSig_retry;
      half = nMETsig * metres_xy;
      nMETsteps = static_cast<int>(std::ceil(nMETsteps * step_scan_factor));
      dmet = (2 * half) / (nMETsteps - 1);
      dphi1 = (2.f * PHI_WIN) / static_cast<float>(nsteps);
      dphi2 = (2.f * PHI_WIN) / static_cast<float>(nsteps);
    }

    for (int i = 0; i < nsteps; i++) {
        const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f)*dphi1);
        for (int j = 0; j < nsteps; j++) {
          const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f)*dphi2);
          // define csc dphi for use in each step in for-loop
          const float sin12 = std::sin(phi2 - phi1);
          if (std::fabs(sin12) < 1e-4f) continue;
          float csc_dphi = 1 / sin12;

          // also now perform MET-scan in grid around measured x/y values
          for (int k = 0; k < nMETsteps; k++) {
            float sMET_x = MET_x - nMETsig * metres_xy + k * dmet;
            for (int l = 0; l < nMETsteps; l++) {
              float sMET_y = MET_y - nMETsig * metres_xy + l * dmet;

              // all of this is doable before placing assumptions on the missing mass
              // since we store all solutions, pre-compute MET weight
              const float dx = sMET_x - MET_x;
              const float dy = sMET_y - MET_y;
            
              // log w_MET = -0.5 * (dx^2/σx^2 + dy^2/σy^2)
              const double log_w_met = -0.5 * (dx*dx * inv_metres_xy2 + dy*dy * inv_metres_xy2);
              const double w_met     = exp(log_w_met); // linear MET weight
              
              // compute pTs of neutrinos
              float pT1 = csc_dphi * ( sMET_x * std::sin(phi2) - sMET_y*std::cos(phi2) );
              float pT2 = -csc_dphi * ( sMET_x * std::sin(phi1) - sMET_y*std::cos(phi1) );
              if (pT1 <= 0.f || pT2 <= 0.f) continue;     // enforce pT > 0 in this window

              
              // now if we have a leptonic leg, need to scan over missing mass
              if (!isLep1 && !isLep2) {
                // using these solutions, solve each quadratic
                std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
                std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);
                for (float z1 : pz1) {
                  // neutrino 1 TLV from (pT1, phi1, z1)
                  const float px1 = pT1 * std::cos(phi1);
                  const float py1 = pT1 * std::sin(phi1);
                  const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1); 
                  TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                  for (float z2 : pz2) {
                    // neutrino 2 TLV from (pT2, phi2, z2)
                    const float px2 = pT2 * std::cos(phi2);
                    const float py2 = pT2 * std::sin(phi2);
                    const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2); 
                    TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                    // full tau four-vectors
                    TLorentzVector tau_full_1 = tau1 + nu1;
                    TLorentzVector tau_full_2 = tau2 + nu2;
                    
                    // angular kinematic weight 
                    const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                    const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());

                    // require tau 4-momenta
                    const float p_tau1 = tau_full_1.P();
                    const float p_tau2 = tau_full_2.P();

                  
                    const float THETA_MAX = 0.3f; 

                    const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                    const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                    const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                    const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                    const float w_event  = w_theta1 * w_theta2 * w_met;

                    // di-tau invariant mass
                    const float mditau = (tau_full_1 + tau_full_2).M();

                    // store every solution 
                    mditau_solutions.push_back({mditau, w_event});
                  }
                }
                
              } else if (isLep1) {
                  // m_miss in [0, m_tau - m_vis]; here m_vis ~ m(ℓ) ~ small; use midpoint sampling
                  const float mLep = tau1.M();                         // visible lepton mass (≈0 or 0.105)
                  const float mMax = std::max(0.f, m_tau - mLep);
                  const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                  std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);

                  for (int t = 0; t < nMass_steps; ++t) {
                    const float m_scan = (t + 0.5f)*dm;               // midpoint
                    if (m_scan <= 0.f) continue;

                    // leptonic prior weight for this m_scan
                    const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                    if (w_miss <= 0.f) continue;

                    std::vector<float> pz1 = solve_pz_quadratic_tau(tau1, phi1, pT1, isLep1,
                                                                    n_charged_tracks_1, m_scan);
                    for (float z1 : pz1) {
                      const float px1 = pT1 * std::cos(phi1);
                      const float py1 = pT1 * std::sin(phi1);
                      const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                      TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                      for (float z2 : pz2) {
                        const float px2 = pT2 * std::cos(phi2);
                        const float py2 = pT2 * std::sin(phi2);
                        const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                        TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                        TLorentzVector tau_full_1 = tau1 + nu1;
                        TLorentzVector tau_full_2 = tau2 + nu2;

                        const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                        const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                        const float p_tau1     = tau_full_1.P();
                        const float p_tau2     = tau_full_2.P();

                        const float THETA_MAX = 0.3f;
                        const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                        const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                        const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                        const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                        const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; // <-- include prior

                        const float mditau   = (tau_full_1 + tau_full_2).M();
                        mditau_solutions.push_back({mditau, w_event});
                      }
                    }
                  }
                } else if (isLep2) {
                    // m_miss in [0, m_tau - m_vis]; here m_vis ~ m(ℓ) ~ small; use midpoint sampling
                    const float mLep = tau2.M();                         // visible lepton mass (≈0 or 0.105)
                    const float mMax = std::max(0.f, m_tau - mLep);
                    const float dm   = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

                    std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);

                    for (int t = 0; t < nMass_steps; ++t) {
                      const float m_scan = (t + 0.5f)*dm;               // midpoint
                      if (m_scan <= 0.f) continue;

                      // leptonic prior weight for this m_scan
                      const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                      if (w_miss <= 0.f) continue;

                      std::vector<float> pz2 = solve_pz_quadratic_tau(tau2, phi2, pT2, isLep2,
                                                                      n_charged_tracks_2, m_scan);
                      for (float z1 : pz1) {
                        const float px1 = pT1 * std::cos(phi1);
                        const float py1 = pT1 * std::sin(phi1);
                        const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                        TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                        for (float z2 : pz2) {
                          const float px2 = pT2 * std::cos(phi2);
                          const float py2 = pT2 * std::sin(phi2);
                          const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                          TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                          TLorentzVector tau_full_1 = tau1 + nu1;
                          TLorentzVector tau_full_2 = tau2 + nu2;

                          const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                          const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                          const float p_tau1     = tau_full_1.P();
                          const float p_tau2     = tau_full_2.P();

                          const float THETA_MAX = 0.3f;
                          const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                          const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                          const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                          const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);

                          const float w_event  = w_theta1 * w_theta2 * w_met * w_miss; // <-- include prior

                          const float mditau   = (tau_full_1 + tau_full_2).M();
                          mditau_solutions.push_back({mditau, w_event});
                        }
                      }
                    }
                  } else {
              // should never make it here..
              std::cout << "Neither? Please check configuration." << std::endl;
            }

            }

          }
          
        }
      }
      // If we found any solutions, don't retry
      if (!mditau_solutions.empty()) break;
  }
  // --- Fallback if MMC scan produced no solutions ---
  if (mditau_solutions.empty()) {
    float m_fb = ditau_mass_collinear_or_vis(tau1, tau2, MET_x, MET_y);
    // Always positive for m_vis at least, but be defensive
    if (m_fb > 0.f && std::isfinite(m_fb)) {
      mditau_solutions.emplace_back(m_fb, 1.0f);
    }
  }

  return mditau_solutions;

}

std::vector<std::pair<float,float>> AnalysisFCChh::solve_ditau_MMC_METScan_angular_lephad_weighted_retry(
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
)
{
  std::vector<std::pair<float,float>> mditau_solutions;

  if (isLep1 && isLep2) {
    return mditau_solutions; // skip dileptonic
  }
  // For had-had increase baseline granularity & window
  //if (!isLep1 && !isLep2) {
  //  nMETsteps = std::max(2, nMETsteps * 2);
  //  nsteps    = std::max(2, nsteps    * 2);
  //  nMETsig   = 10;
  //}

  // also apply preselection here, otherwise we spend too long computing configurations that are nonsense
  if ((n_b_jets_medium != 4) || (n_tau_jets_medium < 1) || (n_tau_jets_medium > 2)) {
    return mditau_solutions;
  }


  auto wrapToPi = [](float a) -> float {
    a = std::fmod(a + M_PI, 2.f*M_PI);
    if (a < 0.f) a += 2.f*M_PI;
    return a - M_PI;
  };

  // --- constants (event-level) ---
  const float isqrt2 = 1.f / std::sqrt(2.f);
  const float metres_xy = metres * isqrt2;
  const float inv_metres_xy2 = 1.f / (metres_xy * metres_xy);
  const float m_tau = 1.77686f;
  const float PHI_WIN = 0.4f;
  const float phi1_c  = tau1.Phi();
  const float phi2_c  = tau2.Phi();

  // leptonic prior
  auto w_lep_miss = [](float mMiss, float mTau, float mLep)->float {
    const float mt2 = mTau*mTau;
    const float x   = (mt2 + mLep*mLep - mMiss*mMiss) / mt2;
    if (x <= 0.f || x >= 1.f || mMiss <= 0.f) return 0.f;
    const float w_x = x*x*(3.f - 2.f*x);
    const float jac = 2.f*mMiss/mt2;
    return w_x * jac;
  };

  // -------------------------------
  // Retry mechanism
  // -------------------------------
  const int   MAX_ATTEMPTS  = 2;        // initial + one retry
  const float RETRY_FACTOR  = 1.5f;     // scale for window and steps on retry

  for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {

    // Scale settings if retrying (only if first pass produced no solutions)
    const int nsteps_try    = (attempt == 0) ? nsteps
                           : std::max(2, (int)std::ceil(nsteps * RETRY_FACTOR));
    const int nMETsteps_try = (attempt == 0) ? nMETsteps
                           : std::max(2, (int)std::ceil(nMETsteps * RETRY_FACTOR));
    const int nMETsig_try   = (attempt == 0) ? nMETsig
                           : std::max(1, (int)std::ceil(nMETsig * RETRY_FACTOR));

    // Step sizes
    const float dphi1 = (2.f*PHI_WIN) / (float)nsteps_try;
    const float dphi2 = (2.f*PHI_WIN) / (float)nsteps_try;

    const float half  = nMETsig_try * metres_xy;
    const float dmet  = (nMETsteps_try > 1) ? (2.f*half) / (float)(nMETsteps_try - 1) : 0.f;

    // -------------------------------
    // Main scan
    // -------------------------------
    for (int i = 0; i < nsteps_try; ++i) {
      const float phi1 = wrapToPi(phi1_c - PHI_WIN + (i + 0.5f)*dphi1);
      const float s1 = std::sin(phi1), c1 = std::cos(phi1);

      for (int j = 0; j < nsteps_try; ++j) {
        const float phi2 = wrapToPi(phi2_c - PHI_WIN + (j + 0.5f)*dphi2);
        const float s2 = std::sin(phi2), c2 = std::cos(phi2);

        const float sin12 = std::sin(phi2 - phi1);
        if (std::fabs(sin12) < 1e-4f) continue;
        const float inv_sin12 = 1.f / sin12;

        // MET grid (square around measured MET)
        for (int k = 0; k < nMETsteps_try; ++k) {
          const float sMET_x = (nMETsteps_try > 1) ? (MET_x - half + k * dmet) : MET_x;
          const float dx = sMET_x - MET_x;

          for (int l = 0; l < nMETsteps_try; ++l) {
            const float sMET_y = (nMETsteps_try > 1) ? (MET_y - half + l * dmet) : MET_y;
            const float dy = sMET_y - MET_y;

            // MET weight (isotropic in x/y with metres_xy)
            const double log_w_met = -0.5 * (dx*dx * inv_metres_xy2 + dy*dy * inv_metres_xy2);
            const double w_met     = std::exp(log_w_met);

            // Solve for neutrino pT along the chosen phis
            float pT1 = (  sMET_x * s2 - sMET_y * c2) * inv_sin12;
            float pT2 = ( -sMET_x * s1 + sMET_y * c1) * inv_sin12;
            if (pT1 <= 0.f || pT2 <= 0.f) continue;

            if (!isLep1 && !isLep2) {
              // had-had
              std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
              std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);
              for (float z1 : pz1) {
                const float px1 = pT1 * c1, py1 = pT1 * s1;
                const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);
                for (float z2 : pz2) {
                  const float px2 = pT2 * c2, py2 = pT2 * s2;
                  const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                  TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                  TLorentzVector tau_full_1 = tau1 + nu1;
                  TLorentzVector tau_full_2 = tau2 + nu2;

                  const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                  const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                  const float p_tau1     = tau_full_1.P();
                  const float p_tau2     = tau_full_2.P();

                  const float THETA_MAX = 0.3f;
                  const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                  const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                  const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                  const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                  const float w_event  = w_theta1 * w_theta2 * (float)w_met;

                  const float mditau = (tau_full_1 + tau_full_2).M();
                  mditau_solutions.emplace_back(mditau, w_event);
                }
              }

            } else if (isLep1) {
              // lep-had (tau1 leptonic)
              const float mLep = tau1.M();
              const float mMax = std::max(0.f, m_tau - mLep);
              const float dm_m = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

              std::vector<float> pz2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);

              for (int t = 0; t < nMass_steps; ++t) {
                const float m_scan = (t + 0.5f) * dm_m;
                if (m_scan <= 0.f) continue;
                const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                if (w_miss <= 0.f) continue;

                std::vector<float> pz1 = solve_pz_quadratic_tau(tau1, phi1, pT1, true,
                                                                n_charged_tracks_1, m_scan);
                for (float z1 : pz1) {
                  const float px1 = pT1 * c1, py1 = pT1 * s1;
                  const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                  TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                  for (float z2 : pz2) {
                    const float px2 = pT2 * c2, py2 = pT2 * s2;
                    const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                    TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                    TLorentzVector tau_full_1 = tau1 + nu1;
                    TLorentzVector tau_full_2 = tau2 + nu2;

                    const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                    const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                    const float p_tau1     = tau_full_1.P();
                    const float p_tau2     = tau_full_2.P();

                    const float THETA_MAX = 0.3f;
                    const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                    const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                    const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                    const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                    const float w_event  = w_theta1 * w_theta2 * (float)w_met * w_miss;

                    const float mditau   = (tau_full_1 + tau_full_2).M();
                    mditau_solutions.emplace_back(mditau, w_event);
                  }
                }
              }

            } else if (isLep2) {
              // lep-had (tau2 leptonic)
              const float mLep = tau2.M();
              const float mMax = std::max(0.f, m_tau - mLep);
              const float dm_m = (nMass_steps > 0) ? (mMax / nMass_steps) : 0.f;

              std::vector<float> pz1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);

              for (int t = 0; t < nMass_steps; ++t) {
                const float m_scan = (t + 0.5f) * dm_m;
                if (m_scan <= 0.f) continue;
                const float w_miss = w_lep_miss(m_scan, m_tau, mLep);
                if (w_miss <= 0.f) continue;

                std::vector<float> pz2 = solve_pz_quadratic_tau(tau2, phi2, pT2, true,
                                                                n_charged_tracks_2, m_scan);
                for (float z1 : pz1) {
                  const float px1 = pT1 * c1, py1 = pT1 * s1;
                  const float E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
                  TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

                  for (float z2 : pz2) {
                    const float px2 = pT2 * c2, py2 = pT2 * s2;
                    const float E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
                    TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

                    TLorentzVector tau_full_1 = tau1 + nu1;
                    TLorentzVector tau_full_2 = tau2 + nu2;

                    const float theta_3D_1 = tau1.Vect().Angle(nu1.Vect());
                    const float theta_3D_2 = tau2.Vect().Angle(nu2.Vect());
                    const float p_tau1     = tau_full_1.P();
                    const float p_tau2     = tau_full_2.P();

                    const float THETA_MAX = 0.3f;
                    const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
                    const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

                    const float w_theta1 = theta_pdf_mixture_trunc(theta_3D_1, p_tau1, K1, THETA_MAX);
                    const float w_theta2 = theta_pdf_mixture_trunc(theta_3D_2, p_tau2, K2, THETA_MAX);
                    const float w_event  = w_theta1 * w_theta2 * (float)w_met * w_miss;

                    const float mditau   = (tau_full_1 + tau_full_2).M();
                    mditau_solutions.emplace_back(mditau, w_event);
                  }
                }
              }
            }
          } // sMET_y
        }   // sMET_x
      }     // j
    }       // i

    // If we found any solutions, don't retry
    if (!mditau_solutions.empty()) break;
    
  } // attempts

  return mditau_solutions;
}



#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>

// now look at implementing the MH algorithm
struct EvalOut {
  double weight;   // target w(theta) up to const (>=0)
  double m_mean;   // weighted mean mtautau over root pairs (only if weight>0)
};

// Evaluate target at θ = (phi1, phi2, sMETx, sMETy)
inline EvalOut evaluate_state(double phi1, double phi2,
                              double sMETx, double sMETy,
                              const TLorentzVector& tau1,
                              const TLorentzVector& tau2,
                              int n_charged_tracks_1,
                              int n_charged_tracks_2,
                              double METx_meas, double METy_meas,
                              double metres_xy, double inv_metres_xy2,
                              const MixCoeffs& K1, const MixCoeffs& K2)
{
  EvalOut out{0.0, 0.0};

  // Guard near-singular Δφ
  const double sdp = std::sin(phi2 - phi1);
  if (std::fabs(sdp) < 1e-6) return out;
  const double csc_dphi = 1.0 / sdp;

  // Solve neutrino transverse magnitudes from MET & azimuths
  const double pT1 =  csc_dphi * ( sMETx * std::sin(phi2) - sMETy * std::cos(phi2) );
  const double pT2 = -csc_dphi * ( sMETx * std::sin(phi1) - sMETy * std::cos(phi1) );
  if (pT1 <= 0.0 || pT2 <= 0.0) return out;

  // Gaussian MET weight (uncorrelated x/y)
  const double dx = sMETx - METx_meas;
  const double dy = sMETy - METy_meas;
  const double w_met = std::exp(-0.5 * (dx*dx + dy*dy) * inv_metres_xy2);

  // Solve pz roots for both taus (your version takes prong count)
  const auto roots1 = solve_pz_quadratic(tau1, phi1, pT1, n_charged_tracks_1);
  const auto roots2 = solve_pz_quadratic(tau2, phi2, pT2, n_charged_tracks_2);
  if (roots1.empty() || roots2.empty()) return out;

  // Accumulate over root pairs
  double sumW  = 0.0;
  double sumMW = 0.0;

  for (double z1 : roots1) {
    const double px1 = pT1 * std::cos(phi1), py1 = pT1 * std::sin(phi1);
    const double E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1);
    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

    for (double z2 : roots2) {
      const double px2 = pT2 * std::cos(phi2), py2 = pT2 * std::sin(phi2);
      const double E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2);
      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

      const TLorentzVector t1 = tau1 + nu1;
      const TLorentzVector t2 = tau2 + nu2;

      const double theta1 = tau1.Vect().Angle(nu1.Vect());
      const double theta2 = tau2.Vect().Angle(nu2.Vect());
      const double p1 = t1.P();
      const double p2 = t2.P();

      // Truncated Gaussian+Moyal mixture with your fitted parameterization
      const float w_theta1 = theta_pdf_mixture_trunc(static_cast<float>(theta1),
                                                     static_cast<float>(p1),
                                                     K1, MMC_THETA_MAX);
      const float w_theta2 = theta_pdf_mixture_trunc(static_cast<float>(theta2),
                                                     static_cast<float>(p2),
                                                     K2, MMC_THETA_MAX);

      const double w = w_met * static_cast<double>(w_theta1) * static_cast<double>(w_theta2);
      if (w <= 0.0) continue;

      const double mditau = (t1 + t2).M();
      sumW  += w;
      sumMW += w * mditau;
    }
  }

  if (sumW <= 0.0) return out;
  out.weight = sumW;
  out.m_mean = sumMW / sumW;
  return out;
}

inline double sqr(double x) { return x * x; }
inline double normal_cdf(double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); }
inline double wrap_to_pi(double a) {
  const double pi = M_PI, two = 2.0 * pi;
  a = std::fmod(a + pi, two);
  if (a < 0) a += two;
  return a - pi;
}

MMC_MH_Result AnalysisFCChh::solve_ditau_MMC_MH(
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
  double sigma_phi,      // proposal stddev for angles (radians)
  double sigma_met,      // proposal stddev for MET x,y; if <0, set to metres/sqrt(2)/2
  unsigned long long seed
)
{
  MMC_MH_Result R;

  /*
  static int event_counter = 0;
  event_counter++;
  

  if (event_counter % 10 != 0) {
      std::cout << "At event #" << event_counter << std::endl;
      return R;
  }
  */

  if (n_iter <= 0) return R;
  if (burn_in < 0) burn_in = 0;
  if (thin <= 0) thin = 1;

  // MET per-axis resolution (assume corr ≈ 0)
  const double metres_xy = metres / std::sqrt(2.0);
  const double inv_metres_xy2 = 1.0 / (metres_xy * metres_xy);

  if (sigma_met <= 0.0) sigma_met = metres_xy * 0.5;
  if (sigma_phi <= 0.0) sigma_phi = 0.15;

  // RNG
  std::mt19937_64 rng;
  if (seed == 0) {
    rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    rng.seed(seed);
  }
  std::normal_distribution<double> nphi(0.0, sigma_phi);
  std::normal_distribution<double> nmet(0.0, sigma_met);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  // Choose angular mixture params by prong
  const MixCoeffs& K1 = (n_charged_tracks_1 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;
  const MixCoeffs& K2 = (n_charged_tracks_2 <= 2) ? COEFFS_1PRONG : COEFFS_3PRONG;

  // ---- Initial state: neutrino phis ~ visible tau phis; MET' = measured MET ----
  double phi1 = wrap_to_pi(tau1.Phi());
  double phi2 = wrap_to_pi(tau2.Phi());
  double mpx  = MET_x_meas;
  double mpy  = MET_y_meas;

  // Evaluate initial state; if invalid, jitter a few times
  EvalOut cur = evaluate_state(phi1, phi2, mpx, mpy,
                               tau1, tau2,
                               n_charged_tracks_1, n_charged_tracks_2,
                               MET_x_meas, MET_y_meas,
                               metres_xy, inv_metres_xy2,
                               K1, K2);

  for (int tries = 0; tries < 100 && cur.weight == 0.0; ++tries) {
    phi1 = wrap_to_pi(phi1 + nphi(rng));
    phi2 = wrap_to_pi(phi2 + nphi(rng));
    mpx  = MET_x_meas + nmet(rng);
    mpy  = MET_y_meas + nmet(rng);
    cur  = evaluate_state(phi1, phi2, mpx, mpy,
                          tau1, tau2,
                          n_charged_tracks_1, n_charged_tracks_2,
                          MET_x_meas, MET_y_meas,
                          metres_xy, inv_metres_xy2,
                          K1, K2);
  }

  int accepts = 0;

  for (int t = 0; t < n_iter; ++t) {
    // Propose new state
    const double phi1_p = wrap_to_pi(phi1 + nphi(rng));
    const double phi2_p = wrap_to_pi(phi2 + nphi(rng));
    const double mpx_p  = mpx + nmet(rng);
    const double mpy_p  = mpy + nmet(rng);

    EvalOut prop = evaluate_state(phi1_p, phi2_p, mpx_p, mpy_p,
                                  tau1, tau2,
                                  n_charged_tracks_1, n_charged_tracks_2,
                                  MET_x_meas, MET_y_meas,
                                  metres_xy, inv_metres_xy2,
                                  K1, K2);

    // Symmetric proposal ⇒ MH ratio = w_prop / w_cur
    double alpha = 0.0;
    if (cur.weight == 0.0 && prop.weight == 0.0) {
      alpha = 0.0; // both invalid
    } else if (cur.weight == 0.0) {
      alpha = 1.0; // jump out of invalid
    } else {
      alpha = std::min(1.0, prop.weight / cur.weight);
    }

    if (uni(rng) < alpha) {
      ++accepts;
      phi1 = phi1_p; phi2 = phi2_p; mpx = mpx_p; mpy = mpy_p;
      cur  = prop;
    }

    // Record samples after burn-in, with thinning
    if (t >= burn_in && ((t - burn_in) % thin == 0) && cur.weight > 0.0) {
      R.masses.push_back(static_cast<float>(cur.m_mean));
    }
  }

  R.accept_rate = static_cast<double>(accepts) / static_cast<double>(n_iter);
  return R;
}



// Evaluate target at θ = (phi1, phi2, sMETx, sMETy, m1, m2)
inline EvalOut evaluate_state_lephad(double phi1, double phi2,
                              double sMETx, double sMETy,
                              double m1, double m2,
                              const TLorentzVector& tau1,
                              const TLorentzVector& tau2,
                              bool isLep1,
                              bool isLep2,
                              int n_charged_tracks_1,
                              int n_charged_tracks_2,
                              double METx_meas, double METy_meas,
                              double metres_xy, double inv_metres_xy2,
                              const MixCoeffs& K1, const MixCoeffs& K2)
{
  EvalOut out{0.0, 0.0};

  // Guard near-singular Δφ
  const double sdp = std::sin(phi2 - phi1);
  if (std::fabs(sdp) < 1e-6) return out;
  const double csc_dphi = 1.0 / sdp;

  // Solve neutrino transverse magnitudes from MET & azimuths
  const double pT1 =  csc_dphi * ( sMETx * std::sin(phi2) - sMETy * std::cos(phi2) );
  const double pT2 = -csc_dphi * ( sMETx * std::sin(phi1) - sMETy * std::cos(phi1) );
  if (pT1 <= 0.0 || pT2 <= 0.0) return out;

  // Gaussian MET weight (uncorrelated x/y)
  const double dx = sMETx - METx_meas;
  const double dy = sMETy - METy_meas;
  const double w_met = std::exp(-0.5 * (dx*dx + dy*dy) * inv_metres_xy2);

  // Solve pz roots for both taus (your version takes prong count)
  const auto roots1 = solve_pz_quadratic_tau(tau1, phi1, pT1, isLep1, n_charged_tracks_1, m1);
  const auto roots2 = solve_pz_quadratic_tau(tau2, phi2, pT2, isLep2, n_charged_tracks_2, m2);

  if (roots1.empty() || roots2.empty()) return out;

  // Accumulate over root pairs
  double sumW  = 0.0;
  double sumMW = 0.0;

  for (double z1 : roots1) {
    const double px1 = pT1 * std::cos(phi1), py1 = pT1 * std::sin(phi1);
    const double E1 = std::sqrt(px1*px1 + py1*py1 + z1*z1 + (isLep1 ? m1*m1 : 0.0));
    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

    for (double z2 : roots2) {
      const double px2 = pT2 * std::cos(phi2), py2 = pT2 * std::sin(phi2);
      const double E2 = std::sqrt(px2*px2 + py2*py2 + z2*z2 + (isLep2 ? m2*m2 : 0.0));
      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

      const TLorentzVector t1 = tau1 + nu1;
      const TLorentzVector t2 = tau2 + nu2;

      const double theta1 = tau1.Vect().Angle(nu1.Vect());
      const double theta2 = tau2.Vect().Angle(nu2.Vect());
      const double p1 = t1.P();
      const double p2 = t2.P();

      // Truncated Gaussian+Moyal mixture with your fitted parameterization
      const float w_theta1 = theta_pdf_mixture_trunc(static_cast<float>(theta1),
                                                     static_cast<float>(p1),
                                                     K1, MMC_THETA_MAX);
      const float w_theta2 = theta_pdf_mixture_trunc(static_cast<float>(theta2),
                                                     static_cast<float>(p2),
                                                     K2, MMC_THETA_MAX);

      const double w = w_met * static_cast<double>(w_theta1) * static_cast<double>(w_theta2);
      if (w <= 0.0) continue;

      const double mditau = (t1 + t2).M();
      sumW  += w;
      sumMW += w * mditau;
    }
  }

  if (sumW <= 0.0) return out;
  out.weight = sumW;
  out.m_mean = sumMW / sumW;
  return out;
}

// Rotate (x,y) to (par,perp) w.r.t. unit vector u=(ux,uy)
inline void xy_to_parperp(double x, double y, double ux, double uy, double& par, double& perp) {
  par  =  x*ux + y*uy;
  perp = -x*uy + y*ux;
}

inline void parperp_to_xy(double par, double perp, double ux, double uy, double& x, double& y) {
  x = par*ux - perp*uy;
  y = par*uy + perp*ux;
}

struct Seed {
  double phi1, phi2, mpx, mpy;
};

struct WalkerConfig {
  int    steps = 5000;         // steps per seed (includes burn-in)
  int    burn_in = 1000;       // per-seed burn-in
  int    thin = 1;
  double T0 = 1.5;             // initial temperature (annealing)
  double Tend = 0.3;           // final temperature
  double sigma_phi0 = 0.20;    // initial φ proposal width (rad)
  double sigma_par0 = 5.0;     // initial MET parallel proposal (GeV)
  double sigma_perp0 = 8.0;    // initial MET perpendicular proposal (GeV)
  double adapt_up = 1.2;       // multiplicative widen when stuck
  double adapt_dn = 0.85;      // multiplicative shrink when too-easy accept
  double target_acc_lo = 0.10; // acceptance tuning band (per block)
  double target_acc_hi = 0.40;
  int    block = 200;          // adaptation block length
  int    max_invalid_resamples = 20;
  double big_jump_prob = 0.10; // occasional big-jump proposals
  double big_jump_scale = 3.0;
};

inline double lin_cool(int t, int Ttot, double T0, double Tend) {
  if (Ttot <= 1) return Tend;
  const double a = (double)t / (double)(Ttot - 1);
  return T0 + a * (Tend - T0);
}

// Propose a step in (phi1,phi2, mp_x, mp_y) using ∥/⊥ Gaussian proposals
struct ProposalStep {
  double phi1_p, phi2_p, mpx_p, mpy_p;
};

inline ProposalStep propose_step(std::mt19937_64& rng,
                                 double phi1, double phi2, double mpx, double mpy,
                                 double sigma_phi, double sigma_par, double sigma_perp,
                                 double big_jump_prob, double big_jump_scale,
                                 double metx_meas, double mety_meas)
{
  std::normal_distribution<double> nphi(0.0, sigma_phi);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  // Direction of measured MET
  const double met_mag = std::hypot(metx_meas, mety_meas);
  double ux=1.0, uy=0.0;
  if (met_mag > 1e-9) { ux = metx_meas / met_mag; uy = mety_meas / met_mag; }

  // Draw ∥/⊥ increments; occasionally make a large jump
  const double jump_scale = (uni(rng) < big_jump_prob) ? big_jump_scale : 1.0;
  std::normal_distribution<double> npar(0.0, sigma_par * jump_scale);
  std::normal_distribution<double> nperp(0.0, sigma_perp * jump_scale);

  double dphi1 = nphi(rng);
  double dphi2 = nphi(rng);
  double dpar  = npar(rng);
  double dperp = nperp(rng);

  double dmx, dmy;
  parperp_to_xy(dpar, dperp, ux, uy, dmx, dmy);

  ProposalStep P;
  P.phi1_p = wrap_to_pi(phi1 + dphi1);
  P.phi2_p = wrap_to_pi(phi2 + dphi2);
  P.mpx_p  = mpx + dmx;
  P.mpy_p  = mpy + dmy;
  return P;
}

// One annealed, adaptive random walk from a given seed
inline void run_one_walker(const Seed& S,
                           const TLorentzVector& tau1,
                           const TLorentzVector& tau2,
                           int nprong1, int nprong2,
                           double METx_meas, double METy_meas,
                           double metres_xy, double inv_metres_xy2,
                           const MixCoeffs& K1, const MixCoeffs& K2,
                           const WalkerConfig& cfg,
                           std::mt19937_64& rng,
                           std::vector<float>& out_masses,
                           int& out_accepts,
                           int& out_steps)
{
  // State
  double phi1 = S.phi1, phi2 = S.phi2, mpx = S.mpx, mpy = S.mpy;

  // Initial evaluation; try to rescue invalid seeds
  EvalOut cur = evaluate_state(phi1, phi2, mpx, mpy,
                               tau1, tau2,
                               nprong1, nprong2,
                               METx_meas, METy_meas,
                               metres_xy, inv_metres_xy2,
                              K1, K2);

  std::normal_distribution<double> rescue_phi(0.0, std::max(0.5*cfg.sigma_phi0, 0.05));
  std::normal_distribution<double> rescue_met(0.0, std::max(0.5*cfg.sigma_par0,  2.0));
  int rescue_tries = 0;
  while (cur.weight == 0.0 && rescue_tries < cfg.max_invalid_resamples) {
    phi1 = wrap_to_pi(phi1 + rescue_phi(rng));
    phi2 = wrap_to_pi(phi2 + rescue_phi(rng));
    mpx  = METx_meas + rescue_met(rng);
    mpy  = METy_meas + rescue_met(rng);
    cur  = evaluate_state(phi1, phi2, mpx, mpy,
                          tau1, tau2,
                          nprong1, nprong2,
                          METx_meas, METy_meas,
                          metres_xy, inv_metres_xy2,
                          K1, K2);
    ++rescue_tries;
  }

  double sigma_phi  = cfg.sigma_phi0;
  double sigma_par  = cfg.sigma_par0;
  double sigma_perp = cfg.sigma_perp0;

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int accepts = 0, steps = 0;
  int acc_in_block = 0, in_block = 0;

  for (int t = 0; t < cfg.steps; ++t) {
    const double T = lin_cool(t, cfg.steps, cfg.T0, cfg.Tend);

    // Propose
    ProposalStep P = propose_step(rng, phi1, phi2, mpx, mpy,
                                  sigma_phi, sigma_par, sigma_perp,
                                  cfg.big_jump_prob, cfg.big_jump_scale,
                                  METx_meas, METy_meas);

    EvalOut prop = evaluate_state(P.phi1_p, P.phi2_p, P.mpx_p, P.mpy_p,
                                  tau1, tau2,
                                  nprong1, nprong2,
                                  METx_meas, METy_meas,
                                  metres_xy, inv_metres_xy2,
                                  K1, K2);

    // Annealed acceptance
    double alpha = 0.0;
    if (cur.weight == 0.0 && prop.weight == 0.0) {
      alpha = 0.0;
    } else if (cur.weight == 0.0) {
      alpha = 1.0; // escape invalid
    } else if (prop.weight == 0.0) {
      alpha = 0.0; // don't go to invalid
    } else {
      const double dlog = std::log(prop.weight) - std::log(cur.weight);
      alpha = std::min(1.0, std::exp(dlog / std::max(T, 1e-6)));
    }

    bool accepted = (uni(rng) < alpha);
    if (accepted) {
      ++accepts;
      phi1 = P.phi1_p; phi2 = P.phi2_p; mpx = P.mpx_p; mpy = P.mpy_p;
      cur  = prop;
    } else if (prop.weight == 0.0) {
      // Try a few quick rescues if we hit invalid region repeatedly
      int bad = 0;
      while (bad < cfg.max_invalid_resamples && cur.weight == 0.0) {
        ProposalStep R = propose_step(rng, phi1, phi2, mpx, mpy,
                                      sigma_phi*cfg.adapt_up, sigma_par*cfg.adapt_up, sigma_perp*cfg.adapt_up,
                                      cfg.big_jump_prob, cfg.big_jump_scale,
                                      METx_meas, METy_meas);
        EvalOut tmp = evaluate_state(R.phi1_p, R.phi2_p, R.mpx_p, R.mpy_p,
                                     tau1, tau2,
                                     nprong1, nprong2,
                                     METx_meas, METy_meas,
                                     metres_xy, inv_metres_xy2,
                                     K1, K2);
        if (tmp.weight > 0.0) {
          phi1 = R.phi1_p; phi2 = R.phi2_p; mpx = R.mpx_p; mpy = R.mpy_p; cur = tmp;
          break;
        }
        ++bad;
      }
    }

    // Record after burn-in with thinning, for valid states
    if (t >= cfg.burn_in && ((t - cfg.burn_in) % cfg.thin == 0) && cur.weight > 0.0) {
      out_masses.push_back(static_cast<float>(cur.m_mean));
    }

    // Adaptation block
    ++in_block;
    if (accepted) ++acc_in_block;
    if (in_block >= cfg.block) {
      const double acc = static_cast<double>(acc_in_block) / static_cast<double>(in_block);
      if (acc < cfg.target_acc_lo) {
        sigma_phi  *= cfg.adapt_up;
        sigma_par  *= cfg.adapt_up;
        sigma_perp *= cfg.adapt_up;
      } else if (acc > cfg.target_acc_hi) {
        sigma_phi  *= cfg.adapt_dn;
        sigma_par  *= cfg.adapt_dn;
        sigma_perp *= cfg.adapt_dn;
      }
      in_block = 0; acc_in_block = 0;
    }

    ++steps;
  }

  out_accepts += accepts;
  out_steps   += steps;
}


MMCPoint AnalysisFCChh::mmc_from_samples_mode(const std::vector<float>& v) {
  MMCPoint out{NAN, NAN, NAN};
  if (v.size() < 8) return out;

  // Range
  auto [mn_it, mx_it] = std::minmax_element(v.begin(), v.end());
  const double lo = *mn_it, hi = *mx_it;
  if (!(hi > lo)) return out;

  // Freedman–Diaconis binning
  std::vector<float> w = v;
  std::nth_element(w.begin(), w.begin()+w.size()/4, w.end());
  const double q1 = w[w.size()/4];
  std::nth_element(w.begin(), w.begin()+3*w.size()/4, w.end());
  const double q3 = w[3*w.size()/4];
  const double iqr = std::max(1e-6, q3 - q1);
  const double h = 2.0 * iqr / std::cbrt(static_cast<double>(v.size()));
  const int nb = std::clamp(static_cast<int>(std::ceil((hi - lo) / std::max(h, 1e-3))), 20, 200);

  // Histogram
  std::vector<int> c(nb, 0);
  const double bw = (hi - lo) / nb;
  for (float x : v) {
    int b = static_cast<int>((x - lo) / bw);
    b = std::clamp(b, 0, nb - 1);
    ++c[b];
  }

  // Peak bin
  int b0 = int(std::max_element(c.begin(), c.end()) - c.begin());

  // Parabolic interpolation with neighbors (optional refinement)
  auto yc = [&](int b){ return static_cast<double>(c[ std::clamp(b,0,nb-1) ]); };
  const double yL = yc(b0-1), y0 = yc(b0), yR = yc(b0+1);
  double delta = 0.0;
  const double denom = (yL - 2*y0 + yR);
  if (std::fabs(denom) > 1e-12) delta = 0.5*(yL - yR)/denom; // in [-0.5,0.5] ideally
  delta = std::clamp(delta, -0.5, 0.5);

  const double mode = (lo + (b0 + 0.5 + delta) * bw);

  // 68% equal–tailed interval
  std::vector<float> s = v;
  std::sort(s.begin(), s.end());
  auto q = [&](double p){
    double idx = p * (s.size()-1);
    size_t i = static_cast<size_t>(std::floor(idx));
    double frac = idx - i;
    double a = s[i], b = s[std::min(i+1, s.size()-1)];
    return static_cast<float>(a + frac*(b - a));
  };
  out.m = static_cast<float>(mode);
  out.lo68 = q(0.16);
  out.hi68 = q(0.84);
  return out;
}

using FourVec = ROOT::Math::PxPyPzEVector;

// xWt with hadronic (W->jj), tau (W->tau nu), and e/mu (W->ℓ nu) hypotheses.
// Returns the minimum chi2 over all combinations.
// Inputs: b-jets, tau-jets, untagged light jets, electrons, muons, MET components.
float AnalysisFCChh::xwt(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& taujets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untaggedjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  float met_px, float met_py
) {
  const int nb = (int)bjets.size();
  const int nu = (int)untaggedjets.size();
  const int nt = (int)taujets.size();
  const int ne = (int)electrons.size();
  const int nm = (int)muons.size();
  if (nb < 1) return -999.f;  // need at least one b for a top candidate

  constexpr float mW_nom       = 80.379f;  // GeV
  constexpr float mt_nom       = 172.5f;   // GeV
  constexpr float sigma_floor  = 1e-3f;

  auto p4b = [&](int ib) { return getTLV_reco(bjets[ib]); };
  auto p4j = [&](int ij) { return getTLV_reco(untaggedjets[ij]); };
  auto p4t = [&](int it) { return getTLV_reco(taujets[it]);     };
  auto p4e = [&](int ie) { return getTLV_reco(electrons[ie]);   };
  auto p4m = [&](int im) { return getTLV_reco(muons[im]);       };

  auto rel_sigma = [&](float m) {
    // simple 10% relative "mass resolution" with a small floor
    return std::max(0.1f * std::abs(m), sigma_floor);
  };
  auto add_in_quad = [](float a, float b){ return std::sqrt(a*a + b*b); };

  // Neutrino from MET (pz=0, massless)
  TLorentzVector nu4;
  if (std::isfinite(met_px) && std::isfinite(met_py)) {
    const double met = std::hypot(met_px, met_py);
    nu4.SetPxPyPzE(met_px, met_py, 0.0, met);
  }

  float best = std::numeric_limits<float>::infinity();

  // ---------------------------
  // A) Hadronic W: W->jj (untagged only)
  // ---------------------------
  if (nu >= 2) {
    for (int ib = 0; ib < nb; ++ib) {
      const TLorentzVector b = p4b(ib);
      for (int i = 0; i < nu; ++i) {
        const TLorentzVector ji = p4j(i);
        for (int k = i+1; k < nu; ++k) {
          const TLorentzVector jk = p4j(k);
          const TLorentzVector Wjj   = ji + jk;
          const TLorentzVector topbj = b  + Wjj;

          const float mW = (float)Wjj.M();
          const float mt = (float)topbj.M();

          const float pullW = (mW - mW_nom) / rel_sigma(mW);
          const float pullt = (mt - mt_nom) / rel_sigma(mt);
          const float chi2  = add_in_quad(pullW, pullt);

          if (chi2 < best) best = chi2;
        }
      }
    }
  }

  // Helper for leptonic/tau W: compute chi2 with l (τ/e/μ) + MET
  auto consider_leptonicW = [&](const TLorentzVector& b, const TLorentzVector& l) {
    if (!std::isfinite(met_px) || !std::isfinite(met_py)) return;

    const float pt_l  = (float)l.Pt();
    const float pt_nu = (float)nu4.Pt();
    const float dphi  = std::abs(TVector2::Phi_mpi_pi(l.Phi() - nu4.Phi()));

    const float mT2 = 2.0f * pt_l * pt_nu * (1.0f - std::cos(dphi));
    const float mT  = std::sqrt(std::max(0.0f, mT2));

    const float mtop = (float)(b + l + nu4).M();

    const float pullW = (mT  - mW_nom) / rel_sigma(mT);
    const float pullt = (mtop - mt_nom) / rel_sigma(mtop);
    const float chi2  = add_in_quad(pullW, pullt);

    if (chi2 < best) best = chi2;
  };

  // ---------------------------
  // B) Tau W: W->τν  (use tau_h + MET)
  // ---------------------------
  if (nt >= 1 && std::isfinite(met_px) && std::isfinite(met_py)) {
    for (int ib = 0; ib < nb; ++ib) {
      const TLorentzVector b = p4b(ib);
      for (int it = 0; it < nt; ++it) {
        consider_leptonicW(b, p4t(it));
      }
    }
  }

  // ---------------------------
  // C) Leptonic W: W->eν and W->μν  (use e/μ + MET)
  // ---------------------------
  if ((ne >= 1 || nm >= 1) && std::isfinite(met_px) && std::isfinite(met_py)) {
    for (int ib = 0; ib < nb; ++ib) {
      const TLorentzVector b = p4b(ib);
      for (int ie = 0; ie < ne; ++ie) consider_leptonicW(b, p4e(ie));
      for (int im = 0; im < nm; ++im) consider_leptonicW(b, p4m(im));
    }
  }

  return std::isfinite(best) ? best : -999.f;
}

float AnalysisFCChh::topness(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  TLorentzVector tau1, TLorentzVector tau2,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& untaggedjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& electrons,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& muons,
  float met_px, float met_py
)
{
  (void)electrons;
  (void)muons;

  const int nb = (int)bjets.size();
  const int nu = (int)untaggedjets.size();
  if (nb < 2 || nu < 2) return -999.f;

  constexpr float mW_nom = 80.379f;
  constexpr float mt_nom = 172.5f;
  constexpr float sigmaW = 10.0f;
  constexpr float sigmat = 20.0f;
  constexpr float invSigmaW2 = 1.0f / (sigmaW * sigmaW);
  constexpr float invSigmat2 = 1.0f / (sigmat * sigmat);

  auto p4b = [&](int ib) { return getTLV_reco(bjets[ib]); };
  auto p4j = [&](int ij) { return getTLV_reco(untaggedjets[ij]); };

  struct Wcand {
    int i;
    int k;
    TLorentzVector p4;
    float m;
  };

  std::vector<Wcand> wcands;
  wcands.reserve(nu * (nu - 1) / 2);
  for (int i = 0; i < nu; ++i) {
    const TLorentzVector ji = p4j(i);
    for (int k = i + 1; k < nu; ++k) {
      const TLorentzVector jk = p4j(k);
      const TLorentzVector w = ji + jk;
      wcands.push_back({i, k, w, (float)w.M()});
    }
  }

  float best_had = std::numeric_limits<float>::infinity();

  for (int ib1 = 0; ib1 < nb; ++ib1) {
    const TLorentzVector b1 = p4b(ib1);
    for (int ib2 = ib1 + 1; ib2 < nb; ++ib2) {
      const TLorentzVector b2 = p4b(ib2);
      for (size_t iw1 = 0; iw1 < wcands.size(); ++iw1) {
        const Wcand& w1 = wcands[iw1];
        for (size_t iw2 = iw1 + 1; iw2 < wcands.size(); ++iw2) {
          const Wcand& w2 = wcands[iw2];
          if (w1.i == w2.i || w1.i == w2.k || w1.k == w2.i || w1.k == w2.k) continue;

          const float mt1 = (float)(b1 + w1.p4).M();
          const float mt2 = (float)(b2 + w2.p4).M();
          const float chi2_a =
            (w1.m - mW_nom) * (w1.m - mW_nom) * invSigmaW2 +
            (mt1 - mt_nom) * (mt1 - mt_nom) * invSigmat2 +
            (w2.m - mW_nom) * (w2.m - mW_nom) * invSigmaW2 +
            (mt2 - mt_nom) * (mt2 - mt_nom) * invSigmat2;

          const float mt1b = (float)(b1 + w2.p4).M();
          const float mt2b = (float)(b2 + w1.p4).M();
          const float chi2_b =
            (w2.m - mW_nom) * (w2.m - mW_nom) * invSigmaW2 +
            (mt1b - mt_nom) * (mt1b - mt_nom) * invSigmat2 +
            (w1.m - mW_nom) * (w1.m - mW_nom) * invSigmaW2 +
            (mt2b - mt_nom) * (mt2b - mt_nom) * invSigmat2;

          const float chi2 = std::min(chi2_a, chi2_b);
          if (chi2 < best_had) best_had = chi2;
        }
      }
    }
  }

  // Mixed hypothesis: one hadronic top (b+jj) + one leptonic top (b+l+MET)
  float best_mixed = std::numeric_limits<float>::infinity();

  std::vector<TLorentzVector> leptons;
  leptons.reserve(2);
  if (std::isfinite(tau1.Pt()) && tau1.E() > 0.0) leptons.push_back(tau1);
  if (std::isfinite(tau2.Pt()) && tau2.E() > 0.0) leptons.push_back(tau2);

  if (!leptons.empty() && std::isfinite(met_px) && std::isfinite(met_py)) {
    const double met = std::hypot(met_px, met_py);
    TLorentzVector nu4;
    nu4.SetPxPyPzE(met_px, met_py, 0.0, met);

    for (int ib1 = 0; ib1 < nb; ++ib1) {
      const TLorentzVector b1 = p4b(ib1);
      for (int ib2 = ib1 + 1; ib2 < nb; ++ib2) {
        const TLorentzVector b2 = p4b(ib2);
        for (const auto& w : wcands) {
          const float mt_had1 = (float)(b1 + w.p4).M();
          const float mt_had2 = (float)(b2 + w.p4).M();
          const float pullW_had =
            (w.m - mW_nom) * (w.m - mW_nom) * invSigmaW2;

          for (const auto& l : leptons) {
            const float pt_l  = (float)l.Pt();
            const float pt_nu = (float)nu4.Pt();
            const float dphi  = std::abs(TVector2::Phi_mpi_pi(l.Phi() - nu4.Phi()));
            const float mT2 = 2.0f * pt_l * pt_nu * (1.0f - std::cos(dphi));
            const float mT  = std::sqrt(std::max(0.0f, mT2));

            const float mt_lep1 = (float)(b1 + l + nu4).M();
            const float mt_lep2 = (float)(b2 + l + nu4).M();

            const float chi2_had1_lep2 =
              pullW_had +
              (mt_had1 - mt_nom) * (mt_had1 - mt_nom) * invSigmat2 +
              (mT - mW_nom) * (mT - mW_nom) * invSigmaW2 +
              (mt_lep2 - mt_nom) * (mt_lep2 - mt_nom) * invSigmat2;

            const float chi2_had2_lep1 =
              pullW_had +
              (mt_had2 - mt_nom) * (mt_had2 - mt_nom) * invSigmat2 +
              (mT - mW_nom) * (mT - mW_nom) * invSigmaW2 +
              (mt_lep1 - mt_nom) * (mt_lep1 - mt_nom) * invSigmat2;

            const float chi2 = std::min(chi2_had1_lep2, chi2_had2_lep1);
            if (chi2 < best_mixed) best_mixed = chi2;
          }
        }
      }
    }
  }

  const float best = std::min(best_had, best_mixed);
  return std::isfinite(best) ? best : -999.f;
}


static inline FourVec toP4(const edm4hep::ReconstructedParticleData& p) {
  return FourVec(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);
}

static inline float deltaPhi(float phi1, float phi2) {
  float d = phi1 - phi2;
  while (d >  M_PI) d -= 2.f * M_PI;
  while (d < -M_PI) d += 2.f * M_PI;
  return d;
}

// RMS functions
float AnalysisFCChh::RMS_mjj(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets) {
  const size_t n = bjets.size();
  if (n < 2) return -999.f;

  double sum_sq = 0.0;
  int count = 0;

  for (size_t i = 0; i < n; ++i) {
    FourVec p4i = toP4(bjets[i]);
    for (size_t j = i+1; j < n; ++j) {
      FourVec p4j = toP4(bjets[j]);
      double mjj = (p4i + p4j).M();
      sum_sq += mjj * mjj;
      ++count;
    }
  }

  return count > 0 ? std::sqrt(sum_sq / count) : -999.f;
}

float AnalysisFCChh::RMS_deta(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets) {
  const size_t n = bjets.size();
  if (n < 2) return -999.f;

  double sum_sq = 0.0;
  int count = 0;

  for (size_t i = 0; i < n; ++i) {
    FourVec p4i = toP4(bjets[i]);
    for (size_t j = i+1; j < n; ++j) {
      FourVec p4j = toP4(bjets[j]);
      double deta = p4i.Eta() - p4j.Eta();
      sum_sq += deta * deta;
      ++count;
    }
  }

  return count > 0 ? std::sqrt(sum_sq / count) : -999.f;
}

float AnalysisFCChh::RMS_dR(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets) {
  const size_t n = bjets.size();
  if (n < 2) return -999.f;

  double sum_sq = 0.0;
  int count = 0;

  for (size_t i = 0; i < n; ++i) {
    FourVec p4i = toP4(bjets[i]);
    for (size_t j = i+1; j < n; ++j) {
      FourVec p4j = toP4(bjets[j]);
      double deta = p4i.Eta() - p4j.Eta();
      double dphi = deltaPhi(p4i.Phi(), p4j.Phi());
      double dR = std::sqrt(deta*deta + dphi*dphi);
      sum_sq += dR * dR;
      ++count;
    }
  }

  return count > 0 ? std::sqrt(sum_sq / count) : -999.f;
}

// with truth variables
ROOT::VecOps::RVec<float> AnalysisFCChh::get_x_fraction_truth(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> visible_particle,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> MET) {
  ROOT::VecOps::RVec<float> results_vec;

  if (visible_particle.size() < 1 || MET.size() < 1) {
    results_vec.push_back(-999.);
    return results_vec;
  }

  // get the components of the calculation
  TLorentzVector met_tlv = getTLV_reco(MET.at(0));
  // TLorentzVector met_tlv = getTLV_MET(MET.at(0));
  TLorentzVector vis_tlv = getTLV_MC(visible_particle.at(0));

  // float x_fraction = vis_tlv.Pt()/(vis_tlv.Pt()+met_tlv.Pt()); // try scalar
  // sum
  float x_fraction =
      vis_tlv.Pt() / (vis_tlv + met_tlv).Pt(); // vector sum makes more sense?

  // std::cout << " Debug m_col: pT_vis truth : " << vis_tlv.Pt() << std::endl;
  // std::cout << " Debug m_col: pT_miss truth : " << met_tlv.Pt() << std::endl;
  // std::cout << " Debug m_col: x truth with vector sum: " << x_fraction <<
  // std::endl; std::cout << " Debug m_col: x with vector sum: " <<
  // vis_tlv.Pt()/(vis_tlv+met_tlv).Pt() << std::endl;

  results_vec.push_back(x_fraction);
  return results_vec;
}


// Thrust
float AnalysisFCChh::thrust(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
)
{
  // Collect p4s from b-jets and tau candidates
  std::vector<TLorentzVector> p4s;
  p4s.reserve(bjets.size() + tau_cands.size());

  for (const auto& obj : bjets) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }
  for (const auto& obj : tau_cands) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }

  if (p4s.empty()) return 0.f;

  // Denominator: sum of |p|
  double denom = 0.0;
  std::vector<double> mags;
  mags.reserve(p4s.size());
  for (const auto& p4 : p4s) {
    double m = p4.P();
    mags.push_back(m);
    denom += m;
  }
  if (denom == 0.0) return 0.f;

  // Initial axis: direction of highest-momentum object
  int iMax = 0;
  for (size_t i = 1; i < mags.size(); ++i) {
    if (mags[i] > mags[iMax]) iMax = i;
  }
  TVector3 n = p4s[iMax].Vect().Unit();

  const int    maxIter = 1000;
  const double tol     = 1e-6;

  for (int iter = 0; iter < maxIter; ++iter) {
    TVector3 v_sum(0., 0., 0.);

    for (const auto& p4 : p4s) {
      TVector3 p = p4.Vect();
      double proj = p.Dot(n);
      double s    = (proj >= 0.0) ? 1.0 : -1.0;
      v_sum += s * p;
    }

    if (v_sum.Mag2() == 0.0) break;

    TVector3 n_new = v_sum.Unit();
    if ((n_new - n).Mag() < tol) {
      n = n_new;
      break;
    }
    n = n_new;
  }

  double num = 0.0;
  for (const auto& p4 : p4s) {
    num += std::abs(p4.Vect().Dot(n));
  }

  double T = num / denom;
  return static_cast<float>(T);
}


// Sphericity
float AnalysisFCChh::sphericity(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
)
{
  std::vector<TLorentzVector> p4s;
  p4s.reserve(bjets.size() + tau_cands.size());

  for (const auto& obj : bjets) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }
  for (const auto& obj : tau_cands) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }

  if (p4s.empty()) return 0.f;

  TMatrixDSym M(3);
  M.Zero();

  double norm = 0.0;
  for (const auto& p4 : p4s) {
    TVector3 p = p4.Vect();
    norm += p.Mag2();

    M(0,0) += p.X()*p.X();
    M(0,1) += p.X()*p.Y();
    M(0,2) += p.X()*p.Z();
    M(1,1) += p.Y()*p.Y();
    M(1,2) += p.Y()*p.Z();
    M(2,2) += p.Z()*p.Z();
  }

  if (norm == 0.0) return 0.f;

  M(1,0) = M(0,1);
  M(2,0) = M(0,2);
  M(2,1) = M(1,2);

  M *= 1.0 / norm;

  TMatrixDSymEigen eigen(M);
  TVectorD evals = eigen.GetEigenValues(); // ascending

  const double lambda1 = evals(0);
  const double lambda2 = evals(1);
  const double lambda3 = evals(2);

  // Optional debug check:
  double sum = lambda1 + lambda2 + lambda3;
  

  const double S = 1.5 * (lambda2 + lambda3);
  if (S > 1) {
    std::cout << "Eigenvalues: "
            << lambda1 << " " << lambda2 << " " << lambda3
            << "  sum = " << sum << std::endl;
  }
  return static_cast<float>(S);
}

// Aplanarity
float AnalysisFCChh::aplanarity(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_cands
)
{
  std::vector<TLorentzVector> p4s;
  p4s.reserve(bjets.size() + tau_cands.size());

  for (const auto& obj : bjets) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }
  for (const auto& obj : tau_cands) {
    TLorentzVector p4 = getTLV_reco(obj);
    if (p4.P() > 0.) p4s.push_back(p4);
  }

  if (p4s.empty()) return 0.f;

  TMatrixDSym M(3);
  M.Zero();

  double norm = 0.0;
  for (const auto& p4 : p4s) {
    TVector3 p = p4.Vect();
    norm += p.Mag2();

    M(0,0) += p.X()*p.X();
    M(0,1) += p.X()*p.Y();
    M(0,2) += p.X()*p.Z();
    M(1,1) += p.Y()*p.Y();
    M(1,2) += p.Y()*p.Z();
    M(2,2) += p.Z()*p.Z();
  }

  if (norm == 0.0) return 0.f;

  M(1,0) = M(0,1);
  M(2,0) = M(0,2);
  M(2,1) = M(1,2);

  M *= 1.0 / norm;

  TMatrixDSymEigen eigen(M);
  TVectorD evals = eigen.GetEigenValues(); // ascending
  const double lambda3 = evals(2);

  const double A = 1.5 * lambda3;
  return static_cast<float>(A);
}

float AnalysisFCChh::mTb_min(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bjets,
  const edm4hep::ReconstructedParticleData& met   // MET as a RecoParticle with px,py in momentum.{x,y}
) {
  if (bjets.empty()) return -999.f;

  // MET components and magnitude
  const float met_px = static_cast<float>(met.momentum.x);
  const float met_py = static_cast<float>(met.momentum.y);
  const float met_pt = std::hypot(met_px, met_py);
  if (!(met_pt > 0.f)) return -999.f;

  const float met_phi = std::atan2(met_py, met_px);

  float best = std::numeric_limits<float>::infinity();

  for (const auto& b : bjets) {
    const float b_px  = static_cast<float>(b.momentum.x);
    const float b_py  = static_cast<float>(b.momentum.y);
    const float b_pt  = std::hypot(b_px, b_py);
    if (!(b_pt > 0.f)) continue;

    const float b_phi = std::atan2(b_py, b_px);
    const float dphi  = deltaPhi(b_phi, met_phi);

    const float mT2 = 2.f * b_pt * met_pt * (1.f - std::cos(dphi));
    const float mT  = (mT2 > 0.f) ? std::sqrt(mT2) : 0.f;

    if (mT < best) best = mT;
  }

  return std::isfinite(best) ? best : -999.f;
}


ROOT::VecOps::RVec<float> AnalysisFCChh::get_mtautau_col(
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> ll_pair_merged,
  float x1, 
  float x2 )
{
  ROOT::VecOps::RVec<float> results_vec;

  // Sanity checks
  if (ll_pair_merged.empty()) return results_vec;
  if (x1 <= 0.f || x2 <= 0.f || x1 >= 1.f || x2 >= 1.f) return results_vec;

  float best_m = -1.f;
  float best_dist = std::numeric_limits<float>::max();

  for (size_t i = 0; i < ll_pair_merged.size(); ++i) {
    float m_vis = getTLV_reco(ll_pair_merged[i]).M();
    float m_col = m_vis / std::sqrt(x1 * x2);

    if (m_col > 0.f && std::isfinite(m_col)) {
      float dist = std::abs(m_col - 125.f);
      if (dist < best_dist) {
        best_m = m_col;
        best_dist = dist;
      }
    }
  }

  // Only store best if valid
  if (best_m > 0.f) {
    results_vec.push_back(best_m);
  }

  return results_vec;
}

// merge the invariant bb mass and the tautau colinear mass for the
// bbtautau(emu) analysis
ROOT::VecOps::RVec<float> AnalysisFCChh::get_mbbtautau_col(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> bb_pair_merged,
    ROOT::VecOps::RVec<float> mtautau_col) {
  ROOT::VecOps::RVec<float> results_vec;

  // here there is no result if any of the arguments is actually not filled or
  // if the mtautau col is at default value -999.
  if (bb_pair_merged.size() < 1 || mtautau_col.size() < 1 ||
      mtautau_col.at(0) <= 0.) {
    results_vec.push_back(-999.);
    return results_vec;
  }

  float mbb = getTLV_reco(bb_pair_merged.at(0)).M();

  results_vec.push_back(mbb + mtautau_col.at(0));
  return results_vec;
}

ROOT::VecOps::RVec<float> AnalysisFCChh::get_m_dihiggs(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> tau_pair_merged,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> yy_pair_merged,
    float x1, 
    float x2 ) {
  ROOT::VecOps::RVec<float> results_vec;

  // Check if input pairs and collinear fractions are valid
  if (tau_pair_merged.size() < 1 || yy_pair_merged.size() < 1) {
    results_vec.push_back(-999.); // Default value for invalid input
    return results_vec;
  }

  // Basic sanity checks
  if (x1 <= 0.f || x2 <= 0.f || x1 >= 1.f || x2 >= 1.f) {
    results_vec.push_back(-999.);
    //std::cout << "Failing with x1x2 being 0<x<1" << std::endl;

    return results_vec;
  }

  // Get the collinear mass of the tau-tau system
  float m_tautau_vis = getTLV_reco(tau_pair_merged.at(0)).M();
  float m_tautau_col = m_tautau_vis / sqrt(x1*x2);
  // Get the invariant mass of the gamma-gamma system
  float m_yy = getTLV_reco(yy_pair_merged.at(0)).M();

  // Calculate the di-Higgs invariant mass
  results_vec.push_back(m_tautau_col + m_yy);

  return results_vec;
}



// truth-matching for HHH
int AnalysisFCChh::find_hhh_signal_match(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets,
  float dR_thres)
{
  // Need at least 4 truth Bs and 2 truth taus to even try
  if (truth_B_fromH.size() < 4 || truth_tauhad_vis_fromH.size() < 2) return false;
  if (reco_b_jets.empty() || reco_tau_jets.empty()) return false;

  // Build TLVs
  std::vector<TLorentzVector> tlv_truth_B; tlv_truth_B.reserve(truth_B_fromH.size());
  for (const auto& b : truth_B_fromH) tlv_truth_B.emplace_back(getTLV_MC(b));

  std::vector<TLorentzVector> tlv_truth_tau; tlv_truth_tau.reserve(truth_tauhad_vis_fromH.size());
  for (const auto& t : truth_tauhad_vis_fromH) tlv_truth_tau.emplace_back(getTLV_MC(t));

  std::vector<TLorentzVector> tlv_reco_b; tlv_reco_b.reserve(reco_b_jets.size());
  for (const auto& r : reco_b_jets) tlv_reco_b.emplace_back(getTLV_reco(r));

  std::vector<TLorentzVector> tlv_reco_tau; tlv_reco_tau.reserve(reco_tau_jets.size());
  for (const auto& r : reco_tau_jets) tlv_reco_tau.emplace_back(getTLV_reco(r));

  auto greedy_match_count = [dR_thres](const std::vector<TLorentzVector>& truth,
                                       const std::vector<TLorentzVector>& reco) -> int {
    if (truth.empty() || reco.empty()) return 0;

    // match in descending pT of truth to stabilize
    std::vector<int> order(truth.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j){
      return truth[i].Pt() > truth[j].Pt();
    });

    std::vector<char> used(reco.size(), 0);
    int matched = 0;

    for (int it : order) {
      double bestDR = 1e9;
      int bestJ = -1;
      for (size_t jr = 0; jr < reco.size(); ++jr) {
        if (used[jr]) continue;
        const double dR = truth[it].DeltaR(reco[jr]);
        if (dR < dR_thres && dR < bestDR) {
          bestDR = dR;
          bestJ = static_cast<int>(jr);
        }
      }
      if (bestJ >= 0) {
        used[bestJ] = 1;
        ++matched;
      }
    }
    return matched;
  };

  // Count matches
  const int nB_matched   = greedy_match_count(tlv_truth_B,   tlv_reco_b);
  const int nTau_matched = greedy_match_count(tlv_truth_tau, tlv_reco_tau);

  // Encode result
  int flag = 0;
  if (nTau_matched >= 2) flag += 2;
  if (nB_matched  >= 4) flag += 4;

  return flag;  // 0, 2, 4, or 6
}


// Bits (keep these consistent with what you read downstream)
enum : int {
  HAS_BB_LR    = 1 << 0,  // 1 : some bb-tagged LR jet contains >=2 truth B hadrons
  HAS_TAUTAU   = 1 << 1,  // 2 : >=2 truth taus matched to reco tau-jets OR ττ LR contains >=2 truth taus
  HAS_2B       = 1 << 2,  // 4 : >=2 truth B hadrons matched to distinct R=0.4 b-jets
  HAS_4B       = 1 << 3,  // 8 : >=4 truth B hadrons total (LR + resolved, no double counting)
  HAS_4B2TAU   = 1 << 4   // 16: condition for "4b2tau" satisfied (>=4 B AND >=2 τ)
};
int AnalysisFCChh::find_hhh_signal_match_LR(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets_R04,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_LR_bb_jets,      // bb-tagged LR jets
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_LR_tautau_jets,  // tautau-tagged LR jets
  float dR_match_b,    // e.g. 0.3–0.4
  float dR_match_tau,  // e.g. 0.2–0.3
  float Rmatch_LR      // e.g. 0.8 or 1.5
) {
  // --- Build TLVs ---
  std::vector<TLorentzVector> tB;   tB.reserve(truth_B_fromH.size());
  for (const auto& x : truth_B_fromH) tB.emplace_back(getTLV_MC(x));

  std::vector<TLorentzVector> tTau; tTau.reserve(truth_tauhad_vis_fromH.size());
  for (const auto& x : truth_tauhad_vis_fromH) tTau.emplace_back(getTLV_MC(x));

  std::vector<TLorentzVector> rBJ;  rBJ.reserve(reco_b_jets_R04.size());
  for (const auto& x : reco_b_jets_R04) rBJ.emplace_back(getTLV_reco(x));

  std::vector<TLorentzVector> rTau; rTau.reserve(reco_tau_jets.size());
  for (const auto& x : reco_tau_jets) rTau.emplace_back(getTLV_reco(x));

  std::vector<TLorentzVector> rLR_bb;  rLR_bb.reserve(reco_LR_bb_jets.size());
  for (const auto& x : reco_LR_bb_jets) rLR_bb.emplace_back(getTLV_reco(x));

  std::vector<TLorentzVector> rLR_tau; rLR_tau.reserve(reco_LR_tautau_jets.size());
  for (const auto& x : reco_LR_tautau_jets) rLR_tau.emplace_back(getTLV_reco(x));

  auto count_in_cone = [](const std::vector<TLorentzVector>& objs,
                          const TLorentzVector& axis, double R) {
    int n = 0;
    for (const auto& o : objs) if (axis.DeltaR(o) < R) ++n;
    return n;
  };

  int flag = 0;

  // -----------------------------
  // Original bits (kept as-is)
  // -----------------------------

  // (1) bb-tagged LR contains >=2 truth B hadrons
  bool bb_in_LR = false;
  if (!tB.empty() && !rLR_bb.empty()) {
    for (const auto& J : rLR_bb) {
      if (count_in_cone(tB, J, Rmatch_LR) >= 2) { bb_in_LR = true; break; }
    }
  }
  if (bb_in_LR) flag |= HAS_BB_LR;

  // (2) >=2 truth B hadrons matched to distinct R=0.4 b-jets (greedy)
  int nB_matched_resolved_for_bit = 0;
  if (!tB.empty() && !rBJ.empty()) {
    std::vector<char> used(rBJ.size(), 0);
    for (size_t it = 0; it < tB.size(); ++it) {
      int bestJ = -1; double bestDR = 1e9;
      for (size_t jr = 0; jr < rBJ.size(); ++jr) {
        if (used[jr]) continue;
        const double dR = tB[it].DeltaR(rBJ[jr]);
        if (dR < dR_match_b && dR < bestDR) { bestDR = dR; bestJ = (int)jr; }
      }
      if (bestJ >= 0) { used[bestJ] = 1; if (++nB_matched_resolved_for_bit >= 2) break; }
    }
  }
  if (nB_matched_resolved_for_bit >= 2) flag |= HAS_2B;

  // (3) ττ condition (either 2 matched τ-jets or ττ LR contains >=2 truth τ)
  int  nTau_matched_resolved_for_bit = 0;
  bool tautau_in_LR = false;

  if (!tTau.empty() && !rTau.empty()) {
    std::vector<char> used(rTau.size(), 0);
    for (size_t it = 0; it < tTau.size(); ++it) {
      int bestJ = -1; double bestDR = 1e9;
      for (size_t jr = 0; jr < rTau.size(); ++jr) {
        if (used[jr]) continue;
        const double dR = tTau[it].DeltaR(rTau[jr]);
        if (dR < dR_match_tau && dR < bestDR) { bestDR = dR; bestJ = (int)jr; }
      }
      if (bestJ >= 0) { used[bestJ] = 1; if (++nTau_matched_resolved_for_bit >= 2) break; }
    }
  }

  if (!tTau.empty() && !rLR_tau.empty()) {
    for (const auto& J : rLR_tau) {
      if (count_in_cone(tTau, J, Rmatch_LR) >= 2) { tautau_in_LR = true; break; }
    }
  }
  if (nTau_matched_resolved_for_bit >= 2 || tautau_in_LR) flag |= HAS_TAUTAU;

  // ---------------------------------------------------
  // New logic: ANY configuration 4b + 2τ without double counting
  // Strategy: assign truth objects to LR jets first (if within Rmatch_LR),
  // then greedily match remaining truth objects to distinct resolved jets.
  // ---------------------------------------------------

  // B hadrons: LR-first assignment
  int nB_in_LR = 0;
  std::vector<char> B_assigned_LR(tB.size(), 0);
  if (!tB.empty() && !rLR_bb.empty()) {
    for (size_t i = 0; i < tB.size(); ++i) {
      for (const auto& J : rLR_bb) {
        if (tB[i].DeltaR(J) < Rmatch_LR) { B_assigned_LR[i] = 1; ++nB_in_LR; break; }
      }
    }
  }

  // Then resolved matching for *unassigned* truth B hadrons
  int nB_resolved = 0;
  if (!tB.empty() && !rBJ.empty()) {
    std::vector<char> used(rBJ.size(), 0);
    for (size_t it = 0; it < tB.size(); ++it) {
      if (B_assigned_LR[it]) continue;  // already counted via LR jet
      int bestJ = -1; double bestDR = 1e9;
      for (size_t jr = 0; jr < rBJ.size(); ++jr) {
        if (used[jr]) continue;
        const double dR = tB[it].DeltaR(rBJ[jr]);
        if (dR < dR_match_b && dR < bestDR) { bestDR = dR; bestJ = (int)jr; }
      }
      if (bestJ >= 0) { used[bestJ] = 1; ++nB_resolved; }
    }
  }

  const int nB_total = nB_in_LR + nB_resolved;
  if (nB_total >= 4) flag |= HAS_4B;

  // Taus: LR-first assignment
  int nTau_in_LR = 0;
  std::vector<char> Tau_assigned_LR(tTau.size(), 0);
  if (!tTau.empty() && !rLR_tau.empty()) {
    for (size_t i = 0; i < tTau.size(); ++i) {
      for (const auto& J : rLR_tau) {
        if (tTau[i].DeltaR(J) < Rmatch_LR) { Tau_assigned_LR[i] = 1; ++nTau_in_LR; break; }
      }
    }
  }

  // Then resolved matching for *unassigned* truth taus
  int nTau_resolved = 0;
  if (!tTau.empty() && !rTau.empty()) {
    std::vector<char> used(rTau.size(), 0);
    for (size_t it = 0; it < tTau.size(); ++it) {
      if (Tau_assigned_LR[it]) continue;
      int bestJ = -1; double bestDR = 1e9;
      for (size_t jr = 0; jr < rTau.size(); ++jr) {
        if (used[jr]) continue;
        const double dR = tTau[it].DeltaR(rTau[jr]);
        if (dR < dR_match_tau && dR < bestDR) { bestDR = dR; bestJ = (int)jr; }
      }
      if (bestJ >= 0) { used[bestJ] = 1; ++nTau_resolved; }
    }
  }

  const int nTau_total = nTau_in_LR + nTau_resolved;

  // Composite requirement: 4b2tau in ANY configuration
  if (nB_total >= 4 && nTau_total >= 2) flag |= HAS_4B2TAU;

  return flag; // now 0..31
}

bool AnalysisFCChh::find_hhh_signal_match_non_unique(
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_B_fromH,
  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tauhad_vis_fromH,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_tau_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_b_jets,
  float dR_thres)
{
  // Quick sanity checks
  if (truth_B_fromH.size() < 4 || truth_tauhad_vis_fromH.size() < 2) return false;
  if (reco_b_jets.empty() || reco_tau_jets.empty()) return false;

  // Build TLVs
  std::vector<TLorentzVector> tlv_truth_B;   tlv_truth_B.reserve(truth_B_fromH.size());
  std::vector<TLorentzVector> tlv_truth_tau; tlv_truth_tau.reserve(truth_tauhad_vis_fromH.size());
  std::vector<TLorentzVector> tlv_reco_b;    tlv_reco_b.reserve(reco_b_jets.size());
  std::vector<TLorentzVector> tlv_reco_tau;  tlv_reco_tau.reserve(reco_tau_jets.size());

  for (const auto& b  : truth_B_fromH)              tlv_truth_B.emplace_back(getTLV_MC(b));
  for (const auto& t  : truth_tauhad_vis_fromH)     tlv_truth_tau.emplace_back(getTLV_MC(t));
  for (const auto& rb : reco_b_jets)                tlv_reco_b.emplace_back(getTLV_reco(rb));
  for (const auto& rt : reco_tau_jets)              tlv_reco_tau.emplace_back(getTLV_reco(rt));

  // Build adjacency lists (edges where dR < dR_thres). Left nodes = truth, right nodes = reco.
  auto buildAdj = [dR_thres](const std::vector<TLorentzVector>& truth,
                             const std::vector<TLorentzVector>& reco)
                    -> std::vector<std::vector<int>>
  {
    std::vector<std::vector<int>> adj(truth.size());
    for (size_t i = 0; i < truth.size(); ++i) {
      for (size_t j = 0; j < reco.size(); ++j) {
        if (truth[i].DeltaR(reco[j]) < dR_thres) adj[i].push_back(static_cast<int>(j));
      }
    }
    return adj;
  };

  const auto adjB   = buildAdj(tlv_truth_B,   tlv_reco_b);
  const auto adjTau = buildAdj(tlv_truth_tau, tlv_reco_tau);

  // Maximum bipartite matching via DFS/Kuhn
  auto maxMatch = [](const std::vector<std::vector<int>>& adj, int nRight) -> int {
    const int nLeft = static_cast<int>(adj.size());
    std::vector<int> matchR(nRight, -1);           // which left node is matched to right j (or -1)
    int matched = 0;

    std::vector<char> seen(nRight);
    std::function<bool(int)> dfs = [&](int u)->bool {
      for (int v : adj[u]) {
        if (seen[v]) continue;
        seen[v] = 1;
        if (matchR[v] == -1 || dfs(matchR[v])) {
          matchR[v] = u;
          return true;
        }
      }
      return false;
    };

    for (int u = 0; u < nLeft; ++u) {
      std::fill(seen.begin(), seen.end(), 0);
      if (dfs(u)) ++matched;
    }
    return matched;
  };

  const int nB_matched   = maxMatch(adjB,   static_cast<int>(tlv_reco_b.size()));
  const int nTau_matched = maxMatch(adjTau, static_cast<int>(tlv_reco_tau.size()));

  // Signal requirement: == 4 B matches and == 2 tauhad matches
  return (nB_matched >= 4) && (nTau_matched >= 2);
}

// double b-tagging
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_bb_tagged(
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> LR_jets,
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> b_tagged_track_jets)
{
  // define output vector of double b-tagging LR jets
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> bb_jets;

  // loop over LR_jets
  for (auto &LR_jet : LR_jets) {
    // construct TLV for LR_jet
    TLorentzVector TLV_LR_jet = getTLV_reco(LR_jet);

    // counter for b-tags in LR jet
    int n_bs = 0;

    // now loop over all track jets
    for (auto &track_jet : b_tagged_track_jets) {
      // construct TLV for track jet
      TLorentzVector TLV_track_jet = getTLV_reco(track_jet);
      // compute dR between LR jet and track jet
      float dR_track_jet_PF = TLV_LR_jet.DeltaR(TLV_track_jet);
      
      // if this track jet is within 0.8, consider it a constituent
      if (dR_track_jet_PF < 0.8) {
        n_bs++;
      }
      
    }
    if (n_bs == 2) {
      std::cout << "Found 2 b-jets " << n_bs << std::endl;
      bb_jets.push_back(LR_jet);
    }

  }
  return bb_jets;
}

// double b-tagging
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::remove_bb_b_overlap(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_tagged_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& smallR_bjets)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> b_jets;
  b_jets.reserve(smallR_bjets.size());

  for (const auto& bjet : smallR_bjets) {
    TLorentzVector tlv_bjet = getTLV_reco(bjet);

    bool overlaps = false;
    for (const auto& LR_jet : bb_tagged_LR_jets) {
      TLorentzVector tlv_LR_jet = getTLV_reco(LR_jet);
      const double dR_bjet_LR = tlv_LR_jet.DeltaR(tlv_bjet);

      // if this b jet is inside the LR jet cone, mark it as overlapping
      if (dR_bjet_LR < 0.8) {
        overlaps = true;
        break;
      }
    }

    // keep only b-jets that don't overlap any LR jet
    if (!overlaps) {
      b_jets.push_back(bjet);
    }
  }

  return b_jets;
}



std::pair<TLorentzVector, TLorentzVector>
AnalysisFCChh::find_4b_hh(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& b_jets)
{
  // number of objecst we consider
  const int nLR = bb_LR_jets.size();
  const int nb  = b_jets.size();

  // if we have two bb-tagged large-R jets: life easy :)
  if (nLR == 2 && nb == 0) {
    TLorentzVector H1 = getTLV_reco(bb_LR_jets[0]);
    TLorentzVector H2 = getTLV_reco(bb_LR_jets[1]);
    if (H2.Pt() > H1.Pt()) std::swap(H1, H2);
    return {H1, H2};
  }

  // if we have one bb-tagged large-R jet: life still easy :)
  if (nLR == 1 && nb == 2) {
    TLorentzVector H_LR = getTLV_reco(bb_LR_jets[0]);
    TLorentzVector H_bb = getTLV_reco(b_jets[0]) + getTLV_reco(b_jets[1]);
    TLorentzVector H1 = H_LR, H2 = H_bb;
    if (H2.Pt() > H1.Pt()) std::swap(H1, H2);
    return {H1, H2};
  }

  // if we have no bb-tagged large-R jets: life slightly more challenging
  // choose b-jet pairs that minimise max(dR)
  if (nLR == 0 && nb == 4) {
    TLorentzVector b0 = getTLV_reco(b_jets[0]);
    TLorentzVector b1 = getTLV_reco(b_jets[1]);
    TLorentzVector b2 = getTLV_reco(b_jets[2]);
    TLorentzVector b3 = getTLV_reco(b_jets[3]);

    struct Cand {
      TLorentzVector H1, H2;
      double maxdR;
    };
    
    auto make = [&](const TLorentzVector& a, const TLorentzVector& b,
                    const TLorentzVector& c, const TLorentzVector& d) {
      const double dR1 = a.DeltaR(b);
      const double dR2 = c.DeltaR(d);
      return Cand{a+b, c+d, std::max(dR1, dR2)};
    };

    Cand c1 = make(b0,b1, b2,b3);
    Cand c2 = make(b0,b2, b1,b3);
    Cand c3 = make(b0,b3, b1,b2);

    const Cand* best = &c1;
    if (c2.maxdR < best->maxdR) best = &c2;
    if (c3.maxdR < best->maxdR) best = &c3;

    TLorentzVector H1 = best->H1, H2 = best->H2;
    if (H2.Pt() > H1.Pt()) std::swap(H1, H2);
    return {H1, H2};
  }

  // Fallback: unsupported multiplicity → return “empty” TLVs
  return {TLorentzVector(), TLorentzVector()};
}

std::vector<std::vector<TLorentzVector>>
AnalysisFCChh::find_4b_hh_pairs(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& bb_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& b_jets)
{
  auto make_group = [](const TLorentzVector& H, std::initializer_list<TLorentzVector> parts)
    -> std::vector<TLorentzVector>
  {
    std::vector<TLorentzVector> g;
    g.reserve(1 + parts.size());
    g.push_back(H);
    for (const auto& p : parts) g.push_back(p);
    return g;
  };

  auto order_by_pt = [](std::vector<std::vector<TLorentzVector>>& G)
  {
    if (G.size() == 2 && !G[0].empty() && !G[1].empty()) {
      if (G[1][0].Pt() > G[0][0].Pt()) std::swap(G[0], G[1]);
    }
  };

  const int nLR = bb_LR_jets.size();
  const int nb  = b_jets.size();

  // Case 1: two bb-tagged large-R jets
  if (nLR == 2 && nb == 0) {
    TLorentzVector lr0 = getTLV_reco(bb_LR_jets[0]);
    TLorentzVector lr1 = getTLV_reco(bb_LR_jets[1]);

    std::vector<std::vector<TLorentzVector>> out;
    out.push_back(make_group(lr0, {lr0}));
    out.push_back(make_group(lr1, {lr1}));
    order_by_pt(out);
    return out;
  }

  // Case 2: one bb-tagged large-R jet + two resolved b-jets
  if (nLR == 1 && nb == 2) {
    TLorentzVector lr = getTLV_reco(bb_LR_jets[0]);
    TLorentzVector b0 = getTLV_reco(b_jets[0]);
    TLorentzVector b1 = getTLV_reco(b_jets[1]);
    TLorentzVector hbb = b0 + b1;

    std::vector<std::vector<TLorentzVector>> out;
    out.push_back(make_group(lr,  {lr}));
    out.push_back(make_group(hbb, {b0, b1}));
    order_by_pt(out);
    return out;
  }

  // Case 3: fully resolved (4 b-jets): pair to minimise max(dR)
  if (nLR == 0 && nb == 4) {
    TLorentzVector b0 = getTLV_reco(b_jets[0]);
    TLorentzVector b1 = getTLV_reco(b_jets[1]);
    TLorentzVector b2 = getTLV_reco(b_jets[2]);
    TLorentzVector b3 = getTLV_reco(b_jets[3]);

    struct Cand {
      TLorentzVector H1, H2;
      std::array<TLorentzVector,2> H1p, H2p;
      double maxdR;
    };

    auto make = [&](const TLorentzVector& a, const TLorentzVector& b,
                    const TLorentzVector& c, const TLorentzVector& d) -> Cand
    {
      const double dR1 = a.DeltaR(b);
      const double dR2 = c.DeltaR(d);
      return Cand{ a+b, c+d, {a,b}, {c,d}, std::max(dR1, dR2) };
    };

    Cand c1 = make(b0,b1, b2,b3);
    Cand c2 = make(b0,b2, b1,b3);
    Cand c3 = make(b0,b3, b1,b2);
    const Cand* best = &c1;
    if (c2.maxdR < best->maxdR) best = &c2;
    if (c3.maxdR < best->maxdR) best = &c3;

    std::vector<std::vector<TLorentzVector>> out;
    out.push_back(make_group(best->H1, {best->H1p[0], best->H1p[1]}));
    out.push_back(make_group(best->H2, {best->H2p[0], best->H2p[1]}));
    order_by_pt(out);
    return out;
  }

  // Fallback: unsupported multiplicity → return empty
  return {};
}


// now also classify the Htautau, somewhat easier since there are only two configurations
TLorentzVector AnalysisFCChh::find_tautau_h(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tautau_LR_jets,
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& tau_jets
)
{
  // define output higgs tlv for storage
  TLorentzVector Htautau;
  // two scenarios 
  if (tautau_LR_jets.size() == 1) {
    Htautau = getTLV_reco(tautau_LR_jets[0]);
  } else if (tau_jets.size() == 2) {
    Htautau = getTLV_reco(tau_jets[0]) + getTLV_reco(tau_jets[1]);
  } else {
    return Htautau;
  }

  return Htautau;

}

// now compute sigma parameters for Dalitz analysis
ROOT::VecOps::RVec<float> AnalysisFCChh::compute_sigma(
  TLorentzVector higgs_1,
  TLorentzVector higgs_2,
  TLorentzVector higgs_3
)
{
  // we have three combinations of higgs to compute sigma
  ROOT::VecOps::RVec<float> sigma;

  /*
    sigma_ij = (s_ij - 4mh^2)/(m_hhh^2 - 9m_h^2)
  */

  float s_12 = (higgs_1 + higgs_2).M2();
  float s_13 = (higgs_1 + higgs_3).M2();
  float s_23 = (higgs_2 + higgs_3).M2();

  float m_hhh2 = (higgs_1 + higgs_2 + higgs_3).M2();

  float m_h2 = 125.25 * 125.25; // via PDG: https://pdg.lbl.gov/2022/tables/rpp2022-sum-gauge-higgs-bosons.pdf#page=5
  
  float sigma_12 = (s_12 - 4*m_h2)/(m_hhh2 - 9*m_h2);
  float sigma_23 = (s_23 - 4*m_h2)/(m_hhh2 - 9*m_h2);
  float sigma_13 = (s_13 - 4*m_h2)/(m_hhh2 - 9*m_h2);

  sigma.push_back(sigma_12); 
  sigma.push_back(sigma_23);
  sigma.push_back(sigma_13);

  return sigma;

}

// truth matching: find a reco part that matches the truth part within cone of
// dR_thres
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matched(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts_all,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if no input part, return nothing (didnt input correct event then)
  if (truth_parts_to_match.size() < 1) {
    return out_vector;
  }

  // currently only want one particle to match, to not get confused with
  // vectors:
  if (truth_parts_to_match.size() != 1) {
    std::cout << "Error! Found more than one truth part in input to "
                 "find_reco_matched() ! Not intended?"
              << std::endl;
    // return out_vector;
  }

  // take the TLV of that particle we want to match:
  TLorentzVector truth_part_tlv = getTLV_MC(truth_parts_to_match.at(0));

  // loop over all reco parts and find if there is one within the dr threshold
  // to the truth part
  for (auto &check_reco_part : reco_parts_all) {
    TLorentzVector check_reco_part_tlv = getTLV_reco(check_reco_part);
    float dR_val = truth_part_tlv.DeltaR(check_reco_part_tlv);

    if (dR_val <= dR_thres) {
      out_vector.push_back(check_reco_part);
    }

    check_reco_part_tlv.Clear();
  }

  return out_vector;
}

// manual implementation of the delphes isolation criterion
ROOT::VecOps::RVec<float> AnalysisFCChh::get_IP_delphes(
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> test_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts_all,
    float dR_min, float pT_min, bool exclude_light_leps) {

  ROOT::VecOps::RVec<float> out_vector;

  if (test_parts.size() < 1) {
    // std::cout << "Debug: No test particles provided, returning empty vector." << std::endl;
    return out_vector;
  }

  // std::cout << "Debug: Number of test particles: " << test_parts.size() << std::endl;
  // std::cout << "Debug: Number of all reconstructed particles: " << reco_parts_all.size() << std::endl;
  // std::cout << "Debug: dR_min: " << dR_min << ", pT_min: " << pT_min << ", exclude_light_leps: " << exclude_light_leps << std::endl;

  for (auto &test_part : test_parts) {
    // first get the pT of the test particle:
    TLorentzVector tlv_test_part = getTLV_reco(test_part);
    float pT_test_part = tlv_test_part.Pt();

    // std::cout << "Debug: Test particle pT: " << pT_test_part << std::endl;

    float sum_pT = 0;

    // loop over all other parts and sum up pTs if they are within the dR cone
    // and above min pT
    for (auto &reco_part : reco_parts_all) {
      TLorentzVector tlv_reco_part = getTLV_reco(reco_part);
      float reco_pT = tlv_reco_part.Pt();
      float dR = tlv_test_part.DeltaR(tlv_reco_part);

      // Skip if the pT of reco part is equal to the pT of test part
      if (reco_pT == pT_test_part) {
        // std::cout << "Debug: Skipping the same particle (equal pT): " << reco_pT << ", dR: " << dR << std::endl;
        tlv_reco_part.Clear();
        continue;
      }

      if (reco_pT > pT_min && dR < dR_min) {
        sum_pT += reco_pT;
        // std::cout << "Debug: Particle within cone, pT: " << reco_pT << ", dR: " << dR << ", Sum pT now: " << sum_pT << std::endl;
      } else {
        // std::cout << "Debug: Particle excluded, pT: " << reco_pT << ", dR: " << dR << std::endl;
      }

      tlv_reco_part.Clear();
    }

    float IP_val = (sum_pT) / pT_test_part;

    // std::cout << "Debug: Calculated IP value: " << IP_val << ", Sum pT: " << sum_pT << ", Test pT: " << pT_test_part << std::endl;

    out_vector.push_back(IP_val);
  }

  return out_vector;
}

// Build from PF (EFlow) objects, not generic RecoParticles.
// Returns IP = sum pT(in cone) / pT(test), per test particle.
// Feed pre-isolation leptons (e.g. MuonNoIso mapped to RP) as test_parts.
// PF inputs are the three EDM4hep RP collections written by your card.
ROOT::VecOps::RVec<float>
AnalysisFCChh::get_IP_delphes_new(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& test_parts,          // e.g. muons (NoIso)
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_tracks,        // EFlowTrack
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_photons,       // EFlowPhoton
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& eflow_neutral_hadr,  // EFlowNeutralHadron
    float dR_max, float pT_min)
{
  ROOT::VecOps::RVec<float> out; out.reserve(test_parts.size());
  if (test_parts.empty()) return out;

  // Optional: if you want to veto e/μ in the cone and do NOT have PDG IDs on PF,
  // pass your reconstructed electrons/muons separately and veto by ΔR there instead.
  auto add_cone_sum = [&](const TLorentzVector& tlv_test,
                          const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& pf,
                          float& sum_pT)
  {
    for (const auto& p : pf) {
      const TLorentzVector tlv = AnalysisFCChh::getTLV_reco(p);
      const float dR = tlv_test.DeltaR(tlv);
      if (dR <= 1e-6) continue;                              // exclude self
      if (tlv.Pt() <= pT_min || dR > dR_max) continue;


      sum_pT += tlv.Pt();
    }
  };

  for (const auto& test : test_parts) {
    const TLorentzVector tlv_test = AnalysisFCChh::getTLV_reco(test);
    const float pT_test = tlv_test.Pt();
    if (pT_test <= 0.f) { out.emplace_back(-1.f); continue; }

    float sum_pT = 0.f;
    add_cone_sum(tlv_test, eflow_tracks,        sum_pT);
    add_cone_sum(tlv_test, eflow_photons,       sum_pT);
    add_cone_sum(tlv_test, eflow_neutral_hadr,  sum_pT);

    out.emplace_back(sum_pT / pT_test);
  }
  return out;
}
// index navigation matching reco and MC particle to get pdg id of reco
// particle, following the ReconstructedParticle2MC::getRP2MC_pdg fuction in
// base FW
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::filter_lightLeps(
    ROOT::VecOps::RVec<int> recind, ROOT::VecOps::RVec<int> mcind,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> mc) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  for (auto &reco_index : recind) {

    std::cout << "Reco index:" << reco_index << std::endl;
    std::cout << "MC index:" << mcind.at(reco_index) << std::endl;

    // testing:
    auto pdg_id = mc.at(mcind.at(reco_index)).PDG;
    float mass = reco.at(reco_index).mass;

    std::cout << "MC PDG ID:" << pdg_id << std::endl;
    std::cout << "Reco mass:" << mass << std::endl;
  }

  return out_vector;
}

// muon mass: 0.105658
// electron mass: 0.000510999  ?

// find the neutrinos that originate from a W-decay (which comes from a higgs,
// and not a b-meson) using the truth info -> to use for truth MET
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getNusFromTau(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids,
    ROOT::VecOps::RVec<podio::ObjectID> daughter_ids,
    TString tau_type) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> nus_list;
  bool from_tau_Z;
  // loop over all truth particles and find neutrinos from Ws that came from
  // higgs (the direction tau->light lepton as child appears to be missing in
  // the tautau samples)
  for (auto &truth_part : truth_particles) {
    if (isNeutrino(truth_part)) {
      if (tau_type.Contains("had")) {
        std::cout << "Doing lep" << std::endl;
        from_tau_Z = isChildOfTauHadFromZ(truth_part, parent_ids, daughter_ids, truth_particles);
      } else if (tau_type.Contains("lep")) {
        std::cout << "Doing lep" << std::endl;
        from_tau_Z = isChildOfTauLepFromZ(truth_part, parent_ids, daughter_ids, truth_particles);
      }
      else {
        std::cout << "Invalid tau type :(" << std::endl;
      }
      if (from_tau_Z) {
        // counter+=1;
        nus_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return nus_list;
}


// find the neutrinos that originate from a tau-decay (which comes from a higgs,
// and not a b-meson) using the truth info -> to use for truth MET
ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::getNusFromW(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> nus_list;

  // loop over all truth particles and find neutrinos from Ws that came from
  // higgs (the direction tau->light lepton as child appears to be missing in
  // the tautau samples)
  for (auto &truth_part : truth_particles) {
    if (isNeutrino(truth_part)) {
      bool from_W_higgs =
          isChildOfWFromHiggs(truth_part, parent_ids, truth_particles);
      if (from_W_higgs) {
        // counter+=1;
        nus_list.push_back(truth_part);
      }
    }
  }
  // std::cout << "Leps from tau-higgs " << counter << std::endl;
  return nus_list;
}

// get truth met -> return as recoparticle so can use instead of reco met in
// other ftcs for some checks
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::getTruthMETObj(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_particles,
    ROOT::VecOps::RVec<podio::ObjectID> parent_ids, TString type) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  ROOT::VecOps::RVec<edm4hep::MCParticleData> selected_nus;

  for (auto &truth_part : truth_particles) {
    if (isNeutrino(truth_part)) {

      // sum only the neutrinos from different higgs decays if requested
      if (type.Contains("hww_only") &&
          isChildOfWFromHiggs(truth_part, parent_ids, truth_particles)) {
        selected_nus.push_back(truth_part);
      }

      else if (type.Contains("htautau_only") &&
               isChildOfTauFromHiggs(truth_part, parent_ids, truth_particles)) {
        // std::cout << "getting truth MET from taus only" << std::endl;
        selected_nus.push_back(truth_part);
      }

      else if (type.Contains("hzz_only") &&
               isChildOfZFromHiggs(truth_part, parent_ids, truth_particles)) {
        selected_nus.push_back(truth_part);
      }

      else if (type.Contains("all_nu")) {
        selected_nus.push_back(truth_part);
      }
    }
  }

  // sum up
  TLorentzVector tlv_total;

  for (auto &nu : selected_nus) {
    TLorentzVector tlv_nu = getTLV_MC(nu);
    tlv_total += tlv_nu;
  }

  edm4hep::ReconstructedParticleData met_obj;
  met_obj.momentum.x = tlv_total.Px();
  met_obj.momentum.y = tlv_total.Py();
  met_obj.momentum.z = 0.;
  met_obj.mass = 0.;

  // std::cout << "Truth MET from building object: " <<
  // sqrt(met_obj.momentum.x*met_obj.momentum.x +
  // met_obj.momentum.y*met_obj.momentum.y) << std::endl;

  out_vector.push_back(met_obj);

  return out_vector;
}

// additonal code for validation of new delphes card:

// helper function to find dR matched mc particle for a single reco particle -
// returns vector of size 1 always, only the one that is closest in dR!
// (technically doesnt need to be vector at this stage ..)
ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::find_mc_matched_particle(
    edm4hep::ReconstructedParticleData reco_part_to_match,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> check_mc_parts,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_vector;

  TLorentzVector reco_part_tlv = getTLV_reco(reco_part_to_match);

  for (auto &check_mc_part : check_mc_parts) {
    TLorentzVector check_mc_part_tlv = getTLV_MC(check_mc_part);

    float dR_val = reco_part_tlv.DeltaR(check_mc_part_tlv);

    if (dR_val <= dR_thres) {

      // check if already sth in the vector - always want only exactly one
      // match!

      if (out_vector.size() > 0) {
        // check which one is closer
        float dR_val_old = reco_part_tlv.DeltaR(getTLV_MC(out_vector.at(0)));

        float pT_diff_old =
            abs(reco_part_tlv.Pt() - getTLV_MC(out_vector.at(0)).Pt());

        if (dR_val < dR_val_old) {
          out_vector.at(0) = check_mc_part;

          if (pT_diff_old < abs(reco_part_tlv.Pt() - check_mc_part_tlv.Pt())) {
            std::cout << "Found case where closest in pT is not closest in dR"
                      << std::endl;
          }
        }
      }

      else {
        out_vector.push_back(check_mc_part);
      }
    }

    check_mc_part_tlv.Clear();
  }

  return out_vector;
}
// function to check if a reco particle has both a b and tau match
bool AnalysisFCChh::find_mc_matched_particle_both(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    float dR_thres)
{
  const TLorentzVector reco_tlv = getTLV_reco(reco_part_to_match);

  bool matched_tau = false;
  for (const auto &tau : check_mc_taus) {
    if (reco_tlv.DeltaR(getTLV_MC(tau)) <= dR_thres) {
      matched_tau = true;
      break;
    }
  }

  bool matched_b = false;
  for (const auto &b : check_mc_bs) {
    if (reco_tlv.DeltaR(getTLV_MC(b)) <= dR_thres) {
      matched_b = true;
      break;
    }
  }

  return matched_tau && matched_b;
}

// function to check if a reco particle has both a b and tau match
bool AnalysisFCChh::find_mc_matched_particle_both_HadronTau(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &mc_parents,
    float dR_thres)
{
  const TLorentzVector reco_tlv = getTLV_reco(reco_part_to_match);

  bool matched_tau = false;
  for (const auto &tau : check_mc_taus) {
    if (reco_tlv.DeltaR(getTLV_MC(tau)) <= dR_thres) {
      if (isFromHadron(tau, mc_parents, mc_particles)) {
        matched_tau = true;
        break;
      }
    }
  }

  bool matched_b = false;
  for (const auto &b : check_mc_bs) {
    if (reco_tlv.DeltaR(getTLV_MC(b)) <= dR_thres) {
      matched_b = true;
      break;
    }
  }

  return matched_tau && matched_b;
}

// function to check if a reco particle has both a b and tau match
// returns PDG of tau parent
int AnalysisFCChh::find_mc_matched_particle_both_PDGID(
    const edm4hep::ReconstructedParticleData &reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &check_mc_bs,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID> &mc_parents,
    float dR_thres)
{
  const TLorentzVector reco_tlv = getTLV_reco(reco_part_to_match);
  int PDG;
  bool matched_tau = false;
  for (const auto &tau : check_mc_taus) {
    if (reco_tlv.DeltaR(getTLV_MC(tau)) <= dR_thres) {
        PDG = tauIsFromWhere(tau, mc_parents, mc_particles);
        matched_tau = true;
        break;
      }
    }
  

  bool matched_b = false;
  for (const auto &b : check_mc_bs) {
    if (reco_tlv.DeltaR(getTLV_MC(b)) <= dR_thres) {
      matched_b = true;
      break;
    }
  }

  if (matched_tau & matched_b) {
    return PDG;
  } else {
    return -999;
  }
}

// function that just looks through the taus not to waste time
std::pair<int, edm4hep::MCParticleData>
AnalysisFCChh::find_highpt_tau_and_parent(
    const edm4hep::ReconstructedParticleData& reco_part_to_match,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& check_mc_taus,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc_particles,
    const ROOT::VecOps::RVec<podio::ObjectID>& mc_parents,
    float dR_thres)
{
  const TLorentzVector reco = getTLV_reco(reco_part_to_match);

  // pick the highest-pT tau within ΔR(reco, tau) <= dR_thres
  const edm4hep::MCParticleData* bestTau = nullptr;
  double bestPt = -1.0;

  for (const auto& tau : check_mc_taus) {
    const TLorentzVector tlv = getTLV_MC(tau);
    if (reco.DeltaR(tlv) <= dR_thres) {
      const double pt = tlv.Pt();
      if (pt > bestPt) {
        bestPt  = pt;
        bestTau = &tau;
      }
    }
  }

  if (bestTau) {
    const int parentPDG = tauIsFromWhere(*bestTau, mc_parents, mc_particles); // abs(PDG) or -999
    return std::make_pair(parentPDG, *bestTau); // copy τ MC record
  }

  return std::make_pair(-999, edm4hep::MCParticleData{});
}

// helper function to find dR matched reco particle for a single truth particle
// - returns vector of size 1 always, only the one that is closest in dR!
// (technically doesnt need to be vector at this stage ..)
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matched_particle(
    edm4hep::MCParticleData truth_part_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_reco_parts,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  TLorentzVector truth_part_tlv = getTLV_MC(truth_part_to_match);

  for (auto &check_reco_part : check_reco_parts) {
    TLorentzVector check_reco_part_tlv = getTLV_reco(check_reco_part);

    float dR_val = truth_part_tlv.DeltaR(check_reco_part_tlv);

    if (dR_val <= dR_thres) {

      // check if already sth in the vector - always want only exactly one
      // match!

      if (out_vector.size() > 0) {
        // check which one is closer
        float dR_val_old = truth_part_tlv.DeltaR(getTLV_reco(out_vector.at(0)));

        float pT_diff_old =
            abs(truth_part_tlv.Pt() - getTLV_reco(out_vector.at(0)).Pt());

        if (dR_val < dR_val_old) {
          out_vector.at(0) = check_reco_part;

          if (pT_diff_old <
              abs(truth_part_tlv.Pt() - check_reco_part_tlv.Pt())) {
            std::cout << "Found case where closest in pT is not closest in dR"
                      << std::endl;
          }
        }
      }

      else {
        out_vector.push_back(check_reco_part);
      }
    }

    check_reco_part_tlv.Clear();
  }

  return out_vector;
}

// same as above but returns the index of the matched particle instead
ROOT::VecOps::RVec<int> AnalysisFCChh::find_reco_matched_index(
    edm4hep::MCParticleData truth_part_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> check_reco_parts,
    float dR_thres) {

  ROOT::VecOps::RVec<int> out_vector;

  TLorentzVector truth_part_tlv = getTLV_MC(truth_part_to_match);

  for (int i = 0; i < check_reco_parts.size(); ++i) {

    edm4hep::ReconstructedParticleData check_reco_part = check_reco_parts[i];

    TLorentzVector check_reco_part_tlv = getTLV_reco(check_reco_part);

    float dR_val = truth_part_tlv.DeltaR(check_reco_part_tlv);

    if (dR_val <= dR_thres) {

      // check if already sth in the vector - always want only exactly one
      // match!

      if (out_vector.size() > 0) {
        // check which one is closer
        edm4hep::ReconstructedParticleData match_old =
            check_reco_parts[out_vector[0]]; // edit this to be TLV directly

        float dR_val_old = truth_part_tlv.DeltaR(getTLV_reco(match_old));

        float pT_diff_old =
            abs(truth_part_tlv.Pt() - getTLV_reco(match_old).Pt());

        if (dR_val < dR_val_old) {
          out_vector.at(0) = i;

          if (pT_diff_old <
              abs(truth_part_tlv.Pt() - check_reco_part_tlv.Pt())) {
            std::cout << "Found case where closest in pT is not closest in dR"
                      << std::endl;
          }
        }
      }

      else {
        out_vector.push_back(i);
      }
    }

    check_reco_part_tlv.Clear();
  }

  return out_vector;
}

// truth -> reco matching for a vector of generic truth particles - this doesnt
// check if the type of particles are the same! -> make sure you give the
// correct collections!
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if no input part, return nothing
  if (truth_parts.size() < 1) {
    return out_vector;
  }

  for (auto &truth_part : truth_parts) {

    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_match_vector =
        find_reco_matched_particle(truth_part, reco_particles, dR_thres);

    if (reco_match_vector.size() > 1) {
      std::cout << "Warning in AnalysisFCChh::find_reco_matches() - Truth "
                   "particle matched to more than one reco particle."
                << std::endl;
    }

    // check that the reco particle is not already in the out_vector
    bool isAlready = false;
    for (auto &out_i : out_vector) {
      if ((getTLV_reco(reco_match_vector[0]).Pt() == getTLV_reco(out_i).Pt()) &&
          (getTLV_reco(reco_match_vector[0]).Eta() ==
           getTLV_reco(out_i).Eta())) {
        isAlready = true;
        // std::cout<<"Already in the array"<<std::endl;
      }
    }
    if (!isAlready) {
      out_vector.append(reco_match_vector.begin(), reco_match_vector.end());
    }
  }

  return out_vector;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matches_unique(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_parts,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  if (truth_parts.empty() || reco_particles.empty()) return out;

  out.reserve(std::min(truth_parts.size(), reco_particles.size()));

  // Track which reco indices are already used
  std::vector<bool> reco_used(reco_particles.size(), false);

  // Precompute TLVs for speed
  std::vector<TLorentzVector> tlv_truth; tlv_truth.reserve(truth_parts.size());
  for (const auto& t : truth_parts) tlv_truth.emplace_back(getTLV_MC(t));
  std::vector<TLorentzVector> tlv_reco; tlv_reco.reserve(reco_particles.size());
  for (const auto& r : reco_particles) tlv_reco.emplace_back(getTLV_reco(r));

  // For each truth, grab the nearest available reco within dR_thres
  for (size_t it = 0; it < truth_parts.size(); ++it) {
    std::vector<TLorentzVector> tlv_truth;
    tlv_truth.reserve(truth_parts.size());
    for (const auto& t : truth_parts) {
      tlv_truth.emplace_back(getTLV_MC(t));
    }

    std::vector<TLorentzVector> tlv_reco;
    tlv_reco.reserve(reco_particles.size());
    for (const auto& r : reco_particles) {
      tlv_reco.emplace_back(getTLV_reco(r));
    }

    for (size_t it = 0; it < truth_parts.size(); ++it) {
      const auto& tv = tlv_truth[it];
      double best_dR = 999.0;
      int best_ir = -1;
      for (size_t ir = 0; ir < reco_particles.size(); ++ir) {
        if (reco_used[ir]) continue;
        double dR = tv.DeltaR(tlv_reco[ir]); 
        if (dR < best_dR) {
          best_dR = dR;
          best_ir = ir;
        }
      }
      if (best_ir >= 0 && best_dR < dR_thres) {
        reco_used[best_ir] = true;
        out.emplace_back(reco_particles[best_ir]);
      }
    }
    
  }

  return out;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_reco_matches_unique(
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles_to_match,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  if (reco_particles_to_match.empty() || reco_particles.empty()) return out;

  out.reserve(std::min(reco_particles_to_match.size(), reco_particles.size()));

  // Track which reco indices are already used
  std::vector<bool> reco_used(reco_particles.size(), false);

  // Precompute TLVs for speed
  std::vector<TLorentzVector> tlv_reco_to_match; tlv_reco_to_match.reserve(reco_particles_to_match.size());
  for (const auto& t : reco_particles_to_match) tlv_reco_to_match.emplace_back(getTLV_reco(t));
  std::vector<TLorentzVector> tlv_reco; tlv_reco.reserve(reco_particles.size());
  for (const auto& r : reco_particles) tlv_reco.emplace_back(getTLV_reco(r));

  // For each reco particle to match, grab the nearest available reco within dR_thres
  for (size_t it = 0; it < reco_particles_to_match.size(); ++it) {
    std::vector<TLorentzVector> tlv_reco_to_match;
    tlv_reco_to_match.reserve(reco_particles_to_match.size());
    for (const auto& t : reco_particles_to_match) {
      tlv_reco_to_match.emplace_back(getTLV_reco(t));
    }

    std::vector<TLorentzVector> tlv_reco;
    tlv_reco.reserve(reco_particles.size());
    for (const auto& r : reco_particles) {
      tlv_reco.emplace_back(getTLV_reco(r));
    }

    for (size_t it = 0; it < reco_particles_to_match.size(); ++it) {
      const auto& tv = tlv_reco_to_match[it];
      double best_dR = 999.0;
      int best_ir = -1;
      for (size_t ir = 0; ir < reco_particles.size(); ++ir) {
        if (reco_used[ir]) continue;
        double dR = tv.DeltaR(tlv_reco[ir]); 
        if (dR < best_dR) {
          best_dR = dR;
          best_ir = ir;
        }
      }
      if (best_ir >= 0 && best_dR < dR_thres) {
        reco_used[best_ir] = true;
        out.emplace_back(reco_particles[best_ir]);
      }
    }
    
  }

  return out;
}

// function to return jets that have both b and tau truth-matched with some dR
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matches_both(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_set1,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_set2,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;
  if (truth_set1.empty() || truth_set2.empty() || reco_particles.empty()) return out;

  // Precompute TLVs
  std::vector<TLorentzVector> tlv_reco; tlv_reco.reserve(reco_particles.size());
  for (const auto& r : reco_particles) tlv_reco.emplace_back(getTLV_reco(r));

  std::vector<TLorentzVector> tlv_truth1; tlv_truth1.reserve(truth_set1.size());
  for (const auto& t : truth_set1) tlv_truth1.emplace_back(getTLV_MC(t));

  std::vector<TLorentzVector> tlv_truth2; tlv_truth2.reserve(truth_set2.size());
  for (const auto& t : truth_set2) tlv_truth2.emplace_back(getTLV_MC(t));

  // Loop over reco jets: keep only if matched to both sets
  for (size_t ir = 0; ir < reco_particles.size(); ++ir) {
    bool match1 = false, match2 = false;

    for (const auto& tv1 : tlv_truth1) {
      if (tv1.DeltaR(tlv_reco[ir]) < dR_thres) { match1 = true; break; }
    }
    for (const auto& tv2 : tlv_truth2) {
      if (tv2.DeltaR(tlv_reco[ir]) < dR_thres) { match2 = true; break; }
    }

    if (match1 && match2) {
      out.emplace_back(reco_particles[ir]);
    }
  }

  return out;
}

auto AnalysisFCChh::find_reco_matches_both_withTruth(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    float dR_thres)
-> std::tuple<
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>,
    ROOT::VecOps::RVec<edm4hep::MCParticleData>,
    ROOT::VecOps::RVec<edm4hep::MCParticleData>
  >
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_reco;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_tau;
  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_b;

  if (truth_b.empty() || truth_tau.empty() || reco_particles.empty())
    return {out_reco, out_tau, out_b};

  // precompute TLVs
  std::vector<TLorentzVector> tlv_reco; tlv_reco.reserve(reco_particles.size());
  for (const auto& r : reco_particles) tlv_reco.emplace_back(getTLV_reco(r));

  std::vector<TLorentzVector> tlv_tau; tlv_tau.reserve(truth_tau.size());
  for (const auto& t : truth_tau) tlv_tau.emplace_back(getTLV_MC(t));

  std::vector<TLorentzVector> tlv_b; tlv_b.reserve(truth_b.size());
  for (const auto& b : truth_b) tlv_b.emplace_back(getTLV_MC(b));

  for (size_t ir = 0; ir < reco_particles.size(); ++ir) {
    int match_tau = -1, match_b = -1;

    for (size_t it = 0; it < tlv_tau.size(); ++it) {
      if (tlv_tau[it].DeltaR(tlv_reco[ir]) < dR_thres) { match_tau = it; break; }
    }
    for (size_t ib = 0; ib < tlv_b.size(); ++ib) {
      if (tlv_b[ib].DeltaR(tlv_reco[ir]) < dR_thres) { match_b = ib; break; }
    }

    if (match_tau >= 0 && match_b >= 0) {
      out_reco.emplace_back(reco_particles[ir]);
      out_tau.emplace_back(truth_tau[match_tau]);
      out_b.emplace_back(truth_b[match_b]);
    }
  }

  return {out_reco, out_tau, out_b};
}


// Return the PDG ID (and index if you want) of the primary ancestor of 'i'.
// Stops when there is no mother or max_depth is reached.
inline int primaryAncestorPDG_singleMother(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc,
    const ROOT::VecOps::RVec<int>& mother,  // size = mc.size(), mother[i] in [-1, ..)
    int i, int max_depth = 100)
{
  int steps = 0;
  int cur = i;
  while (cur >= 0 && cur < (int)mc.size() && mother[cur] >= 0 && steps < max_depth) {
    cur = mother[cur];
    ++steps;
  }
  return (cur >= 0 && cur < (int)mc.size()) ? mc[cur].PDG : 0; // 0 if out-of-range
}

// (optional) also get the ancestor index:
inline std::pair<int,int> primaryAncestorPDGandIndex_singleMother(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc,
    const ROOT::VecOps::RVec<int>& mother, int i, int max_depth = 100)
{
  int steps = 0;
  int cur = i;
  while (cur >= 0 && cur < (int)mc.size() && mother[cur] >= 0 && steps < max_depth) {
    cur = mother[cur];
    ++steps;
  }
  int pdg = (cur >= 0 && cur < (int)mc.size()) ? mc[cur].PDG : 0;
  return {pdg, cur};
}
inline int firstNonTauAncestorPDG_singleMother(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mc,
    const ROOT::VecOps::RVec<int>& mother,
    int i, int max_depth = 100)
{
  int cur = i, steps = 0;
  std::unordered_set<int> visited;

  std::cout << "[DEBUG] Start tracing tau at index " << i
            << " (PDG=" << mc[i].PDG << ")\n";

  while (cur >= 0 && cur < (int)mc.size() && mother[cur] >= 0 && steps < max_depth) {
    if (visited.count(cur)) {
      std::cout << "[DEBUG] Detected cycle at idx=" << cur
                << " (PDG=" << mc[cur].PDG << "), aborting.\n";
      return -1;
    }
    visited.insert(cur);

    int mom = mother[cur];
    if (mom < 0 || mom >= (int)mc.size()) break;

    int momPDG = mc[mom].PDG;
    std::cout << "[DEBUG] Step " << steps
              << ": idx=" << cur << " PDG=" << mc[cur].PDG
              << " → mother idx=" << mom << " PDG=" << momPDG << "\n";

    if (std::abs(momPDG) != 15) {
      std::cout << "[DEBUG] Found non-tau ancestor: PDG=" << momPDG << "\n";
      return momPDG;
    }

    cur = mom;
    ++steps;
  }

  std::cout << "[DEBUG] No non-tau ancestor found (steps=" << steps << ")\n";
  return -1;
}


static ROOT::VecOps::RVec<int>
find_b_nonH_tau_pairs_indices_and_printMotherPDG(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    const ROOT::VecOps::RVec<int>& mother,  // immediate mother index for MC particles
    float dR_thres)
{
  ROOT::VecOps::RVec<int> pairs; // [jet_idx, tau_idx, jet_idx, tau_idx, ...]
  if (truth_b_hadrons.empty() || truth_tau_all.empty() || reco_particles.empty()) {
    return pairs;
  }

  // Precompute TLVs
  std::vector<TLorentzVector> tlv_reco; tlv_reco.reserve(reco_particles.size());
  for (const auto& r : reco_particles) tlv_reco.emplace_back(getTLV_reco(r));

  std::vector<TLorentzVector> tlv_b; tlv_b.reserve(truth_b_hadrons.size());
  for (const auto& b : truth_b_hadrons) tlv_b.emplace_back(getTLV_MC(b));

  std::vector<TLorentzVector> tlv_tau_all; tlv_tau_all.reserve(truth_tau_all.size());
  for (const auto& t : truth_tau_all) tlv_tau_all.emplace_back(getTLV_MC(t));

  std::vector<TLorentzVector> tlv_tau_H; tlv_tau_H.reserve(truth_tau_fromH.size());
  for (const auto& tH : truth_tau_fromH) tlv_tau_H.emplace_back(getTLV_MC(tH));

  // Build veto for τ-from-H (match by near-identical 4-vector)
  const double tiny = 1e-6;
  std::vector<char> is_fromH(truth_tau_all.size(), 0);
  for (size_t i = 0; i < truth_tau_all.size(); ++i) {
    const auto& tv = tlv_tau_all[i];
    for (const auto& tH : tlv_tau_H) {
      if (tv.DeltaR(tH) < tiny && std::abs(tv.Pt() - tH.Pt()) < 1e-6) {
        is_fromH[i] = 1;
        break;
      }
    }
  }

  // So a τ isn't reused across multiple jets
  std::vector<char> tau_used(truth_tau_all.size(), 0);

  // Loop over reco jets
  for (size_t ir = 0; ir < reco_particles.size(); ++ir) {
    const auto& jr = tlv_reco[ir];

    // Must match at least one B hadron
    bool matched_b = false;
    for (const auto& bv : tlv_b) {
      if (jr.DeltaR(bv) < dR_thres) { matched_b = true; break; }
    }
    if (!matched_b) continue;

    // Find nearest available *non-Higgs* τ within ΔR
    double best_dR = 1e9;
    int best_itau = -1;
    for (size_t it = 0; it < truth_tau_all.size(); ++it) {
      if (is_fromH[it] || tau_used[it]) continue;
      double dR = jr.DeltaR(tlv_tau_all[it]);
      if (dR < dR_thres && dR < best_dR) {
        best_dR = dR;
        best_itau = static_cast<int>(it);
      }
    }

    if (best_itau >= 0) {
      tau_used[best_itau] = 1;

      // ---- print immediate mother PDG (or use your primaryAncestorPDG_singleMother) ----
      int momIdx = (best_itau < (int)mother.size()) ? mother[best_itau] : -1;
      int momPDG = (momIdx >= 0 && momIdx < (int)truth_tau_all.size()) ? truth_tau_all[momIdx].PDG : 0;
      std::cout << "[match] jet#" << ir
                << "  tau#" << best_itau
                << "  tau_motherPDG=" << momPDG << std::endl;
      // If you prefer primary ancestor:
      int origPDG = firstNonTauAncestorPDG_singleMother(truth_tau_all, mother, best_itau);
      std::cout << "original mother :) " << origPDG << std::endl;

      // store pair
      pairs.emplace_back(static_cast<int>(ir));
      pairs.emplace_back(best_itau);
    }
  }

  return pairs;
}

ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_jets_b_nonH_tau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    const ROOT::VecOps::RVec<int>& mother,  // immediate mother index
    float dR_thres)
{
  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> jets_out;
  auto pairs = find_b_nonH_tau_pairs_indices_and_printMotherPDG(
      truth_b_hadrons, truth_tau_all, truth_tau_fromH, reco_particles, mother, dR_thres);
  jets_out.reserve(pairs.size() / 2);
  for (size_t k = 0; k + 1 < pairs.size(); k += 2) {
    int ijet = pairs[k];
    jets_out.emplace_back(reco_particles[ijet]);
  }
  return jets_out;
}

// Return the *MC τs* (non-H) that are matched to those b-matched jets.
// Also prints τ mother PDG as a side-effect (once per matched pair).
ROOT::VecOps::RVec<edm4hep::MCParticleData>
AnalysisFCChh::find_mc_taus_b_nonH_tau(
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_b_hadrons,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_all,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& truth_tau_fromH,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& reco_particles,
    const ROOT::VecOps::RVec<int>& mother,  // immediate mother index
    float dR_thres)
{
  ROOT::VecOps::RVec<edm4hep::MCParticleData> taus_out;
  auto pairs = find_b_nonH_tau_pairs_indices_and_printMotherPDG(
      truth_b_hadrons, truth_tau_all, truth_tau_fromH, reco_particles, mother, dR_thres);
  taus_out.reserve(pairs.size() / 2);
  for (size_t k = 0; k + 1 < pairs.size(); k += 2) {
    int itau = pairs[k + 1];
    taus_out.emplace_back(truth_tau_all[itau]);
  }
  return taus_out;
}

// for testing: same as above, but not removing already matched reco objects,
// allowing double match
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matches_no_remove(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if no input part, return nothing
  if (truth_parts.size() < 1) {
    return out_vector;
  }

  for (auto &truth_part : truth_parts) {

    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_match_vector =
        find_reco_matched_particle(truth_part, reco_particles, dR_thres);

    if (reco_match_vector.size() > 1) {
      std::cout << "Warning in AnalysisFCChh::find_reco_matches() - Truth "
                   "particle matched to more than one reco particle."
                << std::endl;
    }

    out_vector.append(reco_match_vector.begin(), reco_match_vector.end());
  }

  return out_vector;
}

// variation of the previous function, with the addition of a set of MC
// particles that must not match with the reco ones!
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_reco_matches_exclusive(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts_exc,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if no input part, return nothing
  if (truth_parts.size() < 1) {
    return out_vector;
  }

  if (truth_parts_exc.size() < 1) {
    // std::cout<<"Use find_reco_matches!"<<std::endl;
    return out_vector;
  }

  for (auto &truth_part : truth_parts) {
    bool excludedMatch = false;
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_match_vector =
        find_reco_matched_particle(truth_part, reco_particles, dR_thres);
    ROOT::VecOps::RVec<edm4hep::MCParticleData> mc_excluded_vector =
        find_mc_matched_particle(reco_match_vector[0], truth_parts_exc, 1.0);
    if (mc_excluded_vector.size() > 0) {
      std::cout << "Excluded MC particle found matching with reco obj!"
                << std::endl;
      continue;
    }
    // if (reco_match_vector.size() > 1){
    //	std::cout << "Warning in AnalysisFCChh::find_reco_matches() - Truth
    // particle matched to more than one reco particle." << std::endl;
    // }

    // check that the reco particle is not already in the out_vector
    bool isAlready = false;
    for (auto &out_i : out_vector) {
      if ((getTLV_reco(reco_match_vector[0]).Pt() == getTLV_reco(out_i).Pt()) &&
          (getTLV_reco(reco_match_vector[0]).Eta() ==
           getTLV_reco(out_i).Eta())) {
        isAlready = true;
        // std::cout<<"Already in the array"<<std::endl;
      }
    }
    if (!isAlready) {
      out_vector.append(reco_match_vector.begin(), reco_match_vector.end());
    }
  }

  return out_vector;
}

ROOT::VecOps::RVec<edm4hep::MCParticleData> AnalysisFCChh::find_truth_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::MCParticleData> out_vector;

  // if no input part, return nothing
  if (truth_parts.size() < 1) {
    return out_vector;
  }

  for (auto &truth_part : truth_parts) {

    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_match_vector =
        find_reco_matched_particle(truth_part, reco_particles, dR_thres);

    if (reco_match_vector.size() > 1) {
      std::cout << "Warning in AnalysisFCChh::find_reco_matches() - Truth "
                   "particle matched to more than one reco particle."
                << std::endl;
    }

    if (reco_match_vector.size() == 1) {
      out_vector.push_back(truth_part);
    }
    // out_vector.append(reco_match_vector.begin(), reco_match_vector.end());
  }

  return out_vector;
}

// same as above just with indices
ROOT::VecOps::RVec<int> AnalysisFCChh::find_reco_match_indices(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_parts,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_particles,
    float dR_thres) {

  ROOT::VecOps::RVec<int> out_vector;

  // if no input part, return nothing
  if (truth_parts.size() < 1) {
    return out_vector;
  }

  for (auto &truth_part : truth_parts) {

    ROOT::VecOps::RVec<int> reco_match_vector =
        find_reco_matched_index(truth_part, reco_particles, dR_thres);

    if (reco_match_vector.size() > 1) {
      std::cout << "Warning in AnalysisFCChh::find_reco_matches() - Truth "
                   "particle matched to more than one reco particle."
                << std::endl;
    }

    out_vector.append(reco_match_vector.begin(), reco_match_vector.end());
  }

  return out_vector;
}

// truth matching: take as input the truth leptons from e.g. HWW decay and check
// if they have a reco match within dR cone - Note: skip taus!
ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
AnalysisFCChh::find_true_signal_leps_reco_matches(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_leps_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_electrons,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_muons,
    float dR_thres) {

  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

  // if no input part, return nothing
  if (truth_leps_to_match.size() < 1) {
    return out_vector;
  }

  // check the flavour of the input truth particle, if it is a tau we skip it
  for (auto &truth_lep : truth_leps_to_match) {

    if (!isLightLep(truth_lep)) {
      // std::cout << "Info: Problem in
      // AnalysisFCChh::find_true_signal_leps_reco_matches() - Found truth tau
      // (or other non light lepton) in attempt to match to reco, skipping." <<
      // std::endl;
      continue;
    }

    // TLorentzVector truth_part_tlv = getTLV_MC(truth_lep);

    // truth particle should thus be either an electron or a muon:

    // checking electrons:
    if (abs(truth_lep.PDG) == 11) {

      ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> ele_match_vector =
          find_reco_matched_particle(truth_lep, reco_electrons, dR_thres);

      // warning if match to more than one particle:
      if (ele_match_vector.size() > 1) {
        std::cout
            << "Warning in AnalysisFCChh::find_true_signal_leps_reco_matches() "
               "- Truth electron matched to more than one reco electron."
            << std::endl;
        std::cout << "Truth electron has pT = " << getTLV_MC(truth_lep).Pt()
                  << " charge = " << truth_lep.charge
                  << " pdg = " << truth_lep.PDG << std::endl;
        // check the pTs of them:
        for (auto &matched_ele : ele_match_vector) {
          std::cout << "Matched electron with pT = "
                    << getTLV_reco(matched_ele).Pt()
                    << " charge = " << matched_ele.charge
                    << " dR distance to truth = "
                    << getTLV_MC(truth_lep).DeltaR(getTLV_reco(matched_ele))
                    << std::endl;
        }
      }

      out_vector.append(ele_match_vector.begin(), ele_match_vector.end());

    }

    // checking muons:
    else if (abs(truth_lep.PDG) == 13) {

      ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> mu_match_vector =
          find_reco_matched_particle(truth_lep, reco_muons, dR_thres);

      // warning if match to more than one particle:
      if (mu_match_vector.size() > 1) {
        std::cout
            << "Warning in AnalysisFCChh::find_true_signal_leps_reco_matches() "
               "- Truth muon matched to more than one reco muon."
            << std::endl;
        std::cout << "Truth muon has pT = " << getTLV_MC(truth_lep).Pt()
                  << " charge = " << truth_lep.charge
                  << " pdg = " << truth_lep.PDG << std::endl;

        // check the pTs of them:
        for (auto &matched_mu : mu_match_vector) {
          std::cout << "Matched muon with pT = " << getTLV_reco(matched_mu).Pt()
                    << " charge = " << matched_mu.charge
                    << "dR distance to truth = "
                    << getTLV_MC(truth_lep).DeltaR(getTLV_reco(matched_mu))
                    << std::endl;
        }
      }

      out_vector.append(mu_match_vector.begin(), mu_match_vector.end());
    }
  }

  return out_vector;
}

// find the indices of reco matched particles, assume here that pdg id filtering
// of the input mc particles is already done and only the matching reco
// collection is passed
ROOT::VecOps::RVec<int> AnalysisFCChh::find_truth_to_reco_matches_indices(
    ROOT::VecOps::RVec<edm4hep::MCParticleData> truth_leps_to_match,
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco_parts,
    int pdg_ID, float dR_thres) {

  ROOT::VecOps::RVec<int> out_vector;

  // if no input part, return nothing
  if (truth_leps_to_match.size() < 1) {
    return out_vector;
  }

  // check the flavour of the input truth particle, if it is a tau we skip it
  for (auto &truth_lep : truth_leps_to_match) {

    // std::cout << "Matching truth particle with pdgID: " << abs(truth_lep.PDG)
    // << std::endl;

    // match only a certain requested type of particle
    if (abs(truth_lep.PDG) != pdg_ID) {
      // std::cout << "Skipping" << std::endl;
      continue;
    }

    // std::cout << "Running the match" << std::endl;

    ROOT::VecOps::RVec<int> reco_match_indices_vector =
        find_reco_matched_index(truth_lep, reco_parts, dR_thres);

    if (reco_match_indices_vector.size() > 1) {
      std::cout
          << "Warning in AnalysisFCChh::find_truth_to_reco_matches_indices() - "
             "Truth particle matched to more than one reco particle."
          << std::endl;
    }

    out_vector.append(reco_match_indices_vector.begin(),
                      reco_match_indices_vector.end());
  }

  return out_vector;
}

// try to get the isoVar when its filled as a usercollection
//  ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  AnalysisFCChh::get_isoVar(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  reco_parts_to_check, ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
//  all_reco_parts){
//  	// first check the index of the particle we want to get the iso var for

// 	ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out_vector;

// 	for (auto & reco_check_part : reco_parts_to_check){
// 		std::cout << "index of the particle is:" <<
// reco_check_part.index << std::endl;

// 	}

// 	return out_vector;

// }


 // end anon namespace

// ============================================================================
// Result type
// ============================================================================


// ============================================================================
// The Metropolis–Hastings MMC sampler
// ============================================================================
// ============================================================================
// ATLAS-style MMC PDF calibration (Angle, Ratio, Mass)
// ============================================================================

#include <TF1.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TKey.h>
#include <TClass.h>
#include <TROOT.h>

namespace {

/// Full ATLAS MMC parametrisations (hadronic + leptonic)
class MmcFullCalibration {
public:
  explicit MmcFullCalibration(const std::string& paramFile);
  ~MmcFullCalibration();

  /// ATLAS-style setters: configure internal TF1s for this tau
  /// tauIndex = 1 or 2; tautype = mmcType (0..5 had, 8 leptonic)
  void setParamAngle(double pt, int tauIndex, int tautype);
  void setParamRatio(double pt, int tauIndex, int tautype);
  void setParamMass (double pt, int tauIndex /* leptons only */);

  /// PDFs (floored at 1e-10, NaNs → 0)
  double thetaProb(double theta, int tauIndex) const;
  double ratioProb(double rho,   int tauIndex) const;
  double massProb (double m,     int tauIndex) const;

private:
  void readInParams(TDirectory* dir);

  // Single-variable shapes (assumed same functional family)
  // [0]*exp(-[2]*(log((x+[3])/[1]))**2)
  TF1* m_formulaAngle1{nullptr};
  TF1* m_formulaAngle2{nullptr};
  TF1* m_formulaRatio1{nullptr};
  TF1* m_formulaRatio2{nullptr};
  TF1* m_formulaMass1{nullptr};
  TF1* m_formulaMass2{nullptr};

  // Parametrisations vs pT
  // lep_numass, lep_angle, lep_ratio, had_angle, had_ratio
  std::vector<TF1*> m_lepNumass;
  std::vector<TF1*> m_lepAngle;
  std::vector<TF1*> m_lepRatio;
  std::vector<TF1*> m_hadAngle;
  std::vector<TF1*> m_hadRatio;

  TFile* m_file{nullptr};
};

// --------------------------------------------------------------------------
// ctor / dtor
// --------------------------------------------------------------------------

MmcFullCalibration::MmcFullCalibration(const std::string& paramFile)
{
  m_file = TFile::Open(paramFile.c_str(), "READ");
  if (!m_file || m_file->IsZombie()) {
    throw std::runtime_error("MmcFullCalibration: cannot open " + paramFile);
  }

  auto makeShape = [](const char* name) {
    return new TF1(name, "[0]*exp(-[2]*(log((x+[3])/[1]))**2)", 0.0, 10.0);
  };

  m_formulaAngle1 = makeShape("fcc_formulaAngle1");
  m_formulaAngle2 = makeShape("fcc_formulaAngle2");
  m_formulaRatio1 = makeShape("fcc_formulaRatio1");
  m_formulaRatio2 = makeShape("fcc_formulaRatio2");
  m_formulaMass1  = makeShape("fcc_formulaMass1");
  m_formulaMass2  = makeShape("fcc_formulaMass2");

  readInParams(m_file);
}

MmcFullCalibration::~MmcFullCalibration()
{
  for (TF1* f : m_lepNumass) delete f;
  for (TF1* f : m_lepAngle)  delete f;
  for (TF1* f : m_lepRatio)  delete f;
  for (TF1* f : m_hadAngle)  delete f;
  for (TF1* f : m_hadRatio)  delete f;

  delete m_formulaAngle1;
  delete m_formulaAngle2;
  delete m_formulaRatio1;
  delete m_formulaRatio2;
  delete m_formulaMass1;
  delete m_formulaMass2;

  if (m_file) m_file->Close();
  delete m_file;
}

// --------------------------------------------------------------------------
// Read parameters in the ATLAS style (DiTauMassTools::readInParams)
// --------------------------------------------------------------------------

void MmcFullCalibration::readInParams(TDirectory* dir)
{
  // This is basically DiTauMassTools::readInParams as seen in Doxygen:
  //   readInParams(dir, aset, lep_numass, lep_angle, lep_ratio, had_angle, had_ratio)
  //
  // Here we fix aset = MMC2024 and rename the vectors to our members.

  const std::string paramcode = "MMC2024MC23";

  TIter next(dir->GetListOfKeys());
  TKey* key = nullptr;

  while ((key = static_cast<TKey*>(next()))) {
    TClass* cl = gROOT->GetClass(key->GetClassName());

    // Recurse into subdirectories
    if (cl->InheritsFrom("TDirectory")) {
      dir->cd(key->GetName());
      TDirectory* subdir = gDirectory;
      readInParams(subdir);
      dir->cd();
      continue;
    }

    // Only TF1 / TGraph are relevant
    if (!cl->InheritsFrom("TF1") && !cl->InheritsFrom("TGraph")) continue;

    std::string total_path = dir->GetPath();
    if (total_path.find(paramcode) == std::string::npos) continue;

    TF1* func = dynamic_cast<TF1*>(dir->Get(key->GetName()));
    if (!func) continue;

    TF1* fclone = new TF1(*func); // detach from file

    const bool isLep   = (total_path.find("lep")   != std::string::npos);
    const bool isHad   = (total_path.find("had")   != std::string::npos);

    if (isLep) {
      if (total_path.find("Angle") != std::string::npos) {
        m_lepAngle.push_back(fclone);
      } else if (total_path.find("Ratio") != std::string::npos) {
        m_lepRatio.push_back(fclone);
      } else if (total_path.find("Mass") != std::string::npos) {
        m_lepNumass.push_back(fclone);
      } else {
        Warning("MmcFullCalibration", "Undefined leptonic PDF term in input file.");
        delete fclone;
      }
    } else if (isHad) {
      if (total_path.find("Angle") != std::string::npos) {
        m_hadAngle.push_back(fclone);
      } else if (total_path.find("Ratio") != std::string::npos) {
        m_hadRatio.push_back(fclone);
      } else {
        Warning("MmcFullCalibration", "Undefined hadronic PDF term in input file.");
        delete fclone;
      }
    } else {
      Warning("MmcFullCalibration", "Undefined decay channel in input file.");
      delete fclone;
    }
  }
}

// --------------------------------------------------------------------------
// Parameter setters (clone of setParamAngle pattern)
// --------------------------------------------------------------------------

static int compact_had_type(int tautype)
{
  // Same grouping as in your code / ATLAS:
  // PanTau: 0..5 had; 4 is catch-all for odd modes
  int type = tautype;
  if (tautype > 4 && tautype < 8) type = 4;
  return type;
}

void MmcFullCalibration::setParamAngle(double pt, int tauIndex, int tautype)
{
  const int type = compact_had_type(tautype);

  if (tautype <= 5) {
    // hadronic taus: 4 parameters per decay type
    if (!m_hadAngle.empty()) {
      for (int i = 0; i < 4; ++i) {
        const int idx = i + type * 4;
        if (idx < 0 || idx >= static_cast<int>(m_hadAngle.size())) continue;
        const double par = m_hadAngle[idx]->Eval(pt);
        TF1* tgt = (tauIndex == 1 ? m_formulaAngle1 : m_formulaAngle2);
        tgt->SetParameter(i, par);
      }
    }
  } else {
    // leptonic taus: assume 4 common parameters vs pT
    if (!m_lepAngle.empty()) {
      for (int i = 0; i < 4 && i < static_cast<int>(m_lepAngle.size()); ++i) {
        const double par = m_lepAngle[i]->Eval(pt);
        TF1* tgt = (tauIndex == 1 ? m_formulaAngle1 : m_formulaAngle2);
        tgt->SetParameter(i, par);
      }
    }
  }
}

void MmcFullCalibration::setParamRatio(double pt, int tauIndex, int tautype)
{
  const int type = compact_had_type(tautype);

  if (tautype <= 5) {
    if (!m_hadRatio.empty()) {
      for (int i = 0; i < 4; ++i) {
        const int idx = i + type * 4;
        if (idx < 0 || idx >= static_cast<int>(m_hadRatio.size())) continue;
        const double par = m_hadRatio[idx]->Eval(pt);
        TF1* tgt = (tauIndex == 1 ? m_formulaRatio1 : m_formulaRatio2);
        tgt->SetParameter(i, par);
      }
    }
  } else {
    if (!m_lepRatio.empty()) {
      for (int i = 0; i < 4 && i < static_cast<int>(m_lepRatio.size()); ++i) {
        const double par = m_lepRatio[i]->Eval(pt);
        TF1* tgt = (tauIndex == 1 ? m_formulaRatio1 : m_formulaRatio2);
        tgt->SetParameter(i, par);
      }
    }
  }
}

/// Leptonic νν mass PDFs – only defined for mmcType = 8
void MmcFullCalibration::setParamMass(double pt, int tauIndex)
{
  if (m_lepNumass.empty()) return;

  for (int i = 0; i < 4 && i < static_cast<int>(m_lepNumass.size()); ++i) {
    const double par = m_lepNumass[i]->Eval(pt);
    TF1* tgt = (tauIndex == 1 ? m_formulaMass1 : m_formulaMass2);
    tgt->SetParameter(i, par);
  }
}

// --------------------------------------------------------------------------
// PDF accessors
// --------------------------------------------------------------------------

static double safe_eval(const TF1* f, double x)
{
  if (!f) return 1e-10;
  double p = f->Eval(x);
  if (!(p > 0.0)) p = 1e-10;
  if (std::isnan(p)) p = 0.0;
  return p;
}

double MmcFullCalibration::thetaProb(double theta, int tauIndex) const
{
  if (theta == 0.0) return 1e-10;
  const TF1* f = (tauIndex == 1 ? m_formulaAngle1 : m_formulaAngle2);
  return safe_eval(f, theta);
}

double MmcFullCalibration::ratioProb(double rho, int tauIndex) const
{
  if (rho <= 0.0) return 1e-10;
  const TF1* f = (tauIndex == 1 ? m_formulaRatio1 : m_formulaRatio2);
  return safe_eval(f, rho);
}

double MmcFullCalibration::massProb(double m, int tauIndex) const
{
  if (m <= 0.0) return 1e-10;
  const TF1* f = (tauIndex == 1 ? m_formulaMass1 : m_formulaMass2);
  return safe_eval(f, m);
}

// --------------------------------------------------------------------------
// Singleton accessor
// --------------------------------------------------------------------------

MmcFullCalibration& fullCalib()
{
  static MmcFullCalibration* inst =
    new MmcFullCalibration("/data/atlas/users/dingleyt/FCChh/MMC/MMC_params_v051224_angle_noLikelihoodFit.root");
  return *inst;
}

// Track→mmcType mapping (unchanged)
inline int mmc_type_from_tracks(int n_tracks)
{
  if (n_tracks <= 0) return 8;   // leptonic proxy
  if (n_tracks == 1) return 0;   // 1-prong had
  if (n_tracks == 2) return 1;   // 1p1n-like
  return 3;                      // 3-prong had
}

} // end anon namespace
// Evaluate target at θ = (phi1, phi2, sMETx, sMETy) with full ATLAS-like PDFs
// Evaluate target at θ = (phi1, phi2, sMETx, sMETy, mnu1, mnu2)
// with full ATLAS-like PDFs (had–had and lep–had)
inline EvalOut evaluate_state_ATLAS(double phi1, double phi2,
                                    double sMETx, double sMETy,
                                    bool   isLep1, bool   isLep2,
                                    double mnu1,  double mnu2,
                                    const TLorentzVector& tau1,
                                    const TLorentzVector& tau2,
                                    int n_charged_tracks_1,
                                    int n_charged_tracks_2,
                                    double METx_meas, double METy_meas,
                                    double metres_xy, double inv_metres_xy2)
{
  EvalOut out{0.0, 0.0};

  // Guard near-singular Δφ
  const double sdp = std::sin(phi2 - phi1);
  if (std::fabs(sdp) < 1e-6) return out;
  const double csc_dphi = 1.0 / sdp;

  // Solve neutrino transverse magnitudes from MET' & azimuths
  const double pT1 =  csc_dphi * ( sMETx * std::sin(phi2) - sMETy * std::cos(phi2) );
  const double pT2 = -csc_dphi * ( sMETx * std::sin(phi1) - sMETy * std::cos(phi1) );
  if (pT1 <= 0.0 || pT2 <= 0.0) return out;

  // MET Gaussian
  const double dx = sMETx - METx_meas;
  const double dy = sMETy - METy_meas;
  const double w_met = std::exp(-0.5 * (dx*dx + dy*dy) * inv_metres_xy2);

  // Unified τ→vis+ν solver for both legs
  const auto roots1 = solve_pz_quadratic_tau(
    tau1,
    static_cast<float>(phi1),
    static_cast<float>(pT1),
    /*isLep=*/isLep1,
    n_charged_tracks_1,
    static_cast<float>(isLep1 ? mnu1 : 0.0)
  );
  const auto roots2 = solve_pz_quadratic_tau(
    tau2,
    static_cast<float>(phi2),
    static_cast<float>(pT2),
    /*isLep=*/isLep2,
    n_charged_tracks_2,
    static_cast<float>(isLep2 ? mnu2 : 0.0)
  );
  if (roots1.empty() || roots2.empty()) return out;

  // mmcType: leptons → 8, hadronic from track multiplicity
  const int mmcType1 = isLep1 ? 8 : mmc_type_from_tracks(n_charged_tracks_1);
  const int mmcType2 = isLep2 ? 8 : mmc_type_from_tracks(n_charged_tracks_2);

  double sumW  = 0.0;
  double sumMW = 0.0;

  auto& calib = fullCalib();

  for (float z1f : roots1) {
    const double z1  = static_cast<double>(z1f);
    const double px1 = pT1 * std::cos(phi1);
    const double py1 = pT1 * std::sin(phi1);
    const double E1  = std::sqrt(px1*px1 + py1*py1 + z1*z1 +
                                 (isLep1 ? mnu1*mnu1 : 0.0));
    TLorentzVector nu1; nu1.SetPxPyPzE(px1, py1, z1, E1);

    for (float z2f : roots2) {
      const double z2  = static_cast<double>(z2f);
      const double px2 = pT2 * std::cos(phi2);
      const double py2 = pT2 * std::sin(phi2);
      const double E2  = std::sqrt(px2*px2 + py2*py2 + z2*z2 +
                                   (isLep2 ? mnu2*mnu2 : 0.0));
      TLorentzVector nu2; nu2.SetPxPyPzE(px2, py2, z2, E2);

      const TLorentzVector t1 = tau1 + nu1;
      const TLorentzVector t2 = tau2 + nu2;

      const double theta1 = tau1.Vect().Angle(nu1.Vect());
      const double theta2 = tau2.Vect().Angle(nu2.Vect());

      // τ pT used for parametrisation
      const double pt1 = t1.Pt();
      const double pt2 = t2.Pt();

      // Set angle / ratio params
      calib.setParamAngle(pt1, 1, mmcType1);
      calib.setParamAngle(pt2, 2, mmcType2);
      calib.setParamRatio(pt1, 1, mmcType1);
      calib.setParamRatio(pt2, 2, mmcType2);

      // Leptonic mass PDFs (mmcType=8)
      if (isLep1) calib.setParamMass(pt1, 1);
      if (isLep2) calib.setParamMass(pt2, 2);

      const double w_theta1 = calib.thetaProb(theta1, 1);
      const double w_theta2 = calib.thetaProb(theta2, 2);

      // ratio = missing / visible
      const double rho1 = nu1.P() / tau1.P();
      const double rho2 = nu2.P() / tau2.P();

      const double w_rho1 = calib.ratioProb(rho1, 1);
      const double w_rho2 = calib.ratioProb(rho2, 2);

      double w_m1 = 1.0;
      double w_m2 = 1.0;
      if (isLep1) w_m1 = calib.massProb(mnu1, 1);
      if (isLep2) w_m2 = calib.massProb(mnu2, 2);

      const double w = w_met * w_theta1 * w_theta2 * w_rho1 * w_rho2 * w_m1 * w_m2;
      if (w <= 0.0) continue;

      const double mditau = (t1 + t2).M();
      sumW  += w;
      sumMW += w * mditau;
    }
  }

  if (sumW <= 0.0) return out;
  out.weight = sumW;
  out.m_mean = sumMW / sumW;
  return out;
}
MMC_MH_Result AnalysisFCChh::solve_ditau_MMC_MH_lephad_ATLAS(
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
)
{
  MMC_MH_Result R;

  // We only handle lep–had here, not lep–lep
  if (isLep1 && isLep2) {
    std::cout << "solve_ditau_MMC_MH_lephad_ATLAS: di-leptonic tau tau not supported." << std::endl;
    return R;
  }

  if (n_iter <= 0) return R;
  if (burn_in < 0) burn_in = 0;
  if (thin <= 0) thin = 1;

  const double metres_xy = metres / std::sqrt(2.0);
  const double inv_metres_xy2 = 1.0 / (metres_xy * metres_xy);

  if (sigma_met <= 0.0) sigma_met = metres_xy * 0.5;
  if (sigma_phi <= 0.0) sigma_phi = 0.15;

  // RNG
  std::mt19937_64 rng;
  if (seed == 0) {
    rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    rng.seed(seed);
  }

  std::normal_distribution<double> nphi(0.0, sigma_phi);
  std::normal_distribution<double> nmet(0.0, sigma_met);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  // Uniform proposal for leptonic νν mass: 0 .. (m_tau - m_vis)
  const double m_tau = 1.77686;
  const double max_m1 = isLep1 ? std::max(0.0, m_tau - tau1.M()) : 0.0;
  const double max_m2 = isLep2 ? std::max(0.0, m_tau - tau2.M()) : 0.0;

  std::uniform_real_distribution<double> n_mass1(0.0, max_m1);
  std::uniform_real_distribution<double> n_mass2(0.0, max_m2);

  // ---- Initial state: neutrino phis ~ visible tau phis; MET' = measured MET ----
  double phi1 = wrap_to_pi(tau1.Phi());
  double phi2 = wrap_to_pi(tau2.Phi());
  double mpx  = MET_x_meas;
  double mpy  = MET_y_meas;
  double m1   = isLep1 ? n_mass1(rng) : 0.0;
  double m2   = isLep2 ? n_mass2(rng) : 0.0;

  // Evaluate initial state; if invalid, jitter a few times
  EvalOut cur = evaluate_state_ATLAS(phi1, phi2, mpx, mpy,
                                     isLep1, isLep2, m1, m2,
                                     tau1, tau2,
                                     n_charged_tracks_1, n_charged_tracks_2,
                                     MET_x_meas, MET_y_meas,
                                     metres_xy, inv_metres_xy2);

  for (int tries = 0; tries < 10000 && cur.weight == 0.0; ++tries) {
    phi1 = wrap_to_pi(phi1 + nphi(rng));
    phi2 = wrap_to_pi(phi2 + nphi(rng));
    mpx  = MET_x_meas + nmet(rng);
    mpy  = MET_y_meas + nmet(rng);
    if (isLep1) m1 = n_mass1(rng);
    if (isLep2) m2 = n_mass2(rng);

    cur = evaluate_state_ATLAS(phi1, phi2, mpx, mpy,
                               isLep1, isLep2, m1, m2,
                               tau1, tau2,
                               n_charged_tracks_1, n_charged_tracks_2,
                               MET_x_meas, MET_y_meas,
                               metres_xy, inv_metres_xy2);
  }

  int accepts = 0;

  for (int t = 0; t < n_iter; ++t) {
    // Propose new state
    const double phi1_p = wrap_to_pi(phi1 + nphi(rng));
    const double phi2_p = wrap_to_pi(phi2 + nphi(rng));
    const double mpx_p  = mpx + nmet(rng);
    const double mpy_p  = mpy + nmet(rng);
    double m1_p = m1;
    double m2_p = m2;
    if (isLep1) m1_p = n_mass1(rng);
    if (isLep2) m2_p = n_mass2(rng);

    EvalOut prop = evaluate_state_ATLAS(phi1_p, phi2_p, mpx_p, mpy_p,
                                        isLep1, isLep2, m1_p, m2_p,
                                        tau1, tau2,
                                        n_charged_tracks_1, n_charged_tracks_2,
                                        MET_x_meas, MET_y_meas,
                                        metres_xy, inv_metres_xy2);

    // Symmetric proposal ⇒ MH ratio = w_prop / w_cur
    double alpha = 0.0;
    if (cur.weight == 0.0 && prop.weight == 0.0) {
      alpha = 0.0;
    } else if (cur.weight == 0.0) {
      alpha = 1.0;
    } else {
      alpha = std::min(1.0, prop.weight / cur.weight);
    }

    if (uni(rng) < alpha) {
      ++accepts;
      phi1 = phi1_p; phi2 = phi2_p;
      mpx  = mpx_p;  mpy  = mpy_p;
      m1   = m1_p;   m2   = m2_p;
      cur  = prop;
    }

    // Record samples after burn-in, with thinning
    if (t >= burn_in && ((t - burn_in) % thin == 0) && cur.weight > 0.0) {
      R.masses.push_back(static_cast<float>(cur.m_mean));
    }
  }

  R.accept_rate = static_cast<double>(accepts) / static_cast<double>(n_iter);
  return R;
}


float AnalysisFCChh::ditau_mass_collinear_then_mT(
    const TLorentzVector& vis1,
    const TLorentzVector& vis2,
    double METx, double METy
)
{

  const TLorentzVector vis = vis1 + vis2;
  const double mvis = vis.M();
  const double pxv  = vis.Px();
  const double pyv  = vis.Py();
  const double ptv  = vis.Pt();

  const double met  = std::sqrt(METx*METx + METy*METy);

  const double ETvis  = std::sqrt(mvis*mvis + ptv*ptv);
  const double ETmiss = met;                        
  const double dot    = pxv*METx + pyv*METy;

  double mT2 = mvis*mvis + 2.0*(ETvis*ETmiss - dot);
  if (mT2 < 0.0) mT2 = 0.0;
  const double mT = std::sqrt(mT2);

  const double px1 = vis1.Px();
  const double py1 = vis1.Py();
  const double px2 = vis2.Px();
  const double py2 = vis2.Py();

  const double denom = px1 * py2 - py1 * px2;
  if (std::fabs(denom) < 1e-6) {
    return static_cast<float>(mT);
  }

  // Solve MET as linear combo of visible τ directions
  const double a = ( METx * py2 - METy * px2) / denom;
  const double b = (-METx * py1 + METy * px1) / denom;

  const double x1 = 1.0 / (1.0 + a);
  const double x2 = 1.0 / (1.0 + b);

  // Require 0 < x_i < 1
  if (!(x1 > 0.0 && x1 < 1.0 && x2 > 0.0 && x2 < 1.0)) {
    return static_cast<float>(-999);
  }

  const double xprod = x1 * x2;
  if (xprod <= 0.0) {
    return static_cast<float>(-999);
  }

  const double m_col = mvis / std::sqrt(xprod);

  return static_cast<float>(m_col);
}

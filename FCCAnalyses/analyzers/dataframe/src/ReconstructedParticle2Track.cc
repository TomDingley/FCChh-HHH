#include "FCCAnalyses/ReconstructedParticle2Track.h"
#include "FCCAnalyses/VertexingUtils.h"

#include <algorithm>
#include <unordered_set>

namespace FCCAnalyses{

namespace ReconstructedParticle2Track{

  ROOT::VecOps::RVec<float> 
  getRP2TRK_mom(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
                ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
    ROOT::VecOps::RVec<float> result;
    for (auto & p: in) {
      if (p.tracks_begin<tracks.size())
        result.push_back(VertexingUtils::get_trackMom(tracks.at(p.tracks_begin)));
      else result.push_back(std::nan(""));
    }
    return result;
  }

  ROOT::VecOps::RVec<float> 
  getRP2TRK_charge(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
                   ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
    ROOT::VecOps::RVec<float> result;
    for (auto & p: in) {
      if (p.tracks_begin<tracks.size())
        result.push_back(p.charge);
      else result.push_back(std::nan(""));
    }
    return result;
  }

  
  int countRP2TRK_ncharged(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
                                ROOT::VecOps::RVec<edm4hep::TrackState>& trackstates,
                                float minPt = 0.f) {
    int cnt = 0;
    for (const auto& p : in) {
      const bool hasTrack = (p.tracks_begin < trackstates.size());

      if (!hasTrack) continue;
      if (p.charge == 0) continue;
      if (minPt > 0.f) {
        const float pt = std::hypot((float)p.momentum.x, (float)p.momentum.y);
        if (pt < minPt) continue;
      }
      ++cnt;
    }
    return cnt;
  }

  

  

  ROOT::VecOps::RVec<float> getRP2TRK_Bz(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& rps, const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks) {
    const double c_light = 2.99792458e8;
    const double a = c_light * 1e3 * 1e-15; //[omega] = 1/mm
    ROOT::VecOps::RVec<float> out;

    for(auto & p: rps) {
      if(p.tracks_begin < tracks.size()) {
	double pt= sqrt(p.momentum.x * p.momentum.x + p.momentum.y * p.momentum.y);
	double Bz= tracks.at(p.tracks_begin).omega / a * pt * std::copysign(1.0, p.charge);
	out.push_back(Bz);
      } else {
	out.push_back(-9.);
      }
    }
    return out;
  }

  float Bz(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& rps, const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks) {
    const double c_light =  2.99792458e8;// speed of light m/sec;
    const double a = c_light * 1e3 * 1e-15; //[omega] = 1/mm

    double Bz = -9;

    for(auto & p: rps) {
      if(p.tracks_begin < tracks.size()) {
        double pt= sqrt(p.momentum.x * p.momentum.x + p.momentum.y * p.momentum.y);
        Bz= tracks.at(p.tracks_begin).omega / a * pt * std::copysign(1.0, p.charge);
      }
    }
    return Bz;
  }

  ROOT::VecOps::RVec<float> XPtoPar_dxy(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
					const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
					const TLorentzVector& V, // primary vertex
					const float& Bz) {

    const double cSpeed = 2.99792458e8 * 1.0e-9;

    ROOT::VecOps::RVec<float> out;

    for (const auto & rp: in) {

      if( rp.tracks_begin < tracks.size()) {

        float D0_wrt0 = tracks.at(rp.tracks_begin).D0;
        float Z0_wrt0 = tracks.at(rp.tracks_begin).Z0;
        float phi0_wrt0 = tracks.at(rp.tracks_begin).phi;

        TVector3 X( - D0_wrt0 * TMath::Sin(phi0_wrt0) , D0_wrt0 * TMath::Cos(phi0_wrt0) , Z0_wrt0);
        TVector3 x = X - V.Vect();
        //std::cout<<"vertex: "<<V.Vect().X()<<", "<<V.Vect().Y()<<", "<<V.Vect().Z()<<", "<<std::endl;
        TVector3 p(rp.momentum.x, rp.momentum.y, rp.momentum.z);

        double a = - rp.charge * Bz * cSpeed;
        double pt = p.Pt();
        double r2 = x(0) * x(0) + x(1) * x(1);
        double cross = x(0) * p(1) - x(1) * p(0);
        double D=-9;
        if (pt * pt - 2 * a * cross + a * a * r2 > 0) {
          double T = TMath::Sqrt(pt * pt - 2 * a * cross + a * a * r2);
      	  if (pt < 10.0) D = (T - pt) / a;
                else D = (-2 * cross + a * r2) / (T + pt);
        }
        //std::cout<<"displ: "<<D<<std::endl;
	      out.push_back(D);

      } else {
	out.push_back(-9.);
      }
    }
    return out;
  }



  ROOT::VecOps::RVec<float> XPtoPar_dz(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
                                        const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
                                        const TLorentzVector& V, // primary vertex
                                        const float& Bz) {

    const double cSpeed = 2.99792458e8 * 1.0e-9; //Reduced speed of light ???

    ROOT::VecOps::RVec<float> out;

    for (const auto & rp: in) {

      if( rp.tracks_begin < tracks.size()) {

        float D0_wrt0 = tracks.at(rp.tracks_begin).D0;
        float Z0_wrt0 = tracks.at(rp.tracks_begin).Z0;
        float phi0_wrt0 = tracks.at(rp.tracks_begin).phi;

        TVector3 X( - D0_wrt0 * TMath::Sin(phi0_wrt0) , D0_wrt0 * TMath::Cos(phi0_wrt0) , Z0_wrt0);
        TVector3 x = X - V.Vect();

        TVector3 p(rp.momentum.x, rp.momentum.y, rp.momentum.z);

        double a = - rp.charge * Bz * cSpeed;
        double pt = p.Pt();
        double C = a/(2 * pt);
        double r2 = x(0) * x(0) + x(1) * x(1);
        double cross = x(0) * p(1) - x(1) * p(0);
        double T = TMath::Sqrt(pt * pt - 2 * a * cross + a * a * r2);
        double D;
        if (pt < 10.0) D = (T - pt) / a;
        else D = (-2 * cross + a * r2) / (T + pt);
        double B = C * TMath::Sqrt(TMath::Max(r2 - D * D, 0.0) / (1 + 2 * C * D));
        if ( TMath::Abs(B) > 1.) B = TMath::Sign(1, B);
        double st = TMath::ASin(B) / C;
        double ct = p(2) / pt;
        double z0;
        double dot = x(0) * p(0) + x(1) * p(1);
        if (dot > 0.0) z0 = x(2) - ct * st;
        else z0 = x(2) + ct * st;

        out.push_back(z0);
      } else {
        out.push_back(-9.);
      }
    }
    return out;
  }

  ROOT::VecOps::RVec<float> XPtoPar_phi(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
					const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
					const TLorentzVector& V, // primary vertex
					const float& Bz) {

    const double cSpeed = 2.99792458e8 * 1.0e-9; //Reduced speed of light ???

    ROOT::VecOps::RVec<float> out;

    for (const auto & rp: in) {

      if( rp.tracks_begin < tracks.size()) {

        float D0_wrt0 = tracks.at(rp.tracks_begin).D0;
        float Z0_wrt0 = tracks.at(rp.tracks_begin).Z0;
        float phi0_wrt0 = tracks.at(rp.tracks_begin).phi;

        TVector3 X( - D0_wrt0 * TMath::Sin(phi0_wrt0) , D0_wrt0 * TMath::Cos(phi0_wrt0) , Z0_wrt0);
        TVector3 x = X - V.Vect();

        TVector3 p(rp.momentum.x, rp.momentum.y, rp.momentum.z);

        double a = - rp.charge * Bz * cSpeed;
        double pt = p.Pt();
        double r2 = x(0) * x(0) + x(1) * x(1);
        double cross = x(0) * p(1) - x(1) * p(0);
        double T = TMath::Sqrt(pt * pt - 2 * a * cross + a * a * r2);
        double phi0 = TMath::ATan2((p(1) - a * x(0)) / T, (p(0) + a * x(1)) / T);

	out.push_back(phi0);

      } else {
        out.push_back(-9.);
      }
    }
    return out;
  }

  ROOT::VecOps::RVec<float> XPtoPar_C(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
				       const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
				       const float& Bz) {

    const double cSpeed = 2.99792458e8 * 1.0e3 * 1.0e-15;
    ROOT::VecOps::RVec<float> out;

    for (const auto & rp: in) {

      if( rp.tracks_begin < tracks.size()) {

        TVector3 p(rp.momentum.x, rp.momentum.y, rp.momentum.z);

        double a = std::copysign(1.0, rp.charge) * Bz * cSpeed;
	double pt = p.Pt();
        double C = a/(2 * pt);

	out.push_back(C);
      } else {
        out.push_back(-9.);
      }
    }
    return out;
  }

  ROOT::VecOps::RVec<float> XPtoPar_ct(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& in,
				       const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
				       const float& Bz) {

    const double cSpeed = 2.99792458e8 * 1.0e-9;
    ROOT::VecOps::RVec<float> out;

    for (const auto & rp: in) {

      if( rp.tracks_begin < tracks.size()) {

        TVector3 p(rp.momentum.x, rp.momentum.y, rp.momentum.z);
	double pt = p.Pt();

        double ct = p(2) / pt;

	out.push_back(ct);

      } else {
        out.push_back(-9.);
      }
    }
    return out;
  }


ROOT::VecOps::RVec<float>
getRP2TRK_D0(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
             ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).D0);
    else result.push_back(-9.);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_D0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					      ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[0]);
    else result.push_back(-9.);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_D0_sig(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					      ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).D0/sqrt(tracks.at(p.tracks_begin).covMatrix[0]));
    else result.push_back(-9.);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_Z0(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					  ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).Z0);
    else result.push_back(-9.);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_Z0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					      ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[9]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_Z0_sig(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					      ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).Z0/sqrt(tracks.at(p.tracks_begin).covMatrix[9]));
    else result.push_back(std::nan(""));
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_phi(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					   ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).phi);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_phi_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					       ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[2]);
    else result.push_back(-9);
  }
  return result;
}


ROOT::VecOps::RVec<float>
getRP2TRK_omega(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					     ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).omega);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_omega_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						 ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[5]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_tanLambda(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						 ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).tanLambda);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_tanLambda_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						     ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[14]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_d0_phi0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						   ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[1]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_d0_omega_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						    ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[3]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_d0_z0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						 ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[6]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_d0_tanlambda_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
							ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[10]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_phi0_omega_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						      ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[4]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_phi0_z0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						   ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[7]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_phi0_tanlambda_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
							  ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[11]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_omega_z0_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
						    ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[8]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_omega_tanlambda_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
							   ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[12]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<float>
getRP2TRK_z0_tanlambda_cov(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
							ROOT::VecOps::RVec<edm4hep::TrackState> tracks) {
  ROOT::VecOps::RVec<float> result;
  for (auto & p: in) {
    if (p.tracks_begin<tracks.size())
      result.push_back(tracks.at(p.tracks_begin).covMatrix[13]);
    else result.push_back(-9);
  }
  return result;
}

ROOT::VecOps::RVec<edm4hep::TrackState>
getRP2TRK( ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in,
					ROOT::VecOps::RVec<edm4hep::TrackState> tracks )
{

  ROOT::VecOps::RVec<edm4hep::TrackState> result ;
  result.reserve( in.size() );

  for (auto & p: in) {
    if (p.tracks_begin >= 0 && p.tracks_begin<tracks.size()) {
	result.push_back(tracks.at(p.tracks_begin) ) ;
    }
  }
 return result ;
}

// returns reco indices of tracks
ROOT::VecOps::RVec<int> 
get_recoindTRK( ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in, 
		ROOT::VecOps::RVec<edm4hep::TrackState> tracks )
{

  ROOT::VecOps::RVec<int> result ;
  
  for (unsigned int ctr=0; ctr<in.size(); ctr++) {
    edm4hep::ReconstructedParticleData p = in.at(ctr);
    if (p.tracks_begin >= 0 && p.tracks_begin<tracks.size()) result.push_back(ctr) ;
  }
 return result ;
}

int getTK_n(ROOT::VecOps::RVec<edm4hep::TrackState> x) {
  int result =  x.size();
  return result;
}

namespace {

inline float track_pt_from_state(const edm4hep::TrackState& state) {
  const double fB = 2.0; // assumed magnetic field in Tesla
  const double C = -0.5 * 1e3 * state.omega;
  const double denom = std::abs(2.0 * C);
  if (denom <= 0.0) return 0.f;
  return static_cast<float>(fB * 0.2998 / denom);
}

inline bool valid_track_range(int begin, int end, int limit) {
  return begin >= 0 && end >= begin && end <= limit;
}

} // namespace

ROOT::VecOps::RVec<int> getRP2TRK_nTracks(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& rps,
                  const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks) {
  ROOT::VecOps::RVec<int> out;
  out.reserve(rps.size());
  for (auto& p : rps) {
    if (p.tracks_begin >= 0 && p.tracks_begin < (int)tracks.size()) {
      // if only a single index: 1 track; if tracks_end available, use that
      out.push_back(1);
    } else {
      out.push_back(0);
    }
  }
  return out;
}

ROOT::VecOps::RVec<int> countTracksPerPFO(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objects,
  const ROOT::VecOps::RVec<int>& rp_trk_index,
  const ROOT::VecOps::RVec<int>& track_states_begin,
  const ROOT::VecOps::RVec<int>& track_states_end,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& track_states
) {
  (void)track_states_end;
  (void)track_states;
  ROOT::VecOps::RVec<int> result;
  result.reserve(objects.size());

  const int nLinks = static_cast<int>(rp_trk_index.size());
  const int nTracks = static_cast<int>(track_states_begin.size());

  for (const auto& obj : objects) {
    int count = 0;
    if (valid_track_range(obj.tracks_begin, obj.tracks_end, nLinks)) {
      for (int link = obj.tracks_begin; link < obj.tracks_end; ++link) {
        int trkIdx = rp_trk_index[link];
        if (trkIdx >= 0 && trkIdx < nTracks) ++count;
      }
    }
    result.push_back(count);
  }
  return result;
}

ROOT::VecOps::RVec<float> sumTrackPtPerPFO(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objects,
  const ROOT::VecOps::RVec<int>& rp_trk_index,
  const ROOT::VecOps::RVec<int>& track_states_begin,
  const ROOT::VecOps::RVec<int>& track_states_end,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& track_states
) {
  ROOT::VecOps::RVec<float> result;
  result.reserve(objects.size());

  const int nLinks = static_cast<int>(rp_trk_index.size());
  const int nTracks = static_cast<int>(track_states_begin.size());
  const int nStates = static_cast<int>(track_states.size());

  for (const auto& obj : objects) {
    float sumPt = 0.f;
    if (valid_track_range(obj.tracks_begin, obj.tracks_end, nLinks)) {
      for (int link = obj.tracks_begin; link < obj.tracks_end; ++link) {
        int trkIdx = rp_trk_index[link];
        if (trkIdx < 0 || trkIdx >= nTracks) continue;
        int stateBegin = track_states_begin[trkIdx];
        int stateEnd = (trkIdx < static_cast<int>(track_states_end.size())) ? track_states_end[trkIdx] : stateBegin + 1;
        if (stateBegin < 0 || stateBegin >= nStates) continue;
        if (stateBegin >= stateEnd && stateBegin + 1 <= nStates) stateEnd = stateBegin + 1;
        const auto& state = track_states[stateBegin];
        sumPt += track_pt_from_state(state);
      }
    }
    result.push_back(sumPt);
  }
  return result;
}

ROOT::VecOps::RVec<float> maxTrackPtPerPFO(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objects,
  const ROOT::VecOps::RVec<int>& rp_trk_index,
  const ROOT::VecOps::RVec<int>& track_states_begin,
  const ROOT::VecOps::RVec<int>& track_states_end,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& track_states
) {
  ROOT::VecOps::RVec<float> result;
  result.reserve(objects.size());

  const int nLinks = static_cast<int>(rp_trk_index.size());
  const int nTracks = static_cast<int>(track_states_begin.size());
  const int nStates = static_cast<int>(track_states.size());

  for (const auto& obj : objects) {
    float maxPt = 0.f;
    if (valid_track_range(obj.tracks_begin, obj.tracks_end, nLinks)) {
      for (int link = obj.tracks_begin; link < obj.tracks_end; ++link) {
        int trkIdx = rp_trk_index[link];
        if (trkIdx < 0 || trkIdx >= nTracks) continue;
        int stateBegin = track_states_begin[trkIdx];
        int stateEnd = (trkIdx < static_cast<int>(track_states_end.size())) ? track_states_end[trkIdx] : stateBegin + 1;
        if (stateBegin < 0 || stateBegin >= nStates) continue;
        if (stateBegin >= stateEnd && stateBegin + 1 <= nStates) stateEnd = stateBegin + 1;
        const auto& state = track_states[stateBegin];
        float pt = track_pt_from_state(state);
        if (pt > maxPt) maxPt = pt;
      }
    }
    result.push_back(maxPt);
  }
  return result;
}

int countExclusiveTrackLinks(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& objects,
  int primaryIndex,
  int otherIndex,
  const ROOT::VecOps::RVec<int>& rp_trk_index
) {
  const int nObjects = static_cast<int>(objects.size());
  const int nLinks = static_cast<int>(rp_trk_index.size());
  if (primaryIndex < 0 || primaryIndex >= nObjects) return 0;

  const auto& primary = objects[primaryIndex];
  if (!valid_track_range(primary.tracks_begin, primary.tracks_end, nLinks)) return 0;

  std::unordered_set<int> otherTracks;
  if (otherIndex >= 0 && otherIndex < nObjects) {
    const auto& other = objects[otherIndex];
    if (valid_track_range(other.tracks_begin, other.tracks_end, nLinks)) {
      for (int link = other.tracks_begin; link < other.tracks_end; ++link) {
        int trkIdx = rp_trk_index[link];
        if (trkIdx >= 0) otherTracks.insert(trkIdx);
      }
    }
  }

  int count = 0;
  for (int link = primary.tracks_begin; link < primary.tracks_end; ++link) {
    int trkIdx = rp_trk_index[link];
    if (trkIdx >= 0 && otherTracks.find(trkIdx) == otherTracks.end()) ++count;
  }
  return count;
}


int get_nChargedTracks_forPFO(const edm4hep::ReconstructedParticleData& p,
                              const ROOT::VecOps::RVec<edm4hep::TrackState>& tracks,
                              bool requireCharged = true)
{
  std::cout << "charge=" << p.charge
            << ", tracks_begin=" << p.tracks_begin
            << ", tracks_end=" << p.tracks_end
            << ", nTracksColl=" << tracks.size() << std::endl;

  if (requireCharged && p.charge == 0) {
    std::cout << "requireCharged=true and charge==0 -> return 0" << std::endl;
    return 0;
  }

  if (p.tracks_begin >= 0 && p.tracks_end >= p.tracks_begin &&
      p.tracks_end <= static_cast<int>(tracks.size()))
  {
    int n = p.tracks_end - p.tracks_begin;
    std::cout << "valid [begin,end) -> nTracksLinked=" << n << std::endl;
    return n;
  }

  if (p.tracks_begin >= 0 && p.tracks_end >= 0) {
    if (p.tracks_end < p.tracks_begin)
      std::cout << "WARNING: tracks_end < tracks_begin -> invalid range" << std::endl;
    else if (p.tracks_begin > static_cast<int>(tracks.size()))
      std::cout << "WARNING: tracks_begin > tracks.size() -> out-of-range begin" << std::endl;
    else if (p.tracks_end > static_cast<int>(tracks.size()))
      std::cout << "WARNING: tracks_end > tracks.size() -> out-of-range end" << std::endl;
  }

  if (p.tracks_begin >= 0 &&
      p.tracks_begin < static_cast<int>(tracks.size()))
  {
    std::cout << "legacy case: only tracks_begin valid -> nTracksLinked=1" << std::endl;
    return 1;
  }

  std::cout << "no valid track link -> return 0" << std::endl;
  return 0;
}


inline double rp_pt(const edm4hep::ReconstructedParticleData& p) {
  return std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y);
}
inline double rp_eta(const edm4hep::ReconstructedParticleData& p) {
  const double pt = rp_pt(p), pz = p.momentum.z;
  if (pt == 0.0) return 0.0;
  return std::asinh(pz / pt);
}
inline double rp_phi(const edm4hep::ReconstructedParticleData& p) {
  return std::atan2(p.momentum.y, p.momentum.x);
}
inline double deltaR(double eta1, double phi1, double eta2, double phi2) {
  double dphi = std::fabs(phi1 - phi2);
  if (dphi > M_PI) dphi = 2*M_PI - dphi;
  const double deta = eta1 - eta2;
  return std::sqrt(deta*deta + dphi*dphi);
}
// Build once per event: for each track, store the max pT of any linked RP
ROOT::VecOps::RVec<float> make_track_pt_proxy(
  const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& rps,
  const ROOT::VecOps::RVec<int>& rp_trk_index // _ReconstructedParticles_tracks.index
){
  ROOT::VecOps::RVec<float> trk_pt;
  
  // Find max track index to size the vector
  int maxIdx = -1;
  for (int v : rp_trk_index) if (v > maxIdx) maxIdx = v;
  trk_pt.assign(maxIdx+1, 0.0f);

  for (size_t rp_i = 0; rp_i < rps.size(); ++rp_i) {
    const auto& p = rps[rp_i];
    if (p.charge == 0) continue;  // want charged particles
    int b = p.tracks_begin, e = p.tracks_end;
    
    if (!(b >= 0 && e >= b && e <= (int)rp_trk_index.size())) continue;

    float pt = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y);
    for (int k = b; k < e; ++k) {
      int trk = rp_trk_index[k];
      if (trk >= 0 && trk < (int)trk_pt.size())
        if (pt > trk_pt[trk]) trk_pt[trk] = pt;
    }
  }
  return trk_pt;
}

int count_tau_tracks_cone_DRpt(
  const edm4hep::ReconstructedParticleData& tau,
  const ROOT::VecOps::RVec<int>& trk_states_begin,
  const ROOT::VecOps::RVec<int>& trk_states_end,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& tstates,
  const ROOT::VecOps::RVec<float>& track_pt_proxy,  // from make_track_pt_proxy
  double dR_max = 0.2,
  double pt_min = 0.5
) {
  auto tau_pt  = std::sqrt(tau.momentum.x*tau.momentum.x + tau.momentum.y*tau.momentum.y);
  auto tau_eta = (tau_pt == 0.0) ? 0.0 : std::asinh(tau.momentum.z / tau_pt);
  auto tau_phi = std::atan2(tau.momentum.y, tau.momentum.x);
  auto deltaR = [](double e1,double p1,double e2,double p2){
    double dphi = std::fabs(p1 - p2); if (dphi > M_PI) dphi = 2*M_PI - dphi;
    double de = e1 - e2; return std::sqrt(de*de + dphi*dphi);
  };

  int n = 0;
  const int nTracks = (int)trk_states_begin.size();
  for (int t = 0; t < nTracks; ++t) {
    int sb = trk_states_begin[t], se = trk_states_end[t]; // beginning and end of track states
    if (!(sb >= 0 && se > sb && se <= (int)tstates.size())) continue;

    if (t >= (int)track_pt_proxy.size() || track_pt_proxy[t] <= pt_min) continue;

    const auto& ts = tstates[sb];
    double eta = std::asinh(ts.tanLambda), phi = ts.phi;
    if (deltaR(eta, phi, tau_eta, tau_phi) < dR_max) ++n;
  }
  return n;
}

///
ROOT::VecOps::RVec<bool> 
hasTRK( ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in ) {

  ROOT::VecOps::RVec<bool> result ;
  result.reserve( in.size() );
  
  for (auto & p: in) {
    if (p.tracks_begin >= 0 && p.tracks_begin != p.tracks_end) result.push_back(true) ;
    else result.push_back(false);
  }
 return result ;
}
int countLinkedTracksForPFO(
  const edm4hep::ReconstructedParticleData& p,
  const ROOT::VecOps::RVec<int>& rp_trk_index,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& track_states
) {
  const int nLinks = static_cast<int>(rp_trk_index.size());
  const int nTracks = static_cast<int>(track_states.size());

  if (p.tracks_begin >= 0 && p.tracks_end >= p.tracks_begin &&
      p.tracks_end <= nLinks)
  {
    int count = 0;
    for (int idx = p.tracks_begin; idx < p.tracks_end; ++idx) {
      const int trk = rp_trk_index[idx];
      if (trk >= 0 && trk < nTracks) {
        ++count;
      }
    }
    return count;
  }

  if (p.tracks_begin >= 0 && p.tracks_begin < nLinks) {
    const int trk = rp_trk_index[p.tracks_begin];
    if (trk >= 0 && trk < nTracks) {
      return 1;
    }
  }

  return 0;
}

inline float pt_from_omega(const edm4hep::TrackState& st, float Bz_Tesla) {
  const float om = st.omega;                    // curvature
  if (Bz_Tesla <= 0.f || !std::isfinite(om) || om == 0.f) return 0.f;
  return 0.299792458f * Bz_Tesla / (std::abs(om) * 1e3f);
}

// --- 2) Build a per-track pT array (size = number of tracks).
//     We use the first state for each track (begin) as the representative state.
ROOT::VecOps::RVec<float> make_track_pt_array(
  const ROOT::VecOps::RVec<int>& track_states_begin,
  const ROOT::VecOps::RVec<int>& track_states_end,       // can be ignored; kept for symmetry
  const ROOT::VecOps::RVec<edm4hep::TrackState>& states,
  float Bz_Tesla
) {
  ROOT::VecOps::RVec<float> pt;
  pt.reserve(track_states_begin.size());
  const int nStates = static_cast<int>(states.size());
  for (int t = 0; t < (int)track_states_begin.size(); ++t) {
    int sb = track_states_begin[t];
    if (sb >= 0 && sb < nStates) pt.push_back(pt_from_omega(states[sb], Bz_Tesla));
    else                         pt.push_back(0.f);
  }
  return pt;
}

// --- 3) Count tracks within ΔR of the tau, with a track-pT threshold.
//     Uses track eta/phi from TrackState (eta = asinh(tanLambda), phi = st.phi).
int count_tau_tracks_cone_DRpt_trackPT(
  const edm4hep::ReconstructedParticleData& tau,
  const ROOT::VecOps::RVec<int>& trk_states_begin,
  const ROOT::VecOps::RVec<int>& trk_states_end,
  const ROOT::VecOps::RVec<edm4hep::TrackState>& tstates,
  const ROOT::VecOps::RVec<float>& track_pt,   // from make_track_pt_array(...)
  double dR_max = 0.2,
  double pt_min = 0.5
) {
  // tau kinematics
  const double tau_pt  = std::hypot(tau.momentum.x, tau.momentum.y);
  const double tau_eta = (tau_pt > 0.0) ? std::asinh(tau.momentum.z / tau_pt) : 0.0;
  const double tau_phi = std::atan2(tau.momentum.y, tau.momentum.x);

  auto dR = [](double e1,double p1,double e2,double p2){
    double dphi = std::fabs(p1 - p2); if (dphi > M_PI) dphi = 2*M_PI - dphi;
    const double de = e1 - e2;
    return std::sqrt(de*de + dphi*dphi);
  };

  int n = 0;
  const int nTracks = static_cast<int>(trk_states_begin.size());
  const int nStates = static_cast<int>(tstates.size());

  for (int t = 0; t < nTracks; ++t) {
    // pT cut on the track itself
    if (t >= (int)track_pt.size() || track_pt[t] <= pt_min) continue;

    int sb = trk_states_begin[t];
    if (sb < 0 || sb >= nStates) continue;  // need a valid state

    const auto& st = tstates[sb];
    const double eta = std::asinh(st.tanLambda);
    const double phi = st.phi;

    if (dR(eta, phi, tau_eta, tau_phi) < dR_max) ++n;
  }
  return n;
}


}//end NS ReconstructedParticle2Track

}//end NS FCCAnalyses

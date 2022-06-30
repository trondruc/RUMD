
%{
#include "rumd/PairPotential.h"
#include "rumd/WallPotential.h"
#include "rumd/CollectiveDensityField.h"
#include "rumd/TetheredGroup.h"
#include "rumd/HarmonicUmbrella.h"
#include "rumd/AxilrodTellerPotential.h"
#include "rumd/EMT_Potential.h"
#include "rumd/BondPotential.h"
#include "rumd/ConstraintPotential.h"
#include "rumd/AnglePotential.h"
#include "rumd/DihedralPotential.h"
  %}

%nodefaultctor Potential; // disable generation of wrapper for default constructor
%nodefaultctor WallPotential; 

class Potential
{
 public:
  void SetVerbose(bool vb);
  void SetID_String(const std::string& set_ID_String);
  std::string GetID_String() const { return ID_String; }
  Potential* Copy();
  bool IsMolecularPotential();
  void ResetInternalData();
};

%extend Potential {
  PyObject* GetParameterList(unsigned num_types) {
    std::vector<float> parameters = $self->GetPotentialParameters(num_types);
    unsigned length = parameters.size();
    PyObject* param_list = PyList_New(length);
    for(unsigned idx = 0; idx < length; idx++) {
      PyList_SET_ITEM(param_list, idx, PyFloat_FromDouble(parameters[idx]));
    }

    return param_list;
  }
}


%nodefaultctor PairPotential; // disable generation of wrapper for default constructor

class PairPotential : public Potential
{
 public:
  void WritePotentials(SimulationBox* simBox);
  float GetNbListSkin();
  float GetMaxCutoff();
  int GetNbListRebuildRequired();
  void SetNbListSkin( float skin );
  void SetNB_Method(const std::string& method);
  std::string GetNB_Method();
  void SetNbListAllocateMax(bool set_allocate_max);
  void SetNbMaxNumNbrs(unsigned maxNumNbrs);
  void CopyExclusionListToDevice();
  void SetAssumeNewton3(bool assume_N3);
  enum CutoffMethod {NS, SP, SF};
  float GetPotentialParameter(unsigned int i, unsigned int j, unsigned int index);
  double GetPotentialEnergy();
  void SetExclusion( unsigned particleI, unsigned particleJ );
  void SetExclusionType( unsigned type0, unsigned type1 );
  unsigned GetActualMaxNumNbrs();
};


%inline {
  int NumberOfPairPotentialParameters() { return NumParam; }
}

%pythoncode %{
  NoShift = PairPotential.NS
  ShiftedPotential = PairPotential.SP
  ShiftedForce = PairPotential.SF
%}


%feature ("autodoc","1") SetParams; // generate docstring for SetParams


// docstring for Pot_IPL_12_6
%feature("autodoc","Standard 12-6 Lennard-Jones (cut and shifted potential)
v(r) =  ( A12/r^12  +  A6/r^6 ) - v_cut
s(r) = 12 A12/r^14 + 6 A6/r^8  = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_IPL_12_6;

class Pot_IPL_12_6 : public PairPotential
{
 public:
  Pot_IPL_12_6( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float A12, float A6, float Rcut);
};


// docstring for Pot_LJ_12_6
%feature("autodoc","Standard 12-6 Lennard-Jones (cut and shifted potential)
v(r) =  4 Epsilon ( (Sigma/r)^12 -    (Sigma/r)^6 ) - v_cut
s(r) = 48 Epsilon ( (Sigma/r)^14 - 0.5(Sigma/r)^8 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_LJ_12_6;

class Pot_LJ_12_6 : public Pot_IPL_12_6
{
 public:
  Pot_LJ_12_6( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


// docstring for Pot_LJ_12_6_smooth_2_4
%feature("autodoc","12-6 Lennard-Jones smoothed at cut_off by second and fourth order term
v(r) = 4 Epsilon (   (Sigma/r)^12 -  (Sigma/r)^6 ) + c0 + c2*(r/Sigma)^2 + c4*(r/Sigma)^4
s(r) = 4 Epsilon ( 12(Sigma/r)^14 - 6(Sigma/r)^8 +  2c2 + 4c4(r/Sigma)^2 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_LJ_12_6_smooth_2_4;

class Pot_LJ_12_6_smooth_2_4 : public PairPotential
{
 public:
  Pot_LJ_12_6_smooth_2_4( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


%feature("autodoc","Generalized Lennard-Jones (cut and shifted potential)
v(r) =   Epsilon/(m -n) ( n*(Sigma/r)^m -  m*(Sigma/r)^n ) - v_cut
s(r) =  Epsilon ( (Sigma/r)^14 - 0.5(Sigma/r)^8 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)\n") Pot_gLJ_m_n; 


class Pot_gLJ_m_n : public PairPotential
{
 public:
  Pot_gLJ_m_n(float set_m, float set_n, CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","Standard 12-6 Lennard-Jones (cut and shifted potential)
with formulated so Sigma is the location of the minimum, consistent with gLJ
v(r) =   Epsilon ( (Sigma/r)^12 -  2 *  (Sigma/r)^6 ) - v_cut
\n") Pot_gLJ_m12_n6; 

class Pot_gLJ_m12_n6 : public Pot_IPL_12_6
{
 public:
  Pot_gLJ_m12_n6( CutoffMethod cutoff_method);
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


%feature("autodoc","12-6 Lennard-Jones plus Gaussian potential cut and shifted.
v(r) =  Epsilon ( (Sigma/r)^12 -  2(Sigma/r)^6 - Epsilon_0 (exp[-(r - r_0)^2 / (2 Sigma_0^2)] ) ) - v_cut
v(r) is a double well with second well located at r_0 with depth Epsilon_0.
s(r) = 48 Epsilon ( (Sigma/r)^14 - 0.5(Sigma/r)^8 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_LJ_12_6_Gauss; 

class Pot_LJ_12_6_Gauss : public PairPotential
{
 public:
  Pot_LJ_12_6_Gauss( CutoffMethod cutoff_method);
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut, float Sigma0, float Epsilon0, float r0);
};


%feature("autodoc","Gaussian core potential
v(r) = Epsilon * exp[-(r/Sigma)^2]
s(r) = ( 2 * Epsilon / Sigma^2 ) * exp[-(r/Sigma)^2] = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_Gauss; 

class Pot_Gauss : public PairPotential
{
 public:
  Pot_Gauss( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","\nBuckingham potential
v(r) = Epsilon{(6/(Alpha - 6))*exp[Alpha(1-r/rm)] - (Alpha/(Alpha -6))*(rm/r)**6}
s(r) = -r^-1*dv/dr = ((6*Alpha*Epsilon)/(Alpha - 6)){r^(-1)*exp[Alpha(1-r/rm)] - rm^(-6)*r^(-8)}
w(r) = s*r^2 = ((6*Alpha*Epsilon)/(Alpha - 6))*{r*exp[Alpha(1-r/rm)] - (rm/r)^6}") Pot_Buckingham; 

class Pot_Buckingham : public PairPotential
{
 public:
  Pot_Buckingham( CutoffMethod cutoff_method);
  void SetParams(unsigned int i, unsigned int j, float rm, float Alpha, float Epsilon, float Rcut);
};


%feature("autodoc","\nIPL potential, n=12. (cut and shifted potential)
v(r) =    Epsilon ( (Sigma/r)^12 ) - v_cut
s(r) = 12 Epsilon ( (Sigma/r)^14 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)\n") Pot_IPL_12; 

class Pot_IPL_12 : public PairPotential
{
 public:
  Pot_IPL_12( CutoffMethod cutoff_method ) ;
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


%feature("autodoc","\nIPL potential, n=9. (cut and shifted potential)
v(r) =    Epsilon ( (Sigma/r)^9 ) - v_cut 
s(r) = 9 Epsilon ( (Sigma/r)^11 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)\n") Pot_IPL_9; 

class Pot_IPL_9 : public PairPotential
{
 public:
  Pot_IPL_9( CutoffMethod cutoff_method ) ;
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


%feature("autodoc","IPL potential, n=18. (cut and shifted potential)
v(r) =    Epsilon ( (Sigma/r)^18 ) - v_cut
s(r) = 18 Epsilon ( (Sigma/r)^20 )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)\n") Pot_IPL_18; 

class Pot_IPL_18 : public PairPotential
{
 public:
  Pot_IPL_18( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","\nIPL potential, abritrary exponent n. (cut and shifted potential)
v(r) =   Epsilon ( (Sigma/r)^n ) - v_cut
s(r) = n Epsilon ( (Sigma/r)^(n+2) )/Sigma^2 = -r^-1*dv/dr (F_ij = s * r_ij)\n") Pot_IPL_n; 

class Pot_IPL_n : public PairPotential
{
 public:
  Pot_IPL_n( float n, CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","Integral of IPLs r^-n with n from 6 to inf
v(r) = Epsilon (Sigma/r)^6 / ln(r/Sigma) - v_cut
s(r) = Epsilon (Sigma/r)^8 ( 6 ln(r/Sigma) + 1 )  /  ( Sigma^2 * ln(r/Sigma)^2 ) = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_IPL6_LN;

class Pot_IPL6_LN : public PairPotential
{
 public:
  Pot_IPL6_LN( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","hard sphere LJ-like potential
v(r) = Epsilon/Epsilon0 ( (Rm/r)^12- a (Rm/r)^6 / ln(r/(R0)) )
with        a = ( 1 - 12ln(R0) ) / ( 1 - 6ln(R0) ) 
     Epsilon0 = 1 / ( 1/6 - ln(R0) )
potential minimum is at (Rm,-Epsilon)") Pot_IPL6_LN;

class Pot_LJ_LN : public PairPotential
{
 public:
  Pot_LJ_LN( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Rm, float R0, float Epsilon, float Rcut);
};

%feature("autodoc","Girifalco potential)") Pot_Girifalco;

class Pot_Girifalco : public PairPotential
{
 public:
  Pot_Girifalco( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float alpha, float beta, float R0, float Rcut);
};

%feature("autodoc","\nYukawa potential
v(r) = epsilon * sigma / r * exp[-r/sigma]
s(r) = (1/r+1/sigma)/r * v(r)") Pot_Yukawa; 

class Pot_Yukawa : public PairPotential
{
 public:
  Pot_Yukawa( CutoffMethod cutoff_method);
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};

%feature("autodoc","Dzugutov potential") Pot_Dzugutov; 

class Pot_Dzugutov : public PairPotential
{
 public:
  Pot_Dzugutov();
  void SetParams(unsigned int i, unsigned int j, float a, float b, float c, float d, float A, float B);
};

%feature("autodoc","Fermi-Jagla potential") Pot_Fermi_Jagla; 

class Pot_Fermi_Jagla : public PairPotential
{
 public:
  Pot_Fermi_Jagla( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float a, float epsilon0, float A0, float A1, float A2, float B0, float B1, float B2, float n, float Rcut);
};

%feature("autodoc","Repulsive-Shoulder") Pot_Repulsive_Shoulder; 

class Pot_Repulsive_Shoulder : public PairPotential
{
 public:
  Pot_Repulsive_Shoulder( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float k0, float Sigma1, float Rcut);
};

%feature("autodoc","Potential for harmonically constraining density fluctuations with a specified wavevector to a given value") CollectiveDensityField; 

class CollectiveDensityField : public Potential
{
 public:
  CollectiveDensityField();
   void SetParams(unsigned int nx, unsigned int ny, unsigned int nz, float kappa, float a);
   float GetCollectiveDensity();
};


%extend CollectiveDensityField {
  PyObject* GetRho_k() {

    float rho_k_re, rho_k_im;
    float rho_k_mag = $self->GetCollectiveDensity(&rho_k_re, &rho_k_im);
    PyObject* result = PyTuple_New(3);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(rho_k_re));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(rho_k_im));
    PyTuple_SET_ITEM(result, 2, PyFloat_FromDouble(rho_k_mag));
    return result;
  }

}


%feature("autodoc","Make a solid block of particles joined by springs to lattice sites; can be moved as a unit") TetheredGroup;

class TetheredGroup : public Potential {
 public:
  TetheredGroup(std::vector<unsigned> solidAtomTypes, float springConstant);
  ~TetheredGroup();

  void SetSpringConstant(float ks);
  void SetDirection(unsigned set_dir);
  void Move(float displacement);
};


class WallPotential : public Potential{};

%feature("autodoc","Smooth Lennard-Jones 9-3 wall potential") Wall_LJ_9_3; 

class Wall_LJ_9_3 : public WallPotential
{
 public:
  Wall_LJ_9_3();
  void SetParams( float wallOne, float wallTwo, float sigma1, float epsilon1, float sigma2, float epsilon2, float rhoWall, float scale );
};

%feature("autodoc","Exponential potential
 v(r) = Epsilon * exp[-(r/Sigma)]
 s(r) = ( Epsilon / ( Sigma * r ) ) * exp[-(r/Sigma)] = -r^-1*dv/dr (F_ij = s * r_ij)") Pot_Exp;

class Pot_Exp : public PairPotential {
 public:
  Pot_Exp( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut);
};


%feature("autodoc","The SAAP potential for the noble elements.
v(r) = [a0*exp(a1*r+a6*r**2)/r+a2*exp(a3*r)+a4]/[1+a5*r**6]

Reference: Deiters and Sadus, J. Chem. Phys. 150, 134504 (2019); https://doi.org/10.1063/1.5085420
Deiters and Sadus gives parameters for He, Ne, Ar, Kr and Xe.

Example of using Argon parameters::
    pot = rumd.Pot_SAAP(cutoff_method = rumd.ShiftedPotential)
    pot.SetParams(i=0, j=0,
                  a0=65214.64725, a1=-9.45234334, a2=-19.42488828, 
                  a3=-1.958381959, a4=-2.379111084, a5=1.051490962, a6=0.0, 
                  Sigma=1.0, Epsilon=1.0, Rcut=4.0)
") Pot_SAAP;

class Pot_SAAP : public PairPotential
{
 public:
  Pot_SAAP( CutoffMethod cutoff_method );
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float a0, float a1, float a2, float a3, float a4, float a5, float a6, float Rcut);
};

%feature("autodoc","""Coulomb potential
 v(r) = Epsilon / r
Truncated using shifted-force cutoff [See J. Phys. Chem. B 116, 5738 (2012)]""" ) Pot_ShiftedForceCoulomb;

class Pot_ShiftedForceCoulomb : public PairPotential {
 public:
  Pot_ShiftedForceCoulomb();
  void SetParams(unsigned int i, unsigned int j, float Epsilon, float Rcut);
};

%feature("autodoc","Potential for Umbrella sampling with overlap order-parameter defined using a pair potential object") HarmonicUmbrella; 

class HarmonicUmbrella : public Potential {
public:
  HarmonicUmbrella(PairPotential* pairPot, float springConst, float Q0 );
  void SetParams(float springConst, float Q0);
  float GetOrderParameter();
  double GetPotentialEnergy();
};


%feature("autodoc","Three-body term in Argon potential") AxilrodTellerPotential; 

class AxilrodTellerPotential : public Potential {
public:
  AxilrodTellerPotential(float v_AT, float Rcut );
};

%feature("autodoc","Effective medium theory many body potential for metals") EMT_Potential; 


class EMT_Potential : public Potential {
public:
  EMT_Potential();
  void SetParams(unsigned type_idx, float E0, float s0, float V0, float eta2, float kappa, float Lambda, float n0);
  float GetNbListSkin();
  float GetMaxCutoff();
  void SetNbListSkin( float skin );
  void SetNB_Method(const std::string& method);
  std::string GetNB_Method();
  void ResetNeighborList();
};


%nodefaultctor BondPotential; // disable generation of wrapper for default constructor

class BondPotential : public Potential {
 public:
  void SetParams(unsigned bond_type, float bond_length, float stiffness, bool exclude);
  void SetExclusions(PairPotential* non_bond_pot );
};

class BondHarmonic : public BondPotential {
 public:
  BondHarmonic();
 
};

class BondFENE : public BondPotential {
 public:
  BondFENE();
};


class ConstraintPotential : public BondPotential {
 public:
  ConstraintPotential();
  void UpdateConstraintLayout();
  unsigned GetNumberOfConstraints();
  void SetParams(unsigned bond_type, float bond_length);
  void SetNumberOfConstraintIterations(unsigned num_iter);
  void SetLinearConstraintMolecules(bool linear);
  void WritePotential(SimulationBox* simBox);
};

%nodefaultctor AnglePotential; // disable generation of wrapper for default constructor

class AnglePotential : public Potential {
 public:
  void SetParams(unsigned angle_type, float theta0, float ktheta);
  void SetExclusions(PairPotential* non_bond_pot );
};

class AngleCosSq : public AnglePotential {
 public:
  AngleCosSq();

};

class AngleSq : public AnglePotential {
 public:
  AngleSq();

};

%nodefaultctor DihedralPotential; // disable generation of wrapper for default constructor

class DihedralPotential : public Potential{
 public:
  void SetParams(unsigned dihedral_type, std::vector<float> coeffs);
  void SetExclusions(PairPotential* non_bond_pot);
};


class DihedralRyckaert : public DihedralPotential {
 public:
  DihedralRyckaert();
  
};

class PeriodicDihedral : public DihedralPotential {
 public:
  PeriodicDihedral();
};

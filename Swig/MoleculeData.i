%{
#include "rumd/MoleculeData.h"
  %}

// turn off keyword arguments for overloaded functions to avoid warnings
%feature("kwargs", "0") MoleculeData::PrintLists;
%feature("kwargs", "0") MoleculeData::SetExclusionMolecule;

class MoleculeData {
 public:
  MoleculeData(Sample *, const std::string& top_filename);
  MoleculeData* Copy();
  ~MoleculeData();

  void ReadTopology(const std::string& top_filename);

  void SetBondParams(unsigned bond_type, float length_param, float stiffness_param, unsigned bond_class);
  void SetAngleParams(unsigned angle_type, float theta0, float ktheta);
  void SetDihedralParams(unsigned dihedral_type, float prm0, float prm1, float prm2, float prm3, float prm4, float prm5);

  void PrintLists();
  void PrintLists(unsigned i);
  void PrintMlist();

  void SetExclusionBond(PairPotential *potential, unsigned etype);
  void SetExclusionMolecule(PairPotential *potential, unsigned molindex);
  void SetExclusionMolecule(PairPotential *potential);

  void SetExclusionAngle(PairPotential* potential);
  void SetExclusionDihedral(PairPotential* potential);

  void SetAngleConstant(unsigned type, float ktheta);
  void SetDihedralRyckaertConstants(unsigned type, float c0, float c1, float c2, 
				    float c3, float c4, float c5);
  void SetPeriodicDihedralConstants(unsigned type, float phiEq, float v1, float v2, float v3);
  void SetRigidBodyVelocities();
  
  unsigned GetNumberOfMolecules();
  unsigned GetMaximumNumberOfBonds();
  unsigned GetNumberOfBonds();
  unsigned GetNumberOfAngles();
  unsigned GetNumberOfDihedrals();

  void WriteMolConf(const std::string& fstr);
  void WriteMolxyz(const std::string&fstr, unsigned mol_id, const std::string& mode);
  void WriteMolxyz_filtered(const std::string&fstr, unsigned filter_uau, const std::string& mode);

};


%extend MoleculeData {
  void Assign(const MoleculeData& M) {
    (*$self) = M;
    }


  PyObject* GetPositions() {

    $self->EvalCM();

    npy_intp dims[2];
    unsigned int nmols = $self->GetNumberOfMolecules();

    dims[0] = nmols;
    dims[1] = 3;

    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    int len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nmols; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_cm[pdx])), len);
        ptr += len;
      }
    
    return posArray;
  }

  PyObject* GetVelocities() {

    $self->EvalVelCM();

    npy_intp dims[2];
    unsigned int nmols = $self->GetNumberOfMolecules();

    dims[0] = nmols;
    dims[1] = 3;

    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    int len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nmols; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_vcm[pdx])), len);
        ptr += len;
      }
    
    return posArray;
  }

  PyObject* GetAngularVelocities() {
    
    $self->EvalSpin();

    npy_intp dims[2];
    unsigned int nmols = $self->GetNumberOfMolecules();

    dims[0] = nmols;
    dims[1] = 3;

    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    int len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nmols; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_omega[pdx])), len);
        ptr += len;
      }
    
    return posArray;
  }

  PyObject* GetMoleculeMasses() {
    
    $self->EvalMolMasses();

    npy_intp dims[2];
    unsigned int nmols = $self->GetNumberOfMolecules();

    dims[0] = nmols;
    dims[1] = 1;

    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    int len = sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nmols; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_cm[pdx].w)), len);
        ptr += len;
      }
    
    return posArray;
  }

  PyObject* GetAngles() {
    
    $self->GetAngles();
    unsigned int nangles = $self->GetNumberOfAngles();

    npy_intp dims[2];
    dims[0] = nangles; 
    dims[1] = 1;

    PyObject *angleArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) angleArray);
    int len = sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nangles; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_angles[pdx])), len);
        ptr += len;
      }
    
    return angleArray;
  }

  PyObject* GetDihedrals() {
    
    $self->GetDihedrals();
    unsigned int ndihedrals = $self->GetNumberOfDihedrals();

    npy_intp dims[2];
    dims[0] = ndihedrals; 
    dims[1] = 1;

    PyObject *dihedArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) dihedArray);
    int len = sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < ndihedrals; pdx++)
      {
        memcpy((void*) ptr, (void*)(&($self->h_dihedrals[pdx])), len);
        ptr += len;
      }
    
    return dihedArray;
  }
  
  PyObject* GetStresses() {

    npy_intp dims[2];

    dims[0] = 6;
    dims[1] = 1;

    PyObject *stressesArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) stressesArray);
    int len = 6*sizeof(float)/sizeof(char);

    memcpy((void*) ptr, (void*)(&($self->symmetricStress[0])), len);   

    return stressesArray;
  }

  
}

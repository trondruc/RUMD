
%{
#include "rumd/Sample.h"
#include "rumd/Potential.h"
%}

%include "numpy.i"

// the following is to allow proper handling of the default argument "w".
// The string typemap does not seem to be applied when there are default
// arguments and I couldn't figure out how to change that

%feature("shadow") Sample::WriteConf(const std::string&filename, 
				     const std::string& mode) %{
  def WriteConf(self, filename, mode="w"):
      return _rumd.Sample_WriteConf(self, filename, mode)
    %}

%rename (SetOutputManagerMetaData_bool) SetOutputManagerMetaData(const std::string&, const std::string&, bool);

%rename (SetOutputManagerMetaData_int) SetOutputManagerMetaData(const std::string&, const std::string&, int);

%feature("shadow") Sample::SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, bool on) %{
def SetOutputManagerMetaData(self, manager_name, key, value):
    if type(value) == type(True):
        return _rumd.Sample_SetOutputManagerMetaData_bool(self, manager_name, key, value)
    else:
        return _rumd.Sample_SetOutputManagerMetaData_int(self, manager_name, key, value)
  %}

// to avoid warnings about keyword arguments not being possible for overloaded functions
%feature("kwargs", "0") Sample::SetOutputManagerMetaData;


%apply (double* IN_ARRAY1, int DIM1) {(double* mass_array, int length)};


class Sample
{
 public:
  Sample( unsigned int set_particles_per_block, unsigned int set_threads_per_particle);
  Sample* Copy();
  int GetNumberOfParticles() const;
  unsigned int GetNumberOfTypes() const;
  unsigned int GetNumberThisType(unsigned int type) const;
  SortingScheme GetSortingScheme() const;
  std::string GetOutputDirectory() const;

  void SetVerbose(bool vb);
  void SetCheckCudaErrors(bool set_check_cuda_errors);
  void InitializeOutputManagers(unsigned long int timeStepIndex);
  void NotifyOutputManagers(unsigned long int timeStepIndex);
  void TerminateOutputManagers();
  void ReadConf( const std::string& filename, bool init_itg=false );
  void ReadRestartConf(int restartBlock, unsigned int numDigitsBlkIdx=4);
  void WriteConf(const std::string&filename, const std::string& mode);

  void EnableBackup(bool make_backup);
  void SetOutputBlockSize(unsigned long int blockSize);
  void SetOutputManagerActive(const std::string& manager_name, bool active);
  void SetLogLinParameters(const std::string& manager_name, unsigned int base, unsigned int maxInterval, long int user_maxIndex=-1);
  void SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, bool on);
  void SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, int value);
  void AddExternalCalculator(ExternalCalculator* calc) {energiesLogLin->AddExternalCalculator(calc)};
  void RemoveExternalCalculator(ExternalCalculator* calc) {energiesLogLin->RemoveExternalCalculator(calc)};
  

  void SetMoleculeData(MoleculeData* md);
  MoleculeData* GetMoleculeData();
  void SetNumberOfDOFs(unsigned int DOFs);
  void UpdateDegreesOfFreedom(unsigned numberOfConstraints=0);
  void SortParticles();
  
  void SetPotential(Potential* potential);
  void AddPotential(Potential* potential);
  void IsotropicScaleSystem( float Rscal );
  void IsotropicScaleSystemCM( float Rscal );
  void AnisotropicScaleSystem( float Rscal, int dir );
  void AffinelyShearSystem( float shear_strain );
  void ScaleVelocities( float factor );
  void SetSimulationBox(SimulationBox* simBox);
  SimulationBox* GetSimulationBox() const;
  void SetIntegrator(Integrator* newItg){ itg = newItg; }

  void SetMass(unsigned type, float mass);
  void SetAllMasses(double* mass_array, int length);

  void SetPB_TP( unsigned int pb, unsigned int tp );
  void SetSortingScheme( SortingScheme ss);
  void SetIncludeKineticStress(bool include_kinetic_stress);

  unsigned int GetParticlesPerBlock();
  unsigned int GetThreadsPerParticle();
  unsigned int GetNumberOfDOFs() const;

  void CalcF(bool calc_stresses = false);
  double GetPotentialEnergy() const;
  double GetVirial() const;
  float GetMass(unsigned int type) const;
  float GetMeanMass() const;

  void SetOutputDirectory(const std::string&filename);
  void AddOutputManager(const std::string& manager_name, LogLinOutputManager* om);
  const Integrator* GetIntegrator() const;
};

%extend Sample {
  void Assign(const Sample& S) {
    (*$self) = S;
  }

  PyObject* GetPositions() {
    const ParticleData& pData = *$self->GetParticleData();
    pData.CopyPosFromDevice();
    int len;
    npy_intp dims[2];
    unsigned int nParticles = $self->GetNumberOfParticles();
    dims[0] = nParticles;
    dims[1] = 3;
    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy((void*) ptr, (void*)(&(pData.h_r[pdx])), len);
      ptr += len;
    }
    
    return posArray;
  }
  PyObject* GetTypes() {
    const ParticleData& pData = *$self->GetParticleData();
    int len;
    npy_intp dims[1];
    unsigned int nParticles = $self->GetNumberOfParticles();
    dims[0] = nParticles;
    PyObject *typeArray = PyArray_SimpleNew(1, dims, NPY_INT32);
    char *ptr = PyArray_BYTES((PyArrayObject*) typeArray);
    len = sizeof(int)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy((void*) ptr, (void*)(&(pData.h_r[pdx].w)), len);
      ptr += len;
    }
    
      return typeArray;
  }



  PyObject* GetVelocities()  {
    const ParticleData& pData = *$self->GetParticleData();
    pData.CopyVelFromDevice();
    int len;
    npy_intp dims[2];
    unsigned int nParticles = $self->GetNumberOfParticles();
    dims[0] = nParticles;
    dims[1] = 3;
    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy((void*) ptr, (void*)(&(pData.h_v[pdx])), len);
      ptr += len;
    }
    
    return posArray;
  }
  PyObject* GetForces()  {
    const ParticleData& pData = *$self->GetParticleData();
    pData.CopyForFromDevice();
    int len;
    npy_intp dims[2];
    unsigned int nParticles = $self->GetNumberOfParticles();
    dims[0] = nParticles;
    dims[1] = 3;
    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy((void*) ptr, (void*)(&(pData.h_f[pdx])), len);
      ptr += len;
    }
    
    return posArray;
  }
  PyObject* GetImages()  {
    const ParticleData& pData = *$self->GetParticleData();
    pData.CopyImagesFromDevice();
    int len;
    npy_intp dims[2];
    unsigned int nParticles = $self->GetNumberOfParticles();
    dims[0] = nParticles;
    dims[1] = 3;
    PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    char *ptr = PyArray_BYTES((PyArrayObject*) posArray);
    len = 3*sizeof(float)/sizeof(char);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy((void*) ptr, (void*)(&(pData.h_im[pdx])), len);
      ptr += len;
    }
    
    return posArray;
  }
  PyObject* GetStress() {
    std::vector<double> stress_vector = $self->GetStress();
    npy_intp dims[1] = {6};
    PyObject *stressArray = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    double* out_ptr = static_cast<double*>(PyArray_DATA((PyArrayObject*)stressArray));
    for(unsigned int idx = 0; idx < 6; idx++)
      out_ptr[idx] = stress_vector[idx];

    return stressArray;
  }

  // Might be better to add this function to the C++  code and then let SWIG
  // generate a wrapper using an appropriate typemap--this would allow better
  // code (as in, less likely to be made incompatible in the future)
  PyObject* SetPositions(PyObject* positions_obj) {
    // must have all declarations before any calls to SWIG_exception
    // (because the latter includes a "goto fail")
    unsigned int nParticles = $self->GetNumberOfParticles();
    const ParticleData& pData = *$self->GetParticleData();
    PyObject *resultobj = 0;
    char *ptr = 0;
    int len = 3*sizeof(float)/sizeof(char);
    npy_intp *dims;
    PyArrayObject *posArray = (PyArrayObject *) PyArray_ContiguousFromObject(positions_obj, NPY_FLOAT, 2, 2);
    if(!posArray)
      SWIG_exception(SWIG_TypeError, "Expected an array of float32");
    
    dims = PyArray_DIMS(posArray);    
    if(dims[0] != $self->GetNumberOfParticles() || dims[1] != 3) {
      SWIG_exception(SWIG_ValueError, "Wrong array shape, must be Nx3");
    }
    
    ptr = PyArray_BYTES((PyArrayObject*) posArray);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy( (void*)(&(pData.h_r[pdx])), (void*) ptr, len);
      ptr += len;
    }
    Py_DECREF(posArray);
    pData.CopyPosToDevice(false); // reset_sorting=false: apply existing sorting if relevant. May want this to be user-decided.

    // the following is copied from typical wrapper code generated by SWIG and
    // is necessary to allow use of SWIG_Exception
    resultobj = SWIG_Py_Void();
    return resultobj;
  fail:
    return NULL;
  }
  PyObject* SetVelocities(PyObject* velocities_obj) {
    // must have all declarations before any calls to SWIG_exception
    // (because the latter includes a "goto fail")
    unsigned int nParticles = $self->GetNumberOfParticles();
    const ParticleData& pData = *$self->GetParticleData();
    PyObject *resultobj = 0;
    char *ptr = 0;
    int len = 3*sizeof(float)/sizeof(char);
    npy_intp *dims;
    PyArrayObject *velArray = (PyArrayObject *) PyArray_ContiguousFromObject(velocities_obj, NPY_FLOAT, 2, 2);
    if(!velArray)
      SWIG_exception(SWIG_TypeError, "Expected an array of float32");
    
    dims = PyArray_DIMS(velArray);    
    if(dims[0] != $self->GetNumberOfParticles() || dims[1] != 3) {
      SWIG_exception(SWIG_ValueError, "Wrong array shape, must be Nx3");
    }
    
    ptr = PyArray_BYTES((PyArrayObject*) velArray);

    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy( (void*)(&(pData.h_v[pdx])), (void*) ptr, len);
      ptr += len;
    }
    Py_DECREF(velArray);
    pData.CopyVelToDevice(); 

    // the following is copied from typical wrapper code generated by SWIG and
    // is necessary to allow use of SWIG_Exception
    resultobj = SWIG_Py_Void();
    return resultobj;
  fail:
    return NULL;
  }
  PyObject* SetTypes(PyObject* types_obj) {
    // must have all declarations before any calls to SWIG_exception
    // (because the latter includes a "goto fail")
    unsigned int nParticles = $self->GetNumberOfParticles();
    ParticleData& pData = *$self->GetParticleData();
    pData.CopyPosFromDevice(); // otherwise will overwrite current positions on device with out-of-date host values
    PyObject *resultobj = 0;
    char *ptr = 0;
    int len = sizeof(float)/sizeof(char);
    npy_intp *dims;
    PyArrayObject *typeArray = (PyArrayObject *) PyArray_ContiguousFromObject(types_obj, NPY_INT32, 1, 1);
    if(!typeArray)
      SWIG_exception(SWIG_TypeError, "Expected an vector of int32");
    
    dims = PyArray_DIMS(typeArray);    
    if(dims[0] != $self->GetNumberOfParticles()) {
      SWIG_exception(SWIG_ValueError, "Wrong vector length, must be N (number of particles)");
    }
    
    ptr = PyArray_BYTES((PyArrayObject*) typeArray);
    for(unsigned int pdx = 0; pdx < nParticles; pdx++) {
      memcpy( (void*)(&(pData.h_r[pdx].w)), (void*) ptr, len);
      pData.h_Type[pdx] = ((unsigned int*)ptr)[0];
      ptr += len;
    }
    Py_DECREF(typeArray);
    pData.CopyPosToDevice(false); // reset_sorting=false: apply existing sorting if relevant. May want this to be user-decided.
    pData.CopyVelFromDevice(); // otherwise will overwrite current velocities on device with out-of-date host values when we update the masses
    pData.UpdateParticleMasses();
    
    // the following is copied from typical wrapper code generated by SWIG and
    // is necessary to allow use of SWIG_Exception
    resultobj = SWIG_Py_Void();
    return resultobj;
  fail:
    return NULL;
  }
  
};

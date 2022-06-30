
%init %{
  // Initialize the numpy module so we can use its C API (otherwise get seg 
  // faults)
  import_array();
  %}

%{
#include "rumd_stats.h"
#include "rumd_rdf.h"
#include "rumd_msd.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
  %}


%include exception.i
%exception {
  try {
    $action
      } catch (const RUMD_Error &e) {
    std::string errorStr("In class ");
    errorStr += e.className;
    errorStr += ", method ";
    errorStr += e.methodName;
    errorStr += ": ";
    errorStr += e.errorStr;
    SWIG_exception(SWIG_RuntimeError,errorStr.c_str());
  } catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown error");
  }
 }

%include stringTypeMap.i
%include vectorTypeMap.i


class Conf {
 public:
  Conf();
  unsigned int GetNumberOfParticles() const;
  void OpenGZ_File(const std::string& filename);
  void CloseCurrentGZ_File();
  void ReadCurrentFile(bool verbose=false);
  ConfigurationMetaData  metaData;
  %extend {
    PyObject* GetPositions() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 3;
      PyObject *posArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      char *ptr = PyArray_BYTES((PyArrayObject*)posArray);
      int len = sizeof(float)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].x)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].y)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].z)), len);
	  ptr += len;
	}
      return posArray;
    } // end GetPositions
    PyObject* GetTypes() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 1;
      PyObject *typeArray = PyArray_SimpleNew(2, dims, NPY_INT);
      char *ptr = PyArray_BYTES((PyArrayObject*)typeArray);
      int len = sizeof(int)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  int type = (int) $self->P[pdx].MyType;
	  memcpy((void*) ptr, (void*)(&type), len);
	  ptr += len;
	}
      return typeArray;
    }
    PyObject* GetImages() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 3;
      PyObject *imArray = PyArray_SimpleNew(2, dims, NPY_INT);
      char *ptr = PyArray_BYTES((PyArrayObject*) imArray);
      int len = 3*sizeof(int)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  int im[3];
	  im[0] = $self->P[pdx].Imx;
	  im[1] = $self->P[pdx].Imy;
	  im[2] = $self->P[pdx].Imz;
	  memcpy((void*) ptr, (void*)(&im), len);
	  ptr += len;
	}
      return imArray;
    }
    PyObject* GetVelocities() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 3;
      PyObject *velArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      char *ptr = PyArray_BYTES((PyArrayObject*) velArray);
      int len = sizeof(float)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].vx)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].vy)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].vz)), len);
	  ptr += len;
	}
      return velArray;
    }
    PyObject* GetForces() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 3;
      PyObject *forArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      char *ptr = PyArray_BYTES((PyArrayObject*) forArray);
      int len = sizeof(float)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].fx)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].fy)), len);
	  ptr += len;
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].fz)), len);
	  ptr += len;
	}
      return forArray;
    }
    PyObject* GetPotentialEnergies() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 1;
      PyObject *peArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      char *ptr = PyArray_BYTES((PyArrayObject*) peArray);
      int len = sizeof(float)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].Ui)), len);
	  ptr += len;
	}
      return peArray;
    }
    PyObject* GetVirials() {
      npy_intp dims[2];
      unsigned int nParticles = $self->GetNumberOfParticles();
      dims[0] = nParticles;
      dims[1] = 1;
      PyObject *virArray = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      char *ptr = PyArray_BYTES((PyArrayObject*) virArray);
      int len = sizeof(float)/sizeof(char);
      for(unsigned int pdx = 0; pdx < nParticles; pdx++)
	{
	  memcpy((void*) ptr, (void*)(&($self->P[pdx].Wi)), len);
	  ptr += len;
	}
      return virArray;
    }
  } // end %extend
 };


%feature("kwargs"); // enable keyword arguments

%feature("autodoc","1");

class rumd_stats {
 public:
  rumd_stats() {}
  void SetVerbose(bool b);
  void SetDirectory(const std::string& directory);
  void SetBaseFilename(const std::string& base_filename);
  void ComputeStats(unsigned int first_block=0, int last_block=-1);
  void PrintStats();
  void WriteStats();
  unsigned int GetCount() const;
  std::string GetColumnLabel(unsigned int colIdx);

  %extend {
    PyObject* GetMeanVals() {
      unsigned int nCols = $self->GetNumCols();

      PyObject *meanValsDict = PyDict_New();
      for(unsigned int col = 0; col< nCols;++col) {
	PyDict_SetItem(meanValsDict, PyString_FromString($self->GetColumnLabel(col).c_str()), PyFloat_FromDouble($self->GetMeanVals()[col]));
      }
      return meanValsDict;
    }
    PyObject* GetMeanSqVals() {
      unsigned int nCols = $self->GetNumCols();

      PyObject *meanSqValsDict = PyDict_New();
      for(unsigned int col = 0; col< nCols;++col) {
	PyDict_SetItem(meanSqValsDict, PyString_FromString($self->GetColumnLabel(col).c_str()), PyFloat_FromDouble($self->GetMeanSqVals()[col]));
      }
      return meanSqValsDict;
    }
    PyObject* GetCovarianceVals() {
      unsigned int nCols = $self->GetNumCols();
      PyObject *covarianceDict = PyDict_New();
      unsigned int cov_idx = 0;
      for(unsigned int idx = 0; idx< nCols;++idx) 
	for(unsigned int jdx = idx+1; jdx< nCols;++jdx) {
	  std::string key = $self->GetColumnLabel(idx) + "-" + $self->GetColumnLabel(jdx);

	  PyDict_SetItem(covarianceDict, PyString_FromString(key.c_str()), PyFloat_FromDouble($self->GetCovarianceVals()[cov_idx]));
	  cov_idx++;
	}
      return covarianceDict;
    }
    PyObject* GetDriftVals() {
      unsigned int nCols = $self->GetNumCols();
      
      PyObject *driftValsDict = PyDict_New();
      for(unsigned int col = 0; col< nCols;++col) {
	PyDict_SetItem(driftValsDict, PyString_FromString($self->GetColumnLabel(col).c_str()), PyFloat_FromDouble($self->GetDriftVals()[col]));
      }
      return driftValsDict;
    }
  } // end %extend
};


class rumd_rdf {
 public:
  rumd_rdf();
  void SetDirectory(const std::string& directory);
  void ComputeAll(int nBins, float min_dt, unsigned int first_block=0, int last_block=-1, unsigned particlesPerMol=1);
  void WriteRDF(const std::string& filename);
  void WriteSDF(const std::string& filename);
  int GetNumTypes() const;
  const double* GetRDF(int type1, int type2);
  const double* GetSDF(int type1, int coord);
  %extend {
    PyObject* GetRDFArray(int type1, int type2) {
      npy_intp nBins = $self->GetNumBins();

      PyObject *rdfArray = PyArray_SimpleNew(1, &nBins, NPY_DOUBLE);
      double* rdfArray_data = (double*) PyArray_DATA((PyArrayObject*)rdfArray);
      const double* rdf = $self->GetRDF(type1, type2);
      for(unsigned int i = 0; i < nBins; ++i)
	rdfArray_data[i] = rdf[i];
      return rdfArray;
    } // end GetRDFArray
    PyObject* GetRadiusValues() {
      npy_intp nBins = $self->GetNumBins();
      PyObject *rValsArray = PyArray_SimpleNew(1, &nBins, NPY_DOUBLE);
      double* rVals_data = (double*) PyArray_DATA((PyArrayObject*)rValsArray);
      const double* rVals = $self->GetRVals();
      for(unsigned int i = 0; i < nBins; ++i)
	rVals_data[i] = rVals[i];
      return rValsArray;
   } // end GetRadiusValues
  } // end % extend
};


%{
  typedef double (*double2)[2]; // needed in order to cast to this type
  %}



class rumd_msd {
 public:
  rumd_msd();
  void SetVerbose(bool v);
  void SetDirectory(const std::string& directory);
  void SetQValues(std::vector<double> qvalues);
  void SetExtraTimesWithinBlock(bool set_etwb);
  void SetSubtractCM_Drift(bool set_subtract_cm_drift);
  void SetAllowTypeChanges(bool set_allow_type_changes);
  void SetNumS4_kValues(unsigned nS4_kValues);
  unsigned int GetNumberOfS4_kValues() const;
  void ComputeAll(unsigned int first_block=0, int last_block=-1, unsigned int particlesPerMol = 1);
  void WriteMSD(const std::string& filename);
  void WriteMSD_CM(const std::string& filename);
  void WriteAlpha2(const std::string& filename);
  void WriteISF(const std::string& filename);
  void WriteISF_CM(const std::string& filename);
  void WriteISF_SQ(const std::string& filename);
  void WriteVAF(const std::string& filename);

  %extend {
    PyObject* GetMSD(int type) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *msdArray = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Copy_MSD_To_Array( (double2) PyArray_DATA((PyArrayObject*)msdArray), type);
      return msdArray;
    } // end GetMSD
    PyObject* GetISF(int type) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *isfArray = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Copy_ISF_To_Array( (double2) PyArray_DATA((PyArrayObject*)isfArray), type);
      return isfArray;
    }
    PyObject* GetVAF(int type) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *vafArray = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Copy_VAF_To_Array( (double2) PyArray_DATA((PyArrayObject*)vafArray), type);
      return vafArray;
    }

    
    PyObject* GetAlpha2(int type) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *alpha2_Array = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Copy_Alpha2_To_Array( (double2) PyArray_DATA((PyArrayObject*)alpha2_Array), type);
      return alpha2_Array;
    }
    PyObject* GetChi4(int type) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *chi4_Array = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Fill_Chi4_Array( (double2) PyArray_DATA((PyArrayObject*)chi4_Array), type);
      return chi4_Array;
    }
    PyObject* GetS4(int type, int k_index) {
      npy_intp shape[2];
      shape[0] = $self->GetNumberOfTimes();
      shape[1] = 2;
      PyObject *S4_Array = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
      $self->Fill_S4_Array( (double2) PyArray_DATA((PyArrayObject*)S4_Array), type, k_index);
      return S4_Array;
    }
  } // end extend
};

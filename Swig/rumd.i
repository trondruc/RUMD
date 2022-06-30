%module(docstring="The rumd module gives access to the C++ classes that make up the bulk of rumd software: Sample, SimulationBox, MoleculeData, potentials, integrators, output managers, etc") rumd


%init %{
  // Initialize the GPU device as part of module initialization
  try {
    Device::GetDevice().Init();
  }
  catch (const RUMD_Error &e){
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    throw;
  }
  catch (const thrust::system::system_error &e) {
    // For some reason could not catch the thrust exception in Device
    std::cerr << "Test using thrust failed (probably  have compiled for incorrect compute capability) : " << e.what() ;
    throw;
  }

  // Initialize the numpy module so we can use its C API (otherwise get segfaults)
  import_array();
  %}

enum SortingScheme {SORT_X, SORT_XY, SORT_XYZ};

%{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "rumd/Device.h" // curiously, leaving this out generates not a compile-error but a linker error....
#include "numpy/arrayobject.h"
#include "rumd/Timer.h"
%}



// the following allows exceptions to be passed to Python
%include exception.i
%{
#include "rumd/RUMD_Error.h"
  %}


%exception {
  try {
    $action
      }  catch (const RUMD_cudaResourceError &e) {
    SWIG_exception(SWIG_ValueError, e.GetCombinedErrorStringAlt().c_str());
  }
  catch (const RUMD_Error &e) {
    SWIG_exception(SWIG_RuntimeError,e.GetCombinedErrorString().c_str());
  } catch(...) {
    std::cerr << "Unrecognized exception" << std::endl;
    throw;
  }
 }




%feature("kwargs"); // enable keyword arguments

%feature("autodoc","1");

%include "stringTypeMap.i"
%include "vectorTypeMap.i"

%include "SimulationBox.i"

%include "Sample.i"

%include "Potentials.i"

%include "Integrators.i"

%include "IO.i"

%include "ExternalCalculators.i"

%include "MoleculeData.i"

%include "UserFunctions.i"


 // the following allows us to access the device name from Python using rumd.Device.GetDeviceName() or (for versions of python < 2.2) rumd.Device_GetDeviceName()
%nodefaultctor Device; // disable generation of wrapper for default constructor

class Device {
 public:
  static Device& GetDevice();
  const std::string GetDeviceName();
  const std::string GetDeviceReport();
  size_t GetDeviceMemory();
  void Synchronize();
  float Time();
  void CheckErrors();
 };


class Timer {
 public:
  Timer();
  double elapsed();
};

%inline %{
  std::string GetVersion() {
    return std::string(VERSION);
  }
  std::string Get_SVN_Revision() {    
    return std::string(SVN_REV);
  }
%}

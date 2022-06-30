#ifndef RUMD_ERROR_H
#define RUMD_ERROR_H

#include <string>

class RUMD_Error 
{
public:
  RUMD_Error(const std::string& className, 
	     const std::string& methodName,
	     const std::string& errorStr) : className(className),
					    methodName(methodName),
					    errorStr(errorStr)

  {}
  virtual ~RUMD_Error() {}
  std::string GetCombinedErrorString() const { return std::string("In class ") + className + ", method " + methodName + ": " + errorStr;}
  std::string GetCombinedErrorStringAlt() const { return errorStr + std::string(" (") + className + "::" + methodName + ")";}
  std::string className;
  std::string methodName;
  std::string errorStr;
};

class RUMD_cudaResourceError : public RUMD_Error 
{
public:
  RUMD_cudaResourceError(const std::string& className, 
			 const std::string& methodName,
			 const std::string& errorStr) : 
    RUMD_Error(className, methodName, errorStr) {}
};


#endif // RUMD_ERROR_H

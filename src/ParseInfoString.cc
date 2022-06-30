#include "rumd/ParseInfoString.h"
#include "rumd/RUMD_Error.h"
#include <cstring>
#include <cstdio>

std::string ParseInfoString(const std::string& infoStr, std::vector<float>&parameterList) {
  parameterList.resize(0);
  if(infoStr.size() == 0)
    return "";
    
  // copy and parse infoStr using strtok_r (necessary to copy because strtok_r
  // modifies the string, and c_str() gives a constant string)
  char* infoStrCopy = new char[infoStr.size()+1];
  strcpy(infoStrCopy, infoStr.c_str());
  char* parameterStr=0;
  char* className = strtok_r(infoStrCopy, ",", &parameterStr);
  // read parameters from rest of infoStr
  char* tokenMarker = 0;
  char* pStr = strtok_r(parameterStr, ",", &tokenMarker);

  while (pStr) {
    float param = 0.f;
    int nItems = sscanf(pStr, "%f",&param);
    if(nItems < 1)
      throw RUMD_Error("[None]","ParseInfoString",std::string("Failed to read parameter from infoStr ") + infoStr );
    parameterList.push_back(param);
    pStr = strtok_r(0, ",", &tokenMarker);
  }

  std::string classNameStr(className); 
  delete [] infoStrCopy;
  return classNameStr;
}
  

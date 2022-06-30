#include "rumd/EnergiesMetaData.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cassert>

using namespace std;

void EnergiesMetaData::ReadMetaData(char *commentLine, bool verbose, std::vector<std::string> &column_labels) {

  int __attribute((unused)) nItems;
  // for tokenizing: there is a two-level hierarchy of tokens,
  // (in some cases, eg list-arguments, three-level)
  // so we need to use strtok_r to allow nested tokenizing
  char *token;
  char *keyword;
  char *argument;
  char *listItem; // for list-arguments (comma-separated items)
  char *tokenMarker1; // highest level tokens
  char *tokenMarker2; // for separating keyword from argument
  char *tokenMarker3; // for list-arguments

  // first set all bool_options to false. If the appropriate string is present 
  // in columns, then the corrresponding option will be set to true
  std::map<std::string, bool>::const_iterator it;
  for(it = Start(); it != End(); it++ )
    Set(it->first, false);
	
  unsigned int N = 0;
  float Dt = 0.;

  // read the "#" sign
  token = strtok_r(commentLine, " \t", &tokenMarker1);
  if(!token || strcmp(token,"#") != 0)
    throw RUMD_Error("EnergiesMetaData","ReadMetaData","No tokens found, or first one is not #");

  // top-level
  token = strtok_r(0, " \t", &tokenMarker1);
  while(token) {
    // now token marks the next major token
    keyword = strtok_r(token, "=", &tokenMarker2);
    argument =  strtok_r(0, "=", &tokenMarker2);
    if(!argument) {
      std::cerr << "No argument for keyword " << keyword << std::endl;
      // get the next major token
      token = strtok_r(0, " \t", &tokenMarker1);
      continue;
    }
    
    if(strcmp(keyword, "ioformat") == 0) {
      nItems = sscanf(argument, "%d", &rumd_energies_ioformat);
      if (verbose) cout << "read ioformat " << rumd_energies_ioformat << endl;
    }
    else if(strcmp(keyword, "N") == 0) {
      nItems = sscanf(argument, "%u", &N);
      if (verbose) cout << "read number of particles N = " << N << endl;
    }
    else if(strcmp(keyword, "Dt") == 0) {
      nItems = sscanf(argument, "%f", &Dt);
      if (verbose) cout << "read Dt " << Dt << endl;
    }
    else if (strcmp(keyword, "timeStepIndex") == 0) {
      nItems = sscanf(argument, "%d", &(timeStepIndex));
      if (verbose) cout << "read timeStepIndex " << timeStepIndex << endl;
    }
    else if (strcmp(keyword, "logLin") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.block));
      if (verbose) cout << "read    logLin.block " << logLin.block << endl;
      listItem = strtok_r(0, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.base));
      if (verbose) cout << "read    logLin.base " << logLin.base << endl;
      listItem = strtok_r(0, ",", &tokenMarker3);
      nItems = sscanf(listItem, "%lu",&(logLin.index));
      if (verbose) cout << "read    logLin.index " << logLin.index;
      listItem = strtok_r(0, ",", &tokenMarker3);
      nItems = sscanf(listItem, "%lu",&(logLin.maxIndex));
      if (verbose) cout << "read    logLin.maxIndex " << logLin.maxIndex << endl;
      listItem = strtok_r(0, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.maxInterval));
      if (verbose) cout << "read    logLin.maxInterval " << logLin.maxInterval << endl;
      assert(0 == strtok_r(0, ",", &tokenMarker3));
      if (verbose)
	cout << "read conf_index: Block " << logLin.block << ", Index " << logLin.index << ", MaxIndex " << logLin.maxIndex << endl;
    }
    else if (strcmp(keyword, "columns") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3);
      while(listItem) {
	bool match = false;
	for(it = Start(); it != End(); it++ )
	  if(strcmp(listItem, GetFileString(it->first).c_str()) == 0){
	    if (verbose) cout << "columns: have " << listItem << endl;
	    match = true;
	    assert(it->second == false); // (should not already be set)
	    Set(it->first, true);
	    column_labels.push_back(listItem);
	  } // if (strcmp ... )	
	if(!match) {
	  cout << "Non-standard column identifier " << listItem <<  endl;
	  bool_options[listItem] = true;
	  fileStr[listItem] = listItem;
	  column_labels.push_back(listItem);
	}
	listItem = strtok_r(0, ",", &tokenMarker3);
      } // end while (listItem)
    } // end if "columns"
    else 
      throw RUMD_Error("EnergiesMetaData","ReadMetaData",std::string("Unknown keyword")+keyword);
 
    // get the next major token
    token = strtok_r(0, " \t", &tokenMarker1);
  } // end while(token)
  
}

void EnergiesMetaData::ShowOptions() const {
  std::map<std::string, bool>::const_iterator it;
    for(it = bool_options.begin(); it != bool_options.end(); it++ )
      cout << it->first << ": " << it->second << endl;
}

void EnergiesMetaData::ShowFileStrings() const {
  std::map<std::string, std::string>::const_iterator it;
    for(it = fileStr.begin(); it != fileStr.end(); it++ )
      cout << it->first << ": " << it->second << endl;
}


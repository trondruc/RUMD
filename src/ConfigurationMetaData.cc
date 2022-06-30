#include "rumd/ConfigurationMetaData.h"
#include "rumd/RUMD_Error.h"
#include "rumd/ParseInfoString.h"

#include <vector>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cassert>

void ConfigurationMetaData::ReadMetaData(char *commentLine, bool verbose) {

  int __attribute__((unused)) nItems;
  unsigned int index;
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
	
  // possible entries in the columns list: 
  char requiredCols[][5] = {"type","x","y","z"};
  unsigned int numReq = sizeof(requiredCols)/(5*sizeof(char));
  char imStr[][5] = {"imx","imy","imz"};
  char velStr[][3] = {"vx","vy","vz"};
  char forStr[][3] = {"fx","fy","fz"};

  integratorInfoStr.assign("");
  simulationBoxInfoStr.assign("");
  bool found_ioformat = false;
  bool found_sim_box = false;
  found_integrator = false; // member variable
  
  // top-level
  token = strtok_r(commentLine, " \t", &tokenMarker1);
  while(token) {
    // now token marks the next major token
    keyword = strtok_r(token, "=", &tokenMarker2);
    argument =  strtok_r(0, "=", &tokenMarker2);

    // ioformat must appear and be read before 
    // anything that depends on it, obviously
    if(strcmp(keyword, "ioformat") == 0) {
      nItems = sscanf(argument, "%d", &rumd_conf_ioformat);
      found_ioformat = true;
      if (verbose) printf("read ioformat %d\n",rumd_conf_ioformat);
    }
    else if (strcmp(keyword, "timeStepIndex") == 0) {
      nItems = sscanf(argument, "%li", &(timeStepIndex));
      if (verbose) printf("read timeStepIndex %li\n",timeStepIndex);
    }
    else if (strcmp(keyword, "logLin") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.block));
      if (verbose) printf("read    logLin.block %lu\n", logLin.block);
      listItem = strtok_r(0, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.base));
      if (verbose) printf("read    logLin.base %lu\n", logLin.base);
      listItem = strtok_r(0, ",", &tokenMarker3);
      nItems = sscanf(listItem, "%lu",&(logLin.index));
      if (verbose) printf("read    logLin.index %lu\n", logLin.index);
      listItem = strtok_r(0, ",", &tokenMarker3);
      nItems = sscanf(listItem, "%lu",&(logLin.maxIndex));
      if (verbose) printf("read    logLin.maxIndex %lu\n", logLin.maxIndex);
      listItem = strtok_r(0, ",", &tokenMarker3); 
      nItems = sscanf(listItem, "%lu",&(logLin.maxInterval));
      if (verbose) printf("read    logLin.maxInterval %lu\n", logLin.maxInterval);
      assert(0 == strtok_r(0, ",", &tokenMarker3));
      if (verbose)
	printf("read conf_index: Block %lu, Index %lu, MaxIndex %lu\n",
	       logLin.block, logLin.index, logLin.maxIndex);
      bool_options["logLin"] = true;
    }
    else if (strcmp(keyword, "numTypes") == 0) {
      unsigned int num_types = 0;
      nItems = sscanf(argument, "%d", &num_types);
      massOfType.resize(num_types);
      if (verbose) std::cout << "Read numTypes " <<  num_types << std::endl;
    }
    else if (strcmp(keyword, "mass") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3);
      index = 0;
      while(listItem) {
	if(index >= massOfType.size())
	  throw RUMD_Error("ConfigurationMetaData", __func__, "More masses than types");
	
	nItems = sscanf(listItem, "%f",&(massOfType[index]));
	index++;
	listItem = strtok_r(0, ",", &tokenMarker3);
      }
      if(index < massOfType.size())
	throw RUMD_Error("ConfigurationMetaData", __func__, "Not enough masses supplied");
      if (verbose) {
	std::cout << "masses: ";
	for(index = 0; index < massOfType.size(); ++index)
	  std::cout << massOfType[index];
	std::cout << std::endl;
      }
    }
    else if (strcmp(keyword, "symbol") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3);
      index = 0;
      while(listItem) {
	assert(index < massOfType.size());
	// not doing anything with symbol yet
	if (verbose) printf("symbol %d is %s\n",index,listItem);
	index++;
	listItem = strtok_r(0, ",", &tokenMarker3);
      }
    }
    else if (strcmp(keyword, "columns") == 0) {
      listItem = strtok_r(argument, ",", &tokenMarker3);
      index = 0;
		
      
      while(listItem && index < numReq) {
	assert(strcmp(listItem, requiredCols[index]) == 0);
	 if (verbose) printf("columns, have required column %d (%s)\n",index,listItem);
	index++;
	listItem = strtok_r(0, ",", &tokenMarker3);
      }
      while(listItem) {
	// check if index < numReq (not all required columns present)?
	if(strcmp(listItem, "imx") == 0){
	  for(index = 0; index < DIM; ++index) {
	    assert(strcmp(listItem, imStr[index]) == 0);
	    if (verbose) printf("columns: have images: %s\n", listItem);
	    listItem = strtok_r(0, ",", &tokenMarker3);
	  }
	  bool_options["images"] = true;
	}
	else if(strcmp(listItem, "vx") == 0) {
	  for(index = 0; index < DIM; index++) {
	    assert(strcmp(listItem, velStr[index]) == 0);
	    if (verbose) printf("columns: have velocities %s\n", listItem);
	    listItem = strtok_r(0, ",", &tokenMarker3);
	  }
	  bool_options["velocities"] = true;
	}
	else if(strcmp(listItem, "fx") == 0) {
	  for(index = 0; (index < DIM) ; index++) {
	    assert(strcmp(listItem, forStr[index]) == 0);
	    if (verbose) printf("columns: have forces %s\n", listItem);
	    listItem = strtok_r(0, ",", &tokenMarker3);
	  }
	  bool_options["forces"] = true;
	}
	else if(strcmp(listItem, "pe") == 0) {
	  if (verbose) printf("columns: have pe\n");
	  bool_options["pot_energies"] = true;
	  listItem = strtok_r(0, ",", &tokenMarker3);
	}
	else if(strcmp(listItem, "vir") == 0) {
	  bool_options["virials"] = true;
	  if (verbose) printf("columns: have vir\n");
	  listItem = strtok_r(0, ",", &tokenMarker3);
	}
      } // end while (listItem) [columns]
    }
    
    // The rest depends on ioformat

    // first items for ioformat == 1
    else if (rumd_conf_ioformat == 1 && strcmp(keyword, "Nose-Hoover-Ps") == 0) {
      nItems = sscanf(argument, "%f", &(Ps));
      found_integrator = true;
      if (verbose) printf("Read Nose-Hoover-Ps %f\n", Ps);
    }
    else if (rumd_conf_ioformat == 1 && strcmp(keyword, "Barostat-Pv") == 0) {
      nItems = sscanf(argument, "%f", &(Pv));
      if (verbose) printf("read Barostat-Pv %f\n", Pv);
    }
    else if (rumd_conf_ioformat == 1 && strcmp(keyword, "dt") == 0) {
      nItems = sscanf(argument, "%f", &dt);
      if (verbose) printf("read time step (dt) %f\n", dt);
    }
    else if (rumd_conf_ioformat == 1 && strcmp(keyword, "boxLengths") == 0) {
      simulationBoxInfoStr.assign(std::string("RectangularSimulationBox,")+argument);
      found_sim_box = true;
    }

    // then items for ioformat == 2
    else if (rumd_conf_ioformat == 2 && strcmp(keyword, "integrator") == 0 ) {
      integratorInfoStr.assign(argument);
      found_integrator = true;
    }
    else if (rumd_conf_ioformat == 2 && strcmp(keyword, "sim_box") == 0 ) {
      simulationBoxInfoStr.assign(argument);
      found_sim_box = true;
    }
    else 
      std::cerr << "Warning: ConfigurationMetaData::ReadMetaData, unrecognized keyword: " << keyword << std::endl;
    
    // get the next major token
    token = strtok_r(0, " \t", &tokenMarker1);
  }

  if(!found_ioformat)
    throw RUMD_Error("ConfigurationMetaData","ReadMetaData","Did not find ioformat");
  if(!found_sim_box)
    throw RUMD_Error("ConfigurationMetaData","ReadMetaData","Did not find box information");

  if(rumd_conf_ioformat == 2 && found_integrator) {
    std::vector<float> integratorParameters;
    std::string integratorClass = ParseInfoString(integratorInfoStr, integratorParameters);
    dt = integratorParameters[0];
  }
}

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd_stats.h"

#include "rumd/EnergiesMetaData.h"
#include "rumd/RUMD_Error.h"
#include <cstring>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <zlib.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdio>

rumd_stats::rumd_stats() : directory("TrajectoryFiles"), 
			   base_filename("energies"),
			   column_labels(),
			   verbose(false),
			   mean_vals(0),
			   meanSq_vals(0),
			   covariance_vals(0),
			   drift_vals(0),
			   init_vals(0),
			   values(0),
			   data_buffer(0),
			   num_cols(0),
			   count(0),
			   buffer_length(0),
			   EMD(),
			   chars_per_data_item(20),
			   numFilenameDigits(4) {}

rumd_stats::~rumd_stats() {
  Free();
  if(data_buffer) delete [] data_buffer;
}

void rumd_stats::Allocate(unsigned int set_num_cols, unsigned int comment_line_length) {
  if (num_cols != set_num_cols) {
    Free();
    num_cols = set_num_cols;
    mean_vals = new double[num_cols];
    meanSq_vals = new double[num_cols];
    covariance_vals = new double[num_cols*(num_cols-1)/2];
    drift_vals = new double[num_cols];
    init_vals = new double[num_cols];
    values = new double[num_cols];
  }

  memset(mean_vals, 0, num_cols*sizeof(double));
  memset(meanSq_vals, 0, num_cols*sizeof(double));
  memset(covariance_vals, 0, (num_cols*(num_cols-1)/2)*sizeof(double));
  memset(drift_vals, 0, num_cols*sizeof(double));
  memset(init_vals, 0, num_cols*sizeof(double));
  memset(values, 0, num_cols*sizeof(double));

  // allocate buffer for reading data lines
  unsigned int new_buffer_length = num_cols*chars_per_data_item + 2;
  // needs to hold comment lines also (+2 for the newline + null-termination;
  // note we already stripped the newline so it is not included in the length)
  if (new_buffer_length < comment_line_length + 2)
    new_buffer_length = comment_line_length + 2;

  if(new_buffer_length != buffer_length) {
    buffer_length = new_buffer_length;
    if(data_buffer)
      delete [] data_buffer;
    data_buffer = new char[buffer_length];
  }
  
}

void rumd_stats::Free()
{
  if(mean_vals) delete [] mean_vals;
  if(meanSq_vals) delete [] meanSq_vals;
  if(covariance_vals) delete [] covariance_vals;
  if(drift_vals) delete [] drift_vals;
  if(init_vals) delete [] init_vals;
  if(values) delete [] values;
}


std::string rumd_stats::GetCommentLine(unsigned block_index)
{
  gzFile gzfp;
  std::ostringstream E0_Name;
  E0_Name << directory << "/" << base_filename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << block_index << ".dat.gz";
  
  if(verbose) std::cout << "Reading comment-line from block " << block_index << ": " << E0_Name.str() << std::endl << std::endl;
  gzfp=gzopen(E0_Name.str().c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_stats", __func__, std::string("Error opening file ")+E0_Name.str() + ": " + strerror(errno));
  

  unsigned buff_length = 100;
  char* strWorkspace = new char[buff_length];

  // read the first line
  bool ok = gzgets(gzfp, strWorkspace, buff_length);
  if(!ok)
    throw RUMD_Error("rumd_stats", __func__, std::string("Error reading first line of file")+E0_Name.str());
  
  while(strlen(strWorkspace) == buff_length - 1) {
    buff_length *= 2;
    std::cout << "Reallocating for comment-line: new buff_length: " << buff_length  << std::endl;
    delete [] strWorkspace;
    strWorkspace = new char[buff_length];
    gzrewind(gzfp); //  (to the start of the file)
    bool ok = gzgets(gzfp, strWorkspace, buff_length);
    if(!ok)
      throw RUMD_Error("rumd_stats", __func__, std::string("Error reading first line of file")+E0_Name.str());
  } // end while
  gzclose(gzfp);

  // replace final newline with a zero and make a C++ string
  strWorkspace[strlen(strWorkspace)-1] = 0;
  std::string commentLine(strWorkspace);
  delete [] strWorkspace;
  return commentLine;
}


void rumd_stats::GetColumnIdentifiers(std::string commentLine)
{  
  char* comment_line_cstr = new char[commentLine.length()+1];
  std::strcpy (comment_line_cstr, commentLine.c_str());
  column_labels.clear();
  EMD.ReadMetaData(comment_line_cstr, verbose, column_labels);

  if(column_labels.size() == 0) {
    std::cout << "Could not find columns keyword. Will assume comment-line simply lists the column IDs" << std::endl;
    ReadCommentLineBasic(commentLine);
    }

   if( column_labels.size() == 0)
    throw RUMD_Error("rumd_stats", __func__, std::string("No data in ") + base_filename + " file");
  
  if(verbose)
    std::cout << "Number of columns is " << column_labels.size() << std::endl;

  delete [] comment_line_cstr;
}

void rumd_stats::ReadCommentLineBasic(std::string commentLine)
{
  // Make a copy as a C-style string
  char* comment_line_cstr = new char[commentLine.length()+1];
  std::strcpy (comment_line_cstr, commentLine.c_str());
  char *token;

  // read the "#" sign
  token = strtok(comment_line_cstr, " \t");
  if(!token || strcmp(token,"#") != 0)
    throw RUMD_Error("rumd_stats",__func__,"No tokens found, or first one is not #");
  column_labels.clear();
  token = strtok(0, " \t");
  while(token) {
    column_labels.push_back(std::string(token));
    token = strtok(0, " \t");
  }
  delete [] comment_line_cstr;
}



void rumd_stats::ComputeStats (unsigned int first_block, int last_block) {
  unsigned int last_saved_block;
  unsigned int last_block_to_read;
  
  gzFile gzfp;
  FILE *fp;
  int err_code = 0;
  
  std::string lastComp_filename = directory+"/LastComplete_" + base_filename +".txt";
  fp=fopen(lastComp_filename.c_str(),"r");
  if (!fp)
    throw RUMD_Error("rumd_stats","ComputeStats",std::string("Error opening file ")+lastComp_filename + ": " + strerror(errno));

  int err=fscanf(fp, "%u", &last_saved_block);
  if (err==EOF)
    throw RUMD_Error("rumd_stats","ComputeStats",std::string("Error reading file ")+lastComp_filename + ": " + strerror(errno));
  fclose(fp);
  if(verbose) std::cout << "last completed energy block is " << last_saved_block << std::endl;

  if(first_block > last_saved_block) {
    std::cerr << "Warning: user-specified first_block out of range; setting to zero" << std::endl;
    first_block = 0;
  }
  if(last_block < 0)
    last_block_to_read = last_saved_block;
  else
    if((unsigned int)last_block > last_saved_block) {
      last_block_to_read = last_saved_block;
      std::cerr << "Warning, specified last_block out of range; using last saved block" << std::endl;
    }
    else
      last_block_to_read = (unsigned int) last_block;

  if(first_block > last_block_to_read)
    std::cerr << "Warning: last_block smaller than first_block: no statistics computed" << std::endl;

  // read comment line for first block from file
  std::string commentLine = GetCommentLine(first_block);
  
  GetColumnIdentifiers(commentLine);
  Allocate( column_labels.size(), commentLine.length() );

  // The main loop. Read all blocks
  count = 0;
  for (unsigned int i=first_block; i<=last_block_to_read; i++) {
    std::ostringstream E_Name;
    E_Name << directory << "/" << base_filename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << i << ".dat.gz";

    if(verbose) std::cout << "Reading block " << i << ": " << E_Name.str() << std::endl;
    gzfp=gzopen(E_Name.str().c_str(),"r");
    if (!gzfp)
      throw RUMD_Error("rumd_stats","ComputeStats",std::string("Error opening file ") + E_Name.str() + ": " + strerror(errno));

    // read first (comment) line, but don't do anything with it (could perhaps
    // check for consistency ... )
    bool ok = gzgets(gzfp, data_buffer, buffer_length);
    if(!ok) {
      const char* error_str = gzerror(gzfp, &err_code);
      throw RUMD_Error("rumd_stats", __func__, std::string("Error reading first line of file ")+E_Name.str() + ": " + error_str);
    }      
    // read data lines
    ok = gzgets(gzfp, data_buffer, buffer_length);
    while(ok) {
      if(strlen(data_buffer) == buffer_length - 1)
	throw RUMD_Error("rumd_stats",__func__,std::string("Buffer length exceeded when reading data line from") + E_Name.str() + "; increase by calling SetCharsPerDataItem()");
      int offset = 0;
      int partial_offset = 0;
      double value = 0.;
      unsigned int idx;
      for(idx = 0; idx < num_cols; ++idx) {
	sscanf(data_buffer+offset, "%lf%n", &value, &partial_offset);
	offset += partial_offset;
	if(count==0) init_vals[idx] = value;
	value -= init_vals[idx];
	mean_vals[idx] += value;
	meanSq_vals[idx] += value * value;
	drift_vals[idx] += value * count;
	values[idx] = value;
      }
      unsigned int jdx, cov_idx = 0;
      for(idx = 0; idx < num_cols; ++idx)
	for(jdx = idx+1; jdx < num_cols; ++jdx)
	  covariance_vals[cov_idx++] += values[idx] * values[jdx];
      count += 1;
      ok = gzgets(gzfp, data_buffer, buffer_length);
    } // end while loop over lines in this file

    gzclose(gzfp);
  } // end loop over blocks
  
  // divide by count, compute variance, drift, etc
  if(count) {
    for(unsigned int idx = 0; idx < num_cols; ++idx) {
      mean_vals[idx] /= count;
      mean_vals[idx] += init_vals[idx];      
      meanSq_vals[idx] /= count;
      meanSq_vals[idx] += init_vals[idx] * (2.*mean_vals[idx]-init_vals[idx]);
      drift_vals[idx] /= count;
    }
    for(unsigned int idx = 0; idx < num_cols; ++idx) {
      drift_vals[idx] -= (mean_vals[idx]-init_vals[idx]) * (count/2.-0.5);
      drift_vals[idx] /= ( ((double) count * (double) count-1.)/12. );
    }
 
    unsigned int cov_idx = 0;
    for(unsigned int idx = 0; idx < num_cols; ++idx)
      for(unsigned int jdx = idx + 1; jdx < num_cols; ++jdx) {
	covariance_vals[cov_idx] /= count;
	covariance_vals[cov_idx] += (init_vals[idx]*mean_vals[jdx] + 
				     init_vals[jdx]*mean_vals[idx] -
				     init_vals[idx] * init_vals[jdx]);
	covariance_vals[cov_idx] -= (mean_vals[idx]*mean_vals[jdx]);
	cov_idx++;
      }
  }
}


void rumd_stats::PrintStats() {
  if(!mean_vals || !meanSq_vals)
    throw RUMD_Error("rumd_stats","PrintStats","Must call ComputeStats first");
  std::cout << "Total line count: " << count << std::endl;
  std::cout << "quantity, mean value, variance, standard deviation" << std::endl;

  for(unsigned int idx = 0; idx < num_cols; ++idx) {
    double variance = (meanSq_vals[idx] - mean_vals[idx]*mean_vals[idx]);
    double std_dev = sqrt(variance);
    printf("%s\t%12.6g\t%12.6g\t%12.6g\n", column_labels[idx].c_str(), mean_vals[idx], variance, std_dev );
  }
}



void rumd_stats::WriteStats() {
  if(!mean_vals || !meanSq_vals)
    throw RUMD_Error("rumd_stats","PrintStats","Must call ComputeStats first");

  std::map<std::string, bool>::const_iterator it;
  std::map<std::string, bool>::const_iterator it2;
  
  std::string filename = base_filename + "_mean.dat";
  FILE* meanFile = fopen(filename.c_str(),"w");
  if(!meanFile)
    throw RUMD_Error("rumd_stats", __func__, std::string("Could not open ") + filename + " for writing");
  filename = base_filename + "_mean_sq.dat";
  FILE* meanSqFile = fopen(filename.c_str(),"w");
  filename = base_filename + "_var.dat";
  FILE* varFile = fopen(filename.c_str(),"w");
  filename = base_filename + "_covar.dat";
  FILE* covarFile = fopen(filename.c_str(),"w");
  filename = base_filename + "_drift.dat"; 
  FILE* driftFile = fopen(filename.c_str(),"w");

  fprintf(meanFile,"#");
  fprintf(meanSqFile,"#");
  fprintf(varFile,"#");
  fprintf(covarFile,"#");
  fprintf(driftFile,"#");

  for(unsigned int idx = 0; idx < num_cols; ++idx) {
    fprintf(meanFile," %s", column_labels[idx].c_str() );
    fprintf(meanSqFile," %s",column_labels[idx].c_str() );
    fprintf(varFile," %s",column_labels[idx].c_str() );
    fprintf(driftFile," %s",column_labels[idx].c_str() );
    
    for (unsigned int jdx = idx + 1; jdx < num_cols; ++jdx)
      fprintf(covarFile, " %s-%s", column_labels[idx].c_str(), column_labels[jdx].c_str());
  }
  fprintf(meanFile,"\n");
  fprintf(meanSqFile,"\n");
  fprintf(varFile,"\n");
  fprintf(driftFile,"\n");
  fprintf(covarFile,"\n");


  unsigned int covar_idx = 0;
  for(unsigned int idx = 0; idx < num_cols; ++idx) {
    double variance = (meanSq_vals[idx] - mean_vals[idx]*mean_vals[idx]);
    fprintf(meanFile," %12.10g",mean_vals[idx]);
    fprintf(meanSqFile," %12.10g",meanSq_vals[idx]);
    fprintf(varFile," %12.10g",variance);
    fprintf(driftFile," %12.10g",drift_vals[idx]);

    for (unsigned int jdx = idx + 1; jdx < num_cols; ++jdx)
      fprintf(covarFile," %12.10g",covariance_vals[covar_idx++]);
  }
  fprintf(meanFile,"\n");
  fprintf(meanSqFile,"\n");
  fprintf(varFile,"\n");
  fprintf(driftFile,"\n");
  fprintf(covarFile,"\n");

  fclose(meanFile);
  fclose(meanSqFile);
  fclose(varFile);
  fclose(driftFile);
  fclose(covarFile);
}

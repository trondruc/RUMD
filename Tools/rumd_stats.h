/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd/EnergiesMetaData.h"
#include "rumd/RUMD_Error.h"
#include <cstring>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <zlib.h>
#include <vector>

class rumd_stats {
public:  
  rumd_stats();
  ~rumd_stats();
  void SetVerbose(bool v) { verbose = v; }
  void SetDirectory(const std::string& directory) {
    this->directory = directory;}
  void SetBaseFilename(const std::string& base_filename) {
    this->base_filename = base_filename; }
  void SetCharsPerDataItem(unsigned int set_chars_per_data_item) {
    chars_per_data_item = set_chars_per_data_item; }
  
  void ComputeStats(unsigned int first_block=0, int last_block=-1);
  void PrintStats();
  void WriteStats();

  unsigned int GetCharsPerDataItem() const { return chars_per_data_item; }
  unsigned int GetNumCols() const { return num_cols; }
  unsigned int GetCount() const { return count; }



  const double* GetMeanVals() const {
    if(!mean_vals) throw RUMD_Error("rumd_stats","GetMeanVals","mean_vals has not been computed yet");
    return mean_vals; 
  }

  const double* GetMeanSqVals() const { 
    if(!meanSq_vals) throw RUMD_Error("rumd_stats","GetMeanSqVals","meanSq_vals has not been computed yet");
    return meanSq_vals;
  }
  
  const double* GetCovarianceVals() const {
    if(!covariance_vals) throw RUMD_Error("rumd_stats","GetCovarianceVals","covariance_vals has not been computed yet");
    return covariance_vals;
  }

  const double* GetDriftVals() const {
    if(!drift_vals) throw RUMD_Error("rumd_stats","GetDriftVals","drift_vals has not been computed yet");
    return drift_vals;
  }

  std::string GetColumnLabel(unsigned int colIdx) {
    if(colIdx >= num_cols)
      throw RUMD_Error("rumd_stats","GetColumnLabel","Index is too large, or yo u haven't called ComputeStats yet.");
    return column_labels[colIdx];
}


private:
  void Allocate(unsigned int set_num_cols, unsigned int comment_line_length);
  void Free();
  std::string GetCommentLine(unsigned block_index);
  void GetColumnIdentifiers(std::string commentLine);
  void ReadCommentLineBasic(std::string commentLine);


  rumd_stats(const rumd_stats&);
  rumd_stats& operator=(const rumd_stats&);
  std::string directory;
  std::string base_filename;
  std::vector<std::string> column_labels;
  bool verbose;
  double* mean_vals;
  double* meanSq_vals;
  double* covariance_vals;
  double* drift_vals;
  double* init_vals;
  double* values;
  char* data_buffer;
  unsigned int num_cols;
  unsigned long int count;
  unsigned int buffer_length;
  EnergiesMetaData EMD;
  unsigned int chars_per_data_item;
  unsigned int numFilenameDigits;
};

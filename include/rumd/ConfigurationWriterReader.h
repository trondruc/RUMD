#ifndef CONFIGURATIONWRITERREADER_H
#define CONFIGURATIONWRITERREADER_H

#include "rumd/ConfigurationMetaData.h"
#include <zlib.h>

class Sample;

class ConfigurationWriterReader {
public:
  ConfigurationMetaData metaData;
  ConfigurationWriterReader() : metaData(), rumd_conf_ioformat(rumd_ioformat), verbose(false), bufferLength(400), max_buffer_length(100000), lineBuffer(0) {lineBuffer = new char[bufferLength];}
  ~ConfigurationWriterReader() {delete [] lineBuffer;}
  void Write(Sample* S, const std::string& filename, const std::string& mode);
  void Read(Sample* S, const std::string& filename);
  void SetVerbose(bool vb) { verbose = vb; }
  
  // the following is temporary
  void SetIO_Format(unsigned int ioformat);
  unsigned int GetIO_Format() { return rumd_conf_ioformat; }
private:
  ConfigurationWriterReader(const ConfigurationWriterReader&);
  ConfigurationWriterReader& operator=(const ConfigurationWriterReader&);

  void InitializeMetaData(Sample* S);

  void Write_ioformat1(Sample* S, const std::string& filename, const std::string& mode);
  void Write_ioformat2(Sample* S, const std::string& filename, const std::string& mode);
  void ReadParticleData(Sample *S, gzFile gp);
  void WriteParticleData(Sample *S, gzFile gp, int precision);
  void ReadLine(gzFile gp);

  //const unsigned int rumd_conf_ioformat;
  unsigned int rumd_conf_ioformat;
  bool verbose;
  unsigned int bufferLength;
  const unsigned int max_buffer_length;
  char* lineBuffer;
};

#endif // CONFIGURATIONWRITERREADER_H

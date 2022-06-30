#include "rumd/RUMD_Error.h"
#include "rumd/ConfigurationWriterReader.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/PairPotential.h"
#include "rumd/IntegratorNVT.h"

#include <vector>
#include <cstdio>
#include <cerrno>
#include <cassert>

void ConfigurationWriterReader::SetIO_Format(unsigned int ioformat){
  if(ioformat < 1 || ioformat > 2) throw RUMD_Error("ConfigurationWriterReader",
						    "SetTrajectoryIO_Format",
						    "Allowed values are 1 and 2");  
  rumd_conf_ioformat = ioformat;			       
}

void ConfigurationWriterReader::Write(Sample* S, const std::string& filename, const std::string& mode) {
  if(rumd_conf_ioformat == 1)
    Write_ioformat1(S, filename, mode);
  else if(rumd_conf_ioformat == 2)
    Write_ioformat2(S, filename, mode);
  else
    throw RUMD_Error("ConfigurationWriterReader","Write","Illegal rumd_conf_ioformat specified");
}

void ConfigurationWriterReader::Write_ioformat1(Sample* S, const std::string& filename, const std::string& mode) {
  // meta-data	such as the box-length, which has meaning independent of
  // the file-format, is taken from sample class itself. We do not check
  // the passed metaData variable for consistency (ConfigurationMetaData has 
  // fields for such data only for reading in configurations)
  
  float dt = 0.;
  float thermostatState = 0;
  if(S->itg) dt = S->itg->GetTimeStep();
  IntegratorNVT* testItg = dynamic_cast<IntegratorNVT*>(S->itg);
  if(testItg){
    thermostatState = testItg->GetThermostatState();
  }
  
  int precision = metaData.Get("precision");

  gzFile gp;
  std::string compressedFilename(filename);
  int len = compressedFilename.size();
  // add .gz if not already present
  if(len < 3 || compressedFilename.substr(len-3,3).compare(".gz"))
    compressedFilename.append(".gz");

  // the mode is append if this is a trajectory file and the index is nonzero  
  gp = gzopen(compressedFilename.c_str(),mode.c_str());

  S->CopySimulationDataFromDevice();

  // first line, number of particles
  gzprintf(gp, "%u\n", S->GetNumberOfParticles()); 
  
  // second line, "comment-line", containing meta-data
  float4 L = S->GetSimulationBox()->GetSimulationBox();
  
  gzprintf(gp, "ioformat=%d dt=%.*f", rumd_conf_ioformat, precision, dt);
  if(metaData.Get("logLin"))
    gzprintf(gp," timeStepIndex=%u logLin=%u,%u,%u,%u,%u", metaData.logLin.nextTimeStep, metaData.logLin.block, metaData.logLin.base,metaData.logLin.index, metaData.logLin.maxIndex, metaData.logLin.maxInterval);

  unsigned int numTypes = S->GetParticleData()->GetNumberOfTypes();
  gzprintf(gp, " boxLengths=%.*f,%.*f,%.*f numTypes=%d Nose-Hoover-Ps=%.*f Barostat-Pv=%.*f", precision, L.x, precision, L.y, precision, 
	  L.z, numTypes, precision, thermostatState, precision, 0.f); 

  gzprintf(gp, " mass=%.*f",precision, S->GetParticleData()->GetMass(0));
  for(unsigned int tdx = 1; tdx < numTypes; ++tdx)
    gzprintf(gp, ",%.*f", precision, S->GetParticleData()->GetMass(tdx));
  

  gzprintf(gp, " columns=type,x,y,z");
  if(metaData.Get("images")) gzprintf(gp, ",imx,imy,imz");
  if(metaData.Get("velocities")) gzprintf(gp, ",vx,vy,vz");
  if(metaData.Get("forces")) gzprintf(gp, ",fx,fy,fz");
  if(metaData.Get("pot_energies")) gzprintf(gp, ",pe");
  if(metaData.Get("virials")) gzprintf(gp, ",vir");
  
  gzprintf(gp, "\n");
  
  WriteParticleData(S, gp, precision);
 
  gzclose(gp); 
}

void ConfigurationWriterReader::Write_ioformat2(Sample* S, const std::string& filename, const std::string& mode) {
  // meta-data	such as the box-length, which has meaning independent of
  // the file-format, is taken from sample class itself. We do not check
  // the passed metaData variable for consistency (ConfigurationMetaData has 
  // fields for such data only for reading in configurations)
  
  int precision = metaData.Get("precision");
  gzFile gp;
  std::string compressedFilename(filename);
  int len = compressedFilename.size();
  // add .gz if not already present
  if(len < 3 || compressedFilename.substr(len-3,3).compare(".gz"))
    compressedFilename.append(".gz");

  // the mode is append if this is a trajectory file and the index is nonzero  
  gp = gzopen(compressedFilename.c_str(),mode.c_str());

  S->CopySimulationDataFromDevice();

  // first line, number of particles
  gzprintf(gp, "%u\n", S->GetNumberOfParticles()); 
  
  // second line, "comment-line", containing meta-data
  gzprintf(gp, "ioformat=%d", rumd_conf_ioformat);
  if(metaData.Get("logLin"))
    gzprintf(gp," timeStepIndex=%u logLin=%u,%u,%u,%u,%u", metaData.logLin.nextTimeStep, metaData.logLin.block, metaData.logLin.base,metaData.logLin.index, metaData.logLin.maxIndex, metaData.logLin.maxInterval);

  // Below is the new stuff for ioformat=2

  unsigned int numTypes = S->GetParticleData()->GetNumberOfTypes();
  gzprintf(gp, " numTypes=%d", numTypes);
    
  // get details from integrator and simulation box. Note that precision here
  // is interpreted (via setprecision) as the number of significant figures; we
  // add 2 to avoid having only 4 sig figs in the box lengths
  if( S->GetIntegrator())
    gzprintf(gp, " integrator=%s", S->GetIntegrator()->GetInfoString(precision+2).c_str());

  gzprintf(gp, " sim_box=%s", S->GetSimulationBox()->GetInfoString(precision+2).c_str());

  gzprintf(gp, " mass=%.*f",precision, S->GetParticleData()->GetMass(0));
  for(unsigned int tdx = 1; tdx < numTypes; ++tdx)
    gzprintf(gp, ",%.*f", precision, S->GetParticleData()->GetMass(tdx));

  gzprintf(gp, " columns=type,x,y,z");
  if(metaData.Get("images")) gzprintf(gp, ",imx,imy,imz");
  if(metaData.Get("velocities")) gzprintf(gp, ",vx,vy,vz");
  if(metaData.Get("forces")) gzprintf(gp, ",fx,fy,fz");
  if(metaData.Get("pot_energies")) gzprintf(gp, ",pe");
  if(metaData.Get("virials")) gzprintf(gp, ",vir");
  gzprintf(gp, "\n");
  
  WriteParticleData(S, gp, precision);

  gzclose(gp); 
}


void ConfigurationWriterReader::WriteParticleData(Sample *S, gzFile gp, int precision) {

  ParticleData& particleData = S->particleData;
  float4* h_r = particleData.h_r;
  float4* h_v = particleData.h_v;
  float4* h_im = particleData.h_im;
  float4* h_f = particleData.h_f;
  float4* h_w = particleData.h_w; 
  unsigned int* h_Type = particleData.h_Type;


  for ( unsigned int i = 0; i < S->GetNumberOfParticles(); i++ ){
    gzprintf(gp, "%d %.*f %.*f %.*f", h_Type[i], precision, h_r[i].x, 
	     precision, h_r[i].y, precision, h_r[i].z);
    if(metaData.Get("images"))
      gzprintf(gp, " %.0f %.0f %.0f", h_im[i].x, h_im[i].y, h_im[i].z);
    if(metaData.Get("velocities"))
      gzprintf(gp, " %.*f %.*f %.*f", precision, h_v[i].x, 
	       precision, h_v[i].y, precision, h_v[i].z);
    if(metaData.Get("forces"))
      gzprintf(gp, " %.*f %.*f %.*f", precision, h_f[i].x, 
	       precision, h_f[i].y, precision, h_f[i].z);
    if(metaData.Get("pot_energies"))
      gzprintf(gp, " %.*f", precision, h_f[i].w);
    if(metaData.Get("virials"))
      gzprintf(gp, " %.*f", precision, h_w[i].w/2.);
    gzprintf(gp,"\n");
    
    // Some consistency check for the neighborList.
    std::vector<Potential*>* vec = S->GetPotentials();
    for ( std::vector<Potential*>::iterator itr = vec->begin(); itr < vec->end(); itr++ ){
      PairPotential* testPairPotential = dynamic_cast<PairPotential*>(*itr);

      if( testPairPotential && h_im[i].w > 0.250001*testPairPotential->GetNbListSkin()*testPairPotential->GetNbListSkin() )
	throw( RUMD_Error("ConfigurationWriterReader","Write","Particles have moved more than half the neighborList skin. Check your starting configuration.") );
    }
  } // end loop over particles
}

void ConfigurationWriterReader::ReadLine(gzFile gp) {
  // reads a line from the compressed file into the lineBuffer array.
  // If there's not enough space it reallocates the latter, up to a point

  int errnum;
  char *ok = gzgets(gp, lineBuffer, bufferLength);
  if(!ok) throw RUMD_Error("ConfigurationWriterReader","ReadLine",
			   std::string("\nError reading line in file ") 
			   + gzerror(gp, &errnum) );

  while(strlen(lineBuffer) == bufferLength - 1) {
    unsigned int new_buffer_length = bufferLength * 2;
    if(new_buffer_length > max_buffer_length) throw RUMD_Error("ConfigurationWriterReader","ReadLine","Tried to read a line exceeding maximum allowed buffer length of 100000 characters: something must be wrong");
    char* new_buffer = new char[new_buffer_length];
    strcpy(new_buffer, lineBuffer);
    delete [] lineBuffer;
    unsigned int current_line_length = bufferLength - 1;
    lineBuffer = new_buffer;
    bufferLength = new_buffer_length;
    ok = gzgets(gp, lineBuffer+current_line_length, bufferLength - current_line_length);
    if(!ok) throw RUMD_Error("ConfigurationWriterReader","ReadLine",
			     std::string("\nError reading line in file ") 
			     + gzerror(gp, &errnum) );
    
  }
  
  // check last character is newline (shouldn't be 
  // possible to get here otherwise, but...)
  int len = strlen(lineBuffer);
  assert(strcmp(&(lineBuffer[len-1]),"\n") == 0);
}

void ConfigurationWriterReader::InitializeMetaData(Sample* __attribute__((unused))S) {
  metaData.Ps = 0.0;
  metaData.rumd_conf_ioformat = rumd_conf_ioformat;
}


void ConfigurationWriterReader::Read(Sample* S, const std::string& filename) {
  unsigned int tdx;
  int nItems;
  unsigned int set_num_part = 0;
    
  gzFile gp;
  gp = gzopen(filename.c_str(),"r");
  if (!gp)
    throw RUMD_Error("ConfigurationWriterReader","Read", std::string("\nError opening file ") + filename + ": " + strerror(errno));

  // read the first line: number of particles, do associated memory alloc
  ReadLine(gp);
  nItems = sscanf(lineBuffer, "%u", &set_num_part);
  if(nItems != 1  || set_num_part <= 0)
    throw RUMD_Error("ConfigurationWriterReader","Read", std::string("\nError reading number of particles from file ")+filename);

  InitializeMetaData(S); // sets default/existing values
  
  // read the comment line and replace final newline with a zero
  ReadLine(gp);
  lineBuffer[strlen(lineBuffer)-1] = 0;
  // read data from comment line into metaData
  metaData.ReadMetaData(lineBuffer, verbose);

  
  // Create SimulationBox.
  SimulationBox* new_sim_box = SimulationBoxFactory().CreateSimulationBox(metaData.simulationBoxInfoStr);

  S->SetSimulationBox(new_sim_box, true);
  S->SetNumberOfParticles(set_num_part);  
  S->GetParticleData()->SetNumberOfTypes(metaData.GetNumTypes()); 
  for(tdx = 0; tdx < S->GetParticleData()->GetNumberOfTypes(); ++tdx)
    S->GetParticleData()->SetMass( tdx, metaData.massOfType[tdx] );
  
  ReadParticleData(S, gp);
  gzclose(gp);
  S->CopySimulationDataToDevice();    
}

void ConfigurationWriterReader::ReadParticleData(Sample *S, gzFile gp) {
  // this code is the same for ioformat 1 and 2 so it is in a comnon function

  float4 zero = { 0.f, 0.f, 0.f, 0.f };

  ParticleData& particleData = S->particleData;
  float4* h_r = particleData.h_r;
  float4* h_v = particleData.h_v;
  float4* h_im = particleData.h_im;
  float4* h_f = particleData.h_f;
  float4* h_w = particleData.h_w; 
  unsigned int* h_Type = particleData.h_Type;
  
  // this is required for the memcpy trick with Types
  assert(sizeof(int) == sizeof(float));

  std::vector<unsigned int> numberOfType(S->GetParticleData()->GetNumberOfTypes());
 
  // read the per-particle
  for ( unsigned int i = 0; i < S->GetNumberOfParticles(); i++ ){
    
    ReadLine(gp);
    int offset = 0;
    int partial_offset = 0;
    
    // type and positions
    sscanf(lineBuffer+offset, "%d %f %f %f%n", &h_Type[i], &h_r[i].x, &h_r[i].y, &h_r[i].z, &partial_offset);
    offset += partial_offset;
    numberOfType[h_Type[i]] += 1;
    // set mass
    if (h_Type[i] >= S->GetParticleData()->GetNumberOfTypes()) {
      printf("%d %d %f %f %f %d\n", i, h_Type[i], h_r[i].x, h_r[i].y, h_r[i].z, S->GetParticleData()->GetNumberOfTypes() );
      assert(h_Type[i] < S->GetParticleData()->GetNumberOfTypes());	    
    }
    // For sending to the device, store type as a float instead of uint

    memcpy((void*)&(h_r[i].w), (void*)&(h_Type[i]),sizeof(int));

    h_v[i].w = 1.0 / S->GetParticleData()->GetMass(h_Type[i]);
    h_im[i] = zero; // initialize images
    if(metaData.Get("images")) {
      sscanf(lineBuffer+offset, "%f %f %f%n", &h_im[i].x, &h_im[i].y, &h_im[i].z, &partial_offset);
      offset += partial_offset;
    }
    if(metaData.Get("velocities"))
      {
	sscanf(lineBuffer+offset, "%f %f %f%n", &h_v[i].x, &h_v[i].y, &h_v[i].z, &partial_offset);
	offset += partial_offset;
      }	
    if(metaData.Get("forces")) {
      sscanf(lineBuffer+offset, "%f %f %f%n", &h_f[i].x, &h_f[i].y, &h_f[i].z, &partial_offset);
      offset += partial_offset;
    }
    if(metaData.Get("pot_energies")) {
      sscanf(lineBuffer+offset, "%f%n", &h_f[i].w, &partial_offset);
      offset += partial_offset;
    }
    if(metaData.Get("virials")) {
      sscanf(lineBuffer+offset, "%f%n",&h_w[i].w, &partial_offset);
      offset += partial_offset;
      h_w[i].w *= 2.;
    }
  } // end loop over particles

  unsigned int check_type_sum = 0;
  for (unsigned int tdx=0;tdx < S->GetParticleData()->GetNumberOfTypes(); ++tdx) {
    check_type_sum += numberOfType[tdx];
    S->GetParticleData()->SetNumberThisType(tdx, numberOfType[tdx]);
  }
  assert ( check_type_sum  == S->GetNumberOfParticles() );


  // Check if the type number of particles is allowed. Temp. solution.
  for( unsigned i=0; i < S->GetNumberOfParticles(); i++ ){
    if( unsigned(h_r[i].w) > 31 ) 
      throw( RUMD_Error("ConfigurationWriterReader","Read", "The particle type number in start.xyz is larger than 31" ) );
  }


}

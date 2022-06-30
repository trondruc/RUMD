#include "rumd_rouse.h"
#include "rumd/ParseInfoString.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

rumd_rouse::rumd_rouse() : Time(0),
			   Count(0),
			   RgRg(0),
			   R0R0(0),
			   R0Rt(0),
			   X0Xt(0),
			   X0X0(0),
			   MaxDataIndex(0),
			   num_mol(0),
			   dt(0.0) {}


void rumd_rouse::AllocateArrays( unsigned int set_MaxDataIndex, unsigned int ppm) {

  if( set_MaxDataIndex != MaxDataIndex  &&  MaxDataIndex > 0 )
    FreeArrays();

  MaxDataIndex = set_MaxDataIndex;
  particlesPerMol = ppm;
  num_mol = numParticles/ppm;

  Count = new unsigned long int[MaxDataIndex];
  Time  = new unsigned long int[MaxDataIndex];
  R0Rt  = new double[MaxDataIndex];
  X0Xt  = new double*[MaxDataIndex];
  X0X0  = new double*[particlesPerMol - 1];

  for (unsigned int i=0; i<MaxDataIndex; i++ )
    X0Xt[i] = new double[particlesPerMol - 1];

  for (unsigned int p=1; p<particlesPerMol; p++ )
    X0X0[p-1] = new double[particlesPerMol - 1];

}


void rumd_rouse::ResetArrays() {

  if(MaxDataIndex == 0)
    throw RUMD_Error("rumd_msd","ResetArrays","Called Reset before arrays have been allocated");

  R0R0 = 0;
  RgRg = 0;

  for (unsigned int i=0; i<MaxDataIndex; i++ ) {
    Count[i] = 0;
    R0Rt[i] = 0;
    for (unsigned int p=1; p<particlesPerMol; p++)
	X0Xt[i][p-1] = 0.0;
  } // loop over i

  for (unsigned int p=1; p<particlesPerMol; p++)
    for (unsigned int q=1; q<particlesPerMol; q++)
      X0X0[p-1][q-1] = 0.0;

}


void rumd_rouse::FreeArrays() {

  for (unsigned int p=1; p<particlesPerMol; p++ )
    delete X0X0[p-1];
  delete X0X0;

  for (unsigned int i=0; i<MaxDataIndex; i++ )
    delete X0Xt[i];
  delete X0Xt;

  delete R0Rt;
  delete Time;
  delete Count;

}

void rumd_rouse::CalcX0X0( Conf &C0 ) {
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double   Lx = sim_box_params_C0[0];
  double   Ly = sim_box_params_C0[1];
  double   Lz = sim_box_params_C0[2];
  double invN = 1.0 / double(particlesPerMol);

  // p denotes rouse mode
  // m denotes molecule
  // i denotes atom within molecule
  for ( unsigned int p=1; p<particlesPerMol; p++ ) {
    for ( unsigned int q=1; q<particlesPerMol; q++ ) {
      for ( unsigned int m=0; m<num_mol; m++ ) {
	unsigned int offset = m*particlesPerMol;
	double Xpx = 0; double Xqx = 0;
	double Xpy = 0; double Xqy = 0;
	double Xpz = 0; double Xqz = 0;

	for ( unsigned int i=0; i<particlesPerMol; i++ ){
	  double cosine = cos( invN * M_PI * p * (double(i)+0.5) );
	  Particle Pi = C0.P[offset+i];
	  Xpx += cosine * ( Pi.x + Lx*Pi.Imx );
	  Xpy += cosine * ( Pi.y + Ly*Pi.Imy );
	  Xpz += cosine * ( Pi.z + Lz*Pi.Imz );

	  cosine = cos( invN * M_PI * q * (double(i)+0.5) );
	  Xqx += cosine * ( Pi.x + Lx*Pi.Imx );
	  Xqy += cosine * ( Pi.y + Ly*Pi.Imy );
	  Xqz += cosine * ( Pi.z + Lz*Pi.Imz );

	}

	X0X0[p-1][q-1] += invN*invN * ( Xpx*Xqx + Xpy*Xqy + Xpz*Xqz );

      } // loop over m
    } // loop over q
  } // loop over p

}

void rumd_rouse::CalcX0Xt( Conf &C0, Conf &Ct, double *Xt ) {
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double   Lx = sim_box_params_C0[0];
  double   Ly = sim_box_params_C0[1];
  double   Lz = sim_box_params_C0[2];
  double invN = 1.0 / double(particlesPerMol);

  // p denotes rouse mode
  // m denotes molecule
  // i denotes atom within molecule
  for ( unsigned int p=1; p<particlesPerMol; p++ ) {
    for ( unsigned int m=0; m<num_mol; m++ ) {
      unsigned int offset = m*particlesPerMol;
      double Xp0x = 0; double Xptx = 0;
      double Xp0y = 0; double Xpty = 0;
      double Xp0z = 0; double Xptz = 0;

      for ( unsigned int i=0; i<particlesPerMol; i++ ){
	double cosine = cos( invN * M_PI * p * (double(i)+0.5) );
	Particle Pi = C0.P[offset+i];
	Xp0x += cosine * ( Pi.x + Lx*Pi.Imx );
	Xp0y += cosine * ( Pi.y + Ly*Pi.Imy );
	Xp0z += cosine * ( Pi.z + Lz*Pi.Imz );

	Pi = Ct.P[offset+i];
	Xptx += cosine * ( Pi.x + Lx*Pi.Imx );
	Xpty += cosine * ( Pi.y + Ly*Pi.Imy );
	Xptz += cosine * ( Pi.z + Lz*Pi.Imz );
      }

      Xt[p-1] += invN*invN*( Xp0x*Xptx + Xp0y*Xpty + Xp0z*Xptz );

    } // loop over m
  } // loop over p

}


void rumd_rouse::CalcR0R0( Conf& C0 ) {
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double   Lx = sim_box_params_C0[0];
  double   Ly = sim_box_params_C0[1];
  double   Lz = sim_box_params_C0[2];

  for ( unsigned int m=0; m<num_mol; m++ ) {  // loop over molecules
    unsigned int offset = m*particlesPerMol;

    // radius of gyration
    double RgRg_m = 0;
    for (unsigned int ai=0; ai < particlesPerMol; ai++ ){        // atoms in  m
      unsigned int i = offset + ai;                              // global index
      for (unsigned int aj = ai+1; aj < particlesPerMol; aj++ ){ // later atoms
	unsigned int j = offset + aj;                            // global index
	RgRg_m += C0.P[i].Rsq(&C0.P[j], Lx, Ly, Lz);
      }
    }
    RgRg += RgRg_m / ( particlesPerMol * particlesPerMol );

    // end-to-end vector
    R0R0 +=  C0.P[offset].Rsq(&C0.P[offset+particlesPerMol-1], Lx, Ly, Lz);

  } // end of loop over molecules

}

void rumd_rouse::CalcR0Rt(Conf &C0, Conf &Ct, double *Rt) {
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double   Lx = sim_box_params_C0[0];
  double   Ly = sim_box_params_C0[1];
  double   Lz = sim_box_params_C0[2];

  // m denotes molecule
  // R denotes the end
  for ( unsigned int m=0; m<num_mol; m++ ) {
    unsigned int offset = m*particlesPerMol;

    Particle P0 = C0.P[offset]; // first particle in chain
    Particle PN = C0.P[offset+particlesPerMol-1];  // last
    double R0x = PN.x-P0.x + Lx*(PN.Imx-P0.Imx);
    double R0y = PN.y-P0.y + Ly*(PN.Imy-P0.Imy);
    double R0z = PN.z-P0.z + Lz*(PN.Imz-P0.Imz);

    P0 = Ct.P[offset]; // first particle in chain at time t
    PN = Ct.P[offset+particlesPerMol-1];  // last at time t
    double Rtx = PN.x-P0.x + Lx*(PN.Imx-P0.Imx);
    double Rty = PN.y-P0.y + Ly*(PN.Imy-P0.Imy);
    double Rtz = PN.z-P0.z + Lz*(PN.Imz-P0.Imz);

    *Rt += ( R0x*Rtx + R0y*Rty + R0z*Rtz );

  } // loop over m

}


void rumd_rouse::ComputeAll(unsigned int first_block, int last_block,  unsigned int particlesPerMol) {
  unsigned int last_block_to_read;
  gzFile gzfp;
  Conf C0, Ct;

  try {
    ReadLastCompleteFile("trajectory");
  }
  catch (const RUMD_Error &e) {
    ReadLastCompleteFile("block");
  }

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
    std::cerr << "Warning: last_block smaller than first_block: no rdfs will be computed" << std::endl;

  // Read metaData from first configuration in first block
  std::string C0_name = GetConfFilename(first_block);

  if(verbose)
    std::cout << "Reading block " << first_block << ": " << C0_name << std::endl << std::endl;

  gzfp=gzopen(C0_name.c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_rouse","ComputeAll",std::string("Error opening file") + C0_name + ": " + strerror(errno) );

  C0.Read(gzfp,C0_name.c_str(),verbose);
  gzclose(gzfp);

  // Store the metaData we need
  dt = C0.metaData.GetDt();
  numParticles = C0.num_part;
  unsigned int MaxIndex = C0.metaData.GetLogLin().maxIndex;
  unsigned int MaxInterval = C0.metaData.GetLogLin().maxInterval;
  if(MaxInterval)
    throw RUMD_Error("rumd_rouse","ComputeAll","Only logarithmically saved data allowed");

  int set_MaxDataIndex = MaxIndex + int(log2(last_block_to_read-first_block+1)+1);

  AllocateArrays(set_MaxDataIndex, particlesPerMol);

  ResetArrays();

  std::cout << "\nReading block: ";
  for (unsigned int i=first_block; i<=last_block_to_read; i++) {

    // Read first configuration of block 'i'
    std::string C0_name = GetConfFilename(i);

    if ((i-first_block)%10==0)
      std::cout << i << " ";
    fflush(stdout);
    gzfp=gzopen(C0_name.c_str(),"r");
    C0.Read(gzfp,C0_name,0);

    // Check that relevant metaData didn't change
    assert(C0.metaData.GetDt()       == dt);
    assert(C0.num_part          == numParticles);
    assert(MaxIndex == C0.metaData.GetLogLin().maxIndex);

    // Calculate X0*X0 (for amplitude and normalization)
    CalcX0X0( C0 );
    CalcR0R0( C0 );

    // Analyze configurations in the same block-file
    for (unsigned int j=1; j<=MaxIndex; j++) {
      Ct.Read(gzfp,C0_name,0);
      Count[j-1]++;
      CalcX0Xt( C0, Ct, X0Xt[j-1] );
      CalcR0Rt( C0, Ct, &R0Rt[j-1] );

    }

    gzclose(gzfp);

    // Analyze configurations in the later block-files
    for (unsigned int j=1; i+(1<<j)<=last_block_to_read; j++) {
      std::string C0_name = GetConfFilename(i+(1<<j));
      gzfp=gzopen(C0_name.c_str(),"r");
      Ct.Read(gzfp,C0_name,0);
      unsigned int jj = MaxIndex+j-1;
      assert(jj<MaxDataIndex);
      Count[jj]++;
      CalcR0Rt( C0, Ct, &R0Rt[jj] );
      CalcX0Xt( C0, Ct, X0Xt[jj] );
      gzclose(gzfp);
     }
  }

  // Normalize with number of used configurations
  //   and number of molecules per simulation
  R0R0 /= float(Count[0])*num_mol;
  RgRg /= float(Count[0])*num_mol;

  for ( unsigned int p=1; p<particlesPerMol; p++ )
    for ( unsigned int q=1; q<particlesPerMol; q++ )
      X0X0[p-1][q-1] /= float(Count[0]*num_mol);

  for ( unsigned int i=0; i<MaxDataIndex; i++ ){
    R0Rt[i] /= float(Count[i]*num_mol);
    for ( unsigned int p=1; p<particlesPerMol; p++ )
      X0Xt[i][p-1] /= float(Count[i]*num_mol);
  }
  std::cout << Count[0] << std::endl;
  std::cout << std::endl;
}


///////////////// NORMALIZATION FUNCTIONS /////////////////

void rumd_rouse::NormalizeR0Rt(){

  // Normalize end-to-end correlation function
  double factor = 1.0/R0R0;
  for ( unsigned int i=0; i<MaxDataIndex; i++ )
      R0Rt[i] *= factor;

}


void rumd_rouse::NormalizeX0Xt(){

  // Normalize mode correlation functions
  for ( unsigned int i=0; i<MaxDataIndex; i++ )
    for ( unsigned int p=1; p<particlesPerMol; p++ )
      X0Xt[i][p-1] /= X0X0[p-1][p-1];

}


void rumd_rouse::NormalizeX0X0(){

  // Normalize cross-correlation matrix
  for ( unsigned int p=1; p<particlesPerMol; p++ )
    for ( unsigned int q=1; q<particlesPerMol; q++ )
      if ( p != q )
	X0X0[p-1][q-1] /= sqrt( X0X0[p-1][p-1] * X0X0[q-1][q-1] );

  for ( unsigned int p=1; p<particlesPerMol; p++ )
    X0X0[p-1][p-1] = 1.0;

}


///////////////// HELPER FUNCTIONS FOR PYTHON INTERFACE /////////////////

void rumd_rouse::Copy_X0Xt_To_Array(double (*X0Xt_array)[2], unsigned int p) {
  if ( MaxDataIndex == 0 )
    throw RUMD_Error("rumd_rouse","Copy_X0Xt_To_Array","No data! Call ComputeAll first");
  if( p == 0  ||  p >= particlesPerMol )
    throw RUMD_Error("rumd_rouse","Copy_X0Xt_To_Array","Invalid p value");

  for (unsigned int i=0; i<MaxDataIndex; i++ ) {
    X0Xt_array[i][0] = (1UL<<i)*dt;
    if ( Count[i]>0 )
      X0Xt_array[i][1] = X0Xt[i][p-1];
    else
      X0Xt_array[i][1] = 0.0;
  }

}

////////////// FILE OUTPUT FUNCTIONS /////////////////////////////////////

void rumd_rouse::WriteR0Rt(const std::string& filename) {
  // Writes the 2-D array with the orientational autocorrelation
  // of the end-to-end vector

  std::ofstream outfile(filename.c_str());

  for (unsigned int i=0; i<MaxDataIndex; i++ )
    if (Count[i]>0)
      outfile << (1UL<<i)*dt << " " << R0Rt[i] << std::endl;

  outfile.close();

}

void rumd_rouse::WriteR0R0(const std::string& filename) {
  // Writes the average end-to-end vector and radius of gyration

  std::ofstream outfile(filename.c_str());

  outfile << "# R0R0 RgRg" << std::endl;
  outfile << " " << R0R0 << " " << RgRg << std::endl;

  outfile.close();

}

void rumd_rouse::WriteX0Xt(const std::string& filename) {
  // Writes the 2-D array with autocorralions of all modes

  std::ofstream outfile(filename.c_str());

  for (unsigned int i=0; i<MaxDataIndex; i++ ) {
    if (Count[i]>0) {
      outfile << (1UL<<i)*dt << " ";
      for (unsigned int p=1; p<particlesPerMol; p++)
	outfile << " " << X0Xt[i][p-1];
      outfile << std::endl;
    }
  }

  outfile.close();

}


void rumd_rouse::WriteX0X0(const std::string& filename) {
  // Writes the variance (amplitude)

  std::ofstream outfile(filename.c_str());

  for (unsigned int p=1; p<particlesPerMol; p++){
    outfile << " " << p;
    outfile << " " << X0X0[p-1][p-1];
    outfile << std::endl;
  }

  outfile.close();

}

void rumd_rouse::WriteXp0Xq0(const std::string& filename) {
  //Writes the covariance

  std::ofstream outfile(filename.c_str());

  for (unsigned int p=1; p<particlesPerMol; p++){
    for (unsigned int q=1; q<particlesPerMol; q++)
      outfile << " " << X0X0[p-1][q-1];
    outfile << std::endl;
  }

  outfile.close();

}

#ifndef RUMD_DATA_H
#define RUMD_DATA_H

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/ConfigurationMetaData.h"
#include "rumd/EnergiesMetaData.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <zlib.h>

class Particle {
 public:
  unsigned int MyType;
  float  x,  y,  z;
  float vx, vy, vz;
  float fx, fy, fz;
  int Imx, Imy, Imz;
  float Ui, Wi;
  Particle() : MyType(0), x(0), y(0), z(0), vx(0),
    vy(0), vz(0), fx(0), fy(0), fz(0), Imx(0), Imy(0), Imz(0), Ui(0), Wi(0) {}
  
  double RsqYZ(Particle *P, float Ly, float Lz) {
    // transverse contribution only, for the Lees-Edwards case
    double dy = y - P->y + Ly*(Imy - P->Imy);
    double dz = z - P->z + Lz*(Imz - P->Imz);

    return dy*dy + dz*dz;
  }

  double Rsq(Particle *P, float Lx, float Ly, float Lz) {

    double dx = x - P->x + Lx*(Imx - P->Imx);
    double dy = y - P->y + Ly*(Imy - P->Imy);
    double dz = z - P->z + Lz*(Imz - P->Imz);

    return dx*dx + dy*dy + dz*dz;
  }

  double Rsq_wrt_cm(Particle *P, float Lx, float Ly, float Lz, double*cm_disp) {

    double dx = x - P->x + Lx*(Imx - P->Imx) - cm_disp[0];
    double dy = y - P->y + Ly*(Imy - P->Imy) - cm_disp[1];
    double dz = z - P->z + Lz*(Imz - P->Imz) - cm_disp[2];

    return dx*dx + dy*dy + dz*dz;
  }


  

  void Disp(Particle *P, float Lx, float Ly, float Lz, double* displacement) {
    displacement[0] = x - P->x + Lx*(Imx - P->Imx);
    displacement[1] = y - P->y + Ly*(Imy - P->Imy);
    displacement[2] = z - P->z + Lz*(Imz - P->Imz);
  }
  
  double RsqMinImageVec(Particle *P, float Lx, float Ly, float Lz, double invLx,  double invLy, double invLz, double* disp, double boxShift=0.0) {
    double dx = x - P->x;
    double dy = y - P->y;
    double dz = z - P->z;
    double y_wrap = rintf( dy * invLy );
    dx -= y_wrap * boxShift;

    dx -= Lx*rintf(dx*invLx);
    dy -= Ly * y_wrap;
    dz -= Lz*rintf(dz*invLz);
    disp[0] = dx;
    disp[1] = dy;
    disp[2] = dz;
    return dx*dx + dy*dy + dz*dz;
  }
  
  double RsqMinImage(Particle *P, float Lx, float Ly, float Lz, double invLx,  double invLy,  double invLz, double boxShift=0.0) {

    // boxShift is amount by which the image in the +y direction is 
    // shifted in the +x direction

    double dx = x - P->x;
    double dy = y - P->y;
    double dz = z - P->z;

    double y_wrap = rintf( dy * invLy );
    dx -= y_wrap * boxShift;

    dx -= Lx*rintf(dx*invLx);
    dy -= Ly * y_wrap;
    dz -= Lz*rintf(dz*invLz);

    return dx*dx + dy*dy + dz*dz;
  }

  double Fsq(Particle *P, float Lx, float Ly, float Lz, float q) {
    double Sum = 0.0;

    Sum += cos( (x - P->x + Lx*(Imx - P->Imx)) * q );
    Sum += cos( (y - P->y + Ly*(Imy - P->Imy)) * q );
    Sum += cos( (z - P->z + Lz*(Imz - P->Imz)) * q );

    return Sum/3.0f;
  }
  
  double FsqYZ(Particle *P,float Ly, float Lz, float q) {
    double Sum = 0.0;

    Sum += cos( (y - P->y + Ly*(Imy - P->Imy)) * q );
    Sum += cos( (z - P->z + Lz*(Imz - P->Imz)) * q );

    return Sum/2.0f;
  }
  


  // Fsq with minimum image instead of keeping track of boundary conditions.
  double FsqMinImage(Particle* P, float Lx, float Ly, float Lz, float q){
    double Sum = 0.0;
    
    double dx = x - P->x;
    double dy = y - P->y;
    double dz = z - P->z;
    
    double invLx = 1.f / Lx;
    double invLy = 1.f / Ly;
    double invLz = 1.f / Lz;    
    
    dx -= Lx*rintf(dx*invLx);
    dy -= Ly*rintf(dy*invLy);
    dz -= Lz*rintf(dz*invLz);
    
    Sum += cos( dx * q );
    Sum += cos( dy * q );
    Sum += cos( dz * q );
    
    return Sum/3.0f;
  }
  
  // Perform a dot product of velocities.
  double VelocityDotProduct(Particle* P){
    return vx * P->vx + vy * P->vy + vz * P->vz;
  }
};

class Conf {
 private:
  // protect from default copy construction and copy assignment 
  Conf(const Conf&);
  Conf& operator=(const Conf&);
  std::string currentFilename;
  gzFile current_gzFile;
  unsigned bufferLength;
  const unsigned int max_buffer_length;
  char* lineBuffer;
 public:
  unsigned int num_part;
  Particle *P;
  ConfigurationMetaData  metaData;
  

  Conf(): 
    currentFilename(""), current_gzFile(0), bufferLength(400), max_buffer_length(100000), lineBuffer(0), num_part(0), P(0), metaData(ConfigurationMetaData()) {
    lineBuffer = new char[bufferLength]; }
  
  void OpenGZ_File(const std::string& filename) {
    currentFilename = filename;
    if(current_gzFile)
      gzclose(current_gzFile);
    current_gzFile = gzopen(currentFilename.c_str(),"r");
  }
  void CloseCurrentGZ_File() {
    if(current_gzFile)
      gzclose(current_gzFile);
    current_gzFile = 0;
  }

  void SetNumberOfParticles(unsigned int num_part_set) {

    assert(num_part_set > 0);
    if (num_part==0) {

      // Allocate particles if not done already
      num_part = num_part_set;
      P = new Particle[num_part];
    } else {

      // If already allocated, check that number didn't change
      assert(num_part_set==num_part);
    }
  }

  unsigned int GetNumberOfParticles() const {return num_part;}

  ~Conf() {
    if(current_gzFile)
      gzclose(current_gzFile);
    if(P)
      delete[] P;
    delete [] lineBuffer;
  }

  void ReadCurrentFile(bool verbose=false) {
    if(!current_gzFile)
      throw RUMD_Error("Conf","ReadCurrentFile","There is no currently open gz file");
    Read(current_gzFile, currentFilename, verbose);
  }

  
  void ReadLine(gzFile gp, const std::string& filename) {
    int errnum;
    char *ok = gzgets(gp, lineBuffer, bufferLength);
    if(!ok) 
      throw RUMD_Error("Conf",__func__,std::string("Failed to read a line from file ") + filename + ": " + gzerror(gp, &errnum));
    
    while(strlen(lineBuffer) == bufferLength - 1) {
      unsigned int new_buffer_length = bufferLength * 2;
      if(new_buffer_length > max_buffer_length) throw RUMD_Error("Conf",__func__,"Tried to read a line exceeding maximum allowed buffer length of 100000 characters: something must be wrong");
      char* new_buffer = new char[new_buffer_length];
      strcpy(new_buffer, lineBuffer);
      delete [] lineBuffer;
      unsigned int current_line_length = bufferLength - 1;
      lineBuffer = new_buffer;
      bufferLength = new_buffer_length;
      ok = gzgets(gp, lineBuffer+current_line_length, bufferLength - current_line_length);
      if(!ok) throw RUMD_Error("Conf",__func__,
			       std::string("\nError reading line in file ") + filename + ": " + gzerror(gp, &errnum) );    
    }
    
  }

  void Read(gzFile gp, const std::string& filename, bool verbose) {

    int nItems;
    unsigned int set_num_part=0;

    ReadLine(gp, filename);
    nItems = sscanf(lineBuffer, "%u", &set_num_part);
    if(nItems != 1)
      throw RUMD_Error("Conf","Read",std::string("Error reading number of particles from  file ")+filename + "\n Read: " + lineBuffer);

    // Set everything associated with number of particles, incl memory alloc
    assert(set_num_part > 0);
    SetNumberOfParticles(set_num_part);

    // read the comment line and replace final newline with a zero
    ReadLine(gp, filename);
    lineBuffer[strlen(lineBuffer)-1] = 0;
    // read data from comment line into metaData
    metaData.ReadMetaData(lineBuffer, verbose);

    // read the per-particle data
    for ( unsigned int i = 0; i < num_part; i++ ){
      ReadLine(gp, filename);
      int offset = 0;
      int partial_offset = 0;

      // type and positions
      sscanf(lineBuffer+offset, "%u %f %f %f%n", &P[i].MyType, &P[i].x, &P[i].y, &P[i].z, &partial_offset);
      offset += partial_offset;
      //printf("%d %f %f %f", P[i].MyType, P[i].x, P[i].y, P[i].z);
      assert(P[i].MyType < metaData.GetNumTypes());

      if(metaData.Get("images")) {
	sscanf(lineBuffer+offset, "%d %d %d%n", &P[i].Imx, &P[i].Imy, &P[i].Imz, &partial_offset);
        offset += partial_offset;
        //printf(" %d %d %d\n", P[i].Imx, P[i].Imy, P[i].Imz);
      } else {
	P[i].Imx=0; P[i].Imy=0; P[i].Imz=0;
      }

      if(metaData.Get("velocities")) {
        sscanf(lineBuffer+offset, "%f %f %f%n", &P[i].vx, &P[i].vy, &P[i].vz, &partial_offset);
        offset += partial_offset;
      }

      if(metaData.Get("forces")) {
	sscanf(lineBuffer+offset, "%f %f %f%n", &P[i].fx, &P[i].fy, &P[i].fz, &partial_offset);
        offset += partial_offset;
      }

      if(metaData.Get("pot_energies")) {
	sscanf(lineBuffer+offset, "%f%n", &P[i].Ui, &partial_offset);
        offset += partial_offset;
      }

      if(metaData.Get("virials")) {
	sscanf(lineBuffer+offset, "%f%n", &P[i].Wi, &partial_offset);
        offset += partial_offset;
      }
    }
    //ok = fgets(lineBuffer, bufferLength, fc); // apparently necessary...
  }


  void UiWi(Conf *Ct, double *MeanUi,  double *MeanWi, double *Ui, double *Wi, unsigned int num_types) {

    for (unsigned int i=0; i<num_part; i++) {
      unsigned int Type = P[i].MyType;
      assert(Type<num_types);
      assert(Type==Ct->P[i].MyType);
      Ui[Type] += (P[i].Ui-MeanUi[Type])*(Ct->P[i].Ui-MeanUi[Type]);
      Wi[Type] += (P[i].Wi-MeanWi[Type])*(Ct->P[i].Wi-MeanWi[Type]);
    }
  }

};
#endif

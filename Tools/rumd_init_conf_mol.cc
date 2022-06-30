#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <time.h>
#include <vector>
#include <assert.h>
#include <algorithm> 
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define LENGTH 256
#define DIM 3
#define EPSILON 1e-10
#define SEED 4622

using namespace std; 


/************************** STRUCTS ****************************/

// Represents a molecule defined via  
// the xyz and top files
typedef struct {

  unsigned nmol_type, nuau;

  float *rel_cm;
  float *masses;
  unsigned *types;

  unsigned *bonds, nbonds;
  unsigned *angles, nangles;
  unsigned *dihedrals, ndihedrals;

  float cm[3];
  float maxsize;

} molecule;



// Lattice on which the molecules are placed
typedef struct {
  float *x, *y, *z;
  float lbox;
} lattice;



/**************** DECLARATIONS/PROTOTYPES *******************/

// Aux. function
void merror(const char *str, const char *fun, int line);
bool setfilepointer(ifstream *fptr, string section);
void pipe( molecule *mol, int ntypes );
float velocity( float temp );
int findnumtypes(molecule *mol, int ntypes_mol);
void findatommasses(molecule *mol, int ntypes, float *masses, int ntypes_atoms);
void printpre(int nmol_tot, int ntypes, float dens, float temp, int randomOrientation, int seed, float userSpacing );
void  printinsect(int n, int ntypes);

// Memory management 
molecule *allmem(unsigned nmoltypes);
void freemem(lattice *lptr, molecule *mptr, unsigned nmols, unsigned *rlist, float *masses);

//Lattice information
float find_spacing(molecule *mols, int ntypes, unsigned nmol_tot, float dens, float userSpacing);
lattice setcubic(unsigned nmol_tot, float spacing);

// Postion information 
unsigned readxyz(molecule *mol, const char *file, unsigned moli, unsigned nmol, float *massMol);
void writexyz(molecule *mol, unsigned moli,  lattice lat, unsigned  lati, int randomOrientation, 
	      float temp, const char *str);
void CalculateRotMatrix(float *rotMatrix, float euler_alpha, float euler_beta, float euler_gamma);
void matProd(float *M, float *A, float *B);

void writexyzheader(unsigned natoms, float lbox, int numtypes, float *masses, const char *str);
unsigned *setmolpos(unsigned ntypes, unsigned nmol, unsigned *nmols_per_type);

// Bond information
void readtop_bonds(molecule *mol, const char *fstr, unsigned molt);
void writebondheader(const char *fstr);
void writebondssect(molecule *mol, unsigned molt, unsigned moli, const char *fstr);


// Angle information
void readtop_angles(molecule *mol, const char *fstr, unsigned molt);
void writeangleheader(const char *fstr);
void writeanglesect(molecule *mol, unsigned molt, unsigned moli, const char *fstr);


// Dihedral information
void readtop_dihedrals(molecule *mol, const char *fstr, unsigned molt);
void writedihedralheader(const char *fstr);
void writedihedralsect(molecule *mol, unsigned molt, unsigned moli, const char *fstr);


/*********************** MAIN ******************************/


int main(int argc, char **argv){

  int c;
  int seed = SEED;
  float temp = 1.0, dens = 0.0, userSpacing = 0.0;
  int randomOrientation = 1;

  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"seed",                required_argument, 0,  's' },
      {"temperature",         required_argument, 0,  't' },
      {"density",             required_argument, 0,  'd' },
      {"spacing",             required_argument, 0,  'p' },
      {"norandomorientation", no_argument,       0,  'r' },
      {"help",                no_argument,       0,  'h' },
      {0,                     0,                 0,  0 }
    };

    c = getopt_long(argc, argv, "s:t:d:r:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 's':
        seed = atoi(optarg);
        break;
      case 't':
        temp = atof(optarg);
        break;
      case 'd':
        dens = atof(optarg);
        break;
      case 'r':
        randomOrientation = 0;
        break;
      case 'p':
        userSpacing = atof(optarg);
        break;
      case 'h':
        cout << "Usage: rumd_init_conf_mol [--help] [--seed N] [--temperature X] [--density X] [--norandomorientation] [--spacing X] <m1.xyz> <m1.top> N1 ..." << endl;
        cout << "  --help: to display the help message. The code exits after it." << endl;
        cout << "  --seed N: to give the seed (integer). The default value is 4622." << endl;
        cout << "  --temperature X: to give the temperature (float). The default value is 1.0. The atom velocities are chosen from a uniform distribution with zero mean and a variance corresponding to the desired temperature." << endl;
        cout << "  --density X: to give the density (float). The default value corresponds to a spacing of twice the size of the largest molecule." << endl;
        cout << "  --norandomorientation: to ask for not randomizing the molecules orientation. By default, the orientations are random." << endl;
        cout << "  --spacing X: to give the spacing between two lattice sites (float). The default value is twice the size of the largest molecule. If both density and spacing are given by the user, that corresponding to the largest spacing is used." << endl;
        cout << "  m1.xyz: xyz file for the atoms in molecule one (string)." << endl;
        cout << "  m1.top: topology file for molecule one (string). If this is an atom, this string should begin with the word atom. The atom topology file, which would be empty, does not need to exist." << endl;
        cout << "  N1: the number of molecules of type one (integer)." << endl;
        cout << "  and so forth for other molecule types." << endl;
        cout << "  The mass of the atom types involved in each molecule type can be given in the comment line of the xyz file of each single molecule. For example, numTypes=2 mass=1.0,2.4, if the molecule contains two atom types of mass 1.0 and 2.4. The default value for the atom mass is 1.0." << endl;
        exit(0);
      default:
        merror("Unknown option",  __func__, __LINE__);
    }
  }

  int nArg = argc-optind; //number of arguments which are not options, that is the triplet xyz, top, N

  if ( (nArg)%3 != 0 || nArg < 3 )
    merror("Usage: rumd_init_conf_mol [--seed N] [--temperature X] [--density X] [--randomorientation N] <m1.xyz> <m1.top> N1 ... ", __func__, __LINE__);

  // Number of different molecule types
  int ntypes = (nArg)/3;

  // Get the number of molecules of each type and the total
  // number of molecules
  unsigned nmol_ptype[ntypes];   int nmol_tot = 0;
  for ( int n=0; n<ntypes; n++ ) {
    nmol_ptype[n] = atoi( argv[n*3 + optind + 2] ); 
    nmol_tot += nmol_ptype[n]; 
  }

  //Initialize random number generator
  srand(seed);

  printpre(nmol_tot, ntypes, dens, temp, randomOrientation, seed, userSpacing);

  // Read the information and calculate the number of 
  // atoms and the total mass of the system 
  molecule *mols = allmem(ntypes);
  unsigned natoms = 0;
  float totalMass = 0.0, massMol;
  for ( int n=0; n<ntypes; n++ ){
    printinsect(n, ntypes);
    natoms += readxyz(mols, argv[n*3 + optind], n, nmol_ptype[n], &massMol)*nmol_ptype[n];
    totalMass += massMol*nmol_ptype[n];
    readtop_bonds(mols, argv[n*3 + optind + 1], n);
    readtop_angles(mols, argv[n*3 + optind + 1], n);
    readtop_dihedrals(mols, argv[n*3 + optind + 1], n);
  }
  
  // Set the lattice sites 
  float spacing = find_spacing(mols, ntypes, nmol_tot, dens, userSpacing);
  lattice lat = setcubic(nmol_tot, spacing);

  // Set the molecules
  unsigned *rlist = setmolpos(ntypes, nmol_tot, nmol_ptype);
    
  // Write the headers in each (temporary) file 
  cout << endl << "Writing output files 'start.xyz.gz' and 'start.top'...";

  int ntypes_atoms = findnumtypes(mols, ntypes);
  float *masses = (float*)malloc(sizeof(float)*ntypes_atoms);
  if ( masses == NULL ) merror("Couldn't allocate memory",  __func__, __LINE__);
  findatommasses(mols, ntypes, masses, ntypes_atoms);
  writexyzheader(natoms, lat.lbox, ntypes_atoms, masses, "start.xyz");
  writebondheader("__bonds.top");
  writeangleheader("__angles.top");
  writedihedralheader("__dihedrals.top");

  // Write the config files
  float velocityFactor = sqrt(3.0*temp*natoms/totalMass);
  for ( int n=0; n<nmol_tot; n++ ){
    writexyz(mols, rlist[n], lat, n, randomOrientation, velocityFactor, "start.xyz");
    writebondssect(mols, rlist[n], n, "__bonds.top");
    writeanglesect(mols, rlist[n], n, "__angles.top");
    writedihedralsect(mols, rlist[n], n, "__dihedrals.top");
  }

  cout << "Done" << endl;

  // Pipe all temp. top files into start.top and remove 
  // temp. files
  pipe(mols, ntypes);

  // Free memory
  freemem(&lat, mols, ntypes, rlist, masses);

  (void)system("gzip -f start.xyz");

  return 0;
}



/************** DEFINITIONS *****************/

void merror(const char *str, const char *fun, int line){

  cerr << fun << " at line " << line << ", " << str << " BAILING OUT" << endl;

  exit(EXIT_FAILURE);

}


molecule *allmem(unsigned nmoltypes){
  
  molecule *mol = (molecule*)malloc(sizeof(molecule)*nmoltypes);
  if ( mol == NULL ) merror("Couldn't allocate memory",  __func__, __LINE__);
  
  return mol;
}



unsigned readxyz(molecule *mol, const char *file, 
		 unsigned moli, unsigned nmol, float *massMol){
  float x, y, z;
  unsigned type;
  int nAtomTypes = 0;
  float *atomMass;
  vector<unsigned> atomType;
  string line;

  atomMass = (float*)malloc(sizeof(float)*nAtomTypes);
  if ( atomMass == NULL )
    merror("Memory allocation error",  __func__, __LINE__);
  for (int a=0; a<nAtomTypes; a++) atomMass[a] = 1.0;

  ifstream fin (file);
  if (!fin.is_open()) merror("Couldn't open file",  __func__, __LINE__);

  getline(fin, line);
  if (fin.fail() || fin.bad()) merror("Reading failure ", __func__, __LINE__);
  if (!(stringstream(line) >> mol[moli].nuau))
    merror("Reading failure",  __func__, __LINE__);

  mol[moli].rel_cm = (float*)malloc(sizeof(float)*mol[moli].nuau*3);
  mol[moli].masses = (float*)malloc(sizeof(float)*mol[moli].nuau);
  mol[moli].types =  (unsigned*)malloc(sizeof(unsigned)*mol[moli].nuau);
  if ( mol[moli].rel_cm == NULL || mol[moli].types == NULL || mol[moli].masses == NULL) 
    merror("Memory allocation error",  __func__, __LINE__);

  getline(fin, line);
  if (fin.fail() || fin.bad()) merror("Reading failure ", __func__, __LINE__);
  if (line.compare(0, 9, "numTypes=") == 0) {
    char str[LENGTH];
    strcpy(str, line.c_str());
    if (sscanf( str, "numTypes=%d mass=", &nAtomTypes) != 1) 
      merror("Reading failure ", __func__, __LINE__);
    atomType.reserve(nAtomTypes);
    atomMass = (float*)realloc(atomMass, sizeof(float)*nAtomTypes);
    if ( atomMass == NULL )
      merror("Memory allocation error",  __func__, __LINE__);

    char* ptrTo2ndEqual = strchr(str,'=');
    ptrTo2ndEqual = strchr(ptrTo2ndEqual+1,'=');
    if (sscanf(ptrTo2ndEqual+1, "%f", &(atomMass[0])) != 1)  {
      merror("Reading failure ", __func__, __LINE__);
    }

    char* ptrToComa = strchr(str,',');
    for (int a=1; a<nAtomTypes; a++){
      if (sscanf(ptrToComa+1, "%f", &(atomMass[a])) != 1)  {
        merror("Reading failure ", __func__, __LINE__);
      }
      ptrToComa = strchr(ptrToComa+1,',');
      if (ptrToComa == NULL) {
        assert(a == nAtomTypes-1);
      }
    }
  }


  float cm[3] = {0.0};
  for ( unsigned n=0; n<mol[moli].nuau; n++ ){
    
    getline(fin, line);
    if (fin.fail() || fin.bad()) merror("Reading failure ", __func__, __LINE__);
    if ( !(stringstream(line) >> type >> x >> y >> z) ) {
      cerr << "Error reading atom " << n << endl;
      merror("Reading failure ",  __func__, __LINE__);
    }
    else { 
      mol[moli].types[n] = type;
      mol[moli].rel_cm[3*n]   = x;
      mol[moli].rel_cm[3*n+1] = y;
      mol[moli].rel_cm[3*n+2] = z;

      cm[0] += x; cm[1] += y; cm[2] += z; 
    }
  }

  fin.close();

  //find all atom types in this molecule
  for ( unsigned n=0; n<mol[moli].nuau; n++ ) {
    int newType = 1;
    for (int a=0; a<(int)atomType.size(); a++) {
      if (mol[moli].types[n] == atomType[a]) {
        newType = 0;
        break;
      }
    }
    if (newType == 1)
      atomType.push_back(mol[moli].types[n]);

  }
  if (nAtomTypes > 0) assert((int)atomType.size() == nAtomTypes);
  sort(atomType.begin(), atomType.end());


  //fill in all missing pieces of information about the atoms (relative position and mass )
  for ( int n=0; n<3; n++ ) cm[n] /= mol[moli].nuau;

  *massMol = 0.0;
  for ( unsigned n=0; n<mol[moli].nuau; n++ ) {
    mol[moli].rel_cm[3*n]   -= cm[0];
    mol[moli].rel_cm[3*n+1] -= cm[1];
    mol[moli].rel_cm[3*n+2] -= cm[2];

    if (nAtomTypes == 0) { 
      mol[moli].masses[n] = 1.0;
      *massMol += 1.0;
    } else {
      for (int a=0; a<nAtomTypes; a++) {
        if (mol[moli].types[n] == atomType[a]) {
          mol[moli].masses[n] = atomMass[a];
          *massMol += mol[moli].masses[n];
          break;
        }
      }
    }

  }  

  // find the size of the molecule (no need to take care of PBC) 
  float maxSize = 0.0;
  for ( unsigned n=0; n<mol[moli].nuau-1; n++ ){
    for ( unsigned m=n+1; m<mol[moli].nuau; m++ ) {

      float distance = 0.0;
      float xn, yn, zn, xm, ym, zm;
      xn = mol[moli].rel_cm[3*n]  ;
      yn = mol[moli].rel_cm[3*n+1];
      zn = mol[moli].rel_cm[3*n+2];
      xm = mol[moli].rel_cm[3*m]  ;
      ym = mol[moli].rel_cm[3*m+1];
      zm = mol[moli].rel_cm[3*m+2];
      distance = pow(xn-xm, 2) + pow(yn-ym, 2) + pow(zn-zm, 2);
      if (distance > maxSize) maxSize = distance;

    }
  }
  mol[moli].maxsize = sqrt(maxSize);

  
  mol[moli].nmol_type = nmol;
  free(atomMass);

  cout << "Read molecule information from " << file << ": " << mol[moli].nmol_type << " molecule(s) with " << mol[moli].nuau << " atom(s). " << endl;
 
  return mol[moli].nuau;
} 

float find_spacing(molecule *mols, int ntypes, unsigned nmol_tot, float dens, float userSpacing) {

  float spacing = 0.0;
  float densitySpacing =  0.0;
  if (dens > 0.0 + EPSILON) {
    int ngdim = ceil( pow(nmol_tot, 1.0/3.0) );
    float lbox = pow(nmol_tot/dens, 1.0/3.0);
    densitySpacing = lbox/ngdim;
  }
  float defaultSpacing = 0.0;
  for ( int n=0; n<ntypes; n++ ) {
    if (mols[n].maxsize > defaultSpacing ) defaultSpacing = 2.0*mols[n].maxsize;
  }

  if (dens <= 0.0 + EPSILON && userSpacing <= 0.0 + EPSILON) {
    spacing = defaultSpacing;
  } else if (dens <= 0.0 + EPSILON) {
    spacing = userSpacing;
  } else if (userSpacing <= 0.0 + EPSILON) {
    spacing = densitySpacing;
  } else {
    spacing = densitySpacing;
    if (userSpacing > densitySpacing) spacing = userSpacing;
  }
  cout << "------------[info] spacing = " <<  spacing << "------------" << endl; 

  return spacing;
}


lattice setcubic(unsigned nmol_tot, float dr){
  lattice lat;

  int ngdim = ceil( pow(nmol_tot, 1.0/3.0) );
  int ngrid = ngdim*ngdim*ngdim;

  lat.z = (float*)malloc(sizeof(float)*ngrid);
  lat.y = (float*)malloc(sizeof(float)*ngrid);
  lat.x = (float*)malloc(sizeof(float)*ngrid);  

  if ( lat.z == NULL || lat.y == NULL || lat.x == NULL )
    merror("Allocation failure",  __func__, __LINE__);

  float lbox = dr*ngdim;
  float hdr = 0.5*dr;

  int i = 0;
  for ( int n=0; n<ngdim; n++ ){
    float z = dr*n + hdr;
    for ( int m=0; m<ngdim; m++ ) {
      float y = dr*m + hdr;
      for ( int k=0; k<ngdim; k++ ) {
	float x = dr*k + hdr;

	lat.x[i] = x; lat.y[i] = y; lat.z[i] = z;
	i ++;
      }
    }
  }

  //to randomize the positions on the lattice
  vector<int> randomLat; randomLat.resize(ngrid);
  for (int n=0; n<ngrid; n++) randomLat[n] = n;
  lattice tmpLat;
  tmpLat.z = (float*)malloc(sizeof(float)*ngrid);
  tmpLat.y = (float*)malloc(sizeof(float)*ngrid);
  tmpLat.x = (float*)malloc(sizeof(float)*ngrid);

  for ( int n=0; n<ngrid; n++ ){
    int rLat = rand()%(ngrid-n);
    int latn = randomLat[rLat];
    randomLat[rLat] = randomLat[ngrid-n-1];
    randomLat.pop_back();
    tmpLat.x[n] = lat.x[latn]; 
    tmpLat.y[n] = lat.y[latn]; 
    tmpLat.z[n] = lat.z[latn]; 
  }
  for ( int n=0; n<ngrid; n++ ){
    lat.x[n] = tmpLat.x[n];
    lat.y[n] = tmpLat.y[n];
    lat.z[n] = tmpLat.z[n];
  }
  free(tmpLat.x);
  free(tmpLat.y);
  free(tmpLat.z);


  lat.lbox = lbox;

  return lat;
}


void writexyz(molecule *mol, unsigned moli,  lattice lat, unsigned  lati, int randomOrientation, 
	      float temp, const char *str){
  
  ofstream fptr;
  fptr.open(str, ios::app);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
  
  float hlbox = 0.5*lat.lbox;  const unsigned i = moli;

  float *rotMatrix = (float*)malloc(sizeof(float)*9);
  float euler_alpha = ((double)rand()/(RAND_MAX-1)*2.0-1.0)*M_PI; //uniform -pi, pi
  float euler_beta  = ((double)rand()/(RAND_MAX-1))*M_PI; //uniform 0, pi
  float euler_gamma = ((double)rand()/(RAND_MAX-1)*2.0-1.0)*M_PI; //uniform -pi, pi
  CalculateRotMatrix(rotMatrix, euler_alpha, euler_beta, euler_gamma);

  for ( unsigned n=0; n<mol[i].nuau; n++ ){
    float relx, rely, relz;
    relx = mol[i].rel_cm[3*n];
    rely = mol[i].rel_cm[3*n+1];
    relz = mol[i].rel_cm[3*n+2];
    if (randomOrientation) {
      relx = rotMatrix[0]*mol[i].rel_cm[3*n] + 
             rotMatrix[1]*mol[i].rel_cm[3*n+1] + 
             rotMatrix[2]*mol[i].rel_cm[3*n+2];
      rely = rotMatrix[3]*mol[i].rel_cm[3*n] +
             rotMatrix[4]*mol[i].rel_cm[3*n+1] +
             rotMatrix[5]*mol[i].rel_cm[3*n+2];
      relz = rotMatrix[6]*mol[i].rel_cm[3*n] +
             rotMatrix[7]*mol[i].rel_cm[3*n+1] +
             rotMatrix[8]*mol[i].rel_cm[3*n+2];
    }

    float x = lat.x[lati] + relx - hlbox;
    float y = lat.y[lati] + rely - hlbox;
    float z = lat.z[lati] + relz - hlbox;
   
    float vx = velocity(temp);
    float vy = velocity(temp);
    float vz = velocity(temp);

    fptr << mol[i].types[n] << ' ' << x << ' ' << y << ' ' <<  z << " 0 0 0 " << vx << ' ' << vy << ' ' << vz << endl;
  }

  fptr.close();
}

void CalculateRotMatrix(float *rotM, float alpha, float beta, float gamma) {
  float *xM = (float*)malloc(sizeof(float)*9);
  float *yM = (float*)malloc(sizeof(float)*9);
  float *zM = (float*)malloc(sizeof(float)*9);
  float *xyM = (float*)malloc(sizeof(float)*9);

  xM[0] = 1.0;         xM[1] = 0.0;         xM[2] = 0.0;
  xM[3] = 0.0;         xM[4] = cos(beta);   xM[5] = -sin(beta);
  xM[6] = 0.0;         xM[7] = sin(beta);   xM[8] = cos(beta);

  yM[0] = cos(alpha);  yM[1] = 0.0;         yM[2] = sin(alpha);
  yM[3] = 0.0;         yM[4] = 1.0;         yM[5] = 0.0;
  yM[6] = -sin(alpha); yM[7] = 0.0;         yM[8] = cos(alpha);

  zM[0] = cos(gamma);  zM[1] = -sin(gamma); zM[2] = 0.0;
  zM[3] = sin(gamma);  zM[4] = cos(gamma);  zM[5] = 0.0;
  zM[6] = 0.0;         zM[7] = 0.0;         zM[8] = 1.0;

  matProd(xyM, xM, yM);
  matProd(rotM, zM, xyM);
}

void matProd(float *M, float *A, float *B) {
  
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      M[3*i+j] = 0.0;
      for (int k=0; k<3; k++) {
        M[3*i+j] += A[3*i + k] * B[3*k + j];
      }
    }
  }
}

unsigned *setmolpos(unsigned ntypes, unsigned nmol, unsigned *nmols_per_type){

  unsigned *rlist = (unsigned*)malloc(sizeof(unsigned)*nmol);
  if ( rlist == NULL ) 
    merror("Couldn't allocate memory", __func__, __LINE__);

  unsigned i=0;
  for ( unsigned n=0; n<ntypes; n++ ){
    for ( unsigned m=0; m<nmols_per_type[n]; m++ ){
      rlist[i] = n;
      i++;
    } 
  }

  for ( unsigned n=0; n<10*nmol; n++ ){
    int a = nmol*(double)rand()/(RAND_MAX-1);
    int b = nmol*(double)rand()/(RAND_MAX-1);
    
    unsigned tmp = rlist[a];
    rlist[a] = rlist[b];
    rlist[b] = tmp;
  }

  return rlist;
}

void freemem(lattice *lptr, molecule *mptr, unsigned nmols, 
	     unsigned *rlist, float *masses){

  free(lptr->x); free(lptr->y); free(lptr->z);

  for ( unsigned n=0; n<nmols; n++ ){
    free(mptr[n].types); 
    free(mptr[n].masses); 
    free(mptr[n].rel_cm);
   
    free(mptr[n].bonds);
    free(mptr[n].angles);
    free(mptr[n].dihedrals);
  }

  free(rlist);
  free(masses);
}


void writexyzheader(unsigned natoms, float lbox, int numtypes, float* masses,
		    const char *str){
  
  ofstream fptr;
  fptr.open(str, ios::trunc);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__); 

  fptr << natoms << endl;
  fptr << "ioformat=2 sim_box=RectangularSimulationBox," << lbox << ',' << lbox << ',' << lbox << " numTypes=" << numtypes << " mass=";
  for (int m=0; m<numtypes-1; m++)
    fptr << masses[m] << ',';
  fptr << masses[numtypes-1];
  fptr << " columns=type,x,y,z,imx,imy,imz,vx,vy,vz" << endl;

  fptr.close();

}

bool setfilepointer(ifstream *fptr, string section) {
  string line;
  bool found = false;

  while (getline(*fptr, line)) {
    if (fptr->fail() || fptr->bad()) merror("Reading failure ", __func__, __LINE__);
    if (line.compare(section) == 0) {
      found = true;
      break;
    }
  }
  
  return found;
}


void readtop_bonds(molecule *mol, const char *fstr, unsigned molt){
  unsigned a, b, btype, dummy;
  string section = "[ bonds ]";
  string line; 
  char atomString[] = "atom";

  mol[molt].nbonds = 0;
  mol[molt].bonds = (unsigned*)malloc(3*sizeof(unsigned)*mol[molt].nbonds);

  if (strncmp(fstr, atomString, 4) != 0) {
    ifstream fptr(fstr);
    if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);

    bool sectionFound = setfilepointer(&fptr, section);

    if (sectionFound) {
      //read comment line
      getline(fptr, line);
      if (fptr.fail() || fptr.bad()) merror("Reading failure ", __func__, __LINE__);

      while(getline(fptr, line)) {
        if (fptr.fail() || fptr.bad()) {
          merror("Reading failure ", __func__, __LINE__);
          cout << "Failure" << endl;
        } else if (line.empty() || fptr.eof()) {
          break;

        } else {
          if (!(stringstream(line) >> dummy >> a >> b >> btype))
            merror("Reading failure",  __func__, __LINE__);
          else {
            (mol[molt].nbonds)++;

            mol[molt].bonds = (unsigned*)realloc(mol[molt].bonds,
                                      3*sizeof(unsigned)*mol[molt].nbonds);
            if ( mol[molt].bonds == NULL )
              merror("Memory allocation failure",  __func__, __LINE__);

            unsigned n = mol[molt].nbonds-1;
            mol[molt].bonds[3*n] = a;
            mol[molt].bonds[3*n+1] = b;
            mol[molt].bonds[3*n+2] = btype;

          }
        }
      }
    }

    fptr.close();
     
    cout << "Read bond information from " << fstr << ": Found " << mol[molt].nbonds << " bond(s)" << endl;
  }
 
}


void writebondheader(const char *fstr){
  
  ofstream fptr;
  fptr.open(fstr, ios::trunc);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
  
  fptr << "[ bonds ]" << endl;
  fptr << ";WARNING generated with rumd_init_conf_mol - NO WARRANTY" << endl;

  fptr.close();
}

void writebondssect(molecule *mol, unsigned molt, unsigned moli, const char *fstr){

  static unsigned atomi = 0; //ACHTUNG!!! 

  ofstream fptr;
  fptr.open(fstr, ios::app);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
 
  for ( unsigned n=0; n<mol[molt].nbonds; n++ ){
    unsigned a = mol[molt].bonds[3*n] + atomi;
    unsigned b = mol[molt].bonds[3*n+1]+ atomi;
    unsigned btype = mol[molt].bonds[3*n+2];

    fptr << moli << ' ' << a << ' ' << b << ' ' << btype << endl;
  }

  fptr.close();

  atomi += mol[molt].nuau;
}



void readtop_angles(molecule *mol, const char *fstr, unsigned molt){
  unsigned a, b, c, atype, dummy;
  string section = "[ angles ]";
  string line;
  char atomString[] = "atom";

  mol[molt].nangles = 0;
  mol[molt].angles = (unsigned*)malloc(4*sizeof(unsigned)*mol[molt].nangles);

  if (strncmp(fstr, atomString, 4) != 0) {
    ifstream fptr(fstr);
    if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);

    bool sectionFound = setfilepointer(&fptr, section);

    if (sectionFound) {
      //read comment line
      getline(fptr, line);
      if (fptr.fail() || fptr.bad()) merror("Reading failure ", __func__, __LINE__);

      while(getline(fptr, line)) {
        if (fptr.fail() || fptr.bad()) merror("Reading failure ", __func__, __LINE__);

        else if (line.empty() || fptr.eof()) {
          break;
        } else {
          if (!(stringstream(line) >> dummy >> a >> b >> c >> atype))
            merror("Reading failure",  __func__, __LINE__);
          else {
            (mol[molt].nangles)++;

            mol[molt].angles = (unsigned*)realloc(mol[molt].angles,
                                      4*sizeof(unsigned)*mol[molt].nangles);
            if ( mol[molt].angles == NULL )
              merror("Memory allocation failure", __func__, __LINE__);

            unsigned n = mol[molt].nangles-1;
            mol[molt].angles[4*n] = a;
            mol[molt].angles[4*n+1] = b;
            mol[molt].angles[4*n+2] = c;
            mol[molt].angles[4*n+3] = atype;

          }
        }
      }
    }

    cout << "Read angle information from " << fstr << ": Found " << mol[molt].nangles << " angles(s)" << endl;

    fptr.close();
  }
    
}




void writeangleheader(const char *fstr){
  
  ofstream fptr;
  fptr.open(fstr, ios::trunc);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
  
  fptr << "[ angles ]" << endl;
  fptr << ";WARNING generated with rumd_init_conf_mol - NO WARRANTY" << endl;

  fptr.close();
}


void writeanglesect(molecule *mol, unsigned molt, unsigned moli, const char *fstr){

  static unsigned atomi = 0; //ACHTUNG!!! 

  ofstream fptr;
  fptr.open(fstr, ios::app);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
  
  for ( unsigned n=0; n<mol[molt].nangles; n++ ){
    unsigned a = mol[molt].angles[4*n] + atomi;
    unsigned b = mol[molt].angles[4*n+1]+ atomi;
    unsigned c = mol[molt].angles[4*n+2]+ atomi;
    unsigned atype = mol[molt].angles[4*n+3];

    fptr << moli << ' ' << a << ' ' <<  b << ' ' <<  c << ' ' <<  atype << endl;
  }

  fptr.close();

  atomi += mol[molt].nuau;
}



void readtop_dihedrals(molecule *mol, const char *fstr, unsigned molt){
  unsigned a, b, c, d, dtype, dummy;
  string section = "[ dihedrals ]";
  string line;
  char atomString[] = "atom";

  mol[molt].ndihedrals = 0;
  mol[molt].dihedrals = (unsigned*)malloc(5*sizeof(unsigned)*mol[molt].ndihedrals);

  if (strncmp(fstr, atomString, 4) != 0) {
    ifstream fptr(fstr);
    if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);

    bool sectionFound = setfilepointer(&fptr, section);

    if (sectionFound) {
      //read comment line
      getline(fptr, line);
      if (fptr.fail() || fptr.bad()) merror("Reading failure ", __func__, __LINE__);

      while(getline(fptr, line)) {
        if (fptr.fail() || fptr.bad()) merror("Reading failure ", __func__, __LINE__);

        else if (line.empty() || fptr.eof()) {
          break;

        } else {
          if (!(stringstream(line) >> dummy >> a >> b >> c >> d >> dtype))
            merror("Reading failure",  __func__, __LINE__);
          else {
            (mol[molt].ndihedrals)++;

            mol[molt].dihedrals = (unsigned*)realloc(mol[molt].dihedrals, 
                                    5*sizeof(unsigned)*mol[molt].ndihedrals);
            if ( mol[molt].dihedrals == NULL ) 
              merror("Memory allocation failure", __func__, __LINE__);
            
            unsigned n = mol[molt].ndihedrals-1;
            mol[molt].dihedrals[5*n] = a; 
            mol[molt].dihedrals[5*n+1] = b; 
            mol[molt].dihedrals[5*n+2] = c; 
            mol[molt].dihedrals[5*n+3] = d; 
            mol[molt].dihedrals[5*n+4] = dtype;

          }
        }
      }
    }

    fptr.close();

    cout << "Read dihedrals information from " << fstr << ": Found " << mol[molt].ndihedrals << " dihedral(s)" << endl;
  }
 
}


void writedihedralheader(const char *fstr){
  
  ofstream fptr;
  fptr.open(fstr, ios::trunc);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
  
  fptr << "[ dihedrals ]" << endl;
  fptr << ";WARNING generated with rumd_init_conf_mol - NO WARRANTY" << endl;

  fptr.close();
}

void writedihedralsect(molecule *mol, unsigned molt, unsigned moli, const char *fstr){

  static unsigned atomi = 0; //ACHTUNG!!! 

  ofstream fptr;
  fptr.open(fstr, ios::app);
  if (!fptr.is_open()) merror("Couldn't open file",  __func__, __LINE__);
 
  for ( unsigned n=0; n<mol[molt].ndihedrals; n++ ){
    unsigned a = mol[molt].dihedrals[5*n] + atomi;
    unsigned b = mol[molt].dihedrals[5*n+1] + atomi;
    unsigned c = mol[molt].dihedrals[5*n+2] + atomi;
    unsigned d = mol[molt].dihedrals[5*n+3] + atomi;
    unsigned dtype = mol[molt].dihedrals[5*n+4];

    fptr <<  moli << ' ' << a << ' ' << b << ' ' << c << ' ' << d << ' ' << dtype << endl;
  }

  fptr.close();

  atomi += mol[molt].nuau;
}

void pipe( molecule *mol, int ntypes ){

  int bonds = 0;  int ang = 0;  int dihed = 0;
  for ( int n=0; n<ntypes; n++ ){
    bonds += mol[n].nbonds;
    ang += mol[n].nangles;
    dihed += mol[n].ndihedrals;
  }

  (void)system("rm -f start.top");

  if ( bonds > 0 ){
    (void)system("cat __bonds.top >> start.top");
    (void)system("echo ""  >> start.top");
  }
  if ( ang > 0 ) {
    (void)system("cat __angles.top >> start.top");
    (void)system("echo ""  >> start.top");
  }
  if ( dihed > 0 ){
    (void)system("cat __dihedrals.top >> start.top");
    (void)system("echo ""  >> start.top");
  }

}


float velocity(float temp){

  return 2.0*temp*((float)rand()/(RAND_MAX - 1) - 0.5);
    
}


int findnumtypes(molecule *mol, int ntypes_mol){

  int type = -1;
  for ( int n=0; n<ntypes_mol; n++ ){
    for ( unsigned m=0; m<mol[n].nuau; m++ )
      if ( (signed)mol[n].types[m] > type ) type = mol[n].types[m];
  }

  return type + 1;

}

void findatommasses(molecule *mol, int ntypes_mol, float *masses, int ntypes_atom) {

  for (int a=0; a<ntypes_atom; a++)
    masses[a] = 1.0;
  for ( int n=0; n<ntypes_mol; n++ ){
    for ( unsigned m=0; m<mol[n].nuau; m++ )
      if (masses[mol[n].types[m]] > 1.0 - EPSILON && masses[mol[n].types[m]] < 1.0 + EPSILON)
        masses[mol[n].types[m]] = mol[n].masses[m];
      else { 
        if (((masses[mol[n].types[m]] > mol[n].masses[m] + EPSILON) || (masses[mol[n].types[m]] < mol[n].masses[m] - EPSILON)) && ((mol[n].masses[m] < 1.0 - EPSILON) || (mol[n].masses[m] > 1.0 + EPSILON))) {
          merror("Inconsistent masses",  __func__, __LINE__);
        }
      }
  }

}


void printpre(int nmol_tot, int ntypes, float dens, float temp, int randomOrientation, int seed, float userSpacing ){

  cout << "This is rumd_init_conf_mol" << endl;
  cout << "This program is a part of RUMD - there is absolutely ";
  cout << "NO WARRANTY" << endl << endl;


  cout << "From arguments: Number of molecules: " << nmol_tot << " [" << ntypes << " type(s)]" << endl;
  cout << "                seed = " << seed << endl;
  cout << "                temperature = " << temp << endl;
  if (dens > EPSILON) cout << "                density = " << dens << endl;
  if (userSpacing > EPSILON) cout << "                spacing = " << dens << endl;
  if (userSpacing < EPSILON && dens < EPSILON) cout << "                density = default" << endl;
  if (randomOrientation) cout << "                molecules orientation to be randomized" << endl;
  else cout << "                molecules orientation not to be randomized" << endl;

}

void  printinsect(int n, int ntypes){
  
  cout << endl << "------------[info] Type " << (n+1) << " of " << ntypes << "------------" << endl;

}

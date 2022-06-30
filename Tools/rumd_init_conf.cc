#include "rumd_init_conf.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

#include <boost/iostreams/filtering_stream.hpp>   // Linker -lboost_iostreams
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include "split.h"    // Split a string into a vector<string>

using namespace std;

/** Constructor class for rumd_init_conf
The constructor class can be given parametes for setting up a basic lattice. 
\param latticeType Set the type of lattice, e.g. sc, fcc or bcc.
\param nx,ny,nz Number of unit-cells in the x, y and z dorection
\param seed Seed for pseudo-random velocites.
*/
rumd_init_conf::rumd_init_conf(string latticeType,unsigned nx, unsigned ny,unsigned nz,unsigned seed) 
:
	lattice_type(latticeType),
	number_of_sites_on_lattice(0),
	type(),
	x(),
	y(),
	z(),
	Lx(nx),
	Ly(ny),
	Lz(nz),
	mass_of_types()
{
	srand(seed);

	// Load a rumd_init_conf from a file if *.xyz or *.xyz.gz is given as the input rumd_init_conf type
	vector<string> fnames = split(latticeType,'.');
	if (fnames.size()>1) {
		// Setup input stream from file
		using namespace boost::iostreams;
		filtering_istream in;
		if(fnames.back()=="xyz"){
			in.push(file_source(latticeType));
		} else if ( fnames.back()=="gz" && (fnames.at(fnames.size()-2)=="xyz" ) ) {
			in.push(gzip_decompressor());
			in.push(file_source(latticeType));
		} else {
			cerr << "error: Unknown name of input file for rumd_init_conf coordinates. Should be an *.xyz, *.xyz.gz or file." << endl;
			abort();
		}
		if(!in.good()){
			cerr << "error: while reading " << latticeType << endl;
			abort();
		}
		string line;

		// Read meta information in header
		getline(in,line); // Number of atoms
		unsigned num_atoms=atoi(line.c_str());
		getline(in,line); // Comment line

		// Find box vectors in comment line
		Lx=1.0;
		Ly=1.0;
		Lz=1.0;
		vector<string> sections = split(line,' ');
		bool read_sim_box = false;
		for(unsigned i = 0 ; i < sections.size() ; i++){
			vector<string> elements = split(sections.at(i),'=');
			if(elements.size()>0 && elements.at(0)=="sim_box" && elements.size()==2){
				vector<string> vars = split(elements.at(1),',');
				if(elements.size()>0 && vars.at(0)=="RectangularSimulationBox" && vars.size()==4){
					Lx = atof(vars.at(1).c_str());
					Ly = atof(vars.at(2).c_str());
					Lz = atof(vars.at(3).c_str());
					read_sim_box = true;
				}
			}
		}
		if(!read_sim_box) cout << "Warning: The file " << latticeType 
			<< " did not contain a sim_box=RectangularSimulationBox,NUM,NUM,NUM in the header. Using default  sim_box=RectangularSimulationBox,1.0,1.0,1.0."
			<< endl;

		// Read atom position
		vector<int> type;
		vector<float> pos_x;
		vector<float> pos_y;
		vector<float> pos_z;
		for(unsigned i=0;i<num_atoms;i++){
			if(!in.good()){
				cerr << "error: while reading " << latticeType << endl;
				abort();
			}
			getline(in,line);
			char * pEnd;
			int    i0;
			double d0, d1, d2;
			i0 = strtol (line.c_str(),&pEnd,10);
			d0 = strtod (pEnd,&pEnd);
			d1 = strtod (pEnd,&pEnd);
			d2 = strtod (pEnd,&pEnd);
			type.push_back(i0);
			pos_x.push_back(d0);
			pos_y.push_back(d1);
			pos_z.push_back(d2);
			/*for(unsigned ix=0;ix<nx;ix++){
				for(unsigned iy=0;iy<ny;iy++){
					for(unsigned iz=0;iz<nz;iz++){
						addParticle(i0,ix*Lx+d0,iy*Ly+d1,iz*Lz+d2);
					}
				}
			}*/
		}
		
		// Add particles
		for(unsigned ix=0; ix<nx; ix++)
		  for(unsigned iy=0; iy<ny; iy++)
			  for(unsigned iz=0; iz<nz; iz++)
				  for(unsigned ip=0; ip<type.size(); ip++)
				    addParticle(type[ip],ix*Lx+pos_x[ip],iy*Ly+pos_y[ip],iz*Lz+pos_z[ip]);
					  

		// Reset to new box size
		Lx*=nx;
		Ly*=ny;
		Lz*=nz;
	} else if(lattice_type=="sc"){			// Simple cubic
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.5,iy+0.5,iz+0.5);
				}
			}
		}
	}
	else if(lattice_type=="rp"){	// Randomly packed
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,Lx*(double)rand()/(double)RAND_MAX,Lx*(double)rand()/(double)RAND_MAX,Lx*(double)rand()/(double)RAND_MAX);
					//addParticle(0,ix,iy,iz);
				}
			}
		}
		// TODO Below MC simulation of hard spheres scales badly (N**2). Can be fixed with a cell list.
		for(double size=0.0;size<1.1;size+=0.001){
			for(unsigned n=0;n<x.size();n++){
				double x_old=x.at(n);
				double y_old=y.at(n);
				double z_old=z.at(n);
				double x_move=0.1*((double)rand()/(double)RAND_MAX-0.5);
				double y_move=0.1*((double)rand()/(double)RAND_MAX-0.5);
				double z_move=0.1*((double)rand()/(double)RAND_MAX-0.5);
				x.at(n)+=x_move;
				y.at(n)+=y_move;
				z.at(n)+=z_move;
				x.at(n)-=floor(x.at(n)/Lx)*Lx;
				y.at(n)-=floor(y.at(n)/Ly)*Ly;
				z.at(n)-=floor(z.at(n)/Lz)*Lz;
				bool overlap = false;
				for(unsigned m=0;m<x.size();m++){
					if(n!=m){
						double xx=x.at(n)-x.at(m);
						xx-=round(xx/Lx)*Lx;xx*=xx;
						double yy=y.at(n)-y.at(m);
						yy-=round(yy/Ly)*Ly;yy*=yy;
						double zz=z.at(n)-z.at(m);
						zz-=round(zz/Lz)*Lz;zz*=zz;
						if(xx+yy+zz<size*size)
							overlap = true;
					}
				}
				if(overlap){
					x.at(n)=x_old;
					y.at(n)=y_old;
					z.at(n)=z_old;
				}
			}
		}
	} else if(lattice_type=="bcc"){	// Body centre cubic
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.0,iy+0.0,iz+0.0);
					addParticle(0,ix+0.5,iy+0.5,iz+0.5);
				}
			}
		}
		set_density(1.0);		
	} else if (lattice_type=="fcc") {	// Face centre cubic
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					//addParticle(0,ix+0.0,iy+0.0,iz+0.0);
					//addParticle(0,ix+0.5,iy+0.5,iz+0.0);
					//addParticle(0,ix+0.5,iy+0.0,iz+0.5);
					//addParticle(0,ix+0.0,iy+0.5,iz+0.5);
					addParticle(0,ix+0.25,iy+0.25,iz+0.25);
					addParticle(0,ix+0.75,iy+0.75,iz+0.25);
					addParticle(0,ix+0.75,iy+0.25,iz+0.75);
					addParticle(0,ix+0.25,iy+0.75,iz+0.75);
				}
			}
		}
		set_density(1.0);
	} else if (lattice_type=="hcp") {	// Hexagonal close packed
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.0,iy+0.0,iz+0.0);
					addParticle(0,ix+0.5,iy+0.5,iz+0.0);
					addParticle(0,ix+0.5,iy+5.0/6.0,iz+0.5);
					addParticle(0,ix+0.0,iy+1.0/3.0,iz+0.5);
				}
			}
		}
		scale_y_coordinates(ny*sqrt(3.0));
		scale_z_coordinates(nz*sqrt(8.0/3.0));
		set_density(1.0);		
	} else if (lattice_type=="hex") {	// Hexagonal layers
		for(unsigned ix=0;ix<nx;ix++) {
			for(unsigned iy=0;iy<ny;iy++) {
				for(unsigned iz=0;iz<nz;iz++) {
					addParticle(0,ix+0.0,iy+0.0,iz+0.0);
					addParticle(0,ix+0.5,iy+0.5,iz+0.0);
				}
			}
		}
		scale_y_coordinates(ny*sqrt(3.0));
		set_density(1.0);		
	} else if (lattice_type=="dc") {	// Diamond cubic rumd_init_conf
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.00,iy+0.00,iz+0.00);
					addParticle(0,ix+0.00,iy+0.50,iz+0.50);
					addParticle(0,ix+0.50,iy+0.00,iz+0.50);
					addParticle(0,ix+0.50,iy+0.50,iz+0.00);
					addParticle(0,ix+0.75,iy+0.75,iz+0.75);
					addParticle(0,ix+0.75,iy+0.25,iz+0.25);
					addParticle(0,ix+0.25,iy+0.75,iz+0.25);
					addParticle(0,ix+0.25,iy+0.25,iz+0.75);
				}
			}
		}
  	set_density(1.0);
	} else if (lattice_type=="NaCl") {	// Rock salt rumd_init_conf
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.0,iy+0.0,iz+0.0);
					addParticle(0,ix+0.5,iy+0.5,iz+0.0);
					addParticle(0,ix+0.5,iy+0.0,iz+0.5);
					addParticle(0,ix+0.0,iy+0.5,iz+0.5);
					addParticle(1,ix+0.5,iy+0.5,iz+0.5);
					addParticle(1,ix+0.0,iy+0.0,iz+0.5);
					addParticle(1,ix+0.0,iy+0.5,iz+0.0);
					addParticle(1,ix+0.5,iy+0.0,iz+0.0);
				}
			}
		}
		set_density(1.0);		
	} else if (lattice_type=="CsCl") {	// Caesium Chloride rumd_init_conf (AuCd or TiNi)
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.00,iy+0.00,iz+0.00);
					addParticle(1,ix+0.50,iy+0.50,iz+0.50);
				}
			}
		}
		set_density(1.0);		
	} else if (lattice_type=="CuZr2") {
		// Mail from Toby Hudson to Ulf R. Pedersen:
		// CuZr2 structure type optimized at rA/rB = 6.0/5.0 (NB - rB<rA) Lx=1.134695 Ly=1.134695 Lz=5.04776
		// A 0.000000 0.000000 0.000000
		// A 0.567348 0.567348 2.523880
		// B 0.000000 0.000000 1.738161
		// B 0.000000 0.000000 3.309599
		// B 0.567348 0.567348 0.785719
		// B 0.567348 0.567348 4.262041
		//    MoSi2, C11b type structure. Pearson symbol is tI6. Space group is I4/mmm.
		double sx=1.134695;double sy=1.134695;double sz=5.04776;
		for(unsigned ix=0;ix<nx;ix++){
			for(unsigned iy=0;iy<ny;iy++){
				for(unsigned iz=0;iz<nz;iz++){
					addParticle(0,ix+0.000000/sx,iy+0.000000/sy,iz+0.000000/sz);
					addParticle(0,ix+0.567348/sx,iy+0.567348/sy,iz+2.523880/sz);
					addParticle(1,ix+0.000000/sx,iy+0.000000/sy,iz+1.738161/sz);
					addParticle(1,ix+0.000000/sx,iy+0.000000/sy,iz+3.309599/sz);
					addParticle(1,ix+0.567348/sx,iy+0.567348/sy,iz+0.785719/sz);
					addParticle(1,ix+0.567348/sx,iy+0.567348/sy,iz+4.262041/sz);
				}
			}
		}
		scale_x_coordinates(ny*sx);scale_z_coordinates(ny*sy);scale_z_coordinates(nz*sz);
	}
	else {
		cerr << "error: unknown rumd_init_conf type " << lattice_type << "." << endl;
		abort();
	}
	
	for(unsigned i=0;i<number_of_types();i++)
		mass_of_types.push_back(1.0);
	// translate rumd_init_conf to origin at center of box
	translate_all_particles(-0.5*Lx,-0.5*Ly,-0.5*Lz);
	number_of_sites_on_lattice=number_of_particles();
}

rumd_init_conf::~rumd_init_conf() {
	type.clear();
	mass_of_types.clear();
	x.clear();
	y.clear();
	z.clear();
}

/**
 * Return random component to velocity vector 
 *   using Marsaglia's polar transform to get Gauss distributed
 *   random numbers (aka "the polar Box–Muller transform").
 */
double rumd_init_conf::random_velocity(double temperature,double mass){
  double u,v,s = 2.0;
  do {
    u = 2.0*(double)rand()/(double)RAND_MAX-1.0;
    v = 2.0*(double)rand()/(double)RAND_MAX-1.0;
    s = u*u + v*v;
  } while ( s >= 1.0 || s == 0.0 );
	return v*sqrt(-2.0*log(s)/s)*sqrt(temperature/mass);
}

/**
 * Add a particle of in_type to the rumd_init_conf
 */
void rumd_init_conf::addParticle(unsigned in_type,double in_x,double in_y,double in_z){
	type.push_back(in_type);
	x.push_back(in_x);
	y.push_back(in_y);
	z.push_back(in_z);
}

/**
 * Randomly swop position of particles
 */
void rumd_init_conf::mix_positions(){
	for(unsigned i=0;i<x.size();i++){
		double this_x=x.at(i);
		double this_y=y.at(i);
		double this_z=z.at(i);
		unsigned j=rand()*x.size()/RAND_MAX;
		x.at(i)=x.at(j);
		y.at(i)=y.at(j);
		z.at(i)=z.at(j);
		x.at(j)=this_x;
		y.at(j)=this_y;
		z.at(j)=this_z;
	}
}

/**
 * Reset the density of the rumd_init_conf
 * \param new_density New density N/V.
 */
void rumd_init_conf::set_density(double new_density){
	double sf=pow( (number_of_particles()/volume())/new_density , 1.0/3.0 );
	scale_coordinates(sf);
}

void rumd_init_conf::scale_coordinates(double sf){
	Lx=sf*Lx;
	Ly=sf*Ly;
	Lz=sf*Lz;
	for (unsigned i=0;i<number_of_particles();i++) {
		x.at(i)=sf*x.at(i);
		y.at(i)=sf*y.at(i);
		z.at(i)=sf*z.at(i);
	}
}

void rumd_init_conf::scale_x_coordinates(double new_Lx){
	double sf=new_Lx/Lx;
	Lx=sf*Lx;
	for (unsigned i=0;i<number_of_particles();i++) {
		x.at(i)=sf*x.at(i);
	}
}

void rumd_init_conf::scale_y_coordinates(double new_Ly){
	double sf=new_Ly/Ly;
	Ly=sf*Ly;
	for (unsigned i=0;i<number_of_particles();i++) {
		y.at(i)=sf*y.at(i);
	}
}

void rumd_init_conf::scale_z_coordinates(double new_Lz){
	double sf=new_Lz/Lz;
	Lz=sf*Lz;
	for (unsigned i=0;i<number_of_particles();i++) {
		z.at(i)=sf*z.at(i);
	}
}

void rumd_init_conf::translate_all_particles(double dx,double dy,double dz){
	for(unsigned n=0;n<x.size();n++){
		x.at(n)+=dx;
		y.at(n)+=dy;
		z.at(n)+=dz;
	}
}

void rumd_init_conf::reset_particle_types(vector<unsigned> num_par){
	mass_of_types.clear();
	// Rename particle types
	unsigned c=0;
	for(unsigned i=0;i<num_par.size();i++){
		mass_of_types.push_back(1.0);
		for(unsigned n=0;n<num_par.at(i);n++)
			if(reset_particle_type(c,i))
				c++;
			else {
				cerr<<"error: Could not reset particle types."<<endl;
				abort();
			}
	}
	// Remove extra particles
	reset_number_of_particles(c);
}

void rumd_init_conf::reset_mass_of_types(vector<double> mass){
	mass_of_types.clear();
	for(unsigned i=0;i<mass.size();i++)
		mass_of_types.push_back(mass.at(i));
	if(number_of_types()!=mass_of_types.size()) {
		cerr << "error: number of masses given is not the same as number of types." << endl;
		abort();
	}
}

/**
 * Reset the type of a particle. 
 * \return False if unsuccessful.
 */
bool rumd_init_conf::reset_particle_type(unsigned i,unsigned new_type){
	bool output=false;
	if(i<type.size()){
		type.at(i)=new_type;
		output=true;
	}
	return output;
}


void rumd_init_conf::reset_number_of_particles(unsigned num_par) {
	while(x.size()>num_par){
		type.pop_back();
		x.pop_back();
		y.pop_back();
		z.pop_back();
	}
}

/**
 * Return the minimum distance found in rumd_init_conf.
 * TODO The get_min_distance() function scales badly, and makes the program slow for large systems.
 * \return The minimum distance found in rumd_init_conf
 */
double rumd_init_conf::get_min_distance(){
	cout << "get_min_distance()" << endl;
	double min_distance=Lx+Lz+Lz;
	for (unsigned i=0;i<number_of_particles()-1;i++) {
		for (unsigned j=i+1;j<number_of_particles();j++) {
			double xx = x.at(i)-x.at(j);
			xx-=round(xx/Lx)*Lx;
			xx*=xx;
			double yy = y.at(i)-y.at(j);
			yy-=round(yy/Ly)*Ly;
			yy*=yy;
			double zz = z.at(i)-z.at(j);
			zz-=round(zz/Lz)*Lz;
			zz*=zz;
			if(xx+yy+zz<min_distance*min_distance){
				min_distance = sqrt(xx+yy+zz);
			}
		}
	}
	return min_distance;
}

/**
 * Scale coordinates so that the minimum distance is given by new_min_distance
 */
void rumd_init_conf::set_min_distance(double new_min_distance){
	scale_coordinates(new_min_distance/get_min_distance());
}

/**
 * Return the number of particles
 */
unsigned rumd_init_conf::number_of_particles(){
	return x.size();
}


/**
 * Return the number of particles types
 */
unsigned rumd_init_conf::number_of_types(){
	unsigned out=0;
	if( type.size()>0 )
		for(unsigned i=0;i<type.size();i++)
			if(type.at(i)>=out) out=type.at(i)+1;
	return out;
}

unsigned rumd_init_conf::number_of_particles_of_type(unsigned test_type){
	unsigned out=0;
	for(unsigned n=0;n<type.size();n++)
		if(type.at(n)==test_type)
			out++;
	return out;
}

/**
 * Return volume of rumd_init_conf.
 * \return Volume, V=Lx*Ly*Lz.
 */
double rumd_init_conf::volume(){
	return Lx*Ly*Lz;
}


/**
 * Write coordinates of particles to xyz-file.
 * \param out Output stream for xyz-file.
 * \param temperature Temperature for the velocities.
 */
void rumd_init_conf::write_xyz(ostream& out,double temperature){

	//stringstream out;
	out << this->number_of_particles()<<endl;
	out << "ioformat=2 numTypes=" << number_of_types();
	out << setprecision(16);
	out << " sim_box=RectangularSimulationBox," << Lx << "," << Ly << "," << Lz;
	out << " mass=";
	for(unsigned i=0;i<mass_of_types.size();i++){
		if(i<mass_of_types.size()-1)
			out << mass_of_types.at(i) << ",";
		else
			out << mass_of_types.at(i);
	}
	out << " columns=type,x,y,z,imx,imy,imz,vx,vy,vz " << endl;
	for (unsigned i=0;i<number_of_particles();i++) {
		out << type.at(i) << " " << x.at(i) << " " << y.at(i) << " " << z.at(i);
		out << " 0 0 0";
		// for(unsigned i=0;i<3;i++)
		if(Lx>0) {
			out << " " << random_velocity(temperature,mass_of_types.at(type.at(i)));
		}else{
			out << " 0.0";
		}
		if(Ly>0) {
			out << " " << random_velocity(temperature,mass_of_types.at(type.at(i)));
		}else{
			out << " 0.0";
		}
		if(Lz>0) {
			out << " " << random_velocity(temperature,mass_of_types.at(type.at(i)));
		}else{
			out << " 0.0";
		}
		out << endl;
	}
}

/** 
* Write a topology file
*/
void rumd_init_conf::write_top(string input_file_name, string output_file_name, unsigned int number_of_unit_cells){
    // Output file
    using namespace boost::iostreams;
		filtering_ostream out;
    out.push(file_sink(output_file_name));    
    
 		// Setup input stream from file
 		vector<string> fnames = split(input_file_name,'.');
		using namespace boost::iostreams;
		filtering_istream in;
		if(fnames.back()=="top"){
			in.push(file_source(input_file_name));
		} else {
			cerr << "error: Unknown name of input file for rumd_init_conf topologt. Should be an *.top file." << endl;
			abort();
		}
		if(!in.good()){
			cerr << "error: while reading " << input_file_name << endl;
			abort();
		}
		string line;

		// Read meta information in header		
		while(in.good()){
		  getline(in,line);
		  string section = line.c_str();
		  vector<string> line_split = split(line, ' ');
		  if(section==""){
		     // Found blank line
		     //cout << "BLANK LINE   " << line << endl;
		  }else if(line_split[0]==";"){
		     // Found comment Line
		     //cout << "COMMENT LINE   " << line << endl;		     
		  }else if(section=="[ bonds ]"){
       // Variables for topology information of one unit cell
		    vector<unsigned> molecule;
		    vector<unsigned> particle_A;
		    vector<unsigned> particle_B;
		    vector<unsigned> bond_type;		  
		    
		    getline(in,line); // Comment line
		    bool found_end_of_section = false;
		    while(in.good() and not found_end_of_section){
		      getline(in,line);
		      //cout << line << endl;
		      vector<string> columns = split(line, ' ');
		      if(columns.size()==4){
		        molecule.push_back(stoi(columns[0].c_str()));
		        particle_A.push_back(stoi(columns[1].c_str()));
		        particle_B.push_back(stoi(columns[2].c_str()));
		        bond_type.push_back(stoi(columns[3].c_str()));
		      }else{
		        found_end_of_section = true;
		      }
		    }
	      // Write Bonds section
        unsigned largest_molecule_id_in_unit_cell = 0;
        for(unsigned i = 0; i<molecule.size(); i++)
          if(molecule[i]>largest_molecule_id_in_unit_cell)
            largest_molecule_id_in_unit_cell = molecule[i];
        
        out << "[ bonds ]" << endl;
        out << "; mol index | atom i | atom j | bond type" << endl;
        
        unsigned N = molecule.size();
        for(unsigned n = 0; n<number_of_unit_cells; n++){
          for(unsigned i = 0; i<N; i++){
            out << molecule[i]+n*(largest_molecule_id_in_unit_cell+1) << " ";
            out << particle_A[i]+n*N << " ";
            out << particle_B[i]+n*N << " ";
            out << bond_type[i] << endl;
          }
        }
	    }else if(line_split[0]=="["){
	       cout << "Warning: Found unknown section " << line << " in " << input_file_name << endl;
 	       cout << "         Only [ bonds ] section is written to " << output_file_name << endl;
	    }
   }
}

/**
 * Print information.
 * \return String with various information about the object.
 */
string rumd_init_conf::info(){
	stringstream out;
	out << "Lattice type:                 " << lattice_type << endl;
	out << "Number of lattice sites:      " << number_of_sites_on_lattice << endl;
	out << "Total number of particles:    " << number_of_particles() << endl;
	out << "Number of types:              " << number_of_types() << endl;
	out << "Number of particles of types:";
	for(unsigned i=0;i<number_of_types();i++)
		out << " " << number_of_particles_of_type(i);
	out << endl;
	out << "Mass of types:               ";
	for(unsigned i=0;i<mass_of_types.size();i++)
		out << " " << mass_of_types.at(i);
	out << endl;
	out << "Lengths of box vectors:       " << Lx << " " << Ly << " " << Lz << endl;
	out << "Box volume:                   " << volume() << endl;
	out << "Number density:               " << number_of_particles()/volume() << endl;
	// out << "Shortest distance in rumd_init_conf: " << get_min_distance() << endl;  // TODO A call to get_min_distance() is removed since it scales badly
	return out.str();
}

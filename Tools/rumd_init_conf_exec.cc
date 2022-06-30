//============================================================================
// Name        : rumd_init_conf.cc
// Author      : Ulf R. Pedersen
// Build       : g++ -O3 rumd_init_conf.h rumd_init_conf.cc rumd_init_conf.cc -lboost_iostreams -o rumd_rumd_init_conf
//============================================================================

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cfloat>
#include <unistd.h>
#include <getopt.h>
#include <vector>
#include <string>
#include <sstream>

#include <boost/iostreams/filtering_stream.hpp>   // Linker -lboost_iostreams
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include "rumd_init_conf.h"

using namespace std;

/**
 * Split a string into a vector string
 */
vector<string> split(string str, char delimiter) {
  vector<string> output;stringstream ss(str);string substr;
  while(getline(ss, substr, delimiter)) output.push_back(substr);
  return output;
}


int main(int argc, char **argv) {

	// Set default values
	string lattice_type = "sc";
	unsigned nx=5;
	unsigned ny=nx;
	unsigned nz=nx;
	vector<unsigned> num_par;
	vector<double> mass;
	bool mix_positions=false;
	bool quiet = false;
	string filename = "start.xyz.gz";
	double density=-1.0;  // Density is not set when negative
	double min_distance=-1.0;
	double Lx=-1.0;
	double Ly=-1.0;
	double Lz=-1.0;
	double temperature=1.0;
	string topology_file = "none.top";
	bool write_topology_file = false;
	unsigned seed=0;

	// Handle command line options
	vector<string> vecstr;
	int c;
	while(1) {
		static struct option long_options[] = {
				{"help",	no_argument, 0, 'h'},
				{"quiet",	no_argument, 0, 'q'},
				{"lattice",	optional_argument, 0, 'l'},
				{"cells",	optional_argument, 0, 'c'},
				{"rho",	optional_argument, 0, 'r'},
				{"minimum_distance", optional_argument, 0, 'd'},
				{"length", optional_argument, 0, 'L'},
				{"num_par", optional_argument, 0, 'N'},
				{"mass", optional_argument, 0, 'u'},
				{"mix_positions", no_argument, 0, 'm'},
				{"temperature", optional_argument, 0, 'T'},
				{"seed", optional_argument, 0, 's'},
				{"topology", optional_argument, 0, 't'},
				{"output",	optional_argument, 0, 'o'},
				{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long (argc, argv, "hql:c:N:u:r:d:L:mT:s:t:o:",long_options,&option_index);
		if(c==-1) break;
		switch (c) {
		/*case 0:
			if (long_options[option_index].flag != 0) break;
			printf ("option %s", long_options[option_index].name);
			if (optarg) printf (" with arg %s", optarg);
			printf ("\n");
			break;*/
		case 'h':
			//cout << endl;
			cout << "Make a configuration used for input to RUMD (www.rumd.org)." << endl << endl;
			cout << "Usage examples:" << endl;
			cout << argv[0] << " --lattice=fcc --cells=6,6,6" << endl;
			cout << argv[0] << " -lfcc -c6,6,6" << endl;
			cout << argv[0] << " -c10 -N800,200 -r1.2 -m -o KobAndersen.xyz.gz" << endl << endl;
			cout << "Optional flags:" << endl;
			cout << " -h, --help                    Print information on using this program." << endl;
			cout << " -q, --quiet                   Hide program output." << endl;
			cout << " -l, --lattice=FILE            Load unitcell from *.xyz or *.xyz.gz file." << endl;
			cout << " -l, --lattice=STR             Set lattice type. Default: sc." << endl;
			cout << "                               Allowed values (n=\"particles in unit cell\"):" << endl;
			cout << "                                 sc     Simple cubic (n=1)." << endl;
			cout << "                                 rp     Randomly packed (n=1). Warning: Slow for large systems." << endl;
			cout << "                                 bcc    Body centered cubic (n=2)." << endl;
			cout << "                                 fcc    Face centered cubic (n=4)." << endl;
			cout << "                                 hcp    Hexagonal close packed (n=4)." << endl;
			cout << "                                 hex    Hexagonal layers in xy-planes (n=2)." << endl;
			cout << "                                 dc     Diamond cubic lattice (n=8)." << endl;
			cout << "                                 NaCl   Rock salt lattice. (n=2x4)" << endl;
			cout << "                                 CsCl   Caesium Chloride lattice (n=2x1)." << endl;
			cout << " -c, --cells=INT               Set number of unit cells. Default: 5." << endl;
			cout << "     --cells=INT,INT,INT       " << endl;
			cout << " -N, --num_par=INT,INT,...     Reset particle types." << endl;
			cout << "                                 Note: The sum of particles must be < or = munber of lattice sites." << endl;
			cout << " -u, --mass=NUM,NUM,...        Set masses of types. Default: 1." << endl;
			cout << " -m, --mix_positions           Make positions of particles on lattice sites random." << endl;
			cout << " -r, --rho=NUM                 Set number density. Default: Use input density." << endl;
			cout << " -L, --length=NUM              Change length of box vectors. Skip resetting when <0."<<endl;
			cout << "     --length=NUM,NUM,NUM        Set one value =0 for a 2D configuration." << endl;
			cout << " -d, --minimum_distance=NUM    Do not allow distances shorter than min_dist. Warning: Slow for large systems." << endl;
			cout << " -T, --temperature=NUM         Temperature of random velocity vectors." << endl;
			cout << " -s, --seed=INT                Seed for pseudo random numbers." << endl;
			cout << " -t, --topology=FILE           Input topology file *.top if input structure is a molecule. Writes mol.top file." << endl;
  		cout << "                               Only the [ bonds ] section is written." << endl;
			cout << " -o, --output=FILE             *.xyz or *.xyz.gz output file." << endl << endl;
			exit(0);
			break;
		case 'q':
			quiet=true;
			break;
		case 'l':
			lattice_type = optarg;
			break;
		case 'c':
			vecstr = split(optarg,',');
			if(vecstr.empty()){
				cerr << "error: unknown input for -c, --cells.\nTry -h or --help for more information." << endl;
				abort();
			}
			if( vecstr.size()==1 ) {
				nx = atoi(optarg);
				ny = nx;
				nz = nx;
			} else if ( vecstr.size()==3 ) {
				nx = atoi(vecstr.at(0).c_str());
				ny = atoi(vecstr.at(1).c_str());
				nz = atoi(vecstr.at(2).c_str());
			} else {
				cerr << "error: unknown input for -c, --cells.\nTry -h or --help for more information." << endl;
				abort();
			}
			break;
		case 'r':
			density=atof(optarg);
			break;
		case 'd':
			min_distance=atof(optarg);
			break;
		case 'L':
			vecstr = split(optarg,',');
			if( vecstr.size()==1 ) {
				Lx = atof(optarg);
				Ly = Lx;
				Lz = Lx;
			} else if (vecstr.size()==3 ) {
				Lx = atof(vecstr.at(0).c_str());
				Ly = atof(vecstr.at(1).c_str());
				Lz = atof(vecstr.at(2).c_str());
			} else {
				cerr << "error: unknown input for -s, --size.\nTry -h or --help for more information." << endl;
				abort();
			}
			break;
		case 'N':
			vecstr = split(optarg,',');
			for(unsigned i=0;i<vecstr.size();i++)
				num_par.push_back(atoi(vecstr.at(i).c_str()));
			break;
		case 'u':
			vecstr = split(optarg,',');
			for(unsigned i=0;i<vecstr.size();i++)
							mass.push_back(atof(vecstr.at(i).c_str()));
			break;
		case 'm':
			mix_positions=true;
			break;
		case 'T':
			temperature = atof(optarg);
			break;
		case 's':
			seed=atoi(optarg);
			break;
		case 't':
		  topology_file = optarg;
		  write_topology_file = true;
		  break;
		case 'o':
			filename = optarg;
			break;
		/*case '?':
			break; */
		default:
			cerr << "Try -h or --help for more information." << endl;
			abort();
		}
	}

	if(!quiet)
			cout << argv[0] << ": Use  -h or --help for more information." << endl;

	// Apply user input to rumd_init_conf object
	rumd_init_conf rumd_init_conf(lattice_type,nx,ny,nz,seed);
	if(mix_positions)
		rumd_init_conf.mix_positions();
	if(!num_par.empty())
		rumd_init_conf.reset_particle_types(num_par);

	// Set masses of types
	if(mass.empty())
		for(unsigned i=0;i<rumd_init_conf.number_of_types();i++)
			mass.push_back(1.0);
	if(rumd_init_conf.number_of_types()!=mass.size()) {
		cerr << "error: number of masses given is not the same as number of types. Try -h or --help for more information." << endl;
		abort();
	}
	rumd_init_conf.reset_mass_of_types(mass);

	// Reset box volume
	if(density>0.0)
		rumd_init_conf.set_density(density);
	if(!(Lx<0.0))
		rumd_init_conf.scale_x_coordinates(Lx);
	if(!(Ly<0.0))
		rumd_init_conf.scale_y_coordinates(Ly);
	if(!(Lz<0.0))
		rumd_init_conf.scale_z_coordinates(Lz);
	if(min_distance>0.0)	// TODO rumd_init_conf.get_min_distance() scales badly, and could make the program slow
		if(min_distance>rumd_init_conf.get_min_distance())
			rumd_init_conf.set_min_distance(min_distance);
	if(!quiet)
		cout << rumd_init_conf.info();


	{	// Write configuration to file
		using namespace boost::iostreams;
		vector<string> fnames = split(filename,'.');
		filtering_ostream out;
		if(fnames.size()<2){
			cerr << "error: Incompatible name of output file. Should be *.xyz or *.xyz.gz" << endl;
			abort();
		}
		if(fnames.back()=="xyz"){
			out.push(file_sink(filename));
		} else if ( fnames.back()=="gz" && fnames.at(fnames.size()-2)=="xyz" )  {
			out.push(gzip_compressor());
			out.push(file_sink(filename));
		} else {
			cerr << "error: Incompatible name of output file. Should be *.xyz or *.xyz.gz" << endl;
			abort();
		}
		rumd_init_conf.write_xyz(out,temperature);
	}
	
	if(write_topology_file){ // Write topology file
	  unsigned int number_of_unit_cells = nx*ny*nz;
  	rumd_init_conf.write_top(topology_file, "mol.top", number_of_unit_cells);
	  cout << "Wrote topology file mol.top using " << topology_file << " as input." << endl;
	}
	if(!quiet)
		cout << "Write configuration to " << filename  << " with temperature " << temperature << endl;


	return 0;
}

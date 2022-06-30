#ifndef rumd_init_conf_H
#define rumd_init_conf_H

#include <vector>
#include <string>
using namespace std;

/// Generate initial particle configuration like start.xyz.gz
/** 
This tool is used to generate initial particle configurations, e.g. start.xyz.gz. 
Note that the tool rumd_init_conf_mol should be used for molecular configurations.
 
# Executable
This class has an executable that can be run from the commanline:
```
Make a configuration used for input to RUMD (www.rumd.org).

Usage examples:
rumd_init_conf --lattice=fcc --cells=6,6,6
rumd_init_conf -lfcc -c6,6,6
rumd_init_conf -c10 -N800,200 -r1.2 -m -o KobAndersen.xyz.gz

Optional flags:
 -h, --help                    Print information on using this program.
 -q, --quiet                   Hide program output.
 -l, --lattice=FILE            Load unitcell from *.xyz or *.xyz.gz file.
 -l, --lattice=STR             Set lattice type. Default: sc.
                               Allowed values (n="particles in unit cell"):
                                 sc     Simple cubic (n=1).
                                 rp     Randomly packed (n=1). Warning: Slow for large systems.
                                 bcc    Body centered cubic (n=2).
                                 fcc    Face centered cubic (n=4).
                                 hcp    Hexagonal close packed (n=4).
                                 hex    Hexagonal layers in xy-planes (n=2).
                                 dc     Diamond cubic lattice (n=8).
                                 NaCl   Rock salt lattice. (n=2x4)
                                 CsCl   Caesium Chloride lattice (n=2x1).
 -c, --cells=INT               Set number of unit cells. Default: 5.
     --cells=INT,INT,INT       
 -N, --num_par=INT,INT,...     Reset particle types.
                                 Note: The sum of particles must be < or = munber of lattice sites.
 -u, --mass=NUM,NUM,...        Set masses of types. Default: 1.
 -m, --mix_positions           Make positions of particles on lattice sites random.
 -r, --rho=NUM                 Set number density. Default: 1.
 -L, --length=NUM              Change length of box vectors. Skip resetting when <0.
     --length=NUM,NUM,NUM        Set one value =0 for a 2D configuration.
 -d, --minimum_distance=NUM    Do not allow distances shorter than min_dist. Warning: Slow for large systems.
 -T, --temperature=NUM         Temperature of random velocity vectors.
 -s, --seed=INT                Seed for pseudo random numbers.
 -o, --output=FILE             *.xyz or *.xyz.gz output file.

```

# Output file

The default output is a start.xyz.gz file for RUMD.
This is an extended xyz-file format 
that have been LZW compressed. The standard xyz-format 
is extended with additional information
in the comment line. The files can be viewed with
```
    zless start.xyz.gz
```
Below is a snippet from a file

```
 864
ioformat=2 numTypes=1 integrator=IntegratorNVT,0.00249999994,1,0.200000003,-0.414839104 sim_box=RectangularSimulationBox,10.2598553,10.2598553,10.2598553 mass=1.000000000 columns=type,x,y,z,imx,imy,imz,vx,vy,vz,fx,fy,fz,pe,vir
0 1.596656084 -5.054388046 4.624642372 0 0 -1 -0.842889726 1.405353189 -0.130556479 21.584344864 10.988825798 -22.567173004 -3.917686224 19.284681320
0 -3.023056746 -0.661342561 -3.348393679 0 0 0 0.027839949 1.621770263 1.421669483 -1.671880007 -6.409437656 -8.846709251 -5.231784821 -9.414367676
0 -2.640017748 3.046555042 -2.370884180 0 -1 0 -1.531434536 -0.844049394 0.150847837 13.054626465 34.658695221 -22.539432526 -4.087572098 21.139272690
0 -2.468223810 0.067353979 3.869976759 0 0 -1 -0.269454956 -0.388110638 0.976019323 8.225893021 16.573566437 -8.846957207 -5.893713474 -10.221490860
0 -0.788988352 -2.636515379 4.565689087 0 0 -1 0.514247775 -0.870343447 0.091416821 -9.201917648 -38.601467133 13.831747055 -3.617202759 17.129533768
0 4.874670506 4.530293465 -1.059690714 -1 -1 0 -1.832157135 1.191907167 1.245251298 10.072606087 -3.495929718 -2.210849524 -5.062928200 -0.151863337
... 
```


# See also
rumd_init_conf_exec.cc

rumd_init_conf_mol

*/
class rumd_init_conf {
private:
	string lattice_type;
	int number_of_sites_on_lattice;
	vector<unsigned> type;
	vector<double> x;
	vector<double> y;
	vector<double> z;
	double Lx;
	double Ly;
	double Lz;
	vector<double> mass_of_types;

	double random_velocity(double temperature,double mass);

public:
	rumd_init_conf(string lattice_type,unsigned nx,unsigned ny,unsigned nz,unsigned seed);
	virtual ~rumd_init_conf();

	void addParticle(unsigned in_type,double in_x,double in_y,double in_z);

	void mix_positions();
	void set_density(double new_density);
	double get_min_distance();
	void set_min_distance(double new_min_distance);
	void scale_coordinates(double sf);
	void scale_x_coordinates(double new_Lx);
	void scale_y_coordinates(double new_Ly);
	void scale_z_coordinates(double new_Lz);
	void translate_all_particles(double dx,double dy,double dz);
	void reset_particle_types(vector<unsigned>);
	void reset_mass_of_types(vector<double>);
	bool reset_particle_type(unsigned particle_index,unsigned new_type);
	void reset_number_of_particles(unsigned num_remove);
	void write_xyz(ostream& out,double temperature);
	void write_top(string input_file_name, string output_file_name, unsigned int number_of_unit_cells);

	unsigned number_of_particles();
	unsigned number_of_types();
	unsigned number_of_particles_of_type(unsigned test_type);
	double volume();
	string info();
};

#endif // rumd_init_conf_H

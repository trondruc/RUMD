Beginners guide: Running your first simulations   {#guideBeginner}
===============================================================================

This tutorial assumes basic GNU/Linux knowledge, a successful
installation of RUMD and a little portion of gumption mixed
with interest in molecular dynamics. For plotting the results 
from simulations it is assumed that xmgrace is installed, but 
other plotting programs can be used if you prefer. 

The basic work-flow of doing simulations is:

* Specify initial condition, i.e., the initial positions 
  and velocities to be used in the simulation. This also involves
  the number of particles (N) and the density (rho=N/V).
* Running the actual simulation. This can be done interactively 
  or by a script in python. It includes defining how particles 
  interact with each other (i.e., the potentials used), what 
  integrator to use (and the associated parameters, e.g., 
  time-step and temperature.)
* Post-processing data analysis.

The following sub-sections will take you through these steps.


Making the initial particle configuration
===============================================================================
When conducting simulations it is often convenient to use the end result 
of one simulation as the initial condition to a new simulation.
However, sometimes you need to generate a fresh initial condition. 
This can be done from a terminal window by executing the commands given below. 

First make a directory where you want your first simulation to be performed. Call
it e.g. `Test`, and go to that directory:
```
    mkdir Test
    cd Test
```


Next, run the rumd_init_conf program (make sure  `<RUMD-home>/Tools/` is 
in your path):
```
    rumd_init_conf --lattice=fcc --cells=6 --rho=0.8
```

This command produces a file called \ref start.xyz.gz that we will 
use as the initial configuration. 
The flag `--lattice=fcc` tells the program to generate a fcc crystal, 
the flag `--cells=6` that is should make six unit cells in each direction,
and the flag `--rho=0.8` that the number density should be 0.8.
The fcc lattice have four particles in the unit cell, 
and the final configuration will have \f& 4*6^3=864 \f& particles
as shown by the output of the program:
```
Lattice type:                 fcc
Number of lattice sites:      864
Total number of particles:    864
Number of types:              1
Number of particles of types: 864
Mass of types:                1
Lengths of box vectors:       10.2599 10.2599 10.2599
Box volume:                   1080
Number density:               0.8
Write configuration to start.xyz.gz with temperature 1

```

We have not specified temperature, and the program will use default value
T=1.0. Note that rumd_init_conf have many options for generating configurations.
A summary of these can be seen by typing
```
    rumd_init_conf --help
```

rumd_init_conf also accept short arguments and the command to generating 
the fcc lattice simply done with
```
    rumd_init_conf -l fcc -c 6 -r 0.8
```
and for help, you can simply write
```
    rumd_init_conf -h
```

View configuration
------------------
The configuration can be viewed with the tool \ref rumd_image
```
    rumd_image
```
Use the `-h` flag to view help page for this tool.
![Face centered cubic lattice generated with rumd_image](fcc864.png)


Executing your first program using the python interface
===============================================================================
In this first example we will simulate a single component
Lennard--Jones liquid. We will work with python interactively, so that you can
 get a feel for what is possible. For production runs you will normally
make a script containing the appropriate python commands. Start python by typing
```
    python3
```

in a terminal. You will get a python prompt that looks like 
```
    >>>
```

Type 
```py
    from rumd import *
```
This will import the main RUMD classes into the global namespace. 

Alternatively (recommended by some to avoid possible name clashes in the global 
namespace) one can type `import rumd`, and then all of the class-names, etc., 
must be prefixed by `rumd`, for example `rumd.IntegratorNVE` (these are actually 
C++ classes, but we can "talk" to them through python). These are in principle 
enough to write scripts and thereby run simulations, but to simplify this 
process, there is an extra python class called `Simulation`: type
```py
    from rumd.Simulation import Simulation
```

This class combines access to the main data-structures 
with the main integration loop and various functions for controlling output 
etc. To start a simulation we first 
create a Simulation object, passing the 
name of a starting configuration file:
```py
    sim = Simulation("start.xyz.gz")
```

Here `sim` is an arbitrary name we choose for the object.
Next we choose an NVT integrator, giving the time-step and temperature:
```py
    itg = IntegratorNVT(timeStep=0.0025, targetTemperature=1.0)
```

Having an integrator object, we need to connect it to the simulation:
```py
    sim.SetIntegrator(itg)
```

Next we need to choose the potential. For 12-6 Lennard-Jones we create a 
Potential object of the type Pot_LJ_12_6, giving it the name `pot`, as follows:
```py
    pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
```
 
The mandatory argument `cutoff_method` specifies how the cut-off of
the pair-potential is to be handled. It must be one of
ShiftedPotential (the most common method, where the potential is
shifted to zero), ShiftedForce or NoShift.
We need to set the parameters, which is done using the method SetParams:
Pot_LJ_12_6::SetParams(unsigned int,unsigned int,float,float,float).
The arguments to this method depend on which Potential class you are
working with, but they can be found by typing
```py
    help(pot.SetParams)
```

which displays a help page generated from the `docstring` of the method
(alternatively type `print pot.SetParams.__doc__|`.
In particular this includes a list of the arguments:
```py
    SetParams(self, unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut)
```

The first one, `self`, represents the object itself, and is not explicitly
given when calling the function. The next two define which particle types we
are specifying parameters for - we just have one type of particle so both will
be zero; the two after that are the standard Lennard-Jones length and energy
parameters, while the last is the cut-off in units of Sigma. Press `Q` to exit
the help screen. We choose the following:
```py
    pot.SetParams(0, 0, 1.0, 1.0, 2.5)
```

Note that we can also use python's "keyword arguments" feature to specify the
arguments, which can then be in any order, for example:
```py
    pot.SetParams(i=0, j=0, Epsilon=1.0, Rcut= 2.5, Sigma=1.0)
```

The potential also needs to be connected to the simulation:

```py
    sim.SetPotential(pot)
```

Now we are ready to run a simulation. To run a simulation with 20000 time steps,
we just write

```py
    sim.Run(20000)
```

Various messages will be printed to the screen while it is running (these can
 be turned off with `sim.SetVerbose(False)`. If we like, we
can get the positions after the 20000 steps by getting them as a numpy array 
(numpy, or numerical python, is a package that provides efficient array-based
numerical methods), and in particular look at the position of particle 0 by typing

```py
    pos = sim.sample.GetPositions()
    print pos[0]
```

However more likely we will run an analysis program on the output
files produced by the simulation during the operation of the `Run`
function. The available analysis programs are described below in subsection 
in their command-line forms. Some of them (eventually all)
can also be called from within python. 
For now let's write a configuration file for possible use as a starting point
for future simulations. Here we go a little deeper into the interface.
Objects of type `Simulation` have an attribute
called "sample" of type Sample 
(the main C++ class representing the sample we are simulating). We call its
WriteConf method as follows:
```py
    sim.WriteConf("end.xyz.gz")
```

Type Ctrl-D to exit python. 


Running a python script
------------------------------
Next we would like to make scripts. Here is the 
script that contains the commands we have just worked through 
(the lines starting with `#` are comments):
\include lennardJones/run.py
The above python script is located at `<RUMD-HOME>/doc/examples/lennardJones/run.py`.

If this script is saved as a file called, for an example `run.py` (it must end in
`.py`), then it is run by typing 
```
    python3 run.py
``` 
This will exit python when finished. To leave python on after the script has completed, 
type `python3 -i run.py` (-i means run in interactive mode).

After the simulation has finished you can view the final configuration with \ref rumd_image
```
    rumd_image -i end.xyz.gz
```

![A simulation of a Lennard-Jones fluid](LJ864.png)


Running a PBS queued job
---------------------------
If the simulation is to be run on a batch queue, 
the appropriate PBS commands should be included in comments at the top as follows:
```py
  #!/usr/bin/python3
  # pass PYTHONPATH environment variable to the batch process
  #PBS -v PYTHONPATH
  #PBS (other PBS commands here)
  #PBS 
  #PBS 

  # this ensures PBS jobs run in the correct directory
  import os, sys
  if "PBS_O_WORKDIR" in os.environ:
      os.chdir(os.environ["PBS_O_WORKDIR"])
      sys.path.append(".")


  from rumd import *
  # ... (etc)
```
Note that the indentation of the two lines following the `if` statement is important!

The python script should be submitted to the queue with the command
```bash 
  qsub run.py
```
however, you can still run the script locally with (for debugging)
```bash 
  python3 run.py
```


Setting options with the python interface
===============================================================================
Here we present some of the options available for controlling simulations.


Choosing the potential and its parameters
-----------------------------------------
Probably the most important thing a user needs to control is the 
potential. Different potentials are represented by different classes; objects
are created just as above, so for example:

```py
  # generalized LJ potential (here m=18, n=4)
  potential = Pot_gLJ_m_n(18,4, cutoff_method=ShiftedPotential) 
  # inverse power-law (IPL) with exponent n=12
  potential = Pot_IPL_12(cutoff_method=ShiftedPotential)
```

To set parameters for the potential, call SetParams() method
as described above. If a binary system is to be simulated, the parameters should be set separately 
for each interaction pair as follows

```py
  # generalized LJ potential (here m=18, n=4)
  potential.SetParams(0, 0, 1.00, 1.00, 1.12246)
  potential.SetParams(0, 1, 0.80, 1.50, 1.12246)
  potential.SetParams(1, 1, 0.88, 0.50, 1.12246)
```

Note that Newton's third law is assumed to hold, so setting the parameters for 
i=0 and j=1 automatically sets those for i=1 and j=0 (we could also have 
called SetParams with the latter).
An overview of the available potentials is given in the user manual.


Choosing the integrator
-----------------------
Perhaps the next most important choice is what kind of integration algorithm
to use. Above we did a constant-temperature (NVT) algorithm (the actual
algorithm is of the Nos{\'e}-Hoover type). For constant energy (NVE) runs we
create the integrator as follows, here passing only the time-step:

```py
  itg = IntegratorNVE(timeStep=0.0025)
```

(Technical note: this creates an object of the same type as IntegratorNVT class,
but here it defaults to NVE mode---in fact in either case one can switch 
thermostatting on or off using the SetThermostatOn() method). 


| Name of Integrator | Description       | Parameters                          |
|--------------------|-------------------|-------------------------------------|
| IntegratorNVE      | NVE Leap-frog     | timeStep                            |
| IntegratorNVT      | Nosè-Hoover NVT   | timeStep, targetTemperature         |
| IntegratorNPTAtomic| NPT Leap-frog     | timeStep, targetTemperature, thermostatRelaxationTime, targetPressure, barostatRelaxationTime |
| IntegratorNVU      | Conserving the total potential energy | dispLength, potentialEnergy |
| IntegratorMMC      | NVT Monte Carlo   | dispLength, targetTemperature       |
| IntegratorIHS      | Energy minimization| timeStep |
| IntegratorSLLOD    | SLLOD equations for atoms | timeStep, strainRate        |
| IntegratorMolecularSLLOD | SLLOD equations for molecules | timeStep, strainRate|


The above integrators are chosen in the usual way with named arguments as 
given in the table. In the case of IntegratorNPTAtomic the user must choose 
suitable relaxation times for the thermostat and the barostat. An example of 
reasonable values for the thermostat and barostat relaxation times for 
the LJ system are `thermostatRelaxationTime=4.0` and `barostatRelaxationTime=40.0`.


Controlling how frequently to write output
------------------------------------------
To control the output written to files we use the Simulation
method SetOutputScheduling. By default there are two output
managers: `energies` and `trajectory`. The first is for the files 
containing potential/kinetic energies and other global
quantities, while the second is for trajectories (configuration files 
containing the particle positions at different times during the run). 
The write-schedule for either manager
can be evenly-spaced in time ("linear", 
logarithmic ("logarithmic"), or a  kind of combination of the two 
("loglin"). Examples of controlling scheduling include

```py
  sim.SetOutputScheduling( "energies",   "linear", interval=50)
  sim.SetOutputScheduling( "energies",   "none" )
  sim.SetOutputScheduling( "trajectory", "logarithmic" )
  sim.SetOutputScheduling( "trajectory", "loglin", base=4, maxInterval=64 )
```

The first causes energies to be output equally spaced (linear) in time, once 
every 50 time steps, while the second turns energy writing off. The third
 will cause logarithmic saving of configurations (where the 
interval between saves doubles 
each time within a "block", starting with 1. The fourth does
logarithmic saving starting with interval 4 until the interval 64 is
reached, after which the interval stays fixed at 64 until the end of
the block. The details of log-lin  are described in a separate
document.  By default energies are linear, written every 256 steps,
and trajectories are logarithmic.


Post processing
===============================================================================
The post processing tools are located in the `<RUMD-HOME>/Tools/`
directory and each one of them has a help text. To see the help text
use the option `-h` or `--help` depending on the tool. 
The actual output files generated by the program are in a
directory called `TrajectoryFiles`. When you start a new simulation in the
same directory, this will be moved to `TrajectoryFiles.rumd_bak` as 
a precaution against accidental overwriting of data. The analysis 
programs should be run in the original directory where the simulation was run
(they know they should look in `TrajectoryFiles` for the data).


Thermodynamic data
------------------
rumd_stats produces mean values, variances and standard
deviations for specific quantities. The rumd_stats stdout for
the simulation just performed is shown below 
```
quantity, mean value, variance, standard deviation
ke	     1.49682	  0.00182708	   0.0427444
pe	    -4.69314	  0.00068112	   0.0260983
p	     1.67351	   0.0151307	    0.123007
T	    0.999033	 0.000813923	   0.0285293
Etot	    -3.19632	  0.00259098	   0.0509017
W	     1.09285	   0.0221461	    0.148816
```
Here `ke` is the kinetic energy, `pe` is the potential energy, `p` is the
pressure, `T` is the kinetic temperature, `Ps` is the Nose Hoover
thermostat, `Etot` is the total kinetic energy and `W` is the virial. The
program writes the mean and variance in the files
`energies_mean.dat` and `energies_var.dat`in one row as the
first column of stdout.

The radial distribution function
--------------------------------
The radial distribution function is computed with rumd_rdf by typing 
```
  rumd_rdf -n 1000 -m 1.0
```
where the first argument is the number of bins in radial
distribution function and the second argument, the minimum time
between successive configurations to use, is to avoid wasting 
time doing calculations that are very similar (we assume here that the
configurations are  uncorrelated after one time unit). Use
```
  rumd_rdf -h
```
for a full list of arguments. 
The output is `rdf.dat` and for binary systems the columns are:
\f[\quad g_{00}(r) \quad  g_{01}(r)\quad  g_{10}(r) \quad  g_{11}(r)\f]
Single component only has two columns. Plot `rdf.dat` with e.g. 
[Xmgrace](http://plasma-gate.weizmann.ac.il/Grace/):
```bash
  xmgrace rdf.dat
```
to obtain figure :
![Radial distribution function made with rumd_rdf.](LJrdf.jpg)


The static structure factor 
---------------------------
The static structure factor can be obtained when the radial
distribution function is computed. It is done with the command rumd_sq:
```
  rumd_sq 2 20 1 
```
where the first argument is the start q value, the second
argument is the final q value and the third argument is the
density. The stdout is the q value for the maximum of the first peak
and it is written in a file called `qmax.dat`. The static
structure factor is written in `Sq.dat` and is structured like
`rdf.dat`. Plot `Sq.dat` to obtain figure :
![Static structure factor computed with rumd_sq](LJSq.jpg)



Mean square displacement
------------------------
The mean square displacement and the self part of the
incoherent intermediate scattering function Fq(t) are calculated with 
the command rumd_msd. This generates a `msd.dat` file with time as the
first column and the mean square displacement as a function of time as
the second column (for binary systems there will be two columns), and a file 
`Fs.dat` with a similar structure. Before it can
be run however, you must create a file called `qvalues.dat` which contains
one wavenumber for each time of particle. Typically these correspond to the 
location of the first peak in the structure factor, so one could copy the file
`qmax.dat` crated by `rumd_sq`:
```
  rumd_sq 2 20 1
  cp qmax.dat qvalues.dat
  rumd_msd
```
`rumd_msd` also calculates gaussian parameter `alpha2.dat`. 
The figures below show plots of the output from rumd_msd:

![Mean square displacement computed with rumd_msd](LJmsd.jpg)

![incoherent intermediate scattering function Fq(t)](LJFs.jpg)


Table of post analysis tools
----------------------------
| Tool       | Description                  | Input file(s)  | Output file(s)   |
|------------|------------------------------|----------------|------------------|
| rumd_stats | Thermodynamic data           |                | energies_mean.dat, energies_var.dat, energies_drift.dat, energies_covar.dat, energies_mean_sq.dat |
| rumd_rdf   | Radial distribution function |                | rdf.dat          | 
| rumd_sq    | Scattering function          |                | Sq.dat, qmax.dat |
| rumd_msd   | Mean squared displacement    | qvalues.dat    | msd.dat, Fs.dat  |

It is evident from the table above that `rumd_rdf` has to be
performed before `rumd_sq` and `rumd_sq` before
`rumd_msd`. If you only are interested in the mean square
displacement and know which q-values to use it is not necessary to run
`rumd_rdf` and `rumd_sq` first. Then you just have to create
a file called `qvalues.dat` with the appropriate q-values before
running `rumd_msd`.


Simulating molecules
===============================================================================
With RUMD you can simulate molecular systems. The intra-molecular 
force field includes bond, angle and torsion interactions. The total
potential energy due to intra-molecular interactions excluding
possible pair potentials is given by 
\f[ 
 U(\vec{r}) = \frac{1}{2}\sum_{\textrm{bonds}}k_s^{(i)} (r_{ij}-l_b^{(i)})^2 + 
\frac{1}{2}\sum_{\textrm{angles}}k_\theta^{(i)} [\cos(\theta)-\cos(\theta^{(i)})]^2 + 
\sum_{\textrm{dihed}} \sum_{n=0}^5 c_n^{(i)} \cos^n(\phi), 
\f]
where \f$ k_s^{(i)} \f$ is the spring force constant for bond type i, \f$ k_\theta^{(i)} \f$ the 
angle force constant for angle force type i, and \f$c_n^{(i)}\f$ the torsion 
coefficients for torsional force type $i$. \f$ l_b^{(i)} \f$ and
\f$ \theta_0^{(i)} \f$ are the zero force bond length and angle,
respectively. 

Beside the standard harmonic bond potential RUMD also supports
simulation of rigid bonds using constraint method as well as the  
Finite Extensible Nonlinear Elastic (FENE) potential 
\f[
U(\vec{r}) = -\frac{1}{2}kR_0^2\sum_{\textrm{bonds}} \ln\left[ 1 -\left(\frac{r_{ij}}{R_0}\right)^2\right],
\f]
where \f$k=30\f$ and \f$R_0=1.5\f$ (in reduced units). At the moment 
the constraint method is applicable for molecules with few
constraints.  


Example: polymers
------------------
In all one starts by creating (or copying and modifying) a topology file
(with extension `.top`). Examples can be found in the
subdirectory `Conf/Mol`, in particular one called
`mol.top`, which is associated with a configuration file  
`ExampleMol.xyz.gz`. Copy both of these files to your test
directory. They specify a system containing 100 polymeric molecules,
each consisting of  10 monomer units. The appropriate lines to include
in the python script include one for reading the topology file and one
for setting the parameters of the (in this case) single bond-type.
```py
  sim = Simulation("ExampleMol.xyz.gz")
  # read topology file
  sim.ReadMoleculeData("mol.top")

  # create integrator object
  itg = IntegratorNVE(timeStep=0.0025)
  sim.SetIntegrator(itg)

  # create pair potential object
  potential = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
  potential.SetParams(i=0, j=0, Epsilon=1.0, Sigma=1.0, Rcut=2.5)
  sim.SetPotential(potential)

  # define harmonic bond and its parameters for bonds of type 0
  sim.SetBondHarmonic(bond_type=0, lbond=0.75, ks=200.0)

  sim.Run(20000)
```
Note that when you specify the bond parameters after
calling SetPotential contributions from those bonds will automatically
be removed from the calculation of the pair potential, otherwise this will not
be the case. For this reason ^it is important that you 
call SetPotential before calling sim.SetBondHarmonic^. If you
wish to keep the pair force interactions between the bonded particles
you can specify this using 
```py
  sim.SetBondHarmonic(bond_type=0, lbond=0.75, ks=200.0, exclude=False)
```

In the case you wish to use the FENE potential you simply use  
```py
  sim.SetBondFENE(bond_type=0)
```
to specify that bond type 0 is a FENE bond type. 
In the FENE potential, the pair interactions between bonded particles
are not excluded. 

As noted above, you can also simulate molecules with rigid bonds. To 
specify that bond type 0 is a rigid bond you add the bond constraint
using the \verb|SetBondConstraint| method
```py
     sim.SetBondConstraint(bond_type=0, lbond=0.75)
```

""" iPython module for performing Molecular Dynamics Simulations with RUMD.

This module allows for interactive simulation using iPython framework and
the Roskilde University Simulation Package (RUMD). The module can perform
(standardized) simulations, and display (standardized) data analysis.
The purpuse is to give an introduction to MD simulations, and the module is
optimized for teaching purposes.

Usage example
-------------
import interactive_md as md
md.LennardJones(steps=16384,Ti=0.8,Tf=0.0)
md.info()
md.visualize()
md.energy()
md.image()
md.rdf()
md.msd()
"""

__author__ = 'Ulf R. Pedersen'


def LennardJones(steps=32768, Ti=0.8, Tf=None, rho=None, p=None):
    """ Run a simulation of the Lennard-Jones model.
    Particles interact via the pair potential r^-12 - r^-6.

    Parameters
    ----------
    steps (int)   : How many steps to be performed
    Ti    (float) : Initial Temperature
    Tf    (float) : Final Temperature
    rho   (float) : Rescale to this density (if not None)
    p     (float) : Perform constant pressure simulation (if not None)
    """
    import os
    import rumd
    from rumd.Simulation import Simulation

    print("steps", steps, "Ti", Ti, "Tf", Tf, "rho", rho, "p", p)

    configuration = "start.xyz.gz"
    if os.path.isfile(configuration):
        print("Use configuration file in directory ", configuration)
    else:
        rumdpath = os.path.dirname(rumd.__file__) + "/../.."
        configuration = rumdpath + \
            "/Conf/SingleComponentLiquid/LJ_N1000_rho0.80_T0.80.xyz.gz"
        print("Use default configuration file: ", configuration)
    sim = Simulation(configuration)
    pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
    pot.SetParams(0, 0, Sigma=1.0, Epsilon=1.0, Rcut=4.0)
    sim.SetPotential(pot)
    if(p is None):
        itg = rumd.IntegratorNVT(timeStep=0.005, targetTemperature=Ti)
    else:
        itg = rumd.IntegratorNPTAtomic(timeStep=0.005,
                                       targetTemperature=Ti,
                                       thermostatRelaxationTime=0.4,
                                       targetPressure=5.0,
                                       barostatRelaxationTime=10.0)
    if(Tf is not None):  # Make temperature ramp
        itg.SetTemperatureChangePerTimeStep((Tf-Ti)/steps)
    sim.SetIntegrator(itg)
    sim.SetOutputMetaData("energies", volume=True)

    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    print("The loaded density is", N/V)
    if(rho is not None):  # Change density
        old_rho=N/V
        scl = pow(rho/old_rho,-1./3.)
        print("scl", scl)
        sim.ScaleSystem(scl)
    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    print("Density is rho =", N/V, " and partial volume is V/N =", V/N)

    sim.Run(steps)
    sim.WriteConf("final.xyz.gz")
    print("Done with simulation. Wrote final configuration to final.xyz.gz.")


def KobAndersen(steps=32768, Ti=1.0, Tf=0.0,rho=None):
    """ Run a simulation of the Kob-Andersen binary Lennard-Jones mixture.
    The composition is 800 large A particles and 200 small B particles.
    The pair interactions are constructed to avoid crystalization
    by having a large AB affinity.
    
    Parameters
    ----------
    steps (int)   : How many steps to be performed
    Ti    (float) : Initial Temperature
    Tf    (float) : Final Temperature
    rho   (float) : Rescale to this density (if not None)
    """
    import os
    import rumd
    from rumd.Simulation import Simulation

    print("steps", steps, "Ti", Ti, "Tf", Tf)

    configuration = "start.xyz.gz"
    if os.path.isfile(configuration):
        print("Use configuration file in directory ", configuration)
    else:
        rumdpath = os.path.dirname(rumd.__file__) + "/../.."
        configuration = rumdpath + "/Conf/KobAndersen/KA_N1000_T1.0.xyz.gz"
        print("Use default configuration file: ", configuration)
    sim = Simulation(configuration)
    pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedForce)
    pot.SetParams(0, 0, Sigma=1.00, Epsilon=1.0, Rcut=2.5)
    pot.SetParams(0, 1, Sigma=0.80, Epsilon=1.5, Rcut=2.5)
    pot.SetParams(1, 0, Sigma=0.80, Epsilon=1.5, Rcut=2.5)
    pot.SetParams(1, 1, Sigma=0.88, Epsilon=0.5, Rcut=2.5)
    sim.SetPotential(pot)
    itg = rumd.IntegratorNVT(timeStep=0.005, targetTemperature=Ti)
    if( Tf is not None ):
        itg.SetTemperatureChangePerTimeStep((Tf-Ti)/steps)
    sim.SetIntegrator(itg)
    
    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    print("The loaded density is",N/V)
    if( rho is not None ):  # Change density
        old_rho=N/V
        scl = pow(rho/old_rho,-1./3.)
        sim.ScaleSystem(scl)
    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    print("Density is rho =", N/V, " and partial volume is V/N =", V/N)

    sim.Run(steps)
    sim.WriteConf("final.xyz.gz")
    print("Done with simulation. Wrote final configuration to final.xyz.gz.")


def water(steps=32768,Ti=2.2698,Tf=None,rho=None):
    """ Run a simulation of the SPC water model.
    The model consists of molecules where each site represent one O atom and two H atoms,
    connected via rigid bonds with an internal angle of 109.47 degree.
    The O atoms interact via a Lennard-Jones pair potential. 
    All atoms are partially charged: q_H = 0.4238 and q_O = -2 x qH
    
    Units in this model are kJ/mol, Aangstrom and the atomic mass, thus
      Boltzman constant is kB=0.0083144621 kJ/(mol K),
      temperature 1 correspond to 120.27 K, and
      one time units is about 1e-13 seconds.

    Parameters
    ----------
    steps (int)   : How many steps to be performed
    Ti    (float) : Initial Temperature
    Tf    (float) : Final Temperature
    rho   (float) : Rescale to this density (if not None)
    """
    import os
    import rumd
    from rumd.Simulation import Simulation
    from shutil import copyfile

    print("steps",steps,"Ti",Ti,"Tf",Tf)
   
    configuration="start.xyz.gz"
    topology="mol.top"
    if (os.path.isfile(configuration) and os.path.isfile('mol.top')):
        print("Use configuration file in directory ",configuration)
    else:
        rumdpath =  os.path.dirname(rumd.__file__) + "/../.."
        configuration=rumdpath + "/Conf/Mol/spcIce100.xyz.gz"
        topology=rumdpath + "/Conf/Mol/spcIce100.top"
        print(topology)
        copyfile(topology,'mol.top')
        print("Use default configuration and topology file: ", \
                configuration,topology)
    
    # Energy is in units of kJ/mol
    # Lengths are in units of Angstrom
    # Masses are in atomic units: 1.660538921e-27 kg 
    # Time unit is 1e-10*sqrt(1.660538921e-27/(1000/6.02214129e23)) ~= 1e-13 sek.
    sim = Simulation(configuration)
    pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedForce)
    pot.SetParams(0,0,Sigma=3.166,Epsilon=0.650,Rcut=9.0)
    pot.SetParams(0,1,Sigma=1.0,Epsilon=0.0,Rcut=9.0)
    pot.SetParams(1,1,Sigma=1.0,Epsilon=0.0,Rcut=9.0)
    sim.SetPotential(pot)
    pot.SetExclusionType(1, 1) # Exclude hydrogen interactions
    pot.SetExclusionType(0, 1)
    # Prefactors for Coulumbs law:
    # O-O :  1389.35457839084*0.8276*0.8276	= 951.599183095512
    # O-H : -1389.35457839084*0.8276*0.4238	= -487.297890038519
    # H-H :  1389.35457839084*0.4238*0.4238	= 249.537029722480
    potCharge = rumd.Pot_IPL_n(n=1,cutoff_method=rumd.ShiftedForce)
    potCharge.SetParams(i=0, j=0, Sigma=1.0, Epsilon= 951.59918, Rcut=9.0)
    potCharge.SetParams(i=1, j=0, Sigma=1.0, Epsilon=-487.29789, Rcut=9.0)
    potCharge.SetParams(i=1, j=1, Sigma=1.0, Epsilon= 249.53703, Rcut=9.0)
    sim.AddPotential(potCharge)
    # Molecule topology
    # H-H bond length = 2*sin(0.5*109.47/180*pi) = 1.63298086184023
    sim.ReadMoleculeData('mol.top')
    sim.SetBondConstraint(bond_type=0, lbond=1.0)
    sim.SetBondConstraint(bond_type=1, lbond=1.63298086184023)
    
    itg = rumd.IntegratorNVT(timeStep=0.01, targetTemperature=Ti)
    if( Tf is not None ):
        itg.SetTemperatureChangePerTimeStep((Tf-Ti)/steps)
    sim.SetIntegrator(itg)
    
    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    if( rho is not None ):  # Change density
        old_rho=N/V
        scl = pow(rho/old_rho,-1./3.)
        sim.ScaleSystem(scl,CM=True)
    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()
    print("Density is rho =", N/V, " and partial volume is V/N =", V/N)
    
    sim.Run(steps)
    sim.WriteConf("final.xyz.gz")
    print("Done with simulation. Wrote final configuration to final.xyz.gz.")


def info():
    """ Brief information about the simulation """
    import os
    print("Directory:",os.getcwd())
    trjdir="TrajectoryFiles/"
    if os.path.isdir(trjdir):
        f = open(trjdir + 'LastComplete_restart.txt', 'rt')
        last_block = int(f.readline().split(' ')[0])
        f.close()
        print("Found an output directory ("+ trjdir +") with " + 
              str(last_block) + " blocks.")
    else:
        print("An output directory was NOT found (did RUMD run?).")
    
    if os.path.isfile("./start.xyz.gz"):
        print("An initial configuration was found (start.xyz.gz).")
    else:
        print("An initial configuration was not found.")
    
    if os.path.isfile("./final.xyz.gz"):
        print("A final configuration was found (final.xyz.gz).")
    else:
        print("A final configuration was NOT found (did RUMD finish?).")

def stats(first_block=0, last_block=-1):
    """ Print thermodynamic statistics 
    The input parameters lets you do analysis on a part of the trajectory.
    """
    import rumd.Tools
    rs = rumd.Tools.rumd_stats()
    rs.ComputeStats(first_block=first_block, last_block=last_block)
    rs.WriteStats()
    rs.PrintStats()


def visualize(viewer='rumd_visualize'):
    """ Visualize MD simulation.
    Possible values for viewer are:
      rumd_visualize
      ruvis
    """
    if viewer == 'rumd_visualize':
        import subprocess
        subprocess.getoutput('xterm -e rumd_visualize')
    elif viewer == 'ruvis':
        import ruvis
        ruvis.view()
    else:
        print('Unknown viewer',viewer)


def rdf(i=0, j=0):
    """ Plot the radial distribution function """
    import rumd
    import matplotlib.pyplot as plt
    rdf_obj=rumd.Tools.rumd_rdf()
    rdf_obj.ComputeAll(1000, 100.0)
    r = rdf_obj.GetRadiusValues()
    g = rdf_obj.GetRDFArray(i,j)
    plt.plot(r,g)
    plt.xlabel('Distance between a pair')
    #plt.xlim([0.0,4.0])
    plt.ylabel('Radial pair distribution function')
    plt.savefig('rdf.png')
    plt.show()


def energy(y='Etot',x='t'):
    """ Plot total energy.
    
    Usage example
    -------------
    Plot energy vs. temperature with energy('Etot','T') 
    """
    import rumd.analyze_energies as analyze
    import matplotlib.pyplot as plt
    nrgs = analyze.AnalyzeEnergies()
    if(x is None or x == 't'):
        nrgs.read_energies(['pe',y])
        vary = nrgs.energies[y]
        plt.plot(vary)
        plt.xlabel('index')
        plt.ylabel(y)
    else:
        nrgs.read_energies([x, y])
        varx = nrgs.energies[x]
        vary = nrgs.energies[y]
        plt.plot(varx, vary)
        plt.xlabel(x)
        plt.ylabel(y)
    plt.savefig('energy.png')
    plt.show()


def msd(i=0):
    """ Plot mean squared displacement of particles """
    import matplotlib.pyplot as plt
    import rumd
    msd_obj = rumd.Tools.rumd_msd()
    msd_obj.SetQValues([7.25, 5.75])
    msd_obj.ComputeAll()
    msd = msd_obj.GetMSD(i)
    x=msd[:,0]
    y=msd[:,1]
    plt.loglog(x,y)
    plt.xlabel("Time")
    plt.ylabel("Mean squared displacement")
    plt.savefig('msd.png')
    plt.show()


def image(conf="final.xyz.gz",width=400,options=""):
    """ Show image of final configuration. 
    Input set configuration, width of image and options for rumd_image
    """
    from subprocess import getoutput
    print(getoutput('rumd_image %s -d -i %s -o image' % (options, conf)))
    height = width*3/4
    print(getoutput('povray -D +P +W%d +H%d +A +HImyPovray.ini +Iimage.pov' % (width,height)))
    from IPython.display import Image, display
    display(Image(filename='./image.png'))
    print("Saved image of configuration %s to image.png." % conf)


def movie(options="-o movie.mp4"):
    """ Generate movie of restart configurations. Use options="-h" for help """
    from subprocess import getoutput 
    print(getoutput('rumd_movie ' + options))
    print("Generated movie of restart configurations.")

def rotate(conf="final.xyz.gz"):
    """ Show movie of the final configuration rotating. """
    import os
    os.system('rumd_image -d -i %s -a 2*pi*clock -o rotate' % conf)
    os.system('povray -D +W400 +H300 +HImyPovray.ini +Irotate.pov +KFF96 +KC +Orotate.png')
    # TODO remove old rotate.mp4
    os.system('ffmpeg -i rotate%02d.png -r 24 rotate.mp4')
    os.system('rm rotate??.png')
    from IPython.display import HTML
    HTML('<iframe width=400 height=300 src="./rotate.mp4"></iframe>')


def copyrun(idir='.', odir='../newrun'):
    """ Copy simulation files to new directory.
    It is assumed that the run script is named run.py
    """

    from os import path,chdir,mkdir,getcwd
    from shutil import copyfile

    if path.isdir(odir):
        print('Directory ' + oname + ' already exist. I refuse to overwrite.')
    else:
        if path.isfile('./final.xyz.gz'):
            mkdir(odir)
            copyfile(idir + '/run.py', odir + '/run.py')
            copyfile(idir + '/mol.top', odir + '/mol.top')
            copyfile(idir + '/final.xyz.gz', odir + '/start.xyz.gz')
            chdir(odir)
            print('Moved into the new directory: ' + getcwd())
        else:
            print('Error: Final configuration final.xyz.gz does NOT exist.')

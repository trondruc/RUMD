""" Provides a function CompressRun which gently compresses/expands a system
represented by a rumdSimulation object to a given desired density"""

import math


def RunCompress(sim, final_density, num_scale_steps=100, niter_per_scale=1000, verbose=False):
    """
    Take an existing simulation object, with integrator and potential(s)
    already set up, and gently change its density to final_density.
    niter_per_scale: time steps to run for each incremental change in density.
    num_scale_steps: number of incremental density changes
    """
    start_density = sim.GetNumberOfParticles() / sim.GetVolume()
    scaleFactor = math.exp(-math.log(final_density/start_density)/3/num_scale_steps)
    have_molecule_data = sim.moleculeData is not None

    for scale_idx in range(num_scale_steps):

        sim.Run(niter_per_scale, suppressAllOutput=True)
        if verbose:
            sim.sample.CalcF()
            density = sim.GetNumberOfParticles() / sim.GetVolume()
            pe = sim.sample.GetPotentialEnergy()
            print("%d %f %f" % (scale_idx, density, pe))


        sim.ScaleSystem(scaleFactor, CM=have_molecule_data)

    # scale again to get exact density
    density = sim.GetNumberOfParticles() / sim.GetVolume()
    scaleFactor = math.exp(-math.log(final_density/density)/3)
    sim.ScaleSystem(scaleFactor, CM=have_molecule_data)
    # Run one more time at final density
    sim.Run(niter_per_scale, suppressAllOutput=True)

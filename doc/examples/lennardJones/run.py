#!/usr/bin/python3
from rumd import *
from rumd.Simulation import Simulation

# create simulation object
sim = Simulation("start.xyz.gz")

# create integrator object
itg = IntegratorNVT(timeStep=0.0025, targetTemperature=1.0, thermostatRelaxationTime=0.2)
sim.SetIntegrator(itg)

# create potential object
pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
pot.SetParams(0, 0, 1.0, 1.0, 2.5)
sim.SetPotential(pot)

# run the simulation
sim.Run(20000)

# write final configuration
sim.WriteConf("end.xyz.gz")

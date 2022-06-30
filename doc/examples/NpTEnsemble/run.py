#!/usr/bin/python3
import rumd
from rumd.Simulation import Simulation

sim = Simulation("start.xyz.gz")

# Kob-Andersen mixture
pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
pot.SetParams(0,0,Sigma=1.0,Epsilon=1.0,Rcut=2.5)
pot.SetParams(1,0,Sigma=0.8,Epsilon=1.5,Rcut=2.5)
pot.SetParams(0,1,Sigma=0.8,Epsilon=1.5,Rcut=2.5)
pot.SetParams(1,1,Sigma=0.88,Epsilon=0.5,Rcut=2.5)
sim.SetPotential(pot)

# Sample configurations in the NpT ensemble
itg = rumd.IntegratorNPTAtomic(timeStep=0.005,targetTemperature=1.028,thermostatRelaxationTime=4.0,targetPressure=10.19,barostatRelaxationTime=40.0)
sim.SetIntegrator(itg)

# Include volume in output
sim.SetOutputMetaData("energies", volume=True)

sim.Run(1000)

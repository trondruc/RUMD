Example: The Lennard-Jones fluid
============
Simulation of the The Lennard-Jones model using the Pot_LJ_12_6 pair potential.

1. Enter a working directory named `LJ` and generate and inspect an initial configuration:
```
    mkdir LJ
    cd LJ
    rumd_init_conf -lfcc -c6 -r0.8 
    rumd_image
```
2. Make a python script name `run.py`
   \include lennardJones/run.py

3. Run an equilibration simulation with
```
    python3 run.py
```
4. Copy the final configurationm, make and view a production run
```
    cp end.xyz.gz start.xyz.gz
    python3 run.py
```
5. Make some basic post analysis
```
    rumd_image -i start.xyz.gz
    rumd_image -i end.xyz.gz
    rumd_plot_energies -bxy 0:2
```

See also
-----------------
This example is explained in details in \ref guideBeginner


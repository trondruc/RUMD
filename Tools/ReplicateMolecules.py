#!/usr/bin/python3

"""
Usage: python3 ReplicateMolecules.py <config_filename> <top_filename> <num_molecules>
"""

import sys
import random
import gzip
import numpy

DIM = 3


# make separate functions for:
# 1 Reading single molecule config and top files
# 2 Generating a lattice


def ReadMoleculeFile(config_filename):
    config_file = open(config_filename)
    molecule_size = int(config_file.readline().strip())
    config_file.readline() #  comment line

    positions = numpy.zeros((molecule_size, DIM))
    types = []
    numTypes = 0
    for adx in range(molecule_size):
        line_items = config_file.readline().split()
        thisType = int(line_items[0])
        if thisType not in types:
            numTypes += 1
        types.append(thisType)
        pos = [float(x) for x in line_items[1:]]
        assert len(pos) == DIM
        positions[adx, :] = pos


    # not using masses so this is not necessarily the center of mass
    # if masses are available they could be passed as the third argument
    # to the average function
    com = numpy.average(positions, 0) # average over atom index (axis 0)



    max_dist_from_com = 0.0
    for adx in range(molecule_size):
        dist_from_com = numpy.sqrt(numpy.sum((positions[adx] - com)**2))
        if dist_from_com > max_dist_from_com:
            max_dist_from_com = dist_from_com

    return positions-com, types, numTypes, max_dist_from_com

def ReadTopologyFile(top_filename, molecule_size):
    top_file = open(top_filename)
    if top_file.readline().strip() != "[ bonds ]":
        raise ValueError("First line of .top file must be \"[ bonds ]\"")
    bondList = [[], [], []]
    nextLine = top_file.readline()

    # Read Bonds
    while nextLine:
        while nextLine.startswith(";"):
            nextLine = top_file.readline()
        if not nextLine:
            break
        if nextLine.startswith("["):
            break
        mol_idx, atom0, atom1, bond_type = [int(item) for item in nextLine.split()]
        if mol_idx != 0:
            raise ValueError("There should be a single molecule whose index is 0")
        bondList[0].append([atom0, atom1, bond_type])
        if atom0 < 0 or atom0 >= molecule_size or atom1 < 0 or atom1 >= molecule_size:
            raise ValueError("Bond contains atom numbers outside the allowed range (0 to %d)" % (molecule_size-1))
        nextLine = top_file.readline()

    # Read angles
    if nextLine.strip() == "[ angles ]":
        nextLine = top_file.readline()
        while nextLine:
            while nextLine.startswith(";"):
                nextLine = top_file.readline()
            if not nextLine:
                break
            if nextLine.startswith("["):
                break
            mol_idx, atom0, atom1, atom2, bond_type = [int(item) for item in nextLine.split()]
            if mol_idx != 0:
                raise ValueError("There should be a single molecule whose index is 0")
            bondList[1].append([atom0, atom1, atom2, bond_type])
            for atom in (atom0, atom1, atom2):
                if atom < 0 or atom >= molecule_size:
                    raise ValueError("Angle contains atom numbers outside the allowed range (0 to %d)" % (molecule_size-1))
            nextLine = top_file.readline()

    # Read dihedrals
    if nextLine.strip() == "[ dihedrals ]":
        nextLine = top_file.readline()
        while nextLine:
            while nextLine.startswith(";"):
                nextLine = top_file.readline()
            if not nextLine:
                break
            if nextLine.startswith("["):
                break
            mol_idx, atom0, atom1, atom2, atom3, bond_type = [int(item) for item in nextLine.split()]
            if mol_idx != 0:
                raise ValueError("There should be a single molecule whose index is 0")
            bondList[2].append([atom0, atom1, atom2, atom3, bond_type])
            for atom in (atom0, atom1, atom2, atom3):
                if atom < 0 or atom >= molecule_size:
                    raise ValueError("Dihedral contains atom numbers outside the allowed range (0 to %d)" % (molecule_size-1))
            nextLine = top_file.readline()

    return bondList

def Get3DRotationMatrix(alpha, beta, gamma):
    C, S = numpy.cos, numpy.sin
    dot = numpy.dot
    array = numpy.array

    xMtx = array([[1., 0., 0.],
                  [0., C(beta), -S(beta)],
                  [0., S(beta), C(beta)]])

    yMtx = array([[C(alpha), 0., S(alpha)],
                  [0., 1., 0.],
                  [-S(alpha), 0., C(alpha)]])

    zMtx = array([[C(gamma), -S(gamma), 0.],
                  [S(gamma), C(gamma), 0.],
                  [0., 0., 1.]])

    # using yxz convention for Euler angles (see Wikipedia entry on
    # rotation matrix)

    return dot(zMtx, (dot(xMtx, yMtx)))

def GetFCCLattice(Nx, Ny, Nz, a):
    array = numpy.array
    latticeSites = numpy.zeros((4*Nx*Ny*Nz, 3))
    siteIdx = 0
    for lx in range(Nx):
        for ly in range(Ny):
            for lz in range(Nz):
                base_pos = array([lx*a, ly*a, lz*a])
                latticeSites[siteIdx, :] = base_pos
                latticeSites[siteIdx+1, :] = base_pos + a*array([0., 0.5, 0.5])
                latticeSites[siteIdx+2, :] = base_pos + a*array([0.5, 0., 0.5])
                latticeSites[siteIdx+3, :] = base_pos + a*array([0.5, 0.5, 0.])
                siteIdx += 4

    return latticeSites


def Replicate(config_filename, top_filename, num_molecules, randomOrientation=True):
    output_config_filename = "start.xyz.gz"
    output_top_filename = "start.top"
    minimum_dist_between_mols = 1.0


    positions, types, numTypes, max_dist_from_com = ReadMoleculeFile(config_filename)
    molecule_size = len(positions)
    bondList = ReadTopologyFile(top_filename, molecule_size)

    # write the new configuration file

    if DIM != 3:
        raise ValueError("For now only fcc lattice supported, so DIM must be 3")
    # first get fcc lattice for the COM positions
    # choose number of lattice sites and lattice constant
    pi = numpy.pi
    a = numpy.sqrt(2.) * (minimum_dist_between_mols + 2.*max_dist_from_com)
    Nx = int(pow(num_molecules/4, 1./3) + 0.999)
    if Nx == 0:
        Nx = 1
    latticeSites = GetFCCLattice(Nx, Nx, Nx, a)

    output_config_file = gzip.open(output_config_filename, "wb")
    output_config_file.write("%d\n" % (num_molecules*molecule_size))
    ioformat = 1
    commentLine = "ioformat=%d boxLengths=%.6f,%.6f,%.6f numTypes=%d columns=type,x,y,z\n" %(ioformat, Nx*a, Nx*a, Nx*a, numTypes)

    output_config_file.write(commentLine)
    siteIdx = 0
    for mol_idx in range(num_molecules):
        # note: I take uniform distributions in the Euler angles for
        # simplicity. This does not sample a uniform distribution in the
        # space of 3D rotations
        if randomOrientation:
            assert DIM == 3
            euler_alpha = random.uniform(-pi, pi)
            euler_beta = random.uniform(0., pi)
            euler_gamma = random.uniform(-pi, pi)
            rotMtx = Get3DRotationMatrix(euler_alpha, euler_beta, euler_gamma)
        else:
            rotMtx = numpy.identity(DIM)

        for atomIdx in range(molecule_size):
            #nextPos = latticeSites[siteIdx] + positions[atomIdx]
            nextPos = latticeSites[siteIdx] + numpy.dot(rotMtx, positions[atomIdx])
            lineOut = "%d" % types[atomIdx]
            for cd in range(DIM):
                lineOut += " %.4f" % nextPos[cd]
            lineOut += "\n"
            output_config_file.write(lineOut)
        siteIdx += 1

    output_config_file.close()

    # write the new topology file
    top_file = open(output_top_filename, "w")

    # Write bonds
    top_file.write("[ bonds ]\n;\n") # comment line required to avoid seg fault!
    base_atom = 0
    for mol_idx in range(num_molecules):
        for bdx in range(len(bondList[0])):
            atom0, atom1, bond_type = bondList[0][bdx]
            top_file.write("%d %d %d %d\n" % (mol_idx,
                                              base_atom+atom0,
                                              base_atom+atom1,
                                              bond_type))
        base_atom += molecule_size

    # Write angles
    if bondList[1] != []:
        top_file.write("[ angles ]\n;\n") # comment line required to avoid seg fault!
        base_atom = 0
        for mol_idx in range(num_molecules):
            for bdx in range(len(bondList[1])):
                atom0, atom1, atom2, bond_type = bondList[1][bdx]
                top_file.write("%d %d %d %d %d\n" % (mol_idx,
                                                     base_atom+atom0,
                                                     base_atom+atom1,
                                                     base_atom+atom2,
                                                     bond_type))
            base_atom += molecule_size

    # Write dihedrals
    if bondList[2] != []:
        top_file.write("[ dihedrals ]\n;\n") # comment line required to avoid seg fault!
        base_atom = 0
        for mol_idx in range(num_molecules):
            for bdx in range(len(bondList[2])):
                atom0, atom1, atom2, atom3, bond_type = bondList[2][bdx]
                top_file.write("%d %d %d %d %d %d\n" % (mol_idx,
                                                        base_atom+atom0,
                                                        base_atom+atom1,
                                                        base_atom+atom2,
                                                        base_atom+atom3,
                                                        bond_type))

            base_atom += molecule_size
    top_file.close()

if __name__ == "__main__":

    if "--help" in sys.argv or len(sys.argv) != 4:
        print(__doc__)
        sys.exit(0)

    Replicate(sys.argv[1], sys.argv[2], int(sys.argv[3]))

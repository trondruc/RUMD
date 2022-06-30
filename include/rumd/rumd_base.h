#ifndef RUMD_BASE_H
#define RUMD_BASE_H

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

////////////////////////////////////////////////////////////
// This header defines constants in the RUMD code.
////////////////////////////////////////////////////////////

// The current version number.
const char VERSION[] = "3.5";

// The dimension of space (future release usage).
const unsigned int DIM = 3;

// Turn on or off periodic boundary in x,y,z-direction.
const bool periodicInX = true;
const bool periodicInY = true;
const bool periodicInZ = true;


// The number of parameters per interaction.
const unsigned int NumParam = 12;

// Defines the size to be allocated in kernels.
const unsigned int simulationBoxSize = 8;

const unsigned int rumd_ioformat = 2;


enum SortingScheme {SORT_X, SORT_XY, SORT_XYZ};

#endif // RUMD_BASE_H

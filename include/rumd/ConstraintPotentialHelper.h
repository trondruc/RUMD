#ifndef CONSTRAINTPOTENTIALHELPER_H
#define CONSTRAINTPOTENTIALHELPER_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/MoleculeData.h"
#include "rumd/Sample.h"

#include <list>

void BuildConstraintGraph( MoleculeData* M, std::list<float4> &constraintList );

#endif // CONSTRAINTPOTENTIALHELPER_H

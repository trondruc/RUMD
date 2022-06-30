#! /bin/bash

test ! -z "$PBS_O_WORKDIR" && cd "$PBS_O_WORKDIR"

mv src/rumd_algorithms.cu .
mv include/rumd/rumd_algorithms.h .

# float -> double
vim -c ":argdo %s/float/double/ge | update" +x src/*.cu
vim -c ":argdo %s/float/double/ge | update" +x src/*.cc
vim -c ":argdo %s/float/double/ge | update" +x src/*.py

vim -c ":argdo %s/float/double/ge | update" +x include/rumd/*.h

vim -c ":argdo %s/float/double/ge | update" +x Swig/*.i
vim -c ":argdo %s/float/double/ge | update" +x Swig/*.cc
vim -c ":argdo %s/float/double/ge | update" +x Swig/*.py

vim -c ":argdo %s/float/double/ge | update" +x Python/*.py

vim -c ":argdo %s/float/double/ge | update" +x Tools/*.cc
vim -c ":argdo %s/float/double/ge | update" +x Tools/*.h
vim -c ":argdo %s/float/double/ge | update" +x Tools/*.py
vim -c ":argdo %s/float/double/ge | update" +x Tools/*.i

# Float -> Double
vim -c ":argdo %s/Float/Double/ge | update" +x src/*.cu
vim -c ":argdo %s/Float/Double/ge | update" +x src/*.cc
vim -c ":argdo %s/Float/Double/ge | update" +x src/*.py

vim -c ":argdo %s/Float/Double/ge | update" +x include/rumd/*.h

vim -c ":argdo %s/Float/Double/ge | update" +x Swig/*.i
vim -c ":argdo %s/Float/Double/ge | update" +x Swig/*.cc
vim -c ":argdo %s/Float/Double/ge | update" +x Swig/*.py

vim -c ":argdo %s/Float/Double/ge | update" +x Python/*.py

vim -c ":argdo %s/Float/Double/ge | update" +x Tools/*.cc
vim -c ":argdo %s/Float/Double/ge | update" +x Tools/*.h
vim -c ":argdo %s/Float/Double/ge | update" +x Tools/*.py

# FLOAT -> DOUBLE
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x src/*.cu
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x src/*.cc
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x src/*.py

vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x include/rumd/*.h

vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Swig/*.i
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Swig/*.cc
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Swig/*.py

vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Python/*.py

vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Tools/*.cc
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Tools/*.h
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Tools/*.py
vim -c ":argdo %s/FLOAT/DOUBLE/ge | update" +x Tools/*.i

# back to float in Python/Autotune.py
vim -c "%s/double/float/g" +x Python/Autotune.py

# PyDouble_FromDouble -> PyFloat_FromDouble
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x src/*.py
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x Swig/*.py
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x Tools/*.py
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x src/*.i
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x Swig/*.i
vim -c ":argdo %s/PyDouble_FromDouble/PyFloat_FromDouble/ge | update" +x Tools/*.i

# int_as_double -> longlong_as_double and associated changes
vim -c ":argdo %s/int_as_double/longlong_as_double/ge | update" +x src/*.cu
vim -c ":argdo %s/double_as_int/double_as_longlong/ge | update" +x src/*.cu
vim -c ":argdo %s/int_as_double/longlong_as_double/ge | update" +x src/*.cc
vim -c ":argdo %s/double_as_int/double_as_longlong/ge | update" +x src/*.cc

vim -c ":argdo %s/\ \ int\ i_val/\ \ unsigned\ long\ long\ int\ i_val/ge | update" +x src/*.cu
vim -c ":argdo %s/\ \ int\ i_val/\ \ unsigned\ long\ long\ int\ i_val/ge | update" +x src/*.cc
vim -c ":argdo %s/\ \ int\ tmp0/\ \ unsigned\ long\ long\ int\ tmp0/ge | update" +x src/*.cc
vim -c ":argdo %s/\ \ int\ tmp0/\ \ unsigned\ long\ long\ int\ tmp0/ge | update" +x src/*.cu
vim -c ":argdo %s/\ \ int\ tmp1/\ \ unsigned\ long\ long\ int\ tmp1/ge | update" +x src/*.cc
vim -c ":argdo %s/\ \ int\ tmp1/\ \ unsigned\ long\ long\ int\ tmp1/ge | update" +x src/*.cu

vim -c ":argdo %s/atomicCAS((int/atomicCAS((unsigned\ long\ long\ int/ge | update" +x src/*.cu
vim -c ":argdo %s/atomicCAS((int/atomicCAS((unsigned\ long\ long\ int/ge | update" +x src/*.cc



# cdouble -> cfloat in include/rumd/rumd_algorithm.h
vim -c "%s/cdouble/cfloat/g" +x include/rumd/rumd_algorithms.h
vim -c "%s/cdouble/cfloat/g" +x src/MoleculeData.cu
vim -c "%s/cdouble/cfloat/g" +x src/rumd_algorithms_CPU.cu

# LOAD in include/rumd/rumd_technical.h
vim -c "%s/__ldg(&(x))/(x)/g" +x include/rumd/rumd_technical.h

# PyFloat_AsDouble and PyFloat_Check in Swig/vectorTypeMap.i
vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x src/*.i
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x src/*.i
vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x Swig/*.i
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x Swig/*.i
vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x Tools/*.i
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x Tools/*.i

vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x src/*.py
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x src/*.py
vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x Swig/*.py
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x Swig/*.py
vim -c ":argdo %s/PyDouble_AsDouble/PyFloat_AsDouble/ge | update" +x Tools/*.py
vim -c ":argdo %s/PyDouble_Check/PyFloat_Check/ge | update" +x Tools/*.py


# CalcF should return a float (change return type in .h, .cc and the type of elapsedTime)
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/AnglePotential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/BondPotential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/ConstraintPotential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/DihedralPotential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/CollectiveDensityField.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/HarmonicUmbrella.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/TetheredGroup.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/WallPotential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/Potential.h
vim -c "%s/double\ CalcF/float\ CalcF/g" +x include/rumd/PairPotential.h

vim -c "%s/double\ AngleCosSq::CalcF/float\ AngleCosSq::CalcF/g" +x src/AnglePotential.cu
vim -c "%s/double\ AngleSq::CalcF/float\ AngleSq::CalcF/g" +x src/AnglePotential.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/AnglePotential.cu

vim -c "%s/double\ BondHarmonic::CalcF/float\ BondHarmonic::CalcF/g" +x src/BondPotential.cu
vim -c "%s/double\ BondFENE::CalcF/float\ BondFENE::CalcF/g" +x src/BondPotential.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/BondPotential.cu

vim -c "%s/double\ ConstraintPotential::CalcF/float\ ConstraintPotential::CalcF/g" +x src/ConstraintPotential.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/ConstraintPotential.cu

vim -c "%s/double\ DihedralRyckaert::CalcF/float\ DihedralRyckaert::CalcF/g" +x src/DihedralPotential.cu
vim -c "%s/double\ PeriodicDihedral::CalcF/float\ PeriodicDihedral::CalcF/g" +x src/DihedralPotential.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/DihedralPotential.cu

vim -c "%s/double\ CollectiveDensityField::CalcF/float\ CollectiveDensityField::CalcF/g" +x src/CollectiveDensityField.cu

vim -c "%s/double\ HarmonicUmbrella::CalcF/float\ HarmonicUmbrella::CalcF/g" +x src/HarmonicUmbrella.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/HarmonicUmbrella.cu

vim -c "%s/double\ TetheredGroup::CalcF/float\ TetheredGroup::CalcF/g" +x src/TetheredGroup.cu
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/TetheredGroup.cu

vim -c "%s/double\ Wall_LJ_9_3::CalcF/float\ Wall_LJ_9_3::CalcF/g" +x src/WallPotential.cu

vim -c "%s/double %(class_name)s::CalcF/float %(class_name)s::CalcF/g" +x src/Generate_PP_FunctionBodies.py
vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x src/Generate_PP_FunctionBodies.py

vim -c "%s/double\ elapsedTime/float\ elapsedTime/g" +x include/rumd/NeighborList.h

# NB Nov 2017
# Back to float in Device::Time
vim -c "%s/double\ Time/float\ Time/g" +x include/rumd/Device.h
vim -c "%s/double\ Device/float\ Device/g" +x src/Device.cu
vim -c "%s/double\ elapsed_time/float\ elapsed_time/g" +x src/Device.cu
# Back to cfloat in rumd_init_conf_exec.cc
vim -c "%s/cdouble/cfloat/g" +x Tools/rumd_init_conf_exec.cc


# scanf and %lf
vim -c "%s/sizeof(int)/sizeof(long\ long\ int)/g" +x src/ConfigurationWriterReader.cu

vim -c "%s/sscanf(listItem,\ \"%f/sscanf(listItem,\ \"%lf/g" +x src/ConfigurationMetaData.cc
vim -c "%s/sscanf(argument,\ \"%f/sscanf(argument,\ \"%lf/g" +x src/ConfigurationMetaData.cc

vim -c "%s/sscanf(lineBuffer+offset,\ \"%d\ %f\ %f\ %f/sscanf(lineBuffer+offset,\ \"%d\ %lf\ %lf\ %lf/g" +x src/ConfigurationWriterReader.cu
vim -c "%s/sscanf(lineBuffer+offset,\ \"%f\ %f\ %f/sscanf(lineBuffer+offset,\ \"%lf\ %lf\ %lf/g" +x src/ConfigurationWriterReader.cu
vim -c "%s/sscanf(lineBuffer+offset,\ \"%f%n/sscanf(lineBuffer+offset,\ \"%lf%n/g" +x src/ConfigurationWriterReader.cu

vim -c "%s/sscanf(argument,\ \"%f/sscanf(argument,\ \"%lf/g" +x src/EnergiesMetaData.cc

vim -c "%s/sscanf(pStr,\ \"%f/sscanf(pStr,\ \"%lf/g" +x src/ParseInfoString.cc

vim -c "%s/sscanf(lineBuffer+offset,\ \"%u\ %f\ %f\ %f/sscanf(lineBuffer+offset,\ \"%u\ %lf\ %lf\ %lf/g" +x Tools/rumd_data.h
vim -c "%s/sscanf(lineBuffer+offset,\ \"%f\ %f\ %f/sscanf(lineBuffer+offset,\ \"%lf\ %lf\ %lf/g" +x Tools/rumd_data.h
vim -c "%s/sscanf(lineBuffer+offset,\ \"%f%n/sscanf(lineBuffer+offset,\ \"%lf%n/g" +x Tools/rumd_data.h


vim -c "%s/sscanf(ptrTo2ndEqual+1,\ \"%f/sscanf(ptrTo2ndEqual+1,\ \"%lf/g" +x Tools/rumd_init_conf_mol.cc
vim -c "%s/sscanf(ptrToComa+1,\ \"%f/sscanf(ptrToComa+1+1,\ \"%lf/g" +x Tools/rumd_init_conf_mol.cc

mv rumd_algorithms.cu src/
mv rumd_algorithms.h include/rumd
echo "DO NOT FORGET TO COMMENT OUT in rumd_algorithms.cu template instantiations of float versions of solveLinearSystems and solveTridiagonalLinearSystems and uncomment the double versions. Also replace curandGenerateNormal with curandGenerateNormalDouble in IntegratorNPTLangevin (this could be included in the script)"

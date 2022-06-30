# Makefile for rumd

NVCCFLAGS = --compiler-options -fno-strict-aliasing,-Wall,-Wextra,-fPIC $(INCLUDE) -DUNIX -O3 $(GENCODE) -dc

NVCCFLAGS += $(NVCCBIN)

# to show messages from assembler include:
# -Xptxas --verbose 

# Workaround for boost >= 1.55 and CUDA < 6.5
# symptom: nvcc: include/boost/assert.hpp:102:47: Error: »noinline« was not defined in this sco
# fixed in nvcc in CUDA 6.5
#NVCCFLAGS += -DBOOST_NOINLINE='__attribute__ ((noinline))'
#
# Workaround for issue with CUDA 7.5 and memcpy
# symptom:
#  /usr/include/string.h: In function 'void* __mempcpy_inline(void*, const void*, size_t)':
#  /usr/include/string.h:652:42: error: 'memcpy' was not declared in this scope
#    return (char *) memcpy (__dest, __src, __n) + __n;
#NVCCFLAGS += -D_FORCE_INLINES

LIBS= -lcudart -lz -lboost_filesystem -lcurand

CXXFLAGS = -O3 -Wall -Wextra -Weffc++ $(INCLUDE) 

# dependency analysis
SOURCES = $(wildcard *.cc)
CU_SOURCES = $(wildcard *.cu)

DEPENDS = .depend.mk

build: $(DEPENDS) library

$(DEPENDS):
	$(CXX) -MM $(SOURCES) $(INCLUDE)  >>$(DEPENDS)
	$(CXX) -MM -x c++ $(filter-out PairPotential.cu CalculateMolecularStress.cu, $(CU_SOURCES)) $(INCLUDE)  >>$(DEPENDS)
	echo "PairPotential.o: PairPotential.cu ../include/rumd/rumd_base.h ../include/rumd/rumd_utils.h ../include/rumd/PairPotential.h  ../include/rumd/RUMD_Error.h ../include/rumd/Potential.h  ../include/rumd/KernelPlan.h ../include/rumd/NeighborList.h  ../include/rumd/rumd_technical.h ../include/rumd/ParticleData.h  ../include/rumd/SimulationBox.h PairPotentialFunctionBodies.inc" >>$(DEPENDS)
	echo "CalculateMolecularStress.o: CalculateMolecularStress.cu ../include/rumd/MoleculeData.h  ../include/rumd/Sample.h ../include/rumd/rumd_base.h  ../include/rumd/rumd_technical.h ../include/rumd/LogLinOutputManager.h  ../include/rumd/LogLin.h ../include/rumd/SimulationBox.h  ../include/rumd/RUMD_Error.h ../include/rumd/ParticleData.h ../include/rumd/rumd_base.h ../include/rumd/NeighborList.h ../include/rumd/KernelPlan.h ../include/rumd/PairPotential.h  ../include/rumd/Potential.h ../include/rumd/ConstraintPotential.h ../include/rumd/BondPotential.h ../include/rumd/AnglePotential.h ../include/rumd/DihedralPotential.h \
	 MolecularStress_Instantiation.inc" >>$(DEPENDS)



-include $(DEPENDS)

%.o: %.cc 
	@echo "[CXX] -c" $<
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $<

# pattern rule for object files from CUDA source
%.o : %.cu
	@echo "[NVCC] -c" $<
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

# special case, uses generated code
PairPotential.o:
	@echo "[NVCC] -c" $<
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

PairPotentialFunctionBodies.inc: Generate_PP_FunctionBodies.py ../include/rumd/PairPotential.h
	python3 $<

MolecularStress_Instantiation.inc: Generate_MolSts_Inst.py ../include/rumd/PairPotential.h
	python3 $<

# shared code for all executables

O_DEPS = $(patsubst %.cc, %.o, $(SOURCES)) $(patsubst %.cu, %.o, $(CU_SOURCES))

library: ../lib/librumd.a

# The device linker needs an architecture (and only one may be specified).
# It should be at least the compute capability of the actual architecture
../lib/rumd_dlink.o: $(O_DEPS)
	$(NVCC) $(NVCCBIN) -arch=$(DEVICE_LINK_ARCH) -Xcompiler '-fPIC' -dlink $^ -o $@

../lib/librumd.a: $(O_DEPS) ../lib/rumd_dlink.o
	$(NVCC) $(NVCCBIN) -arch=$(DEVICE_LINK_ARCH) --lib -o $@ $^

# making the shared library allows only one of the device-linked files
../lib/librumd.so: $(O_DEPS)  ../lib/rumd_dlink.o
	$(NVCC) $(NVCCBIN) --shared -o $@ $^

clean:
	@for f in $(O_DEPS) $(DEPENDS) ../lib/librumd.a ../lib/rumd_dlink.o PairPotentialFunctionBodies.inc MolecularStress_Instantiation.inc ; \
	do if [ -f $$f ] ; then rm $$f ; fi ; done

.PHONY: build install clean library

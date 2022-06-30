# Makefile for Tools


CXXFLAGS = -O3 -Wall -Wextra -Weffc++ -fPIC

PYTHON3_INCLUDE = -DHAVE_CONFIG_H $(shell python3-config --includes)

CPPFLAGS += $(INCLUDE) 

SRC=$(filter-out tools_wrap.cc,$(wildcard *.cc))
OBJS=$(patsubst %.cc, %.o, $(SRC))

DEPENDS = .depend.mk

EXECUTABLES=rumd_rdf rumd_msd rumd_stats rumd_init_conf rumd_init_conf_mol rumd_vhc rumd_bonds rumd_rouse

SWIG_INCLUDE = -I../Swig


all: $(EXECUTABLES) ../Python/rumd/_Tools.so

$(DEPENDS):
	$(CXX) -MM $(SRC) $(INCLUDE)  >$(DEPENDS)

-include $(DEPENDS)

rumd_init_conf_mol: LIBS = -lm
rumd_init_conf_mol: rumd_init_conf_mol.o

rumd_rdf: LIBS = -lz -lrumd
rumd_rdf: rumd_rdf_exec.o rumd_rdf.o

rumd_msd: LIBS = -lz -lrumd
rumd_msd: rumd_msd_exec.o rumd_msd.o

rumd_stats: LIBS = -lz -lrumd
rumd_stats: rumd_stats_exec.o rumd_stats.o

rumd_init_conf: LIBS = -lboost_iostreams
rumd_init_conf: rumd_init_conf_exec.o rumd_init_conf.o

rumd_vhc: LIBS = -lz -lrumd
rumd_vhc: rumd_vhc_exec.o rumd_vhc.o

rumd_bonds: LIBS = -lz -lrumd
rumd_bonds: rumd_bonds_exec.o rumd_bonds.o

rumd_rouse: LIBS = -lz -lrumd
rumd_rouse: rumd_rouse_exec.o rumd_rouse.o

TOOLOBJ = rumd_stats.o rumd_rdf.o rumd_msd.o

%: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cc
	@echo "[CXX] -c" $<
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<

# rule with sideeffect, writes Tools.py to ../Python/rumd
tools_wrap.cc: Tools.i rumd_stats.h rumd_rdf.h rumd_msd.h
	$(SWIG) -Wall -c++ -python -outdir ../Python/rumd -module Tools -o $@ $(SWIG_INCLUDE) Tools.i

tools_wrap.o: tools_wrap.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(PYTHON3_INCLUDE) -c $< -o $@

../Python/rumd/_Tools.so: LIBS=-lz -lrumd
../Python/rumd/_Tools.so: tools_wrap.o $(TOOLOBJ)
	$(CXX) -shared tools_wrap.o $(TOOLOBJ) $(LDFLAGS) $(LIBS) -o $@


clean:	
	-rm -f $(OBJS) $(EXECUTABLES) tools_wrap.cc tools_wrap.o ../Python/rumd/_Tools.so ../Python/rumd/Tools.py .depend.mk

.PHONY: all clean

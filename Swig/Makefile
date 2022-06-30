# Makefile for Swig (interfacing to Python)

CXXFLAGS = -O3 -Wall -Wextra $(INCLUDE)
LIBS = -lz -lrumd  -lcudart -lcurand
SVNDEF = -D'SVN_REV="$(shell svnversion -n . || echo -n unknown)"'

PYTHON3_INCLUDE = -DHAVE_CONFIG_H $(shell python3-config --includes)

SWIG_SRCINC=$(filter-out rumd.i,$(wildcard *.i))

all: ../Python/rumd/_rumd.so

rumd_wrap.cc: rumd.i $(SWIG_SRCINC)
	$(SWIG) -Wall -c++ -python -module rumd -o $@ $(INCLUDE) $<
	mv rumd.py ../Python/rumd/rumdswig.py

rumd_wrap.o: rumd_wrap.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(PYTHON3_INCLUDE) $(SVNDEF) -fPIC -c $< -o $@

../Python/rumd/_rumd.so: rumd_wrap.o ../lib/librumd.a
	$(CXX) $(LDFLAGS) -shared rumd_wrap.o  $(LIBS) -o $@

clean:
	rm -f rumd_wrap.o rumd_wrap.cc ../Python/rumd/_rumd.so ../Python/rumd/rumdswig.py

.PHONY: all clean

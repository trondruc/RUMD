# Makefile for RUMD

DESTDIR = /

CC = gcc
CXX = g++
SWIG = swig
NVCC = nvcc

INCLUDE += -I$(PWD)/include
LDFLAGS += -L$(PWD)/lib

# if you installed cuda using the nvidia installer, the paths are usually
# INCLUDE += -I/usr/local/cuda/include
# LDFLAGS += -L/usr/local/cuda/lib64

# building with CUDA 8.0, 9.0, 9.1, 9.2
#GENCODE = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61
#DEVICE_LINK_ARCH = sm_61

# building with CUDA 10.1
GENCODE = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75
DEVICE_LINK_ARCH = sm_75

# nvcc from CUDA 8.0 doesn't work with gcc >= 6 for compiling host code, but can work with clang-3.8
NVCCREL=$(word 2,$(shell nvcc --version | grep release | cut -d, -f2))
ifeq ($(NVCCREL),8.0)
CLANG=$(shell which clang-3.8)
ifneq ($(CLANG),)
	NVCCBIN = -ccbin $(CLANG)
endif
endif

export DESTDIR CC CXX SWIG NVCC NVCCBIN DESTDIR INCLUDE LDFLAGS GENCODE DEVICE_LINK_ARCH

all:
	$(MAKE) -C src
	$(MAKE) -C Tools
	$(MAKE) -C Swig

test:
	$(MAKE) -C Test/ConsistencyTests

testclean:
	@$(MAKE) -C Test/ConsistencyTests testclean

testmemcheck:
	@$(MAKE) -C Test/ConsistencyTests testmemcheck

testclean_memcheck:
	@$(MAKE) -C Test/ConsistencyTests testclean_memcheck

teststress:
	@$(MAKE) -C Test/ConsistencyTests teststress

clean:
	@$(MAKE) -C src clean
	@$(MAKE) -C Python clean
	@$(MAKE) -C Tools clean
	@$(MAKE) -C Swig clean
	@$(MAKE) -C doc clean
	@$(MAKE) -C Test/ConsistencyTests clean

doc:
	$(MAKE) -C doc

.PHONY: all clean doc

#!/usr/bin/make
# Makefile
# (c) Mohammad Mofrad, 2018
# (e) m.hasanzadeh.mofrad@gmail.com 
# make MACROS=-DHAS_WEIGHT to enable weights

CXX = g++
MPI_CXX = mpicxx
CXX_FLAGS = -std=c++14 -fpermissive
#THREADED = -fopenmp -D_GLIBCXX_PARALLEL
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native
#DEBUG = -g -fsanitize=undefined,address -lasan -lubsan
obj=main

.PHONY: all clean

all: clean app

app:
	$(MPI_CXX) $(CXX_FLAGS) $(DEBUG) $(OPTIMIZE) $(THREADED) $(MACROS) -o $(obj) $(obj).cpp
clean:
	rm -rf $(obj)
	

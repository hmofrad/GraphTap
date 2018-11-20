#!/usr/bin/make
# Makefile
# (c) Mohammad Mofrad, 2018
# (e) m.hasanzadeh.mofrad@gmail.com 
# make MACROS=-DHAS_WEIGHT to enable weights

CXX = g++
MPI_CXX = mpicxx
CXX_FLAGS = -std=c++14 -fpermissive
MACROS=-DHAS_WEIGHT
# Definitely Turn this on for faster binaries
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native

# Do not turn on the DEBUG flag unless using mpich
#DEBUG = -g -fsanitize=undefined,address -lasan -lubsan

.PHONY: all clean

objs  = deg pr tc cc bfs sssp

all: $(objs)

$(objs): %: src/apps/%.cpp
	@mkdir -p bin
	$(MPI_CXX) $(CXX_FLAGS) $(DEBUG) $(OPTIMIZE)           -o bin/$@   -I src $<
	$(MPI_CXX) $(CXX_FLAGS) $(DEBUG) $(OPTIMIZE) $(MACROS) -o bin/$@_w -I src $<

clean:
	rm -rf bin

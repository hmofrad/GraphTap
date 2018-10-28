#!/usr/bin/make
# Makefile
# (c) Mohammad Mofrad, 2018
# (e) m.hasanzadeh.mofrad@gmail.com 
# make MACROS=-DHAS_WEIGHT to enable weights

CXX = g++
MPI_CXX = mpicxx
CXX_FLAGS = -std=c++14 -fpermissive

# Definitely Turn this on for fast binaries
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native

# Do not turn on the DEBUG flag unless using mpich
#DEBUG = -g -fsanitize=undefined,address -lasan -lubsan

.PHONY: all clean

objs = deg pr tc cc bfs # deg pr tc bfs sssp
# deg pr tc cc
#objs = deg pr tc cc

all: $(objs)

$(objs): %: src/apps/%.cpp
	@mkdir -p bin
	$(MPI_CXX) $(CXX_FLAGS) $(DEBUG) $(OPTIMIZE) $(THREADED) $(MACROS) -o bin/$@ -I src $<
    
clean:
	rm -rf bin
#!/usr/bin/make
# Makefile
# (c) Mohammad Hasanzadeh Mofrad, 2019
# (e) m.hasanzadeh.mofrad@gmail.com
# make TIMING=-DTIMING to enable time counters
# TIMING = -DTIMING

CXX = g++
MPI_CXX = mpicxx
SKIPPED_CXX_WARNINGS = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized
CXX_FLAGS = -std=c++14 -fpermissive $(SKIPPED_CXX_WARNINGS)
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native
#DEBUG = -g  -fsanitize=undefined,address -lasan -lubsan

.PHONY: dir all test misc clean
objs   = deg pr cc bfs
objs_w = sssp

all: dir $(objs) $(objs_w)

dir:
	@mkdir -p bin

$(objs): %: src/apps/%.cpp
	$(MPI_CXX) $(CXX_FLAGS) $(OPTIMIZE) $(DEBUG) $(TIMING) -o bin/$@   -I src $<

$(objs_w): %: src/apps/%.cpp
	$(MPI_CXX) $(CXX_FLAGS) $(OPTIMIZE) $(DEBUG) $(TIMING) -DHAS_WEIGHT -o bin/$@ -I src $<

test: dir
	$(CXX) $(CXX_FLAGS) $(OPTIMIZE) -o bin/main src/singlenode/main.cpp

misc: dir
	$(CXX) $(CXX_FLAGS) $(OPTIMIZE) -o bin/converter src/misc/converter.cpp

clean:
	rm -rf bin

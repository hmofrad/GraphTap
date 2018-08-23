#!/usr/bin/make

CXX = g++
MPI_CXX = mpicxx
CXX_FLAGS = -std=c++14 -fpermissive
DEBUG = -g -fsanitize=undefined,address -lasan -lubsan
obj=main

.PHONY: all clean

all: clean app

app:
	$(MPI_CXX) $(CXX_FLAGS) $(DEBUG) -o $(obj) $(obj).cpp
clean:
	rm -rf $(obj)
	

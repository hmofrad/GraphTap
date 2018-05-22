#!/usr/bin/make

CXX = g++
MPI_CXX = mpicxx

obj=proc2tile

.PHONY: all clean

all: clean app

app:
	$(MPI_CXX) -o $(obj) $(obj).cpp
clean:
	rm -rf $(obj)
	

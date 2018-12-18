/*
 * spmspv.cpp: Unit test kernels 
                   Compressed Sparse Column (CSC)  for SpMV/SpMSpV
            Double Compressed Sparse Column (DCSC) for SpMV/SpMSpV
   Optmized Double Compressed Sparse Column (ODCSC) (LA3) for SpMSpV
            Triple Compressed Sparse Column (TCSC) (GraphTap) for SpMSpV
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 * Compile commnad:
 * g++ -o spmspv spmspv.cpp -std=c++14
 */
 
#include <iostream>
#include <vector>

#include "csc.hpp"
#include "csc_.hpp"
#include "dcsc.hpp"
#include "dcsc_.hpp"
#include "tcsc.hpp"

int main(int argc, char **argv)
{ 
    if(argc != 5)
    {
        std::cout << "\"Usage: " << argv[0] << " <CSC SpMV|DCSC SpMV|CSC SpMSpV|DCSC SpMSpV|ODCSC(LA3)|TCSC SpMSpV(GT)|> <iterations> <file_path> <num_vertices>\"" << std::endl;
        exit(1);
    }

    std::string file_path = argv[1];
    uint32_t nvertices = std::atoi(argv[2]) + 1; // For vertex id 0
    uint32_t niters = std::atoi(argv[3]);
    int which = std::atoi(argv[4]);
    
    if(which == 0)
    {
        CSC csc(file_path, nvertices, niters);
        csc.run_pagerank();
    }
    else if(which == 1)
    {
        DCSC dcsc(file_path, nvertices, niters);
        dcsc.run_pagerank();
    }
    else if(which == 2)
    {
        CSC_ csc(file_path, nvertices, niters);
        csc.run_pagerank();
    }
    else if(which == 3)
    {
        DCSC_ dcsc(file_path, nvertices, niters);
        dcsc.run_pagerank();
    }
    else if(which == 4)
    {
        TCSC tcsc(file_path, nvertices, niters);
        tcsc.run_pagerank();
    }
    else if(which == 5)
    {
        //ODCSC odcsc(file_path, nvertices, niters);
        //odcsc.run_pagerank();
    }
    return(0);
}

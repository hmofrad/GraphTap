/*
 * spmv.cpp: Unit test for SpMV Kernels
                   Compressed Sparse Column (CSC)  for SpMV/SpMSpV
   Double          Compressed Sparse Column (DCSC) for SpMV/SpMSpV
   Optmized Double Compressed Sparse Column (ODCSC) (LA3) for SpMSpV
            Triple Compressed Sparse Column (TCSC) (GraphTap) for SpMSpV
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 * Standalone compile commnad:
 * g++ -o spmspv spmspv.cpp -std=c++14
 */
 
#include <iostream>
#include <cstdlib> 
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <sys/mman.h>
#include <cstring> 
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <chrono>


struct Triple
{
    uint32_t row;
    uint32_t col;
};


struct ColSort
{
    bool operator()(const struct Triple &a, const struct Triple &b)
    {
        return((a.col == b.col) ? (a.row < b.row) : (a.col < b.col));
    }
};
/* CSC/TCSC arrays */
uint32_t nnz;
uint32_t ncols_plus_one;
uint32_t *A;
uint32_t *IA;
uint32_t *JA;

/* CSC/TCSC SpMV vectors */
std::vector<uint32_t> y;
std::vector<uint32_t> x;

uint32_t num_vertices;
uint32_t num_iter;
uint32_t iter;
uint32_t nOps = 0;
std::vector<struct Triple> *triples;
std::vector<uint32_t> values;
uint32_t value = 0;
uint64_t size = 0;
uint64_t extra = 0;

std::chrono::steady_clock::time_point begin;
std::chrono::steady_clock::time_point end;

#include "csc.cpp"
#include "csc_d.cpp"
#include "dcsc.cpp"
#include "odcsc.cpp"
#include "tcsc.cpp"

void read_binary(std::string filepath)
{
    // Open graph file.
    std::ifstream fin(filepath.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file");
        exit(1); 
    }
    
    // Obtain filesize
    uint64_t nedges = 0, filesize = 0, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    struct Triple triple;
    struct Triple pair;
    while (offset < filesize)
    {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        
        if(fin.gcount() != sizeof(struct Triple))
        {
            fprintf(stderr, "read() failure\n");
            exit(1);
        }
        
        nedges++;
        offset += sizeof(struct Triple);
        //printf("row=%d col=%d\n", triple.row, triple.col);
        triples->push_back(triple);
    }
    fin.close();
    assert(offset == filesize);
    
    printf("[x]I/O for %s is done: Read %lu edges\n", filepath.c_str(), nedges);
}

int main(int argc, char **argv)
{ 
    if(argc != 5)
    {
        std::cout << "\"Usage: " << argv[0] << " <CSC|DCSC|ODCSC(LA3)|TCSC(GraphTap)> <iterations> <file_path> <num_vertices>\"" << std::endl;
        exit(1);
    }
    
    printf("[x]SpMV kernel unit test...\n");
    int which = std::atoi(argv[1]);
    num_iter = std::atoi(argv[2]);
    std::string file_path = argv[3];
    num_vertices = std::atoi(argv[4]) + 1; // For 0
    triples = new std::vector<struct Triple>;
    read_binary(file_path);

    ColSort f_col;
    std::sort(triples->begin(), triples->end(), f_col);
    
    if(which == 0)
    {
        run_csc();
        //walk_csc();
        begin = std::chrono::steady_clock::now();
            init_csc_vecs();
            for(iter = 0; iter < num_iter; iter++)
                spmv_csc();
            done_csc();
        end = std::chrono::steady_clock::now();
        std::cout << "CSC SpMV ";
    }
    if(which == 1)
    {
        filtering_csc_d(num_vertices);
        run_csc_d();
        //walk_csc_d();
        begin = std::chrono::steady_clock::now();
            init_csc_d_vecs();
            for(iter = 0; iter < num_iter; iter++)
                spmv_csc_d();
            done_csc_d();
        end = std::chrono::steady_clock::now();
        std::cout << "CSC SpMSpV ";
    }
    else if(which == 2)
    {
        filtering_dcsc(num_vertices);
        run_dcsc();
        //walk_dcsc();
        begin = std::chrono::steady_clock::now();
            init_dcsc_vecs();
            for(iter = 0; iter < num_iter; iter++)
                spmv_dcsc();
            done_dcsc();
        end = std::chrono::steady_clock::now();
        std::cout << "DCSC SpMV ";
    }
    else if(which == 3)
    {
        filtering_dcsc(num_vertices);
        run_dcsc();
        //walk_dcsc();
        begin = std::chrono::steady_clock::now();
            init_dcsc_vecs();
            for(iter = 0; iter < num_iter; iter++)
                spmv_dcsc();
            done_dcsc();
        end = std::chrono::steady_clock::now();
        std::cout << "DCSC SpMSpV ";
    }
    else if(which == 4)
    {
        triples_regulars = new std::vector<struct Triple>;
        triples_sources = new std::vector<struct Triple>;
        classification_odcsc(num_vertices);
        run_odcsc();
        //walk_odcsc_regulars();
        //walk_odcsc_sources();
        begin = std::chrono::steady_clock::now();
            init_odcsc_vecs();
            for(uint32_t i = 0; i < num_iter; i++)
                spmv_odcsc();
            done_odcsc();
        end = std::chrono::steady_clock::now();
        
        triples_regulars->clear();
        triples_sources->clear();
        extra = ((nentries_regulars * sizeof(Edge)) + (nentries_sources * sizeof(Edge)));
        std::cout << "odcsc (LA3) SpMSpV ";
    }
    else if(which == 5)
    {
        filtering(num_vertices);
        run_tcsc();
        //walk_tcsc();
        begin = std::chrono::steady_clock::now();
            init_tcsc_vecs();
            for(iter = 0; iter < num_iter; iter++)
                spmv_tcsc();
            done_tcsc();
        end = std::chrono::steady_clock::now();
        std::cout << "TCSC (GraphTap) SpMSpV ";
    }
    triples->clear();
    
    std::cout << "Stats:" << std::endl;
    std::cout << "    Utilized Memory: " << size / 1e9 << " G" << std::endl;
    if(std::atoi(argv[1]) == 2)
    std::cout << "    Extra    Memory: " << extra / 1e9 << " G (extra per iteration)" << std::endl;
    std::cout << "    Elapsed time:    " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1e6 << " sec" << std::endl;
    std::cout << "    Final value:     " << value <<std::endl;
    std::cout << "    Num SpMV Ops:    " << nOps <<std::endl;
    return(0);
}

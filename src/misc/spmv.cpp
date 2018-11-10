/*
 * spmv.cpp: Unit test for SpMV Kernel
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 * Standalone compile commnad:
 * g++ -o spmv spmv.cpp -std=c++14
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
uint32_t num_vertices;
uint32_t num_iter;
uint32_t iter;
std::vector<struct Triple> *triples;
std::vector<uint32_t> values;
uint32_t value = 0;
uint64_t size = 0;
uint64_t extra = 0;

std::chrono::steady_clock::time_point begin;
std::chrono::steady_clock::time_point end;

#include "graphtap.cpp"
#include "la3.cpp"

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
    
    printf("[x]I/O is done: Read %lu edges\n", nedges);
}

int main(int argc, char **argv)
{ 
    if(argc != 5)
    {
        std::cout << "\"Usage: " << argv[0] << " <GraphTap(0)|LA3(1)> <iterations> <file_path> <num_vertices>\"" << std::endl;
        exit(1);
    }
    
    printf("[x]SpMV kernel unit test...\n");
    bool which = std::atoi(argv[1]) == 0 ? false : true;
    num_iter = std::atoi(argv[2]);
    std::string file_path = argv[3];
    num_vertices = std::atoi(argv[4]) + 1; // For 0
    triples = new std::vector<struct Triple>;
    read_binary(file_path);

    ColSort f_col;
    std::sort(triples->begin(), triples->end(), f_col);
    if(not which)
    {
        filtering(num_vertices);
        csc();
        
        begin = std::chrono::steady_clock::now();
            init();
            for(iter = 0; iter < num_iter; iter++)
                spmv();
            done();
        end = std::chrono::steady_clock::now();
        
    }
    else
    {
        triples_regulars = new std::vector<struct Triple>;
        triples_sources = new std::vector<struct Triple>;
        classification(num_vertices);
        csc_la3();
        begin = std::chrono::steady_clock::now();
            init_la3();
            for(uint32_t i = 0; i < num_iter; i++)
                spmv_la3();
            done_la3();
        end = std::chrono::steady_clock::now();
        
        triples_regulars->clear();
        triples_sources->clear();
        extra = ((nentries_regulars * sizeof(Edge)) + (nentries_sources * sizeof(Edge)));
    }
    triples->clear();
    
    std::cout << "Stats:" << std::endl;
    std::cout << "    Memory: " << size / 1e3 << " K" << std::endl;
    if(std::atoi(argv[1]))
    std::cout << "    Memory: " << extra / 1e3 << " K (extra per iteration)" << std::endl;
    std::cout << "    Time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1e6 <<std::endl;
    std::cout << "    Value: " << value <<std::endl;
	return(0);
}
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

std::vector<struct Triple> *triples;
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
    
    printf("\n%s: Read %lu edges\n", filepath.c_str(), nedges);
}

int main(int argc, char **argv)
{ 
    if(argc != 4)
    {
        std::cout << "\"Usage: " << argv[0] << " <GraphTap(0)|LA3(1)> <file_path> <num_vertices>\"" << std::endl;
        exit(1);
    }
    
    printf("SpMV kernel unit test...\n");
    std::string file_path = argv[2]; 
    uint32_t num_vertices = std::atoi(argv[3]) + 1; // For 0
    triples = new std::vector<struct Triple>;
    read_binary(file_path);
    //std::sort(tile.triples->begin(), tile.triples->end(), f_row);
    //for(auto& triple: *triples)
    //    printf("%d %d\n", triple.row, triple.col);

    ColSort f_col;
    std::sort(triples->begin(), triples->end(), f_col);
    //for(auto &triple: *triples)
    //    printf("row=%d col=%d\n", triple.row, triple.col);
    if(not std::atoi(argv[1]))
    {
        filtering(num_vertices);
        init_csc(triples->size(), nnz_cols);
        popu_csc();
        //walk_csc();
        
        begin = std::chrono::steady_clock::now();
        spmv();
        end = std::chrono::steady_clock::now();
    }
    else
    {
        triples_regulars = new std::vector<struct Triple>;
        triples_sources = new std::vector<struct Triple>;
        classification(num_vertices);
        
        init_csc_regulars(triples_regulars->size(), nnz_ingoings);
        popu_csc_regulars();
        //walk_csc_regulars();
        y_regulars.resize(nnz_regulars);
        
        init_csc_sources(triples_sources->size(), nnz_outgoings);
        popu_csc_sources();
        //walk_csc_sources();
        y_sources.resize(nnz_sources);
        
        begin = std::chrono::steady_clock::now();
        spmv_regulars(0);
        spmv_regulars(regulars_sinks_offset);
        spmv_sources(0);
        spmv_sources(sources_sinks_offset);
        end = std::chrono::steady_clock::now();
        
        value = y_regulars_value + y_sources_value;
        triples_regulars->clear();
        triples_sources->clear();
        extra = ((nentries_regulars * sizeof(Edge)) + (nentries_sources * sizeof(Edge)));
        //std::cout << "CSC extra memory: " << ((nentries_regulars * sizeof(CSCEntry)) + (nentries_sources * sizeof(CSCEntry))) / 1e3 << "KB" << std::endl;
        
    }
            
        std::cout << "Memory: " << size / 1e3 << " K" << std::endl;
        if(std::atoi(argv[1]))
            std::cout << "Memory: " << extra / 1e3 << " K (extra per iteration)" << std::endl;
        std::cout << "Time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1e6 <<std::endl;
        std::cout << "Value: " << value <<std::endl;
        ;
        
        
        
    //elapsed_secs = 
    //std::time_t temp = difftime (end, start);

    //std::cout << temp << std::endl;
    //printf("Time=%f, Value=%d\n",  elapsed_secs, value);
    triples->clear();
    //classification(num_vertices);
	return(0);
}
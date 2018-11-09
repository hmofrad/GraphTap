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

struct Triple
{
    uint32_t row;
    uint32_t col;
};

std::vector<struct Triple> *triples;

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
    
    printf("SpMV kernel uint test...\n");
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
        kernel();
    }
    else
    {
        triples_regulars = new std::vector<struct Triple>;
        triples_sinks = new std::vector<struct Triple>;
        classification(num_vertices);
        init_csc_regulars(triples_regulars->size(), nnz_ingoings);
        init_csc_sinks(triples_sinks->size(), nnz_ingoings);
        popu_csc_regulars();
        triples_regulars->clear();
        triples_sinks->clear();
        
    }
    triples->clear();
    
    //classification(num_vertices);
	return(0);
}
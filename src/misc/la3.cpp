/*
 * la3.cpp: LA3 SpMV implementation
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


std::vector<struct Triple> *triples_regulars;
std::vector<struct Triple> *triples_sinks;

struct CSCEntry
{
  uint32_t global_idx;
  uint32_t idx;
};

// Vertex classification
uint32_t nnz_outgoings;
std::vector<char> outgoings;
std::vector<uint32_t> outgoings_val;
uint32_t nnz_ingoings;
std::vector<char> ingoings;
std::vector<uint32_t> ingoings_val;
uint32_t nnz_regulars;
std::vector<char> regulars;
std::vector<uint32_t> regulars_val;
uint32_t nnz_sources;
std::vector<char> sources;
std::vector<uint32_t> sources_val;
uint32_t nnz_sinks;
std::vector<char> sinks;
std::vector<uint32_t> sinks_val;
uint32_t nnz_isolates;
std::vector<char> isolates;
std::vector<uint32_t> isolates_val;

uint32_t regulars_sinks_offset = 1;
uint32_t sinks_sinks_offset = 1;

uint32_t nentries_regulars;
uint32_t ncols_regulars;
uint32_t* colptrs_regulars;
uint32_t* colidxs_regulars;
CSCEntry* entries_regulars;

uint32_t nentries_sinks;
uint32_t ncols_sinks;
uint32_t* colptrs_sinks;
uint32_t* colidxs_sinks;
CSCEntry* entries_sinks;

void classification(uint32_t num_vertices)
{
    outgoings.resize(num_vertices);
    outgoings_val.resize(num_vertices);
    ingoings.resize(num_vertices);
    ingoings_val.resize(num_vertices);
    regulars.resize(num_vertices);
    regulars_val.resize(num_vertices);
    sources.resize(num_vertices);
    sources_val.resize(num_vertices);
    sinks.resize(num_vertices);
    sinks_val.resize(num_vertices);
    isolates.resize(num_vertices);
    isolates_val.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        outgoings[triple.row] = 1;
        ingoings[triple.col]  = 1;
    }
    

    uint32_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0;

    for(uint32_t o = 0; o < num_vertices; o++)
    {
        if(outgoings[o])
        {
            outgoings_val[o] = i;
            i++;
        }
        if(ingoings[o])
        {
            ingoings_val[o] = j;
            printf("%d ", j);
            j++;
            
        }
        if(outgoings[o] and ingoings[o])
        {
            regulars[o] = 1;
            regulars_val[o] = k;
            k++;
        }
        if(outgoings[o] and not ingoings[o])
        {
            sources[o] = 1;
            sources_val[o] = l;
            l++;
        }
        if(not outgoings[o] and ingoings[o])
        {
            sinks[o] = 1;
            sinks_val[o] = m;
            m++;
        }
        if(not outgoings[o] and not ingoings[o])
        {
            isolates[o] = 1;
            isolates_val[o] = n;
            n++;
        }
    }
    printf("\n");
    nnz_outgoings = i;
    nnz_ingoings = j;
    nnz_regulars = k;
    nnz_sources = l;
    nnz_sinks = m;
    nnz_isolates = n;
    
    
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        printf("i=%d: out=%d, in=%d, reg=%d, src=%d, snk=%d, iso=%d\n", i, outgoings[i], ingoings[i], regulars[i], sources[i], sinks[i], isolates[i]);
    }
    
    printf("regulars->regulars\n");
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and regulars[triple.col])
        {
            triples_regulars->push_back(triple);
            printf("%d %d\n", triple.row, triple.col);
            regulars_sinks_offset++;
        }
    }

    printf("regulars->sinks\n");
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and sinks[triple.col])
        {
            triples_regulars->push_back(triple);
            printf("%d %d\n", triple.row, triple.col);
        }
    }
    
    printf("sources->regulars\n");
    for(auto &triple: *triples)
    {
        if(sources[triple.row] and regulars[triple.col])
        {
            triples_sinks->push_back(triple);
            printf("%d %d\n", triple.row, triple.col);
            sinks_sinks_offset++;
        }
    }
    printf("sources->sinks\n");
    for(auto &triple: *triples)
    {
        if(sources[triple.row] and sinks[triple.col])
        {
            triples_sinks->push_back(triple);
            printf("%d %d\n", triple.row, triple.col);
        }
    }
    

    
    printf("nnz_outgoings=%d\nnnz_ingoings=%d\nnnz_regulars=%d\nnnz_sources=%d\nnnz_sinks=%d\nnnz_isolates=%d\n", 
            nnz_outgoings, nnz_ingoings, nnz_regulars, nnz_sources, nnz_sinks, nnz_isolates);
    
        
}


void init_csc_regulars(uint32_t nnz_, uint32_t ncols_)
{
    nentries_regulars = nnz_;
    ncols_regulars = ncols_ + 1;
    
    colptrs_regulars = (uint32_t*) mmap(nullptr, (ncols_regulars) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colptrs_regulars != nullptr);
    memset(colptrs_regulars, 0, ncols_regulars * sizeof(uint32_t));        
    colidxs_regulars = (uint32_t*) mmap(nullptr, (ncols_regulars) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colidxs_regulars != nullptr);
    memset(colidxs_regulars, 0, ncols_regulars * sizeof(uint32_t));        
    entries_regulars = (CSCEntry*) mmap(nullptr, nentries_regulars * sizeof(CSCEntry), PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(entries_regulars != nullptr);    
    memset(entries_regulars, 0, nentries_regulars * sizeof(CSCEntry));        
}

void init_csc_sinks(uint32_t nnz_, uint32_t ncols_)
{
    nentries_sinks = nnz_;
    ncols_sinks = ncols_ + 1;
    
    colptrs_sinks = (uint32_t*) mmap(nullptr, (ncols_sinks) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colptrs_sinks != nullptr);
    memset(colptrs_sinks, 0, (ncols_sinks) * sizeof(uint32_t));
    colidxs_sinks = (uint32_t*) mmap(nullptr, (ncols_sinks) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colidxs_sinks != nullptr);
    memset(colidxs_sinks, 0, (ncols_sinks) * sizeof(uint32_t));
    entries_sinks = (CSCEntry*) mmap(nullptr, nentries_sinks * sizeof(CSCEntry), PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(entries_sinks != nullptr); 
    memset(entries_sinks, 0, nentries_sinks * sizeof(CSCEntry));    
}


void popu_csc_regulars()
{
    //colptrs_regulars // JA
    //colidxs_regulars // Extra
    //entries_regulars // A, IA
    uint32_t i = 0;
    uint32_t j = 1;
    //int c = 1;
    //int last_col = 0;
    colptrs_regulars[0] = 0;
    for(auto &triple: *triples_regulars)
    {
        //if(i == 0)
        //    last_col = ingoings_val[triple.col];
        /*
        if(c < regulars_sinks_offset)
        {
            
            printf("1.<%d j=%d> <idx=%d colidxs=%d> <g_idx=%d %d>\n", i, j, triple.row, triple.col, regulars_val[triple.row], ingoings_val[triple.col]);

            
        }
        else
        {
            printf("2.<%d j=%d> <id=%d colidxs=%d> <gid=%d %d> %d c=%d\n", i, j, triple.row, triple.col, sinks_val[triple.row], ingoings_val[triple.col], regulars_sinks_offset, c);
            //break;
            
        }
        */
        
        //if((j - 1) != ingoings_val[triple.col])
        if(colidxs_regulars[j-1] != triple.col)//ingoings_val[triple.col])// and last_col != ingoings_val[triple.col])

        {
            //printf("%d %d %d \n", j, triple.col, ingoings_val[triple.col]);
           // printf("?? %d %d\n", j, ingoings_val[triple.col]);
            j++;
            colptrs_regulars[j] = colptrs_regulars[j - 1];
            //printf("%d %d %d \n", j, triple.col, ingoings_val[triple.col]);
        }  
        
        colptrs_regulars[j]++;
        colidxs_regulars[j-1] = triple.col;
        printf("(i=%d j=%d colidxs=%d)\n", i, j, triple.col);
        
        entries_regulars[i].idx = triple.row;
        //if(c < regulars_sinks_offset)
            entries_regulars[i].global_idx = regulars_val[triple.row];
        //else
        //{
          //  entries_regulars[i].global_idx = sinks_val[triple.row];
           // printf(">>>>>%d %d\n", c, i);
        //}
        
        //printf("j=%d colptr=%d colidx=%d idx=%d gidx=%d\n", j, colptrs_regulars[j], colidxs_regulars[j], entries_regulars[i].idx, entries_regulars[i].global_idx);        
        
        
        //IA[i] = pair.row;
        i++;     
        //c++;
       // last_col = ingoings_val[triple.col];
        //break;
        //    break;
        
    }
    
    for (uint32_t j = 0; j < ncols_regulars - 1; j++)
    {
        printf("j=%d, %d %d\n", j, colptrs_regulars[j], colptrs_regulars[j + 1]);
        for (uint32_t i = colptrs_regulars[j]; i < colptrs_regulars[j + 1]; i++)
        {
            auto& entry = entries_regulars[i];
            printf("i=%d, gidx=%d, idx=%d, j=%d, col=%d\n", i, entry.global_idx, entry.idx, j, colidxs_regulars[j]);
        //auto& entry = entries[i];
        }
        //printf("\n");
        //printf("i=%d, colidxs[i]=%d, colptrs[i]=%d, %d\n", i, colidxs[i], colptrs[i],  colptrs[i + 1] -  colptrs[i]);
    }
    
    /*
                JA[0] = 0;
            for (auto& triple : *(tile.triples))
            {
                pair = rebase(triple);
                while((j - 1) != pair.col)
                {
                    j++;
                    JA[j] = JA[j - 1];
                }            
                // In case weights are there
                #ifdef HAS_WEIGHT
                A[i] = triple.weight;
                #endif
                
                JA[j]++;
                IA[i] = pair.row;
                i++;
            }
            while((j + 1) < (tile_width + 1))
            {
                j++;
                JA[j] = JA[j - 1];
            }
    
    
    */

    /*
    for(auto &triple: *triples)
    {

        //printf("%d %d %d %d, %d %d %d %d\n", i, triple.row, rows[triple.row], rows_val[triple.row], j, triple.col, cols[triple.col], cols_val[triple.col]);
        
        while((j - 1) != cols_val[triple.col])
        {
            j++;
            JA[j] = JA[j - 1];
        }  
                
        A[i] = 1;
        JA[j]++;
        IA[i] = rows[triple.row];
        i++;
    }

    while((j + 1) < (nnz_cols + 1))
    {
        j++;
        JA[j] = JA[j - 1];
    }   
    */
}
/*
 * edcsc.cpp: EDCSC SpMV implementation (LA3)
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
#include <unordered_set>

std::vector<struct Triple> *triples_regulars;
std::vector<struct Triple> *triples_sources;

struct CSCEntry
{
  uint32_t global_idx;
  uint32_t idx;
  uint32_t weight;
};

struct Edge
{
  const uint32_t src, dst;

  const char weight;

  Edge() : src(0), dst(0), weight(1) {}

  Edge(const uint32_t src, const uint32_t dst, const char weight)
      : src(src), dst(dst), weight(weight) {}
};

std::vector<uint32_t> y_regulars;
std::vector<uint32_t> y_sources;
std::vector<uint32_t> x_regulars;
std::vector<uint32_t> x_sources;
std::vector<uint32_t> x_all;
std::vector<char> cols_all;
std::vector<uint32_t> cols_all_val;
uint32_t nnz_cols_all = 0;
std::vector<uint32_t> regulars2cols_all;
std::vector<uint32_t> cols_all2regulars;
uint32_t nnz_regulars2cols_all = 0;
std::vector<uint32_t> sources2cols_all;
std::vector<uint32_t> cols_all2sources;
uint32_t nnz_sources2cols_all = 0;

std::vector<uint32_t> regulars2ingoings;
std::vector<uint32_t> ingoings2regulars;
uint32_t nnz_regulars2ingoings = 0;

std::vector<uint32_t> ingoings2outgoigns;
std::vector<uint32_t> outgoigns2ingoings;
uint32_t nnz_ingoings2outgoigns = 0;

std::vector<uint32_t> regulars2outgoings;
std::vector<uint32_t> outgoings2regulars;
uint32_t nnz_regulars2outgoings = 0;

std::vector<uint32_t> sources2ingoings;
std::vector<uint32_t> ingoings2sources;
uint32_t nnz_sources2ingoings = 0;


std::vector<uint32_t> sources2outgoings;
std::vector<uint32_t> outgoings2sources;
uint32_t nnz_sources2outgoings = 0;

std::vector<uint32_t> regulars2vals;
std::vector<uint32_t> sources2vals;


uint32_t y_regulars_value = 0;
uint32_t y_sources_value = 0;

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

uint32_t nnz_regulars_cols;
uint32_t nnz_regulars_sinks_cols;
uint32_t regulars_sinks_offset;
uint32_t nnz_sources_cols;
uint32_t nnz_sources_sinks_cols;
uint32_t sources_sinks_offset;

uint32_t nentries_regulars;
uint32_t ncols_regulars;
uint32_t* colptrs_regulars;
uint32_t* colidxs_regulars;
CSCEntry* entries_regulars;

uint32_t nentries_sources;
uint32_t ncols_sources;
uint32_t* colptrs_sources;
uint32_t* colidxs_sources;
CSCEntry* entries_sources;

void classification_edcsc(uint32_t num_vertices)
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
    cols_all.resize(num_vertices);
    cols_all_val.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        outgoings[triple.row] = 1;
        ingoings[triple.col]  = 1;
    }
    
    uint32_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, p = 0;
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
            j++;
            
        }
        if(outgoings[o] and ingoings[o])
        {
            regulars2vals.push_back(o);
            regulars[o] = 1;
            regulars_val[o] = k;
            k++;
        }
        if(outgoings[o] and not ingoings[o])
        {
            sources2vals.push_back(o);
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
        if(outgoings[o] or ingoings[o])
        {
            cols_all[o] = 1;
            cols_all_val[o] = p;
            p++;
        }
    }

    nnz_outgoings = i;
    nnz_ingoings = j;
    nnz_regulars = k;
    nnz_sources = l;
    nnz_sinks = m;
    nnz_isolates = n;
    nnz_cols_all = p;
    
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        if(regulars[i] and ingoings[i])
        {
            regulars2ingoings.push_back(regulars_val[i]);
            ingoings2regulars.push_back(ingoings_val[i]);
            nnz_regulars2ingoings++;
        }
        
        if(sources[i] and outgoings[i])
        {
            sources2outgoings.push_back(sources_val[i]);
            outgoings2sources.push_back(outgoings_val[i]);
            nnz_sources2outgoings++;
        }
        if(ingoings[i] and outgoings[i])
        {
            ingoings2outgoigns.push_back(ingoings_val[i]);
            outgoigns2ingoings.push_back(outgoings_val[i]);
            nnz_ingoings2outgoigns++;
        }
        if(regulars[i] and outgoings[i])
        {
            regulars2outgoings.push_back(regulars_val[i]);
            outgoings2regulars.push_back(outgoings_val[i]);
            nnz_regulars2outgoings++;
        }
        if(sources[i] and ingoings[i])
        {
            sources2ingoings.push_back(sources_val[i]);
            ingoings2sources.push_back(ingoings_val[i]);
            nnz_sources2ingoings++;
        }
        
        if(regulars[i] and ingoings[i])
        {
            regulars2cols_all.push_back(regulars_val[i]);
            cols_all2regulars.push_back(ingoings_val[i]);
            nnz_regulars2cols_all++;
        }
        if(sources[i] and ingoings[i])
        {
            sources2cols_all.push_back(sources_val[i]);
            cols_all2sources.push_back(ingoings_val[i]);
            nnz_sources2cols_all++;
        }
    }

    std::unordered_set<uint32_t> uniques;
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and regulars[triple.col])
        {
            triples_regulars->push_back(triple);
            uniques.insert(triple.col);
        }
    }
    regulars_sinks_offset = uniques.size();
    nnz_regulars_cols = regulars_sinks_offset;
    nnz_regulars_sinks_cols = nnz_ingoings - regulars_sinks_offset;
    uniques.clear();
    
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and sinks[triple.col])
            triples_regulars->push_back(triple);
    }

    for(auto &triple: *triples)
    {
        if(sources[triple.row] and regulars[triple.col])
        {
            triples_sources->push_back(triple);
            uniques.insert(triple.col);
        }
    }
    sources_sinks_offset = uniques.size();
    nnz_sources_cols = sources_sinks_offset;
    nnz_sources_sinks_cols = nnz_outgoings - sources_sinks_offset;
    
    uniques.clear();
    for(auto &triple: *triples)
    {
        if(sources[triple.row] and sinks[triple.col])
            triples_sources->push_back(triple);
    }
    //printf("nnz_outgoings=%d, nnz_ingoings=%d, nnz_regulars=%d, nnz_sources=%d, nnz_sinks=%d, nnz_isolates=%d\n", nnz_outgoings, nnz_ingoings, nnz_regulars, nnz_sources, nnz_sinks, nnz_isolates);
    //printf("nnz_regulars_cols=%d, nnz_regulars_sinks_cols=%d, regulars_sinks_offset=%d, nnz_sources_cols=%d, nnz_sources_sinks_cols=%d, sources_sinks_offset=%d\n", nnz_regulars_cols, nnz_regulars_sinks_cols, regulars_sinks_offset, nnz_sources_cols, nnz_sources_sinks_cols, sources_sinks_offset);
}


void init_edcsc_regulars(uint32_t nnz_, uint32_t ncols_)
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

    size = (ncols_regulars * (sizeof(uint32_t) +  sizeof(uint32_t))) + (nentries_regulars * sizeof(CSCEntry));
}

void popu_edcsc_regulars()
{
    uint32_t i = 0;
    uint32_t j = 1;
    colptrs_regulars[0] = 0;
    for(auto &triple: *triples_regulars)
    {
        if((i != 0) and colidxs_regulars[j-1] != triple.col)
        {
            j++;
            colptrs_regulars[j] = colptrs_regulars[j - 1];
        }  
        colptrs_regulars[j]++;
        colidxs_regulars[j-1] = triple.col;
        entries_regulars[i].idx = triple.row;
        entries_regulars[i].global_idx = regulars_val[triple.row];
        entries_regulars[i].weight = 1;
        i++;     
    }
    
    while((j + 1) < (ncols_regulars + 1))
    {
        j++;
        colptrs_regulars[j] = colptrs_regulars[j - 1];
    }
}

void walk_edcsc_regulars()
{
    for(uint32_t j = 0; j < ncols_regulars - 1; j++)
    {
        printf("j=%d\n", j);
        for (uint32_t i = colptrs_regulars[j]; i < colptrs_regulars[j + 1]; i++)
        {
            auto& entry = entries_regulars[i];
            auto edge = Edge(colidxs_regulars[j], entry.idx, entry.weight);
            printf("   i=%d, global_index=%d, index=%d, weight=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, entry.weight, j, colidxs_regulars[j]);
        }
    }
}

void spmv_edcsc_regulars(uint32_t offset)
{
    uint32_t ncols = 0;
    if(offset)
        ncols = ncols_regulars - 1;  
    else
        ncols = nnz_regulars_cols;
    for(uint32_t j = offset; j < ncols; j++)
    {
        for (uint32_t i = colptrs_regulars[j]; i < colptrs_regulars[j + 1]; i++)
        {
            auto& entry = entries_regulars[i];
            auto edge = Edge(colidxs_regulars[j], entry.idx, entry.weight);
            y_regulars[entry.global_idx] += entry.weight * x_regulars[j];
            nOps++;
        }
    }
}


void init_edcsc_sources(uint32_t nnz_, uint32_t ncols_)
{
    nentries_sources = nnz_;
    ncols_sources = ncols_ + 1;
    
    colptrs_sources = (uint32_t*) mmap(nullptr, (ncols_sources) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colptrs_sources != nullptr);
    memset(colptrs_sources, 0, (ncols_sources) * sizeof(uint32_t));
    colidxs_sources = (uint32_t*) mmap(nullptr, (ncols_sources) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colidxs_sources != nullptr);
    memset(colidxs_sources , 0, (ncols_sources) * sizeof(uint32_t));
    entries_sources = (CSCEntry*) mmap(nullptr, nentries_sources * sizeof(CSCEntry), PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(entries_sources != nullptr); 
    memset(entries_sources, 0, nentries_sources * sizeof(CSCEntry)); 

    size += (ncols_sources * (sizeof(uint32_t) +  sizeof(uint32_t))) + (nentries_sources * sizeof(CSCEntry));
}

void popu_edcsc_sources()
{
    uint32_t i = 0;
    uint32_t j = 1;
    colptrs_sources[0] = 0;
    for(auto &triple: *triples_sources)
    {
        if((i != 0) and colidxs_sources[j-1] != triple.col)
        {
            j++;
            colptrs_sources[j] = colptrs_sources[j - 1];
        }
        colptrs_sources[j]++;
        colidxs_sources[j-1] = triple.col;
        entries_sources[i].idx = triple.row;
        entries_sources[i].global_idx = sources_val[triple.row]; 
        entries_sources[i].weight = 1;
        i++;     
    }
    while((j + 1) < (ncols_sources + 1))
    {
        j++;
        colptrs_sources[j] = colptrs_sources[j - 1];
    }
}

void walk_edcsc_sources()
{
    for(uint32_t j = 0; j < ncols_sources - 1; j++)
    {
        printf("j=%d\n", j);
        for (uint32_t i = colptrs_sources[j]; i < colptrs_sources[j + 1]; i++)
        {
            auto& entry = entries_sources[i];
            auto edge = Edge(colidxs_sources[j], entry.idx, entry.weight);
            printf("   i=%d, global_index=%d, index=%d, weight=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, entry.weight, j, colidxs_sources[j]);
        }
    }
}

void run_edcsc()
{
    
    init_edcsc_regulars(triples_regulars->size(), nnz_ingoings);
    popu_edcsc_regulars();
    init_edcsc_sources(triples_sources->size(), nnz_outgoings);
    popu_edcsc_sources();    
    printf("[x]Compression is done\n");
}

void spmv_edcsc_sources(uint32_t offset)
{
    uint32_t ncols = 0;
    if(offset)
        ncols = ncols_sources - 1;    
    else
        ncols = nnz_sources_cols;
    
    for(uint32_t j = offset; j < ncols; j++)
    {
        for (uint32_t i = colptrs_sources[j]; i < colptrs_sources[j + 1]; i++)
        {
            auto& entry = entries_sources[i];
            auto edge = Edge(colidxs_sources[j], entry.idx, entry.weight);
            y_sources[entry.global_idx] += entry.weight * x_sources[j];
            nOps++;
        }
    }
}

void init_edcsc_vecs()
{
    values.resize(num_vertices);
    
    y_regulars.resize(nnz_regulars);
    x_regulars.resize(ncols_regulars, 1);
    
    y_sources.resize(nnz_sources);
    x_sources.resize(ncols_sources, 1);
}

void spmv_edcsc()
{
    spmv_edcsc_regulars(0);
    spmv_edcsc_regulars(regulars_sinks_offset);

    spmv_edcsc_sources(0);
    spmv_edcsc_sources(sources_sinks_offset);

    for(uint32_t i = 0; i < nnz_regulars2ingoings; i++)
    {
        x_regulars[ingoings2regulars[i]] = y_regulars[regulars2ingoings[i]];
        x_regulars[ingoings2regulars[i]] = 1;
    }

    for(uint32_t i = 0; i < nnz_regulars2outgoings; i++)
    {
       x_sources[outgoings2regulars[i]] = y_regulars[regulars2outgoings[i]];
       x_sources[outgoings2regulars[i]] = 1;
    }
    
    for(uint32_t i = 0; i < nnz_sources2ingoings; i++)
    {
        x_sources[ingoings2sources[i]] = y_sources[sources2ingoings[i]];
        x_sources[ingoings2sources[i]] = 1;
    }
    
    for(uint32_t i = 0; i < nnz_sources2outgoings; i++)
    {
        x_regulars[sources2outgoings[i]] = y_sources[sources2outgoings[i]];
        x_regulars[sources2outgoings[i]] = 1;
    }
   
}

void done_edcsc()
{
    
    for(uint32_t i = 0; i < nnz_regulars; i++)
        values[regulars2vals[i]] += y_regulars[i];
    
    for(uint32_t i = 0; i < nnz_sources; i++)
        values[sources2vals[i]] += y_sources[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];   
}
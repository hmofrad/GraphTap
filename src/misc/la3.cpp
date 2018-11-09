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
#include <unordered_set>

std::vector<struct Triple> *triples_regulars;
std::vector<struct Triple> *triples_sources;

struct CSCEntry
{
  uint32_t global_idx;
  uint32_t idx;
  char weight;
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

    nnz_outgoings = i;
    nnz_ingoings = j;
    nnz_regulars = k;
    nnz_sources = l;
    nnz_sinks = m;
    nnz_isolates = n;
    
    /*
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        printf("i=%d: out=%d, in=%d, reg=%d, src=%d, snk=%d, iso=%d\n", i, outgoings[i], ingoings[i], regulars[i], sources[i], sinks[i], isolates[i]);
    }
    */
    std::unordered_set<uint32_t> uniques;
    //printf("regulars->regulars\n");
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and regulars[triple.col])
        {
            triples_regulars->push_back(triple);
            //printf("%d %d\n", triple.row, triple.col);
            uniques.insert(triple.col);
        }
    }
    regulars_sinks_offset = uniques.size();
    nnz_regulars_cols = regulars_sinks_offset;
    nnz_regulars_sinks_cols = nnz_ingoings - regulars_sinks_offset;
    uniques.clear();
    //printf("regulars->sinks\n");
    for(auto &triple: *triples)
    {
        if(regulars[triple.row] and sinks[triple.col])
        {
            triples_regulars->push_back(triple);
            //printf("%d %d\n", triple.row, triple.col);
        }
    }
    
    //printf("sources->regulars\n");
    for(auto &triple: *triples)
    {
        if(sources[triple.row] and regulars[triple.col])
        {
            triples_sources->push_back(triple);
            //printf("%d %d\n", triple.row, triple.col);
            uniques.insert(triple.col);
        }
    }
    sources_sinks_offset = uniques.size();
    nnz_sources_cols = sources_sinks_offset;
    nnz_sources_sinks_cols = nnz_outgoings - sources_sinks_offset;
    uniques.clear();
    //printf("sources->sinks\n");
    for(auto &triple: *triples)
    {
        if(sources[triple.row] and sinks[triple.col])
        {
            triples_sources->push_back(triple);
            //printf("%d %d\n", triple.row, triple.col);
        }
    }
    

    
    //printf("nnz_outgoings=%d, nnz_ingoings=%d, nnz_regulars=%d, nnz_sources=%d, nnz_sinks=%d, nnz_isolates=%d\n", nnz_outgoings, nnz_ingoings, nnz_regulars, nnz_sources, nnz_sinks, nnz_isolates);
    //printf("nnz_regulars_cols=%d, nnz_regulars_sinks_cols=%d, regulars_sinks_offset=%d, nnz_sources_cols=%d, nnz_sources_sinks_cols=%d, sources_sinks_offset=%d\n", nnz_regulars_cols, nnz_regulars_sinks_cols, regulars_sinks_offset, nnz_sources_cols, nnz_sources_sinks_cols, sources_sinks_offset);
        
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

void popu_csc_regulars()
{
    uint32_t i = 0;
    uint32_t j = 1;
    colptrs_regulars[0] = 0;
    for(auto &triple: *triples_regulars)
    {
       // printf("1.<%d j=%d> <idx=%d colidxs=%d> <g_idx=%d %d>off=%d,%d\n", i, j, triple.row, triple.col, regulars_val[triple.row], ingoings_val[triple.col], regulars_sinks_offset, nnz_ingoings);
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
    
    
    /*
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
    */
}

void walk_csc_regulars()
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

void spmv_regulars(uint32_t offset)
{
    //if(accumulator_has_not_initialized)
    //std::vector<uint32_t> y(nnz_regulars);
    std::vector<uint32_t> x(ncols_regulars - 1);
    //uint32_t ncols = ncols_regulars;
    
    uint32_t ncols = 0;
    if(offset)
    {
       // x.resize(nnz_regulars_sinks_cols);
        ncols = ncols_regulars - 1;    
    }
    else
    {
       // x.resize(nnz_regulars_cols);
        ncols = nnz_regulars_cols;
    }
    
    std::fill(x.begin(), x.end(), 1);
    
    
    
    
    //printf("%d %d %d\n\n", ncols, nnz_regulars_cols, nnz_regulars);
    for(uint32_t j = offset; j < ncols; j++)
    {
        ///printf("j=%d\n", j);
        for (uint32_t i = colptrs_regulars[j]; i < colptrs_regulars[j + 1]; i++)
        {
            auto& entry = entries_regulars[i];
            auto edge = Edge(colidxs_regulars[offset + j], entry.idx, entry.weight);
            y_regulars[entry.global_idx] += entry.weight * x[j];

            //y[IA[i]] += A[i] * x[j];
            //printf("   i=%d, global_index=%d, index=%d, weight=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, entry.weight, j, colidxs_regulars[j]);
            //printf("   i=%d, global_index=%d, index=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, j, colidxs_regulars[j]);
        }
    }
    
    
    if(offset)
    {
        //uint32_t value = 0;
        for(uint32_t i = 0; i < nnz_regulars; i++)
            y_regulars_value += y_regulars[i];
        //printf("value=%d\n", value);
    }
    
    
}


void init_csc_sources(uint32_t nnz_, uint32_t ncols_)
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
}

void popu_csc_sources()
{
    uint32_t i = 0;
    uint32_t j = 1;
    colptrs_sources[0] = 0;
    for(auto &triple: *triples_sources)
    {
        //printf("2.<%d j=%d> <idx=%d colidxs=%d> <gidx=%d %d>off=%d,%d\n", i, j, triple.row, triple.col, sources_val[triple.row], outgoings_val[triple.col], sources_sinks_offset, nnz_outgoings);
     
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
}

void walk_csc_sources()
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

void spmv_sources(uint32_t offset)
{
    //if(accumulator_has_not_initialized)
    //std::vector<uint32_t> y(nnz_regulars);
    std::vector<uint32_t> x(ncols_sources - 1);
    //uint32_t ncols = ncols_regulars;
    
    uint32_t ncols = 0;
    if(offset)
    {
       // x.resize(nnz_regulars_sinks_cols);
        ncols = ncols_sources - 1;    
    }
    else
    {
       // x.resize(nnz_regulars_cols);
        ncols = nnz_sources_cols;
    }
    
    std::fill(x.begin(), x.end(), 1);
    
    
    
    
    //printf("%d %d %d %d\n\n", ncols, nnz_regulars_cols, nnz_sources, y_sources.size());
    for(uint32_t j = offset; j < ncols; j++)
    {
        ///printf("j=%d\n", j);
        for (uint32_t i = colptrs_sources[j]; i < colptrs_sources[j + 1]; i++)
        {
            auto& entry = entries_sources[i];
            auto edge = Edge(colidxs_sources[offset + j], entry.idx, entry.weight);

            y_sources[entry.global_idx] += entry.weight * x[j];
            //y[IA[i]] += A[i] * x[j];
            //printf("   i=%d, global_index=%d, index=%d, weight=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, entry.weight, j, colidxs_sources[j]);
            //printf("   i=%d, global_index=%d, index=%d, j=%d, col_index=%d\n", i, entry.global_idx, entry.idx, j, colidxs_regulars[j]);
        }
    }
    
    
    if(offset)
    {
        //uint32_t value = 0;
        for(uint32_t i = 0; i < nnz_sources; i++)
            y_sources_value += y_sources[i];
        //printf("value=%d\n", value);
    }
    
    
}


/*
void kernel()
{
    std::vector<uint32_t> y(nnz_rows);
    std::vector<uint32_t> x(nnz_cols);
    std::fill(x.begin(), x.end(), 1);
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            printf("%d %d %d %d %d\n", IA[i], j, y[IA[i]], A[i], x[j]);
            y[IA[i]] += A[i] * x[j]; 
        }
    }
    
    uint32_t value = 0;
    for(uint32_t i = 0; i < nnz_rows; i++)
        value += y[i];
    printf("value=%d\n", value);
}
*/




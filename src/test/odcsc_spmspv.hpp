/*
 * odcsc.cpp: ODCSC SpMSpV implementation (LA3)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef ODCSC_SPMSPV_HPP
#define ODCSC_SPMSPV_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "odcsc_base.hpp"  

class ODCSC {  
    public:
        ODCSC() {};
        ODCSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~ODCSC() {};
        void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair> *pairs = nullptr;
        std::vector<struct Pair> *pairs_regulars = nullptr;
        std::vector<struct Pair> *pairs_sources = nullptr;
        //std::vector<struct Pair> *pairs_sinks = nullptr;
        struct ODCSC_BASE *odcsc_regulars = nullptr;
        struct ODCSC_BASE *odcsc_sources = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x_regulars;
        std::vector<double> x_sources;
        std::vector<double> x_sinks;
        std::vector<double> y;
        std::vector<double> y_regulars;
        std::vector<double> y_sources;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        
        uint32_t nnzrows_ = 0;
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzrows_regulars_ = 0;
        std::vector<char> rows_regulars;
        std::vector<uint32_t> rows_regulars_all;
        std::vector<uint32_t> rows_regulars_nnz;
        uint32_t nnzrows_sources_ = 0;
        std::vector<char> rows_sources;
        std::vector<uint32_t> rows_sources_all;
        std::vector<uint32_t> rows_sources_nnz;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        uint32_t nnzcols_regulars_ = 0;
        std::vector<char> cols_regulars;
        std::vector<uint32_t> cols_regulars_all;
        std::vector<uint32_t> cols_regulars_nnz;        
        uint32_t nnzcols_sinks_ = 0;
        std::vector<char> cols_sinks;
        std::vector<uint32_t> cols_sinks_all;
        std::vector<uint32_t> cols_sinks_nnz;
        /*
        uint32_t nnzrows_isolates_ = 0;
        std::vector<char> rows_isolates;
        std::vector<uint32_t> rows_isolates_all;
        std::vector<uint32_t> rows_isolates_nnz;
        */
        void construct_filter();
        void destruct_filter();
        
        void populate();
        void walk();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
        virtual void message_nnzcols();        
        virtual uint64_t spmv_nnzrows_nnzcols();
        virtual void update();
        virtual void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};
 
void ODCSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    pairs_regulars = new std::vector<struct Pair>;
    pairs_sources = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    construct_filter();
    odcsc_regulars = new struct ODCSC_BASE(nedges, nnzcols_);
    odcsc_sources = new struct ODCSC_BASE(nedges, nnzcols_);
    populate();
    pairs_regulars->clear();
    pairs_regulars->shrink_to_fit();
    pairs_regulars = nullptr;
    pairs_sources->clear();
    pairs_sources->shrink_to_fit();
    pairs_sources = nullptr;
    //walk();
    construct_vectors_degree();
    (void)spmv_nnzrows_nnzcols();
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] =  y[i];
    //v = y;
    //(void)checksum();
    printf("%f\n", checksum());
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete odcsc_regulars;
    odcsc_regulars = nullptr;
    delete odcsc_sources;
    odcsc_sources = nullptr;

    // PageRank program
    pairs = new std::vector<struct Pair>;
    pairs_regulars = new std::vector<struct Pair>;
    pairs_sources = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    construct_filter();
    odcsc_regulars = new struct ODCSC_BASE(nedges, nnzcols_);
    odcsc_sources = new struct ODCSC_BASE(nedges, nnzcols_);
    populate();
    space();
    pairs_regulars->clear();
    pairs_regulars->shrink_to_fit();
    pairs_regulars = nullptr;
    pairs_sources->clear();
    pairs_sources->shrink_to_fit();
    pairs_sources = nullptr;
    construct_vectors_pagerank();
    for(uint32_t i = 0; i < nrows; i++) {
        if(rows[i] == 1)
            d[i] = v[i];
    }
    //d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    if(niters == 1)
    {
        std::fill(x_regulars.begin(), x_regulars.end(), 0);
        std::fill(x_sources.begin(), x_sources.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzrows_nnzcols();
        update();
    }
    else
    {
        ;
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "ODCSC SpMSpV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete odcsc_regulars;
    odcsc_regulars = nullptr;
    delete odcsc_sources;
    odcsc_sources = nullptr;
    
    /*
    column_sort(pairs);
    construct_filter();
    dcsc = new struct ODCSC_BASE(nedges, nnzcols_);
    populate();        
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    construct_vectors_pagerank(); 
    for(uint32_t i = 0; i < nrows; i++) {
        if(rows[i] == 1)
            d[i] = v[i];
    }
    //d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    for(uint32_t i = 0; i < niters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        noperations += spmv();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "DCSC SpMV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete dcsc;
    dcsc = nullptr;
    */
}


void ODCSC::construct_filter() {
    nnzrows_ = 0;
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto& pair:* pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;        
    }
    nnzrows_regulars_ = 0;
    rows_all.resize(nvertices);
    rows_regulars_all.resize(nvertices);
    rows_sources.resize(nvertices);
    nnzrows_sources_ = 0;
    rows_regulars.resize(nvertices);    
    rows_sources_all.resize(nvertices);    
    nnzcols_regulars_ = 0;
    cols_regulars.resize(nvertices);
    cols_regulars_all.resize(nvertices);    
    nnzcols_sinks_ = 0;
    cols_sinks.resize(nvertices);
    cols_sinks_all.resize(nvertices);
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = nnzrows_;
            nnzrows_++;
            if(cols[i] == 0) {
                rows_sources[i] = 1;
                rows_sources_nnz.push_back(i);
                rows_sources_all[i] = nnzrows_sources_;
                nnzrows_sources_++;
            }
            if(cols[i] == 1) {
                rows_regulars[i] = 1;
                rows_regulars_nnz.push_back(i);
                rows_regulars_all[i] = nnzrows_regulars_;
                nnzrows_regulars_++;   
            }
        }
        if(cols[i] == 1) {
            nnzcols_++;
            if(rows[i] == 0) {
                cols_sinks_nnz.push_back(i);
                cols_sinks_all[i] = nnzcols_sinks_;
                nnzcols_sinks_++;
            }
            if(rows[i] == 1) {
                cols_regulars_nnz.push_back(i);
                cols_regulars_all[i] = nnzcols_regulars_;
                nnzcols_regulars_++;
            }
        }
    } 
}

void ODCSC::destruct_filter() {
    nnzrows_ = 0;
    rows.clear();
    rows.shrink_to_fit();
    rows_all.clear();
    rows_all.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit(); 
    nnzrows_regulars_ = 0;
    rows_regulars.clear();
    rows_regulars.shrink_to_fit();
    rows_regulars_all.clear();
    rows_regulars_all.shrink_to_fit();
    rows_regulars_nnz.clear();
    rows_regulars_nnz.shrink_to_fit();
    nnzrows_sources_ = 0;
    rows_sources.clear();
    rows_sources.shrink_to_fit();
    rows_sources_all.clear();
    rows_sources_all.shrink_to_fit();
    rows_sources_nnz.clear();
    rows_sources_nnz.shrink_to_fit();
    nnzcols_ = 0;
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_regulars_ = 0;
    cols_regulars.clear();
    cols_regulars.shrink_to_fit();
    cols_regulars_all.clear();    
    cols_regulars_all.shrink_to_fit();    
    nnzcols_sinks_ = 0;
    cols_sinks.clear();
    cols_sinks.shrink_to_fit();
    cols_sinks_all.clear();
    cols_sinks_all.shrink_to_fit();
}
 
void ODCSC::populate() {
    for(auto& pair: *pairs) {
        if(rows_regulars[pair.row])
            pairs_regulars->push_back(pair);
        else if(rows_sources[pair.row]) 
            pairs_sources->push_back(pair);
        //else if(rows_sinks[pair.col])
          //  pairs_sinks->push_back(pair);
        else {
            printf("Invalid edge\n");
            exit(0);
        }
    }
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    //printf("reg=%lu src=%lu all=%lu\n", pairs_regulars->size(), pairs_sources->size(), pairs_regulars->size() + pairs_sources->size());
    // Regulars
    column_sort(pairs_regulars);
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES; // ENTRIES  
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA; // COL_PTR
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC; // COL_IDX
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    auto& p = pairs_regulars->front();
    JC[0] = p.col;
    for(auto& pair:* pairs_regulars) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                 
        ENTRIES[i].idx = pair.row;
        ENTRIES[i].global_idx = rows_all[pair.row];//rows_regulars_all[pair.row];
        ENTRIES[i].weight = 1;        
        JA[j]++;
        i++;
    }
    // Sources
    column_sort(pairs_sources);
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES; // ENTRIES  
    JA = (uint32_t *) odcsc_sources->JA; // COL_PTR
    JC = (uint32_t *) odcsc_sources->JC; // COL_IDX
    i = 0;
    j = 1;
    JA[0] = 0;
    p = pairs_sources->front();
    JC[0] = p.col;
    for(auto& pair:* pairs_sources) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                 
        ENTRIES[i].idx = pair.row;
        ENTRIES[i].global_idx = rows_all[pair.row];//rows_sources_all[pair.row];
        ENTRIES[i].weight = 1;        
        JA[j]++;
        i++;
    }
    //printf("nnzrows_regulars_=%d nnzrows_sources_=%d %d\n", nnzrows_regulars_, nnzrows_sources_, rows_sources_nnz[10]);
}    

void ODCSC::walk() {
    // Regulars
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            printf("    i=%d,%d, j=%d, value=%d\n", entry.idx, entry.global_idx, JC[j], entry.weight);
        }
    }
    // Sources
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES;
    JA = (uint32_t*) odcsc_sources->JA;
    JC = (uint32_t*) odcsc_sources->JC;
    nnzcols =  odcsc_sources->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            printf("    i=%d,%d, j=%d, value=%d\n", entry.idx, entry.global_idx, JC[j], entry.weight);
        }
    }
}   


void ODCSC::construct_vectors_degree() {
    v.resize(nrows);
    x_regulars.resize(nnzcols_, 1);
    x_sources.resize(nnzcols_, 1);
    y.resize(nnzrows_);
}

void ODCSC::destruct_vectors_degree() {
    x_regulars.clear();
    x_regulars.shrink_to_fit();
    x_sources.clear();
    x_sources.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void ODCSC::construct_vectors_pagerank() {
    x_regulars.resize(nnzcols_);
    x_sources.resize(nnzcols_);
    y.resize(nnzrows_);
    d.resize(nrows);
}

void ODCSC::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x_regulars.clear();
    x_regulars.shrink_to_fit();
    x_sources.clear();
    x_sources.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void ODCSC::message_nnzcols() {
    uint32_t* JC_R = (uint32_t*) odcsc_regulars->JC;
    uint32_t* JC_S = (uint32_t*) odcsc_sources->JC;
    uint32_t nnzcols = odcsc_regulars->nnzcols;
    for(uint32_t i = 0; i < nnzcols; i++) {
        if(d[JC_R[i]])
            x_regulars[i] = v[JC_R[i]]/d[JC_R[i]];
        if(d[JC_S[i]]) 
            x_sources[i] = v[JC_S[i]]/d[JC_S[i]];
    }
}

uint64_t ODCSC::spmv_nnzrows_nnzcols() {
    // Regulars
    uint64_t noperations = 0;
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++)
    {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_regulars[j]);
            noperations++;
        }
    }
    // Sources
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES;
    JA = (uint32_t*) odcsc_sources->JA;
    JC = (uint32_t*) odcsc_sources->JC;
    nnzcols =  odcsc_sources->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++)
    {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_sources[j]);
            noperations++;
        }
    }
    return(noperations);
}

void ODCSC::update() {
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void ODCSC::space() {
    total_size += odcsc_regulars->size + odcsc_sources->size;
    total_size += sizeof(uint32_t) * rows_nnz.size();
}

double ODCSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void ODCSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void ODCSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations << std::endl;
    std::cout << "Final value    : " << checksum() << std::endl;
}

#endif
 
/* 
 
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

void classification_odcsc(uint32_t num_vertices)
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


void init_odcsc_regulars(uint32_t nnz_, uint32_t ncols_)
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

void popu_odcsc_regulars()
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

void walk_odcsc_regulars()
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

void spmv_odcsc_regulars(uint32_t offset)
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


void init_odcsc_sources(uint32_t nnz_, uint32_t ncols_)
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

void popu_odcsc_sources()
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

void walk_odcsc_sources()
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

void run_odcsc()
{
    
    init_odcsc_regulars(triples_regulars->size(), nnz_ingoings);
    popu_odcsc_regulars();
    init_odcsc_sources(triples_sources->size(), nnz_outgoings);
    popu_odcsc_sources();    
    printf("[x]Compression is done\n");
}

void spmv_odcsc_sources(uint32_t offset)
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

void init_odcsc_vecs()
{
    values.resize(num_vertices);
    
    y_regulars.resize(nnz_regulars);
    x_regulars.resize(ncols_regulars, 1);
    
    y_sources.resize(nnz_sources);
    x_sources.resize(ncols_sources, 1);
}

void spmv_odcsc()
{
    spmv_odcsc_regulars(0);
    spmv_odcsc_regulars(regulars_sinks_offset);

    spmv_odcsc_sources(0);
    spmv_odcsc_sources(sources_sinks_offset);

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

void done_odcsc()
{
    
    for(uint32_t i = 0; i < nnz_regulars; i++)
        values[regulars2vals[i]] += y_regulars[i];
    
    for(uint32_t i = 0; i < nnz_sources; i++)
        values[sources2vals[i]] += y_sources[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];   
}
*/
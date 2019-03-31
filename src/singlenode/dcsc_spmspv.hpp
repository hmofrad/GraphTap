/*
 * dcsc.hpp: DCSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DCSC_SPMSPV_HPP
#define DCSC_SPMSPV_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "dcsc_spmv.hpp" 
 
class DCSC__ : protected DCSC {
    using DCSC::DCSC;
    public:
        virtual void run_pagerank();
    protected:
        uint32_t nnzrows_ = 0;
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        void construct_filter();
        void destruct_filter();
        
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();    
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
};

void DCSC__::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    dcsc = new struct DCSC_BASE(nedges, nnzcols_);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors_degree();
    (void)spmv();
    v = y;
    //(void)checksum();
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete dcsc;
    dcsc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    dcsc = new struct DCSC_BASE(nedges, nnzcols_);
    populate();        
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    construct_vectors_pagerank(); 
    space();
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
    stats(t, "DCSC SpMSpV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete dcsc;
    dcsc = nullptr;
}

void DCSC__::construct_filter() {
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    for(uint32_t i = 0; i < nvertices; i++) {
        if(cols[i] == 1)
            nnzcols_++;
    }
}

void DCSC__::destruct_filter() {
    rows.clear();
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_ = 0;
}

void DCSC__::construct_vectors_degree() {
    v.resize(nrows);
    x.resize(nnzcols_, 1);
    y.resize(nrows);
}

void DCSC__::destruct_vectors_degree() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void DCSC__::construct_vectors_pagerank() {
    x.resize(nnzcols_);
    y.resize(nrows);
    d.resize(nrows);
}

void DCSC__::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

uint64_t DCSC__::spmv() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) dcsc->A;
    uint32_t* IA = (uint32_t*) dcsc->IA;
    uint32_t* JA = (uint32_t*) dcsc->JA;
    uint32_t nnzcols = dcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
        }
    }
    return(noperations);
}

void DCSC__::update() {
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void DCSC__::space() {
    total_size += dcsc->size;
    total_size += (2 * sizeof(uint32_t) * x.size()) + (2 * sizeof(uint32_t) + y.size());
}

#endif
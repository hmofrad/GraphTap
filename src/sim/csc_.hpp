/*
 * csc_.hpp: CSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CSC__HPP
#define CSC__HPP 

#include <chrono>

#include "pair.hpp" 
#include "io.cpp" 
#include "base_csc.hpp" 
#include "csc.hpp"

class CSC_ : protected CSC {
    using CSC::CSC;
    public:
        virtual void run_pagerank();
    protected:
        virtual void message();
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        void construct_filter();
        void destruct_filter();
        void destroy_vectors();
        std::vector<char> rows;
        std::vector<uint32_t> rows_nnz;
        std::vector<uint32_t> rows_all;
        uint32_t nnz_rows;
        std::vector<char> cols;
        std::vector<uint32_t> cols_nnz;
        std::vector<uint32_t> cols_all;        
        uint32_t nnz_cols;
};

void CSC_::run_pagerank() {    
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    csc = new struct Base_csc(nedges, nvertices);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors();
    (void)spmv();
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_nnz[i]] =  y[i];
    //(void)checksum();
    //display();
    delete csc;
    csc = nullptr;
    destruct_filter();
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    csc = new struct Base_csc(nedges, nvertices);
    populate();
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    destroy_vectors();
    x.resize(nnz_cols);
    y.resize(nnz_rows);
    d.resize(nrows);
    d = v;
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
    stats(t, "CSC SpMSpV");
    display();
    delete csc;
    csc = nullptr;
    destruct_vectors();
}

void CSC_::construct_filter() {
    rows.resize(nvertices);
    cols.resize(nvertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_all.resize(nvertices);
    cols_all.resize(nvertices);
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = rows_nnz.size() - 1;
        }
        if(cols[i] == 1) {
            cols_nnz.push_back(i);
            cols_all[i] = cols_nnz.size() - 1;
        }
    }
    nnz_rows = rows_nnz.size();
    nnz_cols = cols_nnz.size();
}

void CSC_::destruct_filter() {
    rows.clear();
    rows.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit();
    rows_all.clear();
    rows_all.shrink_to_fit();
    nnz_rows = 0;
    cols.clear();
    cols.shrink_to_fit();
    cols_nnz.clear();
    cols_nnz.shrink_to_fit();
    cols_all.clear();
    cols_all.shrink_to_fit();
    nnz_cols = 0;
}

void CSC_::destroy_vectors() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void CSC_::message() {
    for(uint32_t i = 0; i < nnz_cols; i++)
        x[i] = d[cols_nnz[i]] ? (v[cols_nnz[i]]/d[cols_nnz[i]]) : 0;   
}

uint64_t CSC_::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t ncols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[rows_all[IA[i]]] += (A[i] * x[cols_all[j]]);
            num_operations++;
        }
    }
    return(num_operations);
}

void CSC_::update() {
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void CSC_::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * rows_all.size()) + (sizeof(uint32_t) * rows_nnz.size());
    total_size += (sizeof(uint32_t) * cols_all.size()) + (sizeof(uint32_t) * cols_nnz.size());
}
#endif
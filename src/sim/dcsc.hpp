/*
 * dcsc.cpp: DCSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DCSC_HPP
#define DCSC_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "base_csc.hpp" 
#include "csc.hpp"
 
class DCSC : protected CSC{
    using CSC::CSC;    
    public:
        virtual void run_pagerank();
    protected:
        virtual void populate();
        virtual void message();
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        void filter();
        void destroy_filter();
        void destroy_vectors();
        
        std::vector<char> cols;
        std::vector<uint32_t> cols_vals_nnz;
        std::vector<uint32_t> cols_vals_all;        
        uint32_t nnz_cols;
};

void DCSC::run_pagerank() {
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    filter();
    csc = new struct Base(num_edges, nnz_cols);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    v.resize(num_rows);
    x.resize(nnz_cols, 1);
    y.resize(num_rows);
    (void)spmv();
    v = y;
    //(void)checksum();
    //display();
    delete csc;
    csc = nullptr;
    destroy_filter();

    // PageRank program
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    filter();
    csc = new struct Base(num_edges, nnz_cols);
    populate();
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    destroy_vectors();
    x.resize(nnz_cols);
    d.resize(num_rows);
    d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    for(uint32_t i = 0; i < num_iterations; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        num_operations += spmv();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "DCSC SpMV");
    display();
    delete csc;
    csc = nullptr;
}

void DCSC::filter() {
    cols.resize(num_vertices);
    for(auto &pair: *pairs)
        cols[pair.col] = 1;
    cols_vals_all.resize(num_vertices);
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        if(cols[i] == 1)
        {
            cols_vals_nnz.push_back(i);
            cols_vals_all[i] = cols_vals_nnz.size() - 1;
        }
    }
    nnz_cols = cols_vals_nnz.size();
}

void DCSC::destroy_filter() {
    cols.clear();
    cols.shrink_to_fit();
    cols_vals_nnz.clear();
    cols_vals_nnz.shrink_to_fit();
    cols_vals_all.clear();
    cols_vals_all.shrink_to_fit();
    nnz_cols = 0;
}

void DCSC::destroy_vectors() {
    x.clear();
    x.shrink_to_fit();
}

void DCSC::populate() {
    uint32_t *A  = (uint32_t *) csc->A;  // Weight      
    uint32_t *IA = (uint32_t *) csc->IA; // ROW_INDEX
    uint32_t *JA = (uint32_t *) csc->JA; // COL_PTR
    uint32_t ncols = csc->  ncols_plus_one - 1;
    
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &pair: *pairs) {
        if((j - 1) != cols_vals_all[pair.col]) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
}

void DCSC::message() {
    for(uint32_t i = 0; i < nnz_cols; i++)
        x[i] = d[cols_vals_nnz[i]] ? (v[cols_vals_nnz[i]]/d[cols_vals_nnz[i]]) : 0;   
}

uint64_t DCSC::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t num_cols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < num_cols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            num_operations++;
        }
    }
    return(num_operations);
}

void DCSC::update() {
    for(uint32_t i = 0; i < num_rows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void DCSC::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * cols_vals_all.size()) + (sizeof(uint32_t) * cols_vals_nnz.size());
}
#endif
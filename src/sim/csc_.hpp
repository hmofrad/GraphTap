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
        
        void filter();
        void destroy_filter();
        void destroy_vectors();
        std::vector<char> rows;
        std::vector<uint32_t> rows_vals_nnz;
        std::vector<uint32_t> rows_vals_all;
        uint32_t nnz_rows;
        std::vector<char> cols;
        std::vector<uint32_t> cols_vals_nnz;
        std::vector<uint32_t> cols_vals_all;        
        uint32_t nnz_cols;
};

void CSC_::run_pagerank() {    
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    filter();
    csc = new struct Base(num_edges, num_vertices);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    v.resize(num_rows);
    x.resize(nnz_cols, 1);
    y.resize(nnz_rows);
    (void)spmv();
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] =  y[i];
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
    csc = new struct Base(num_edges, num_vertices);
    populate();
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    destroy_vectors();
    x.resize(nnz_cols);
    y.resize(nnz_rows);
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
    stats(t, "CSC SpMSpV");
    display();
    delete csc;
    csc = nullptr;
}

void CSC_::filter() {
    rows.resize(num_vertices);
    cols.resize(num_vertices);
    for(auto &pair: *pairs)
    {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_vals_all.resize(num_vertices);
    cols_vals_all.resize(num_vertices);
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        if(rows[i] == 1)
        {
            rows_vals_nnz.push_back(i);
            rows_vals_all[i] = rows_vals_nnz.size() - 1;
        }
        if(cols[i] == 1)
        {
            cols_vals_nnz.push_back(i);
            cols_vals_all[i] = cols_vals_nnz.size() - 1;
        }
    }
    nnz_rows = rows_vals_nnz.size();
    nnz_cols = cols_vals_nnz.size();
}

void CSC_::destroy_filter() {
    rows.clear();
    rows.shrink_to_fit();
    rows_vals_nnz.clear();
    rows_vals_nnz.shrink_to_fit();
    rows_vals_all.clear();
    rows_vals_all.shrink_to_fit();
    nnz_rows = 0;
    cols.clear();
    cols.shrink_to_fit();
    cols_vals_nnz.clear();
    cols_vals_nnz.shrink_to_fit();
    cols_vals_all.clear();
    cols_vals_all.shrink_to_fit();
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
        x[i] = d[cols_vals_nnz[i]] ? (v[cols_vals_nnz[i]]/d[cols_vals_nnz[i]]) : 0;   
}

uint64_t CSC_::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t num_cols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < num_cols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[rows_vals_all[IA[i]]] += (A[i] * x[cols_vals_all[j]]);
            num_operations++;
        }
    }
    return(num_operations);
}

void CSC_::update() {
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void CSC_::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * rows_vals_all.size()) + (sizeof(uint32_t) * rows_vals_nnz.size());
    total_size += (sizeof(uint32_t) * cols_vals_all.size()) + (sizeof(uint32_t) * cols_vals_nnz.size());
}
#endif
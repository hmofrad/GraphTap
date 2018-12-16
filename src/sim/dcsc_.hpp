/*
 * dcsc_.cpp: DCSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DCSC__HPP
#define DCSC__HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "base_dcsc.hpp" 
#include "csc.hpp"
 
class DCSC_ : protected CSC {
    using CSC::CSC;    
    public:
        virtual void run_pagerank();
    protected:
        struct Base_dcsc *dcsc = nullptr;
        virtual void populate();
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

void DCSC_::run_pagerank() {
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    filter();
    dcsc = new struct Base_dcsc(num_edges, nnz_cols);
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
    delete dcsc;
    dcsc = nullptr;
    destroy_filter();
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    filter();
    dcsc = new struct Base_dcsc(num_edges, nnz_cols);
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
    for(uint32_t i = 0; i < num_iterations; i++) {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        num_operations += spmv();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "DCSC SpMSpV");
    display();
    delete dcsc;
    dcsc = nullptr;
}


void DCSC_::filter() {
    rows.resize(num_vertices);
    cols.resize(num_vertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_vals_all.resize(num_vertices);
    cols_vals_all.resize(num_vertices);
    for(uint32_t i = 0; i < num_vertices; i++) {
        if(rows[i] == 1) {
            rows_vals_nnz.push_back(i);
            rows_vals_all[i] = rows_vals_nnz.size() - 1;
        }
        if(cols[i] == 1) {
            cols_vals_nnz.push_back(i);
            cols_vals_all[i] = cols_vals_nnz.size() - 1;
        }
    }
    nnz_rows = rows_vals_nnz.size();
    nnz_cols = cols_vals_nnz.size();
}

void DCSC_::destroy_filter() {
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

void DCSC_::destroy_vectors() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void DCSC_::populate() {
    uint32_t *A  = (uint32_t *) dcsc->A;  // Weight      
    uint32_t *IA = (uint32_t *) dcsc->IA; // ROW_INDEX
    uint32_t *JA = (uint32_t *) dcsc->JA; // COL_PTR
    uint32_t ncols = dcsc->nnzcols_plus_one - 1;
    
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

void DCSC_::message() {
    for(uint32_t i = 0; i < nnz_cols; i++)
        x[i] = d[cols_vals_nnz[i]] ? (v[cols_vals_nnz[i]]/d[cols_vals_nnz[i]]) : 0;   
}

uint64_t DCSC_::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) dcsc->A;
    uint32_t *IA = (uint32_t *) dcsc->IA;
    uint32_t *JA = (uint32_t *) dcsc->JA;
    uint32_t num_cols = dcsc->nnzcols_plus_one - 1;
    for(uint32_t j = 0; j < num_cols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[rows_vals_all[IA[i]]] += (A[i] * x[j]);
            num_operations++;
        }
    }
    return(num_operations);
}

void DCSC_::update() {
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void DCSC_::space() {
    total_size += dcsc->size;
    total_size += (sizeof(uint32_t) * rows_vals_all.size()) + (sizeof(uint32_t) * rows_vals_nnz.size());
    total_size += (sizeof(uint32_t) * cols_vals_all.size()) + (sizeof(uint32_t) * cols_vals_nnz.size());
}
#endif
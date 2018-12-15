/*
 * dcsc.cpp: DCSC SpMV implementation
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

struct Edge_
{
  const uint32_t dst;

  Edge_() : dst(0) {}

  Edge_(const uint32_t dst)
      : dst(dst) {}
};


uint32_t nnz_rows_;
std::vector<char> rows_;
std::vector<uint32_t> rows_val_;
std::vector<uint32_t> rows2vals_;
uint32_t nnz_cols_;
std::vector<char> cols_;
std::vector<uint32_t> cols_val_;
std::vector<uint32_t> cols2vals_;
std::vector<int> rows2cols_;
uint32_t nnz_rows2cols_;
std::vector<int> cols2rows_;


void filtering_dcsc(uint32_t num_vertices)
{
    rows_.resize(num_vertices);
    rows_val_.resize(num_vertices);
    cols_.resize(num_vertices);
    cols_val_.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        rows_[triple.row] = 1;
        cols_[triple.col] = 1;
    }
    
    uint32_t i = 0, j = 0;
    for(uint32_t k = 0; k < num_vertices; k++)
    {
        if(rows_[k] and cols_[k])
        {
            rows2cols_.push_back(i);
            cols2rows_.push_back(j);
        }
        
        if(rows_[k])
        {
            rows2vals_.push_back(k);
            rows_val_[k] = i;
            i++;
        }
        if(cols_[k])
        {
            cols2vals_.push_back(j);
            cols_val_[k] = j;
            j++;
        }
        //printf("%d %d %d\n", k, rows[k], cols[k]);
    }
    nnz_rows_ = i;
    nnz_cols_ = j;
    nnz_rows2cols_ = rows2cols_.size();
    printf("[x]Filtering is done\n");
}

void init_dcsc(uint32_t nnz_, uint32_t ncols)
{
    nnz = nnz_;
    ncols_plus_one = ncols + 1;
    if((A = (uint32_t *) mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(uint32_t));
    
    if((IA = (uint32_t *) mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(uint32_t));

    if((JA = (uint32_t *) mmap(nullptr, ncols_plus_one * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    size =  nnz * (sizeof(uint32_t) + sizeof(uint32_t)) + (ncols_plus_one * sizeof(uint32_t)) + (nnz_cols_ * sizeof(uint32_t));
}

void popu_dcsc()
{
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &triple: *triples)
    {
        while((j - 1) != cols_val_[triple.col])
        {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_val_[triple.row];
        i++;
    }
}

void run_dcsc()
{
    init_dcsc(triples->size(), nnz_cols_);
    popu_dcsc();
    printf("[x]Compression is done\n");
}

void walk_dcsc()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            printf("   i=%d j=%d\n", IA[i], j);
        }
    }
}

void init_dcsc_vecs()
{
    values.resize(num_vertices);
    y.resize(num_vertices);
    x.resize(nnz_cols_, 1);
}

void spmv_dcsc()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            auto edge = Edge_(cols2vals_[j]);
            y[IA[i]] += A[i] * x[j]; 
            nOps++;
            
        }
    }
    for(uint32_t i = 0; i < nnz_rows2cols_; i++)
    {
        x[cols2rows_[i]] = y[rows2cols_[i]];
        x[cols2rows_[i]] = 1;
    }
}

void done_dcsc()
{
    
    for(uint32_t i = 0; i < num_vertices; i++)
            values[i] = y[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];
    
    
}

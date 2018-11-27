/*
 * tcsc.cpp: TCSC SpMV implementation (GraphTap)
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

// Vertex filtering 
uint32_t nnz_rows;
std::vector<char> rows;
std::vector<uint32_t> rows_val;
std::vector<uint32_t> rows2vals;
uint32_t nnz_cols;
std::vector<char> cols;
std::vector<uint32_t> cols_val;
std::vector<int> rows2cols;
uint32_t nnz_rows2cols;
std::vector<int> cols2rows;


void filtering(uint32_t num_vertices)
{
    rows.resize(num_vertices);
    rows_val.resize(num_vertices);
    cols.resize(num_vertices);
    cols_val.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        rows[triple.row] = 1;
        cols[triple.col] = 1;
    }
    
    uint32_t i = 0, j = 0;
    for(uint32_t k = 0; k < num_vertices; k++)
    {
        if(rows[k] and cols[k])
        {
            rows2cols.push_back(i);
            cols2rows.push_back(j);
        }
        
        if(rows[k])
        {
            rows2vals.push_back(k);
            rows_val[k] = i;
            i++;
        }
        if(cols[k])
        {
            cols_val[k] = j;
            j++;
        }
        //printf("%d %d %d\n", k, rows[k], cols[k]);
    }
    nnz_rows = i;
    nnz_cols = j;
    nnz_rows2cols = rows2cols.size();
    printf("[x]Filtering is done\n");
}

void init_tcsc(uint32_t nnz_, uint32_t ncols)
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
    size =  nnz * (sizeof(uint32_t) + sizeof(uint32_t)) + (ncols_plus_one * sizeof(uint32_t));
}

void popu_tcsc()
{
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &triple: *triples)
    {
        while((j - 1) != cols_val[triple.col])
        {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_val[triple.row];
        i++;
    }
}

void run_tcsc()
{
    init_tcsc(triples->size(), nnz_cols);
    popu_tcsc();
    printf("[x]Compression is done\n");
}

void walk_tcsc()
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

void init_tcsc_vecs()
{
    values.resize(num_vertices);
    y.resize(nnz_rows);
    x.resize(nnz_cols, 1);
}

void spmv_tcsc()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            y[IA[i]] += A[i] * x[j]; 
            nOps++;
            
        }
    }
    for(uint32_t i = 0; i < nnz_rows2cols; i++)
    {
        x[cols2rows[i]] = y[rows2cols[i]];
        x[cols2rows[i]] = 1;
    }
}

void done_tcsc()
{
    
    for(uint32_t i = 0; i < nnz_rows; i++)
            values[rows2vals[i]] = y[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];
    
    
}

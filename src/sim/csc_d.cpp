/*
 * csc_d.cpp: CSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
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

uint32_t nnz_rows__;
std::vector<char> rows__;
std::vector<uint32_t> rows_val__;
std::vector<uint32_t> rows2vals__;
uint32_t nnz_cols__;
std::vector<char> cols__;
std::vector<uint32_t> cols_val__;
std::vector<int> rows2cols__;
uint32_t nnz_rows2cols__;
std::vector<int> cols2rows__;


void filtering_csc_d(uint32_t num_vertices)
{
    rows__.resize(num_vertices);
    rows_val__.resize(num_vertices);
    cols__.resize(num_vertices);
    cols_val__.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        rows__[triple.row] = 1;
        cols__[triple.col] = 1;
    }
    
    uint32_t i = 0, j = 0;
    for(uint32_t k = 0; k < num_vertices; k++)
    {
        if(rows__[k] and cols__[k])
        {
            rows2cols__.push_back(i);
            cols2rows__.push_back(j);
        }
        
        if(rows__[k])
        {
            rows2vals__.push_back(k);
            rows_val__[k] = i;
            i++;
        }
        if(cols__[k])
        {
            cols_val__[k] = j;
            j++;
        }
        //printf("%d %d %d\n", k, rows[k], cols[k]);
    }
    nnz_rows__ = i;
    nnz_cols__ = j;
    nnz_rows2cols__ = rows2cols__.size();
    printf("[x]Filtering is done\n");
}


void init_csc_d(uint32_t nnz_, uint32_t ncols)
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
    size =  nnz * (sizeof(uint32_t) + sizeof(uint32_t)) + (ncols_plus_one * sizeof(uint32_t)) + (nnz_rows__ * sizeof(uint32_t));
}

void popu_csc_d()
{
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &triple: *triples)
    {
        while((j - 1) != triple.col)
        {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = triple.row;
        i++;
    }
    while((j + 1) < (num_vertices + 1))
    {
        j++;
        JA[j] = JA[j - 1];
    }
}

void run_csc_d()
{
    init_csc_d(triples->size(), num_vertices);
    popu_csc_d();
    printf("[x]Compression is done\n");
}

void walk_csc_d()
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

void init_csc_d_vecs()
{
    values.resize(num_vertices);
    y.resize(nnz_rows__);
    x.resize(nnz_cols__, 1);
}

void spmv_csc_d()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            y[rows_val__[IA[i]]] += A[i] * x[cols_val__[j]]; 
            nOps++;
        }
    }
    
    for(uint32_t i = 0; i < nnz_rows2cols__; i++)
    {
        x[cols2rows__[i]] = y[rows2cols__[i]];
        x[cols2rows__[i]] = 1;
    }
}

void done_csc_d()
{
    
    for(uint32_t i = 0; i < nnz_rows__; i++)
            values[rows2vals__[i]] = y[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];
    
    
}

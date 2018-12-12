/*
 * dcsc_d_d.cpp: dcsc_d SpMSpV implementation
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


uint32_t nnz_rows___;
std::vector<char> rows___;
std::vector<uint32_t> rows_val___;
std::vector<uint32_t> rows2vals___;
uint32_t nnz_cols___;
std::vector<char> cols___;
std::vector<uint32_t> cols_val___;
std::vector<int> rows2cols___;
uint32_t nnz_rows2cols___;
std::vector<int> cols2rows___;


void filtering_dcsc_d(uint32_t num_vertices)
{
    rows___.resize(num_vertices);
    rows_val___.resize(num_vertices);
    cols___.resize(num_vertices);
    cols_val___.resize(num_vertices);
    
    for(auto &triple: *triples)
    {
        rows___[triple.row] = 1;
        cols___[triple.col] = 1;
    }
    
    uint32_t i = 0, j = 0;
    for(uint32_t k = 0; k < num_vertices; k++)
    {
        if(rows___[k] and cols___[k])
        {
            rows2cols___.push_back(i);
            cols2rows___.push_back(j);
        }
        
        if(rows___[k])
        {
            rows2vals___.push_back(k);
            rows_val___[k] = i;
            i++;
        }
        if(cols___[k])
        {
            cols_val___[k] = j;
            j++;
        }
        //printf("%d %d %d\n", k, rows[k], cols[k]);
    }
    nnz_rows___ = i;
    nnz_cols___ = j;
    nnz_rows2cols___ = rows2cols___.size();
    printf("[x]Filtering is done\n");
}

void init_dcsc_d(uint32_t nnz_, uint32_t ncols)
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
    size =  nnz * (sizeof(uint32_t) + sizeof(uint32_t)) + (ncols_plus_one * sizeof(uint32_t)) + (nnz_cols___ * sizeof(uint32_t)) + (nnz_rows___ * sizeof(uint32_t));
    
}

void popu_dcsc_d()
{
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &triple: *triples)
    {
        while((j - 1) != cols_val___[triple.col])
        {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_val___[triple.row];
        i++;
    }
}

void run_dcsc_d()
{
    init_dcsc_d(triples->size(), nnz_cols___);
    popu_dcsc_d();
    printf("[x]Compression is done\n");
}

void walk_dcsc_d()
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

void init_dcsc_d_vecs()
{
    values.resize(num_vertices);
    y.resize(nnz_rows___);
    x.resize(nnz_cols___, 1);
}

void spmv_dcsc_d()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            y[rows_val___[IA[i]]] += A[i] * x[j]; 
            nOps++;
            
        }
    }
    
    for(uint32_t i = 0; i < nnz_rows2cols___; i++)
    {
        x[cols2rows___[i]] = y[rows2cols___[i]];
        x[cols2rows___[i]] = 1;
    }
}

void done_dcsc_d()
{
    
    for(uint32_t i = 0; i < nnz_rows___; i++)
            values[rows2vals___[i]] = y[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];
    
    
}

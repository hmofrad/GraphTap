/*
 * graphtap.cpp: GraphTap SpMV implementation
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
uint32_t nnz_cols;
std::vector<char> cols;
std::vector<uint32_t> cols_val;

uint32_t nnz;
uint32_t ncols_plus_one;
char *A;
uint32_t *IA;
uint32_t *JA;

struct ColSort
{
    bool operator()(const struct Triple &a, const struct Triple &b)
    {
        return((a.col == b.col) ? (a.row < b.row) : (a.col < b.col));
    }
};

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
        if(rows[k])
        {
            rows_val[k] = i;
            i++;
        }
        if(cols[k])
        {
            cols_val[k] = j;
            j++;
        }
    }
    
    nnz_rows = i;
    nnz_cols = j;
}

void init_csc(uint32_t nnz_, uint32_t ncols)
{
    nnz = nnz_;
    ncols_plus_one = ncols + 1;
    if((A = (char *) mmap(nullptr, nnz * sizeof(char), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(char));
    
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
}

void popu_csc()
{
    uint32_t i = 0;
    uint32_t j = 1;
    
    JA[0] = 0;

    for(auto &triple: *triples)
    {

        //printf("%d %d %d %d, %d %d %d %d\n", i, triple.row, rows[triple.row], rows_val[triple.row], j, triple.col, cols[triple.col], cols_val[triple.col]);
        
        while((j - 1) != cols_val[triple.col])
        {
            j++;
            JA[j] = JA[j - 1];
        }  
                
        A[i] = 1;
        JA[j]++;
        IA[i] = rows[triple.row];
        i++;
    }

    while((j + 1) < (nnz_cols + 1))
    {
        j++;
        JA[j] = JA[j - 1];
    }
}


void kernel()
{
    std::vector<uint32_t> y(nnz_rows);
    std::vector<uint32_t> x(nnz_cols);
    std::fill(x.begin(), x.end(), 1);
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            printf("%d %d %d %d %d\n", IA[i], j, y[IA[i]], A[i], x[j]);
            y[IA[i]] += A[i] * x[j]; 
        }
    }
    
    uint32_t value = 0;
    for(uint32_t i = 0; i < nnz_rows; i++)
        value += y[i];
    printf("value=%d\n", value);
}

/*
 * csc.cpp: CSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include "base.hpp" 

void populate_csc(struct CSC *csc) {
    uint32_t *A  = (uint32_t *) csc->A;  // Weight      
    uint32_t *IA = (uint32_t *) csc->IA; // ROW_INDEX
    uint32_t *JA = (uint32_t *) csc->JA; // COL_PTR
    uint32_t ncols = csc->  ncols_plus_one - 1;
    
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &pair: *pairs) {
        while((j - 1) != pair.col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (ncols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
    printf("[x]CSC is done: Data structure size is %lu.\n", csc->size);
}

void walk_csc(struct CSC *csc) {
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t ncols = csc->ncols_plus_one - 1;
    
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], j, A[i]);
        }
    }
}


void spmv_csc(struct CSC *csc, std::vector<double>& x, std::vector<double>& y) {
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t ncols = csc->ncols_plus_one - 1;
    //printf("START %d\n", ncols);
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            //y[IA[i]] += A[i] * x[j]; 
            y[IA[i]] += x[j]; 
            //if(j < 10)
              //  printf("%f %d %f\n" ,y[IA[i]], A[i], x[j]);
            //nOps++;
        }
    }
        //for(uint32_t i = 0; i < 10; i++)    
       // printf("DONE\n");
}

void copy_csc_v_x(const std::vector<double> v, const std::vector<double> d, std::vector<double>& x) {
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
    //{
        x[i] = d[i] ? (v[i]/d[i]) : 0;
    //for(uint32_t i = 0; i < 10; i++)    
      //  printf("%f %f\n", d[i], x[i]);
    
}

void copy_csc_y_v(const std::vector<double> y, std::vector<double>& v) {
    uint32_t nrows = y.size();
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = y[i];
}

void values_csc(const std::vector<double> v, int num_vals = 10)
{
    for(uint32_t i = 0; i < num_vals; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void checksum_csc(const std::vector<double> v, const std::string preamble) {
    double value = 0.0;
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    std::cout << "[x]" << preamble << " checksum: " <<  value << std::endl;
}

void update_csc(std::vector<double>& v, std::vector<double> y, double alpha) {
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void run_pagerank_csc(std::string file_path, uint32_t num_vertices, int num_iters) {
    printf("[x]CSC SpMV kernel unit test...\n");
    // Degree program
    pairs = new std::vector<struct Pair>;
    read_binary(file_path);
    column_sort();
    struct CSC *csc = new struct CSC(pairs->size(), num_vertices);
    populate_csc(csc);
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk_csc(csc);
    std::vector<double> v(num_vertices);
    std::vector<double> x(num_vertices, 1);
    std::vector<double> y(num_vertices);
    spmv_csc(csc, x, y);
    copy_csc_y_v(y, v);
    values_csc(v);
    checksum_csc(v, "Degree");
    delete csc;
    csc = nullptr;
    
    // PageRank program
    double alpha = 0.15;
    pairs = new std::vector<struct Pair>;
    read_binary(file_path, true);
    column_sort();
    csc = new struct CSC(pairs->size(), num_vertices);
    populate_csc(csc);
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    std::vector<double> d(num_vertices);
    d = v;
    std::fill(v.begin(), v.end(), alpha);
    
    for(int i = 0; i < 20; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        copy_csc_v_x(v, d, x);
        spmv_csc(csc, x, y);
        update_csc(v, y, alpha);
        
    }
    values_csc(v);
    checksum_csc(v, "PageRank");
    
    
    
    
    

    
    //csc(pairs->size(), num_vertices);
    
}

/* 
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

void init_csc(uint32_t nnz_, uint32_t ncols)
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

void popu_csc()
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

void run_csc()
{
    init_csc(triples->size(), num_vertices);
    popu_csc();
    printf("[x]Compression is done\n");
}

void walk_csc()
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

void init_csc_vecs()
{
    values.resize(num_vertices);
    y.resize(num_vertices);
    x.resize(num_vertices, 1);
}

void spmv_csc()
{
    for(uint32_t j = 0; j < ncols_plus_one - 1; j++)
    {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
        {
            y[IA[i]] += A[i] * x[j]; 
            nOps++;
        }
    }
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        x[i] = y[i];
        x[i] = 1;
    }
}

void done_csc()
{
    
    for(uint32_t i = 0; i < num_vertices; i++)
            values[i] = y[i];
    
    for(uint32_t i = 0; i < num_vertices; i++)
        value += values[i];
}
*/

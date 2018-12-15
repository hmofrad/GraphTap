/*
 * base_csc.hpp: Base class for column compressed data structures
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BASE_CSC_HPP
#define BASE_CSC_HPP

#include <sys/mman.h>
#include <cstring> 

struct Base
{
    Base(uint64_t nnz_, uint32_t ncols_plus_one_);
    ~Base();
    uint64_t nnz;
    uint32_t ncols_plus_one;
    uint64_t size;
    
    void *A;  // VAL
    void *IA; // ROW_INDEX
    void *JA; //COL_PTR
};

Base::Base(uint64_t nnz_, uint32_t ncols_plus_one_)
{
    nnz = nnz_;
    ncols_plus_one = ncols_plus_one_;
    if((A = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(uint32_t));
    
    if((IA = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(uint32_t));

    if((JA = mmap(nullptr, ncols_plus_one * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, ncols_plus_one * sizeof(uint32_t));
    
    size = (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint32_t)) + (ncols_plus_one * sizeof(uint32_t));
}

Base::~Base()
{
    if(munmap(A, nnz * sizeof(uint32_t)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(IA, nnz * sizeof(uint32_t)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, ncols_plus_one * sizeof(uint32_t)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}
#endif
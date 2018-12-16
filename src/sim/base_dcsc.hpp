/*
 * Base_dcsc.hpp: Base_csc class for column compressed data structures
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BASE_DCSC_HPP
#define BASE_DCSC_HPP

#include <sys/mman.h>
#include <cstring> 

struct Base_dcsc {
    public:
        Base_dcsc(uint64_t nnz_, uint32_t nnzcols_);
        ~Base_dcsc();
        uint64_t nnz;
        uint32_t nnzcols_plus_one;
        uint64_t size;
        void *A;  // VAL
        void *IA; // ROW_INDEX
        void *JA; //COL_PTR
        void *JC; //COL_INDEX
};

Base_dcsc::Base_dcsc(uint64_t nnz_, uint32_t nnzcols_) {
    nnz = nnz_;
    nnzcols_plus_one = nnzcols_ + 1;
    if((A = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(uint32_t));
    if((IA = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(uint32_t));
    if((JA = mmap(nullptr, nnzcols_plus_one * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, nnzcols_plus_one * sizeof(uint32_t));
    if((JC = mmap(nullptr, nnzcols_ * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC, 0, (nnzcols_plus_one - 1) * sizeof(uint32_t));
    size = (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint32_t)) + (nnzcols_plus_one * sizeof(uint32_t)) + ((nnzcols_plus_one - 1) * sizeof(uint32_t));
}

Base_dcsc::~Base_dcsc() {
    if(munmap(A, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    if(munmap(IA, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    if(munmap(JA, nnzcols_plus_one * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    if(munmap(JC, (nnzcols_plus_one - 1) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}
#endif
/*
 * Base_dcsc.hpp: Base_csc class for column compressed data structures
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BASE_ODCSC_HPP
#define BASE_ODCSC_HPP

#include <sys/mman.h>
#include <cstring> 

struct CSCEntry
{
  uint32_t global_idx;
  uint32_t idx;
  uint32_t weight;
};

struct Edge
{
  const uint32_t src, dst;

  const char weight;

  Edge() : src(0), dst(0), weight(1) {}

  Edge(const uint32_t src, const uint32_t dst, const char weight)
      : src(src), dst(dst), weight(weight) {}
};

struct Base_odcsc {
    public:
        Base_odcsc(uint64_t nnz_, uint32_t nnzcols_);
        ~Base_odcsc();
        uint64_t nnz;
        uint32_t nnzcols;
        uint64_t size;
        void *A;  // WEIGHT
        void *IA; // ROW_IDX
        void *JA; // COL_PTR
        void *JC; // COL_IDX
};

Base_odcsc::Base_odcsc(uint64_t nnz_, uint32_t nnzcols_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    
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
    
    if((JA = mmap(nullptr, (nnzcols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, (nnzcols + 1) * sizeof(uint32_t));
    
    if((JC = mmap(nullptr, nnzcols * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC, 0, nnzcols * sizeof(uint32_t));
    
    size = (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint32_t)) + ((nnzcols + 1) * sizeof(uint32_t)) + (nnzcols * sizeof(uint32_t));
}

Base_odcsc::~Base_odcsc() {
    if(munmap(A, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(IA, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, (nnzcols + 1) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JC, nnzcols * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}
#endif
/*
 * compressed_column.hpp: Column compressed storage implementaion
 * Compressed Sparse Column (CSC)
 * Double Compressed Sparse Column (DCSC)
 * Triple Compressed Sparse Column (TCSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP

#include <sys/mman.h>
#include <cstring> 
 
enum Compression_type
{
    _CSR_,
  _CSC_, // Compressed Sparse Col
 _DCSC_, // Compressed Sparse Col
 _TCSC_, // Compressed Sparse Col
};

template<typename Weight, typename Integer_Type>
class Compressed_column
{
    public:
        Compressed_column() {}
        ~Compressed_column() {}
        virtual void populate();
};

/*
class NullHasher : public ReversibleHasher
{
    public:
        NullHasher() {}
        long hash(long v) const { return v; }
        long unhash(long v) const { return v; }
};
*/




template<typename Weight, typename Integer_Type>
struct CSC
{
    CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_);
    ~CSC();
    Integer_Type nnz;
    Integer_Type ncols_plus_one;
    
    void *A;  // VAL
    void *IA; // ROW_INDEX
    void *JA; //COL_PTR
};

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_)
{
    nnz = nnz_;
    ncols_plus_one = ncols_plus_one_;
    #ifdef HAS_WEIGHT
    if((A = mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(A, 0, nnz * sizeof(Weight));
    #endif
    #ifdef PREFETCH
    #ifdef HAS_WEIGHT
    madvise(A, nnz * sizeof(Weight), MADV_SEQUENTIAL);
    #endif
    #endif
    
    if((IA = mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(IA, 0, nnz * sizeof(Integer_Type));
    #ifdef PREFETCH
    madvise(IA, nnz * sizeof(Integer_Type), MADV_SEQUENTIAL);
    #endif

    if((JA = mmap(nullptr, ncols_plus_one * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(JA, 0, ncols_plus_one * sizeof(Integer_Type));
    #ifdef PREFETCH
    madvise(JA, ncols_plus_one * sizeof(Integer_Type), MADV_SEQUENTIAL);
    #endif
}

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::~CSC()
{
    #ifdef HAS_WEIGHT
    if(munmap(A, nnz * sizeof(Weight)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
    #endif
    
    if(munmap(IA, nnz * sizeof(Integer_Type)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
    
    if(munmap(JA, ncols_plus_one * sizeof(Integer_Type)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}
#endif
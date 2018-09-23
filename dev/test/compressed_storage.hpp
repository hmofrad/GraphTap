/*
 * compressed_storage.hpp: Compressed storage implementaion
 * Compressed Sparse Row (CSR)
 * Compressed Sparse Column (CSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP

#include <sys/mman.h>
#include <cstring> 
 
enum Compression_type
{
  _CSR_, // Compressed Sparse Row
  _CSC_, // Compressed Sparse Col
};


/* Compressed Sparse Column */
template<typename Weight, typename Integer_Type>
struct CSR
{
    CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_);
    ~CSR();
    Integer_Type nnz;
    Integer_Type nrows_plus_one;
    uint64_t nbytes;
    
    void *A;
    void *IA;
    void *JA;
};

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_)
{
    nnz = nnz_;
    nrows_plus_one = nrows_plus_one_;
    /* #define HAS_WEIGHT is a hack over partial specialization becuase
       we didn't want to duplicate the code for Empty weights though! */
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
    
    if((IA = mmap(nullptr, nrows_plus_one * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(IA, 0, nrows_plus_one * sizeof(Integer_Type));
    #ifdef PREFETCH
    madvise(IA, nrows_plus_one * sizeof(Integer_Type), MADV_SEQUENTIAL);
    #endif
    
    if((JA = mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(JA, 0, nnz * sizeof(Integer_Type));
    #ifdef PREFETCH
    madvise(JA, nnz * sizeof(Integer_Type), MADV_SEQUENTIAL);
    #endif
}

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::~CSR()
{
    #ifdef HAS_WEIGHT
    if(munmap(A, nnz * sizeof(Weight)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
    #endif
    
    if(munmap(IA, nrows_plus_one * sizeof(Integer_Type)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
    
    if(munmap(JA, nnz * sizeof(Integer_Type)) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}


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



/* 
template<typename Weight, typename Integer_Type>
struct CSR
{
    CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_);
    ~CSR();
    Integer_Type nnz;
    Integer_Type nrows_plus_one;
    
    struct Basic_Storage<Weight, Integer_Type> *A;
    struct Basic_Storage<Integer_Type, Integer_Type> *IA;
    struct Basic_Storage<Integer_Type, Integer_Type> *JA;
    
    void populate();
    void del_csr();
};

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_)
{
    nnz = nnz_;
    nrows_plus_one = nrows_plus_one_;
    #ifdef HAS_WEIGHT
    A = new struct Basic_Storage<Weight, Integer_Type>(nnz);
    #endif
    #ifdef PREFETCH
    #ifdef HAS_WEIGHT
    madvise(A->data, A->nbytes, MADV_SEQUENTIAL);
    #endif
    #endif
    
    
    IA = new struct Basic_Storage<Integer_Type, Integer_Type>(nrows_plus_one);
    #ifdef PREFETCH
    madvise(IA->data, IA->nbytes, MADV_SEQUENTIAL);
    #endif
    
    JA = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz);
    #ifdef PREFETCH
    madvise(JA->data, JA->nbytes, MADV_SEQUENTIAL);
    #endif
}

template<typename Weight, typename Integer_Type>
void CSR<Weight, Integer_Type>::populate()
{

    // #define HAS_WEIGHT is a hack over partial specialization becuase
       //we didn't want to duplicate the code for Empty weights though! 
    #ifdef HAS_WEIGHT
    Weight *A = A->data;
    #endif
    Integer_Type *IA = (Integer_Type *) IA->data;
    Integer_Type *JA = (Integer_Type *) JA->data;
    IA[0] = 0;

}

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::~CSR()
{
    #ifdef HAS_WEIGHT
    delete A;
    #endif
    delete IA;
    delete JA;
}

template<typename Weight, typename Integer_Type>
void CSR<Weight, Integer_Type>::del_csr()
{
    //A->del_storage();
    #ifdef HAS_WEIGHT
    A->del_storage();
    delete A;
    #endif
    IA->del_storage();
    delete IA;
    JA->del_storage();
    delete JA;
}
*/
/*

template<typename Weight, typename Integer_Type>
struct CSC
{
    CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_);
    ~CSC();
    Integer_Type nnz;
    Integer_Type ncols_plus_one;
    
    struct Basic_Storage<Weight, Integer_Type> *VAL;
    struct Basic_Storage<Integer_Type, Integer_Type> *ROW_INDEX;
    struct Basic_Storage<Integer_Type, Integer_Type> *COL_PTR;
    
    void populate();
    void del_csc();
};

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_)
{
    nnz = nnz_;
    ncols_plus_one = ncols_plus_one_;
    #ifdef HAS_WEIGHT
    VAL = new struct Basic_Storage<Weight, Integer_Type>(nnz);
    #endif
    #ifdef PREFETCH
    #ifdef HAS_WEIGHT
    madvise(VAL->data, VAL->nbytes, MADV_SEQUENTIAL);
    #endif
    #endif
    
    ROW_INDEX = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz);
    #ifdef PREFETCH
    madvise(ROW_INDEX->data, ROW_INDEX->nbytes, MADV_SEQUENTIAL);
    #endif

    COL_PTR = new struct Basic_Storage<Integer_Type, Integer_Type>(ncols_plus_one);
    #ifdef PREFETCH
    madvise(COL_PTR->data, COL_PTR->nbytes, MADV_SEQUENTIAL);
    #endif
}

template<typename Weight, typename Integer_Type>
void CSC<Weight, Integer_Type>::populate()
{

}

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::~CSC()
{
    #ifdef HAS_WEIGHT
    delete VAL;
    #endif
    delete ROW_INDEX;
    delete COL_PTR;
}

template<typename Weight, typename Integer_Type>
void CSC<Weight, Integer_Type>::del_csc()
{
    #ifdef HAS_WEIGHT
    VAL->del_storage();
    delete VAL;
    #endif
    ROW_INDEX->del_storage();
    delete ROW_INDEX;
    COL_PTR->del_storage();
    delete COL_PTR;
}
*/
#endif
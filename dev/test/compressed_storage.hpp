/*
 * compressed_storage.hpp: Compressed storage implementaion
 * Compressed Sparse Row (CSR)
 * Compressed Sparse Column (CSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP
 
enum Compression_type
{
  _CSR_, // Compressed Sparse Row
  _CSC_, // Compressed Sparse Col
  _TCSR_,// Triply Compressed Sparse Row
  _TCSC_ // triply Compressed Sparse Col
};
 
#include "basic_storage.hpp"
 
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

    /* #define HAS_WEIGHT is a hack over partial specialization becuase
       we didn't want to duplicate the code for Empty weights though! */
    #ifdef HAS_WEIGHT
    Weight *A = A->data;
    #endif
    Integer_Type *IA = (Integer_Type *) IA->data;
    Integer_Type *JA = (Integer_Type *) JA->data;
    IA[0] = 0;
    /*
    for (auto& triple : *(tile.triples))
    {
        pair = rebase(triple);
        while((j - 1) != pair.row)
        {
            j++;
            IA[j] = IA[j - 1];
        }            
        // In case weights are there
        if(has_weight)
        {
            A[i] = triple.weight;
        }
        IA[j]++;
        JA[i] = pair.col;    
        i++;
        //printf("%d %d %d\n", triple.row, triple.col, triple.weight);
    }
    
    while(j < tile_height)
    {
        j++;
        IA[j] = IA[j - 1];
    }
    */
}

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::~CSR()
{
    //delete A;
    //delete IA;
    //delete JA;
}

template<typename Weight, typename Integer_Type>
void CSR<Weight, Integer_Type>::del_csr()
{
    //A->del_storage();
    #ifdef HAS_WEIGHT
    delete A;
    #endif
    delete IA;
    delete JA;
}


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
    //delete VAL;
    //delete ROW_INDEX;
    //delete COL_PTR;
}

template<typename Weight, typename Integer_Type>
void CSC<Weight, Integer_Type>::del_csc()
{
    #ifdef HAS_WEIGHT
    delete VAL;
    #endif
    delete ROW_INDEX;
    delete COL_PTR;
}

#endif
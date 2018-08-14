/*
 * compressed_storage.hpp: Compressed storage implementaion
 * Compressed Sparse Row (CSR)
 * Compressed Sparse Column (CSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

enum Compression_type
{
  _CSR_,
  _CSC_
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
};

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_)
{
    nnz = nnz_;
    nrows_plus_one = nrows_plus_one_;
    A = new struct Basic_Storage<Weight, Integer_Type>(nnz);
    
    IA = new struct Basic_Storage<Integer_Type, Integer_Type>(nrows_plus_one);
    
    JA = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz);
}

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::~CSR()
{
    delete A;
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
};

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_)
{
    nnz = nnz_;
    ncols_plus_one = ncols_plus_one_;
    VAL = new struct Basic_Storage<Weight, Integer_Type>(nnz);
    
    ROW_INDEX = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz);
    
    COL_PTR = new struct Basic_Storage<Integer_Type, Integer_Type>(ncols_plus_one);
}

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::~CSC()
{
    delete VAL;
    delete ROW_INDEX;
    delete COL_PTR;
}

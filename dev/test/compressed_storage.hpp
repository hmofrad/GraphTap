/*
 * csr.hpp: csr implementaion
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include "storage.hpp"
 
template<typename Weight, typename Integer_Type>
struct CSR
{
    CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_);
    ~CSR();
    Integer_Type nnz;
    Integer_Type nrows_plus_one;
    
    struct basic_storage<Weight, Integer_Type> *A;
    struct basic_storage<Integer_Type, Integer_Type> *IA;
    struct basic_storage<Integer_Type, Integer_Type> *JA;
};

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::CSR(Integer_Type nnz_, Integer_Type nrows_plus_one_)
{
    nnz = nnz_;
    nrows_plus_one = nrows_plus_one_;
    A = new struct basic_storage<Weight, Integer_Type>(nnz);
    
    IA = new struct basic_storage<Integer_Type, Integer_Type>(nrows_plus_one);
    
    JA = new struct basic_storage<Integer_Type, Integer_Type>(nnz);
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
    
    struct basic_storage<Weight, Integer_Type> *VAL;
    struct basic_storage<Integer_Type, Integer_Type> *ROW_INDEX;
    struct basic_storage<Integer_Type, Integer_Type> *COL_PTR;
};

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_)
{
    nnz = nnz_;
    ncols_plus_one = ncols_plus_one_;
    VAL = new struct basic_storage<Weight, Integer_Type>(nnz);
    
    ROW_INDEX = new struct basic_storage<Integer_Type, Integer_Type>(nnz);
    
    COL_PTR = new struct basic_storage<Integer_Type, Integer_Type>(ncols_plus_one);
}

template<typename Weight, typename Integer_Type>
CSC<Weight, Integer_Type>::~CSC()
{
    delete VAL;
    delete ROW_INDEX;
    delete COL_PTR;
}

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
    A = new struct basic_storage<Weight, Integer_Type>(nnz_);
    
    IA = new struct basic_storage<Integer_Type, Integer_Type>(nrows_plus_one_);
    
    JA = new struct basic_storage<Integer_Type, Integer_Type>(nnz_);
    
    //A_n = nnz;
    //A_nbytes 
        
}

template<typename Weight, typename Integer_Type>
CSR<Weight, Integer_Type>::~CSR()
{
    delete A;
    delete IA;
    delete JA;
}
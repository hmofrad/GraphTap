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
  _CSC_, // Compressed Sparse Col
 _DCSC_, // Compressed Sparse Col
 _TCSC_, // Compressed Sparse Col
};

template<typename Weight, typename Integer_Type>
struct Compressed_column {
    public:
        //Compressed_column() {}
        virtual ~Compressed_column() {}
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, Integer_Type tile_height, Integer_Type tile_width){};
};


template<typename Weight, typename Integer_Type>
struct CSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        CSC_BASE(uint64_t nnz_, Integer_Type ncols_);
        ~CSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, Integer_Type tile_height, Integer_Type tile_width);
        uint64_t nnz;
        Integer_Type ncols;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
};

template<typename Weight, typename Integer_Type>
CSC_BASE<Weight, Integer_Type>::CSC_BASE(uint64_t nnz_, Integer_Type ncols_) {
    nnz = nnz_;
    ncols = ncols_;
    #ifdef HAS_WEIGHT
    if((A = (Weight*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(Integer_Type));
    #endif
    
    if((IA = (Integer_Type*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(Integer_Type));
    
    if((JA = (Integer_Type*) mmap(nullptr, (ncols + 1) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, (ncols + 1) * sizeof(Integer_Type));
}

template<typename Weight, typename Integer_Type>
CSC_BASE<Weight, Integer_Type>::~CSC_BASE() {
    #ifdef HAS_WEIGHT
    if(munmap(A, nnz * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    #endif
    
    if(munmap(IA, nnz * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, (ncols + 1) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}

template<typename Weight, typename Integer_Type>
void CSC_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, Integer_Type tile_height, Integer_Type tile_width) {
    struct Triple<Weight, Integer_Type> pair;
    uint32_t i = 0; // Row Index
    uint32_t j = 1; // Col index
    JA[0] = 0;
    for (auto& triple : *triples)
    {
        pair  = {(triple.row % tile_height), (triple.col % tile_width)};
        while((j - 1) != pair.col)
        {
            j++;
            JA[j] = JA[j - 1];
        }            
        #ifdef HAS_WEIGHT
        A[i] = triple.weight;
        #endif
        
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (ncols + 1))
    {
        j++;
        JA[j] = JA[j - 1];
    }
}


template<typename Weight, typename Integer_Type>
struct CSC
{
    CSC(Integer_Type nnz_, Integer_Type ncols_plus_one_);
    ~CSC();
    Integer_Type nnz;
    Integer_Type ncols_plus_one;
    
    void* A;  // WEIGHT
    void* IA; // ROW_IDX
    void* JA; // COL_PTR
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
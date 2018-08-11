/*
 * ds.hpp: Basic data structures implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <stdint.h>
#include <sys/mman.h>
#include <cstring>
#include <vector>


/*
 * Triple data structure for storing edges
 */
template <typename Weight, typename Integer_Type = uint32_t>
struct Triple
{
    Integer_Type row;
    Integer_Type col;
    Weight weight;
    Triple(Integer_Type row = 0, Integer_Type col = 0, Weight weight = 0);
    ~Triple();
    void set_weight(Weight &w);
    Weight get_weight();
};

template <typename Weight, typename Integer_Type>
Triple<Weight, Integer_Type>::Triple(Integer_Type row, Integer_Type col, Weight weight)
      : row(row), col(col), weight(weight) {};

template <typename Weight, typename Integer_Type>
Triple<Weight, Integer_Type>::~Triple() {};
      
template <typename Weight, typename Integer_Type>
void Triple<Weight, Integer_Type>::set_weight(Weight &w)
{
    weight = w;
}

template <typename Weight, typename Integer_Type>
Weight Triple<Weight, Integer_Type>::get_weight()
{
    return(weight);
}

#include <type_traits>
struct Empty {};
//typedef typename Integer_Type IT;

//(std::is_same<Weight, Empty>::value)

/* Triple that supports empty weights */
/*
template <>
struct Triple <Empty>
{
    uint32_t row;
    union {
        uint32_t col;
        Empty weight;
    };
    void set_weight(Empty& w) {};
    bool get_weight() {return(1);};
};
*/
template <typename Integer_Type>
struct Triple <Empty, Integer_Type>
{
    Integer_Type row;
    union {
        Integer_Type col;
        Empty weight;
    };
    void set_weight(Empty& w) {};
    bool get_weight() {return(true);};
};

/*
 * Functor for passing to std::sort
 * It sorts the Triples using their row index
 * and then their column index
 */
template <typename Weight, typename Integer_Type>
struct Functor
{
    bool operator()(const struct Triple<Weight, Integer_Type> &a, const struct Triple<Weight, Integer_Type> &b)
    {
        return((a.row == b.row) ? (a.col < b.col) : (a.row < b.row));
    }
};



/*
 * Basic storage with mmap
 * We use mmap as it returns a contiguous piece of memory
 */
template <typename Weight = void, typename Integer_Type = uint32_t>
struct basic_storage
{
    basic_storage(Integer_Type n_);
    ~basic_storage();
    Integer_Type n;
    uint64_t nbytes;
    void *data;
    //void allocate(Integer_Type n_);
    //void free();
};

template <typename Weight, typename Integer_Type>
basic_storage<Weight, Integer_Type>::basic_storage(Integer_Type n_)
{
    n = n_;
    nbytes = n_ * sizeof(Weight);
    assert(nbytes == (n * sizeof(Weight)));
    if((data = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(data, 0, nbytes);
}

template <typename Weight, typename Integer_Type>
basic_storage<Weight, Integer_Type>::~basic_storage()
{
    if(munmap(data, nbytes) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}

template <>
struct basic_storage <Empty>
{
    basic_storage(uint64_t n_);
    ~basic_storage();
    uint64_t n;
    uint64_t nbytes;
    void *data;
    /*
    void allocate(uint64_t n_)
    {
        n = n_;
        nbytes = n_;
        assert(nbytes == (n * sizeof(Empty)));
        if((data = mmap(nullptr, nbytes, PROT_READ | 
                                         PROT_WRITE, MAP_ANONYMOUS |
                                         MAP_PRIVATE, -1, 0)) == (void*) -1)
        {    
            fprintf(stderr, "Error mapping memory\n");
            Env::exit(1);
        }
        memset(data, 0, nbytes);
    };
    void free()
    {
        if(munmap(data, nbytes) == -1)
        {
            fprintf(stderr, "Error unmapping memory\n");
            Env::exit(1);
        }
    };
    */
};

//template <>
basic_storage<Empty>::basic_storage(uint64_t n_)
{
    assert(n_ > 0);
    n = n_;
    nbytes = n_;
    assert(nbytes == (n * sizeof(Empty)));
    if((data = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS
                                               | MAP_PRIVATE, -1, 0)) == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        Env::exit(1);
    }
    memset(data, 0, nbytes);
}

//template <>
basic_storage<Empty>::~basic_storage()
{
    if(munmap(data, nbytes) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}


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
    
    Integer_Type A_n;
    uint64_t A_nbytes;
    Weight* A_data;
    
    Integer_Type* IA_n;
    uint64_t IA_nbytes;
    Integer_Type* IA_data;
    
    Integer_Type* JA_n;
    uint64_t JA_nbytes;
    Integer_Type* JA_data;
    
    void allocate(Integer_Type nnz_, Integer_Type nrows_plus_one_);
    void free();
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

/*
template<typename Weight, typename Integer_Type>
void CSR<Weight, Integer_Type>::allocate(uint32_t nnz_, uint32_t nrows_plus_one_)
{        
    nnz = nnz_;
    nrows_plus_one = nrows_plus_one_;
    A = mmap(nullptr, (nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    //(void *) -1)
    //{
      //  fprintf(stderr, "Error mapping memory\n");
        //exit(1);
    //}
    memset(CSR<Weight>::A, 0, CSR<Weight>::nnz * sizeof(uint32_t));
    CSR<Weight>::IA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nrows_plus_one) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(CSR<Weight>::IA, 0, CSR<Weight>::nrows_plus_one * sizeof(uint32_t));
    CSR<Weight>::JA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(CSR<Weight>::JA, 0, CSR<Weight>::nnz * sizeof(uint32_t));
}    
    

template<typename Weight>
void CSR<Weight>::free()
{
    munmap(CSR<Weight>::A, (this->nnz) * sizeof(uint32_t));
    //{
    //if(munmap(address, size) == -1) {
    //    fprintf(stderr, "Error unmapping memory\n");
      //  exit(1);
    //}

    munmap(CSR<Weight>::IA, (this->nrows_plus_one) * sizeof(uint32_t));
    munmap(CSR<Weight>::JA, (this->nnz) * sizeof(uint32_t));    
}
*/







/*
 * Index vector to code the sparsity of vectors. The indices will later
 * be used to traverse the received segments from other ranks. This requires
 * significant more storage compared to having a bit vector but we can iterate
 * over the received segment faster. 
 * Another way of implementing this indexing process is to use bit vector,
 * where bit vector is used to identify the non-zero elements and then use this
 * information to index the received regment. Bit vector has better storage,
 * but index vector has better performance.
 */
template <typename Type>
struct IV : basic_storage<Type> {};












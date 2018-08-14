/*
 * storage.hpp: Basic storage with mmap
 * We use mmap as it returns a contiguous piece of memory
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include <sys/mman.h>
#include <cstring>

template <typename Weight = void, typename Integer_Type = uint32_t>
struct Basic_Storage
{
    Basic_Storage(Integer_Type n_);
    ~Basic_Storage();
    Integer_Type n;
    uint64_t nbytes;
    void *data;
    //void allocate(Integer_Type n_);
    //void free();
};

template <typename Weight, typename Integer_Type>
Basic_Storage<Weight, Integer_Type>::Basic_Storage(Integer_Type n_)
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
Basic_Storage<Weight, Integer_Type>::~Basic_Storage()
{
    if(munmap(data, nbytes) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}

template <typename Integer_Type>
struct Basic_Storage <Empty, Integer_Type>
{
    Basic_Storage(Integer_Type n_);
    ~Basic_Storage();
    Integer_Type n;
    Integer_Type nbytes;
    void *data;
};

template <typename Integer_Type>
Basic_Storage<Empty, Integer_Type>::Basic_Storage(Integer_Type n_)
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

template <typename Integer_Type>
Basic_Storage<Empty, Integer_Type>::~Basic_Storage()
{
    if(munmap(data, nbytes) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        Env::exit(1);
    }
}

/*
 * vector.hpp: Vector and segment implementations
 * A vector is consists of multiple segments.
 * The number of segments is equal to the number
 * of column gorups.
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

//#include "basic_storage.hpp"

template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
struct Segment
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vector;
    
    //template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    //friend class Vertex_Program;
    Segment();
    Segment(Integer_Type n_, Integer_Type g_);
    ~Segment();

    void allocate(Integer_Type n_, Integer_Type g_);
    void del_seg();
    
    //struct Basic_Storage<Fractional_Type, Integer_Type> *D;
    void *D;
    Integer_Type n;
    uint64_t nbytes;
    bool allocated;

    uint32_t g;
    
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(Integer_Type n_, Integer_Type g_)
{
    n = n_;
    nbytes = n_ * sizeof(Fractional_Type);
    g = g_;
    /*
    if(n)
    {
        D = malloc(nbytes);
        allocated = true;
    }
    else
        allocated = true;
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(){};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment()
{
    //delete D;
    
    //printf("Segment %d %d %d\n", Env::rank, g, n);
    //if(n)
        //free(D);
    //delete[] (Fractional_Type *) D;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::allocate(Integer_Type n_, Integer_Type g_)
{
    n = n_;
    g = g_;
    D = nullptr;
    nbytes = n * sizeof(Fractional_Type);
    if(n)
    {
        //D = malloc(nbytes);
        D = new Fractional_Type[n];
        memset(D, 0, nbytes);
        allocated = true;
    }
    else
        allocated = false;
    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::del_seg()
{
    //D->del_storage();
    //delete D;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    public:
        Vector();
        //Vector(Integer_Type nelems_, std::vector<int32_t> &local_segments_);
        Vector(std::vector<Integer_Type> &nitems_, std::vector<int32_t> &local_segments_);
        ~Vector();
        void del_vec();
        
        //std::vector<struct Segment<Weight, Integer_Type, Fractional_Type>> segments;
        std::vector<int32_t> local_segments;
        std::vector<Integer_Type> nitems;
        //std::vector<void *> data;
        Fractional_Type **data;
        
        std::vector<bool> allocated;
        uint32_t vector_length;
        //Integer_Type nelems;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector() {};


template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector()
{
    for(uint32_t i = 0; i < vector_length; i++)
    {
        if(allocated[i])
        {
            
            free(data[i]);
            //free(data1[i]);
            //delete[] (Fractional_Type *) data[i];
            data[i] =nullptr;
        }
    }
    free(data);
    
    
    //data.clear();
    //data.shrink_to_fit();
}
    
    //printf("~vector %d\n", Env::rank);




/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nelems_, std::vector<int32_t> &local_segments_)
{
    nelems = nelems_;
    local_segments = local_segments_;
    vector_length = local_segments.size();
    // Reserve the 1D vector of segments. 
    segments.resize(vector_length);
    for(uint32_t i = 0; i < vector_length; i++)
    {
        segments[i].allocate(nelems);
        segments[i].allocated = true;
        segments[i].g = local_segments[i];
        #ifdef PREFETCH
        madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
        #endif
    }
}
*/

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(std::vector<Integer_Type> &nitems_, std::vector<int32_t> &local_segments_)
{
    //printf("Vector rank=%d nelems=%lu ls=%lu\n", Env::rank, nelems_.size(), local_segments_.size());
    assert(nitems_.size() == local_segments_.size());
    
    nitems = nitems_;
    local_segments = local_segments_;
    vector_length = local_segments.size();
    // Reserve the 1D vector of segments. 
    //segments.resize(vector_length);
    //data.resize(vector_length);
    allocated.resize(vector_length);
    data = (Fractional_Type **) malloc(vector_length * sizeof(Fractional_Type *));
    for(uint32_t i = 0; i < vector_length; i++)
    {
        //nitems.push_back(nitems_[i]);
        uint64_t nbytes = nitems[i] * sizeof(Fractional_Type);
        if(nbytes)
        {
            //void *D = (Fractional_Type *) malloc(nbytes);
            //void *D = new Fractional_Type[nitems[i]];
            //memset(D, 0, nbytes);
            //data.push_back(D);
            //allocated.push_back(true);    
            //data[i] = (Fractional_Type *) malloc(nbytes);
            //memset(data[i], 0, nbytes);
            
            data[i] = (Fractional_Type *) malloc(nbytes);
            memset(data[i], 0, nbytes);
            
            allocated[i] = true;
        }
        else
        {
            data[i] = nullptr;
            allocated[i] = false;
            //allocated.push_back(false);
        }
        
        
        
        
        //Segment<Weight, Integer_Type, Fractional_Type> segment(nelems_[i], local_segments[i]);
        //segment.g = local_segments[i];
        //segments[i].allocate(nitems_[i], local_segments[i]);
       
       // void *D = malloc(nbytes);
        //data.push_back(
        //segments.push_back(segment);
        //segments[i].g = local_segments[i];
        
       // segments[i] = &segment;
        /*
        if(nelems_[i])
        {
            //if(!Env::rank)
            //    printf(">>>%d\n", nelems_[i]);   
            segments[i].allocated = true;
            segments[i].allocate(nelems_[i]);
        }
        else
        {
            segments[i].allocated = false;
            segments[i].allocate(nelems_[i]);
        }
        */
        //segments[i].g = local_segments[i];
        
        //#ifdef PREFETCH
        //madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
        //#endif
    }
}

    
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vector<Weight, Integer_Type, Fractional_Type>::del_vec()
{
    for(uint32_t i = 0; i < vector_length; i++)
    {
        //if(segments[i].allocated)
        //{
            //segments[i].del_seg();
        //}
    }
}
#endif
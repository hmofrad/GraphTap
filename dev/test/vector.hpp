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

template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
struct Segment
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vector;
    
    //template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    //friend class Vertex_Program;
    
    Segment();
    ~Segment();

    void allocate(Integer_Type n);
    void del_seg();
    
    struct Basic_Storage<Fractional_Type, Integer_Type> *D;

    uint32_t g, cg, rg;
    bool allocated;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::allocate(Integer_Type n)
{
    D = new struct Basic_Storage<Fractional_Type, Integer_Type>(n);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::del_seg()
{
    delete D;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    public:
        Vector();
        Vector(Integer_Type nelems_, std::vector<int32_t> &local_segments_);
        Vector(std::vector<Integer_Type> &nelems_, std::vector<int32_t> &local_segments_);
        ~Vector();
        void del_vec();
        
        std::vector<struct Segment<Weight, Integer_Type, Fractional_Type>> segments;
        std::vector<int32_t> local_segments;
        uint32_t vector_length;
        Integer_Type nelems;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector() {};

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

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(std::vector<Integer_Type> &nelems_, std::vector<int32_t> &local_segments_)
{
    //printf("Vector rank=%d nelems=%lu ls=%lu\n", Env::rank, nelems_.size(), local_segments_.size());
    assert(nelems_.size() == local_segments_.size());
    
    nelems = -1;
    local_segments = local_segments_;
    vector_length = local_segments.size();
    // Reserve the 1D vector of segments. 
    segments.resize(vector_length);
    
    for(uint32_t i = 0; i < vector_length; i++)
    {
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
        
        segments[i].g = local_segments[i];
        #ifdef PREFETCH
        madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
        #endif
    }
}

    
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vector<Weight, Integer_Type, Fractional_Type>::del_vec()
{
    for(uint32_t i = 0; i < vector_length; i++)
    {
        //if(segments[i].allocated)
        //{
            segments[i].del_seg();
        //}
    }
}
#endif
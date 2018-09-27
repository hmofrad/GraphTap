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

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Matrix;
    
    template<typename Weight___, typename Integer_Type___, typename Fractional_Type___>
    friend class Vertex_Program;
    
    public:
        Vector();
        Vector(std::vector<Integer_Type> &nitems_, std::vector<int32_t> &local_segments_);
        ~Vector();
        std::vector<int32_t> local_segments;
        std::vector<Integer_Type> nitems;
        Fractional_Type **data;
        std::vector<bool> allocated;
        uint32_t vector_length;
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
            data[i] = nullptr;
        }
    }
    free(data);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(std::vector<Integer_Type> &nitems_, std::vector<int32_t> &local_segments_)
{
    assert(nitems_.size() == local_segments_.size());    
    nitems = nitems_;
    local_segments = local_segments_;
    vector_length = local_segments.size();
    allocated.resize(vector_length);
    data = (Fractional_Type **) malloc(vector_length * sizeof(Fractional_Type *));
    for(uint32_t i = 0; i < vector_length; i++)
    {
        uint64_t nbytes = nitems[i] * sizeof(Fractional_Type);
        if(nbytes)
        {
            data[i] = (Fractional_Type *) malloc(nbytes);
            memset(data[i], 0, nbytes);            
            allocated[i] = true;
        }
        else
        {
            data[i] = nullptr;
            allocated[i] = false;
        }
    }
}
#endif
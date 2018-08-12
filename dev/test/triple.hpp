/*
 * triple.hpp: Triple implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

//#include <iostream>
//#include <stdint.h>

//#include <cstring>


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

struct Empty {};
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
 * Index vector to code the sparsity of vectors. The indices will later
 * be used to traverse the received segments from other ranks. This requires
 * significant more storage compared to having a bit vector but we can iterate
 * over the received segment faster. 
 * Another way of implementing this indexing process is to use bit vector,
 * where bit vector is used to identify the non-zero elements and then use this
 * information to index the received regment. Bit vector has better storage,
 * but index vector has better performance.
 */
//template <typename Type>
//struct IV : basic_storage<Type> {};












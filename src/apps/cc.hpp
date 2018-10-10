/*
 * cc.hpp: Triangle counting benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CC_HPP
#define CC_HPP 

#include "vp/vertex_state.hpp"
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class CC_state : public Vertex_State<Weight, Integer_Type, Fractional_Type>
{
    public:        
        bool init_func(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }
        
        Fractional_Type message_func(Fractional_Type &v, Fractional_Type &s) 
        {
            return(v);
        }

        void add(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        void combine_func(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 = (y1 < y2) ? y1 : y2;
        }
        
        bool apply_func(Fractional_Type &y, Fractional_Type &v) 
        {
            Fractional_Type t = v;
            v = (y < v) ? y : v;
            return(t != v);
        }      
};

#endif
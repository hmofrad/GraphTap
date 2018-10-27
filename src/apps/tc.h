/*
 * tc.h: Triangle counting benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TC_H
#define TC_H 

#include "vp/vertex_program.hpp"

/* HAS_WEIGHT macro will be defined by compiler. 
   make MACROS=-DHAS_WEIGHT */   
using em = Empty; // Weight (default is Empty)
#ifdef HAS_WEIGHT
using wp = uint32_t;
#else
using wp = em;
#endif

/*  Integer precision controls the number of vertices
    the engine can possibly process. */
using ip = uint32_t;

/* Fractional precision controls the precision of values.
   E.g. vertex rank in PageRank. */
using fp = double;

struct TC_State
{
    ip vid;
    ip get_state(){return(vid);};
    std::vector<ip> neighbors;
    std::vector<ip> get_neighbors(){return(neighbors);};
    ip get_inf(){return(0);};
    std::string print_state()
    {
        std::string str = "neighbors=";
        for(auto vid : neighbors)
        {
            str += std::to_string(vid) + ", ";
        }
        return(str);
    };
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class TC_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, TC_State>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, TC_State>::Vertex_Program;
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        {
            return(true);
        }      
};
#endif
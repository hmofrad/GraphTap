/*
 * deg.h: Degree benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DEG_H
#define DEG_H 

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
   E.g. vertex rank in PageRank*/
using fp = uint32_t;

#include "vp/vertex_program.hpp"

struct Degree_State
{
    //Degree_State(){};
    //~Degree_State(){};
    ip degree = 0;
    ip get_state(){return degree;};
    std::string print_state(){return("Degree=" + std::to_string(degree));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Degree_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, Degree_State>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, Degree_State>::Vertex_Program;  // inherit constructors
        
        //virtual bool initializer(Degree_State &s, const Fractional_Type &v2)
        //{
        //    s.degree = 0;
        //    return(true);
        //}
        /*
        virtual bool initializer(Fractional_Type &v1, const Fractional_Type &v2)
        {
            
            v1 = v2;
            return(true);
        }
        */
        virtual Fractional_Type messenger(Degree_State &s) 
        {
            return(1);
        }
        
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        virtual bool applicator(Degree_State &s, const Fractional_Type &y) 
        {
            s.degree = y;
            return(true);
        }    
};
#endif
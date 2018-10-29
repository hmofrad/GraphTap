/* 
 * bfs.h: Breadth First Search (BFS) benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef SSSP_H
#define SSSP_H

#include "vp/vertex_program.hpp"

#define INF 2147483647

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
using fp = uint32_t;

struct SSSP_State
{
    ip distance = INF;
    //ip hops = INF;
    //ip vid = 0;
    ip get_state(){return(distance);};
    ip get_inf(){return(INF);};    
    std::string print_state(){return("Distance=" + std::to_string(distance));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class SSSP_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, SSSP_State>
{
    public:  
        Integer_Type root = 0;
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, SSSP_State>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Integer_Type vid, SSSP_State &state) 
        {
            if(vid == root)
            {
                //state.vid = vid;
                state.distance = 0;
                //printf("%d\n", vid);
                return(true);
            }
            else
            {
                //state.vid = vid;
                state.distance = INF; // Not necessary
                return(false);
            }
            
            
        }
        
        virtual Fractional_Type messenger(SSSP_State &state) 
        //virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            return(state.distance);
        }

        //#ifdef HAS_WEIGHT
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2, const Fractional_Type &w) 
        {
            Fractional_Type tmp = y2 + w;
            y1 = (y1 < tmp) ? y1 : tmp;
        }
        //#else
            
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) 
        {
            /*
            if(y2 != INF)
            {
                if(y2 < y1)     
                    y1 = y2;
            }
            */
            y1 = (y1 < y2) ? y1 : y2;
            //printf("%d %d \n", y1 ,y2);
        }
        
        //#endif
        
        //virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        virtual bool applicator(SSSP_State &state, const Fractional_Type &y)
        {
            Fractional_Type tmp = state.distance;
            #ifdef HAS_WEIGHT
            state.distance = (y < state.distance) ? y : state.distance;
            #else
            state.distance = (y < state.distance) ? y + 1 : state.distance;
            #endif
            //state.distance = (y < state.distance) ? y : state.distance;
            return(tmp != state.distance);
            
            //#ifdef HAS_WEIGHT
            /*
            Fractional_Type t = v;
            v = (y < v) ? y : v;
            return(t != v); 
            */
            
            //#else
              /*
            if(state.distance != INF)
                return(false); // already visited
            else
            {
                if(y != INF)
                {
                    
                    #ifdef HAS_WEIGHT
                    state.distance = (y < state.distance) ? y : state.distance;
                    #else
                    state.distance = (y < state.distance) ? y + 1 : state.distance;
                    //v = (y < v) ? y + 1: v;    
                    //v = y + 1;
                    #endif    
                    
                    return(true);
                }
                else
                    return(false);
            }
            */
            /*
            
            if(v != INF)
                return(false); // already visited
            else
            {
                if(y != INF)
                {
                    #ifdef HAS_WEIGHT
                    v = (y < v) ? y : v;
                    #else
                    v = (y < v) ? y + 1: v;    
                    //v = y + 1;
                    #endif    
                    return(true);
                }
                else
                    return(false);
            }
            
                A tmp = s.distance;
    s.distance = std::min(s.distance, y);
    return tmp != s.distance;
            */
            //#endif
            
        }      
};
#endif
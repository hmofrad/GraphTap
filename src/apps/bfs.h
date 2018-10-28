/* 
 * bfs.h: Breadth First Search (BFS) benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BFS_H
#define BFS_H

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

struct BFS_State
{
    ip parent = 0;
    ip hops = INF;
    ip vid = 0;
    ip get_state(){return(hops);};
    ip get_inf(){return(INF);};    
    std::string print_state(){return("Parent=" + std::to_string(parent) + ",Hops=" + std::to_string(hops));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class BFS_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, BFS_State>
{
    public:  
        Integer_Type root = 0;
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, BFS_State>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Integer_Type vid, BFS_State &state) 
        {
            if(vid == root)
            {
                state.vid = vid;
                state.hops = 0;
                return(true);
            }
            else
            {
                state.vid = vid;
                state.hops = INF; // Not necessary
                return(false);
            }
        }
        
        virtual Fractional_Type messenger(BFS_State &state) 
        {
            return(state.vid);
        }

        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) 
        {
            if(y2 != INF)
            {
                //if(y2 > y1 or y1 == INF)
                if(y2 < y1)     
                    y1 = y2;
            }
            ///else
                
            
            //printf("%d %d\n", y1, y2);
        }
        
        virtual bool applicator(BFS_State &state, const Fractional_Type &y, Integer_Type iteration)
        {
            
            if(state.hops != INF)
                return(false); // already visited
            else
            {
                if(y != INF)
                {
                    state.hops = iteration + 1;
                //v = y;
                    state.parent = y;
                    return(true);
                }
                else
                    return(false);
            }
            
            //if (s.hops != INF)
            //return false;  // already visited
            //s.hops = (dist_t) (iter + 1);
            //s.parent = y;
            //return true;
            
        }      
};
#endif
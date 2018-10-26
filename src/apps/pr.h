/*
 * pr.h: PageRank benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef PR_H
#define PR_H 

#include "deg.h"

fp tol = 1e-5;
fp alpha = 0.15;


struct PageRank_State : Degree_State
{
    fp rank = alpha;
    ip get_state(){return(rank);};
    std::string print_state(){return("Rank=" + std::to_string(rank) + ",Degree=" + std::to_string(degree));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class PageRank_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, PageRank_State>
//class PageRank_Program : public Vertex_Program<wp, ip, fp, PageRank_State>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, PageRank_State>::Vertex_Program;  // inherit constructors
        //using Vertex_Program<wp, ip, fp, PageRank_State>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Integer_Type vid, PageRank_State &state, const State &other)
        //virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            state.degree = ((const Degree_State&) other).degree;
            state.rank = alpha; //  Not necessary
            //printf("%d %d %f\n", vid, state.degree, state.rank);
            return(true);
        }

        virtual Fractional_Type messenger(PageRank_State &state) 
        {
            return( (state.degree) ? (state.rank / state.degree) : 0 );
        }

        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        //virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        virtual bool applicator(PageRank_State &state, const Fractional_Type &y) 
        {
            Fractional_Type tmp = state.rank;
            state.rank = alpha + (1.0 - alpha) * y;
            return (fabs(state.rank - tmp) > tol);         
        }
};

#endif
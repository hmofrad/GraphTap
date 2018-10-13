/*
 * pr.h: PageRank benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef PR_H
#define PR_H 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class PageRank_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program;  // inherit constructors
        Fractional_Type tol = 1e-5;
        Fractional_Type alpha = 0.15;
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = alpha;
            return(true);
        }

        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            if(v and s)
            {
                return(v / s);
            }
            else
            {
                return(0);
            }
        }
        
        virtual void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        {
            Fractional_Type tmp = v;
            v = alpha + (1.0 - alpha) * y;
            return (fabs(v - tmp) < tol);            
        }
};
#endif
/*
 * pr.h: PageRank benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef PR_H
#define PR_H 

#include <cmath>

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class PR_state
{
    public: 
        PR_state(){};
        ~PR_state(){};   
        Fractional_Type tol = 1e-5;
        Fractional_Type alpha = 0.15;
        bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }

        Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            return(1);
        }
        
        Fractional_Type messenger1(Fractional_Type &v, Fractional_Type &s) 
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
        
        void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        bool applicator(Fractional_Type &v, Fractional_Type &y) 
        {
            v = y;
            return(true);
        }  

        bool applicator1(Fractional_Type &v, Fractional_Type &y) 
        {
            Fractional_Type tmp = v;
            v = alpha + (1.0 - alpha) * y;
            return (fabs(v - tmp) > tol);  
        }
};
#endif
/*
 * deg.h: Degree benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DEG_H
#define DEG_H 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class DEG_state
{
    public: 
        DEG_state(){};
        ~DEG_state(){};   
        
        bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }

        Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            return(1);
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
};
#endif
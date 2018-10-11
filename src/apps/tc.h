/*
 * tc.h: Triangle counting benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TC_H
#define TC_H 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class TC_state
{
    public: 
        TC_state(){};
        ~TC_state(){};    
        bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }

        void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            ;
        }
        
        bool applicator(Fractional_Type &y, Fractional_Type &v) 
        {
            return(true);
        }      
};
#endif
/*
 * tc.h: Triangle counting benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TC_H
#define TC_H 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class TC_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            v1 = v2;
            return(true);
        }
        virtual bool applicator(Fractional_Type &y, Fractional_Type &v) 
        {
            return(true);
        }      
};
#endif
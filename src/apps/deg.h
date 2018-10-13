/*
 * deg.h: Degree benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DEG_H
#define DEG_H 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Degree_Program : public Vertex_Program <Weight, Integer_Type, Fractional_Type>
{
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2)
        {
            v1 = v2;
            return(true);
        }
        
        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            return(1);
        }
        
        virtual void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 += y2;
        }
        
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        {
            v = y;
            return(true);
        }    
};
#endif
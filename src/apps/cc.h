/*
 * cc.h: Connected component benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CC_H
#define CC_H

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class CC_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type>
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
            return(v);
        }

        virtual void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            y1 = (y1 < y2) ? y1 : y2;
        }
        
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y) 
        {
            Fractional_Type t = v;
            v = (y < v) ? y : v;
            return(t != v);
        }      
};
#endif
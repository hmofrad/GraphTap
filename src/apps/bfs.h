/* 
 * bfs.h: Breadth First Search (BFS) benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CC_H
#define CC_H

//#define INF 2147483647

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class BFS_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type>
{
    public:  
        Integer_Type root = 0;
        using Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) 
        {
            if(v2 == root)
            {
                v1 = 0;
                return(true);
            }
            else
            {
                v1 = INF;
                return(false);
            }
        }
        
        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) 
        {
            return(v);
        }

        virtual void combiner(Fractional_Type &y1, Fractional_Type &y2) 
        {
            if(y2 != INF)
                y1 = y2;
        }
        
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y, Integer_Type iteration) 
        {
            
            if(v != INF)
                return(false); // already visited
            else
            {
                if(y != INF)
                {
                    v = iteration + 1;
                //v = y;
                //parent = y;
                    return(true);
                }
                else
                    return(false);
            }
        }      
};
#endif
/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "vector.hpp"
 
template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
class Vertex_Program
{
    public:
        //Vertex_Program();
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph);
        ~Vertex_Program();
        void test();
    
    private:
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        Vector<Weight, Integer_Type, Fractional_Type>* X;
        Vector<Weight, Integer_Type, Fractional_Type>* Y;
        Vector<Weight, Integer_Type, Fractional_Type>* V;
        Vector<Weight, Integer_Type, Fractional_Type>* S;
};
/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program ()
                : X(nullptr), Y(nullptr), V(nullptr), S(nullptr) {};
                */
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph)
{
    A = Graph.A;
    
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
              std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
             
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->local_col_segments);
              
    Y = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
   
   
    V = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
                   
    S = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
             
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    delete X;
    delete Y;
    delete V;
    delete S;   
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::test()
{
    if(!Env::rank)
    printf("Testing this\n");

uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
              std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
//Y = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
  //                  A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
}


 
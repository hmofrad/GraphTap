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
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s);
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
    
    private:
        
        
        
        
        void apply();
        void populate(Fractional_Type value, Segment<Weight, Integer_Type, Fractional_Type> &segment);
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
    
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

    /* The values inside vector's segments are initialized as zero
       because the basic storage would memset the segments for us. */
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(Fractional_Type value,
            Segment<Weight, Integer_Type, Fractional_Type> &segment)
{
    auto *data = (Fractional_Type *) segment.D->data;
    uint32_t nitems = segment.D->n;
    uint32_t nbytes = segment.D->nbytes;
    if(value)
    {
        for(uint32_t i = 0; i < nitems; i++)
        {
            data[i] = value;
        }
    } 
    else
    {
        memset(data, 0, nbytes);
    }
}                

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, 
                           Fractional_Type y, Fractional_Type v, Fractional_Type s)
{
    
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
        /*
        uint32_t x_nitems = x_seg.D->n;
        uint32_t x_nbytes = x_seg.D->nbytes;
        
        if(x)
        {
            for(uint32_t i = 0; i < x_nitems; i++)
            {
                x_data[i] = x;
            }
        } 
        else
        {
            memset(x_data, 0, x_nbytes);
        }
        */
    }    

    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
        
        //auto *y_data = (Fractional_Type *) y_seg.D->data;
        
        /*
        uint32_t y_nitems = y_seg.D->n;
        uint32_t y_nbytes = y_seg.D->nbytes;
        
        if(v)
        {
            for(uint32_t i = 0; i < y_nitems; i++)
            {
                y_data[i] = y;
            }
        } 
        else
        {
            memset(y_data, 0, y_nbytes);
        }
        */
    }    
    
    
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    populate(v, v_seg);
    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    populate(s, s_seg);
    
    /*
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    uint32_t v_nitems = v_seg.D->n;
    uint32_t v_nbytes = v_seg.D->nbytes;
    
    if(v)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            v_data[i] = v;
        }
    } 
    else
    {
        memset(v_data, 0, v_nbytes);
    }
*/

    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{

    //(*f)();
    //uint32_t vi = V->owned_segment;
    //auto &v_seg = V->segments[vi];
        //auto *v_data = (Fractional_Type *) v_seg.D->data;
    //uint32_t v_nitems = v_seg.D->n;
    //uint32_t v_nbytes = v_seg.D->nbytes;
    
    
    uint32_t xi = X->owned_segment;
    auto &x_seg = X->segments[xi];
    auto *x_data = (Fractional_Type *) x_seg.D->data;
    uint32_t x_nitems = x_seg.D->n;
    uint32_t x_nbytes = x_seg.D->nbytes;
    
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    uint32_t v_nitems = v_seg.D->n;
    uint32_t v_nbytes = v_seg.D->nbytes;
    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    uint32_t s_nitems = s_seg.D->n;
    uint32_t s_nbytes = s_seg.D->nbytes;
    
    for(uint32_t i = 0; i < x_nitems; i++)
    {
        x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
    }
    
    
    
    
    MPI_Request request;
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_ROW))
    {
        uint32_t leader = x_seg.leader_rank;
        for(uint32_t j = 0; j < A->tiling->colgrp_nranks - 1; j++)
        {
            uint32_t other_rank = A->follower_colgrp_ranks[j];
            MPI_Isend(x_data, x_nbytes, MPI_BYTE, other_rank, x_seg.cg, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }
    }
    else if(A->tiling->tiling_type == Tiling_type::_1D_COL)
    {
        ;
    }
    else
    {
        fprintf(stderr, "Invalid tiling\n");
        Env::exit(1);
    }
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::gather()
{   
    uint32_t xi = X->owned_segment;
    
    MPI_Request request;
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_ROW))
    {    
        for(uint32_t s: A->local_col_segments)
        {
            if(s != xi)
            {

                auto& xj_seg = X->segments[s];
                auto *xj_data = (Fractional_Type *) xj_seg.D->data;
                //uint32_t xj_nitems = xj_seg.D->n;
                uint32_t xj_nbytes = xj_seg.D->nbytes;
                MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, xj_seg.leader_rank, xj_seg.cg, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    }
    else if(A->tiling->tiling_type == Tiling_type::_1D_COL)
    {
        ;
    }
    else
    {
        fprintf(stderr, "Invalid tiling\n");
        Env::exit(1);
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    Env::barrier();   
}

/*
template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::combine(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t xi, yi, yj, si, vi;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = X->segments[xi];
        auto *x_data = (Fractional_Type *) x_seg.D->data;
        uint32_t x_nitems = x_seg.D->n;
        uint32_t x_nbytes = x_seg.D->nbytes;
        
        yi = Y->owned_segment;
        auto &y_seg = Y->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        uint32_t y_nitems = y_seg.D->n;
        uint32_t y_nbytes = y_seg.D->nbytes;
    
        bool has_weight = (std::is_same<Weight, Empty>::value) ? false : true;
        
        // Local computation, no need to put it on X
        if(tile.allocated)
        {
            uint32_t k = 0;
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
            uint32_t nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
            uint32_t nnz_per_row;
            for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
            {
                nnz_per_row = IA[i + 1] - IA[i];
                for(uint32_t j = 0; j < nnz_per_row; j++)
                {
                    y_data[i] += x_data[JA[k]];
                    k++;
                }
            }
        }
    }
}
*/
//template<typename T,
//typename std::enable_if<!std::is_same<Weight,Empty>::value>::type* = nullptr>

//template<typename Weight, typename std::enable_if<!std::is_base_of<Empty, Weight>::value, Empty>::type* = nullptr, typename Integer_Type, typename Fractional_Type>
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t xi, yi, yj, si, vi;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = X->segments[xi];
        auto *x_data = (Fractional_Type *) x_seg.D->data;
        uint32_t x_nitems = x_seg.D->n;
        uint32_t x_nbytes = x_seg.D->nbytes;
        
        yi = Y->owned_segment;
        auto &y_seg = Y->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        uint32_t y_nitems = y_seg.D->n;
        uint32_t y_nbytes = y_seg.D->nbytes;
    
        bool has_weight = (std::is_same<Weight, Empty>::value) ? false : true;
        
        // Local computation, no need to put it on X
        if(tile.allocated)
        {
            uint32_t k = 0;
            Weight *A = nullptr;
            
            if(has_weight)
            {
                A = (Weight *) tile.csr->A->data;
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                uint32_t nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                uint32_t nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    nnz_per_row = IA[i + 1] - IA[i];
                    for(uint32_t j = 0; j < nnz_per_row; j++)
                    {
                        y_data[i] += A[k] * x_data[JA[k]];
                        k++;
                    }
                }            
            }
            else
            {
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                uint32_t nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                uint32_t nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    nnz_per_row = IA[i + 1] - IA[i];
                    for(uint32_t j = 0; j < nnz_per_row; j++)
                    {
                        y_data[i] += x_data[JA[k]];
                        k++;
                    }
                }            
            }
        }
        bool communication = (tile.nth) / A->tiling->rank_nrowgrps; 
        if(communication)
        {
            uint32_t leader = Y->segments[tile.rg].leader_rank;
            if(Env::rank == leader)
            {
                uint32_t vi = V->owned_segment;
                auto &v_seg = V->segments[vi];
                auto *v_data = (Fractional_Type *) v_seg.D->data;
                uint32_t v_nitems = v_seg.D->n;
                uint32_t v_nbytes = v_seg.D->nbytes;
                
                
                if(A->tiling->tiling_type == Tiling_type::_1D_ROW)
                {   
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        v_data[i] = (*f)(0, y_data[i], 0, 0);
                    }
                }
                else if((A->tiling->tiling_type == Tiling_type::_2D_)
                     or (A->tiling->tiling_type == Tiling_type::_1D_COL))
                {
                    MPI_Status status;
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                    {
                        uint32_t other_rank = A->follower_rowgrp_ranks[j];
                        yj = A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Y->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        uint32_t yj_nitems = yj_seg.D->n;
                        uint32_t yj_nbytes = yj_seg.D->nbytes;
                        MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &status);
                    }
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                    {
                        yj = A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Y->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        uint32_t yj_nitems = yj_seg.D->n;
                        uint32_t yj_nbytes = yj_seg.D->nbytes;                        
                        for(uint32_t i = 0; i < yj_nitems; i++)
                        {
                            y_data[i] += yj_data[i];
                        }
                    }
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        v_data[i] = (*f)(0, y_data[i], 0, 0); 
                    }
                }
            }
            else
            {
                MPI_Send(y_data, y_nbytes, MPI_BYTE, leader, pair.row, Env::MPI_WORLD);
            }
            memset(y_data, 0, y_nbytes);
            /*
            Triple<Weight, Integer_Type> pair;
            Triple<Weight, Integer_Type> pair1;
            uint32_t vi = V->owned_segment;
            auto &v_seg = V->segments[vi];
            auto *v_data = (Fractional_Type *) v_seg.D->data;
            uint32_t v_nitems = v_seg.D->n;
            uint32_t v_nbytes = v_seg.D->nbytes;
            
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                pair.row = i;
                pair.col = 0;
                auto pair1 = A->base(pair, v_seg.rg, v_seg.cg);
                printf("R(%d),S[%d]=%f\n",  Env::rank, pair1.row, v_data[i]);
            }
            */
        }
        
    }
}





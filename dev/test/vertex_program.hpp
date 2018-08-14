/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "vector.hpp"
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vertex_Program
{
    public:
        Vertex_Program();
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph);
        ~Vertex_Program();
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s);
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Vertex_Program<Weight,
                  Integer_Type, Fractional_Type> *VProgram);
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void free();
    
    protected:
        void print(Segment<Weight, Integer_Type, Fractional_Type> &segment);
        void apply();
        void populate(Fractional_Type value, Segment<Weight, Integer_Type, Fractional_Type> &segment);
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        Vector<Weight, Integer_Type, Fractional_Type> *X;
        Vector<Weight, Integer_Type, Fractional_Type> *Y;
        Vector<Weight, Integer_Type, Fractional_Type> *V;
        Vector<Weight, Integer_Type, Fractional_Type> *S;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};
                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight,
                       Integer_Type, Fractional_Type> &Graph)
                       : X(nullptr), Y(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
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
    Integer_Type nitems = segment.D->n;
    Integer_Type nbytes = segment.D->nbytes;
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init( 
        Fractional_Type x, Fractional_Type y, Fractional_Type v, 
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
          std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->local_col_segments);
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
    }    
    
    Y = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    }   

    V = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    populate(v, v_seg);
    
    Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
    uint32_t vi_ = VP->V->owned_segment;
    auto &v_seg_ = VP->V->segments[vi_];
    auto *v_data_ = (Fractional_Type *) v_seg_.D->data;
    Integer_Type v_nitems_ = v_seg_.D->n;
    Integer_Type v_nbytes_ = v_seg_.D->nbytes;
    
    S = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
               A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
    assert(vi_ == si);
    assert(v_nitems_ == s_nitems);
    assert(v_nbytes_ == s_nbytes);
    
    for(uint32_t i = 0; i < s_nitems; i++)
    {
        s_data[i] = v_data_[i];
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, 
                           Fractional_Type y, Fractional_Type v, Fractional_Type s)
{    
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
              std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
              
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->local_col_segments);
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
    }                    

    Y = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    } 
    
    V = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    populate(v, v_seg);
                   
    S = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    populate(s, s_seg);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t xi = X->owned_segment;
    auto &x_seg = X->segments[xi];
    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
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
                Integer_Type xj_nbytes = xj_seg.D->nbytes;
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
        Integer_Type x_nitems = x_seg.D->n;
        Integer_Type x_nbytes = x_seg.D->nbytes;
        
        yi = Y->owned_segment;
        auto &y_seg = Y->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
    
        bool has_weight = (std::is_same<Weight, Empty>::value) ? false : true;
        
        // Local computation, no need to put it on X
        if(tile.allocated)
        {
            if(A->compression == Compression_type::_CSR_)
            {
                uint32_t k = 0;
                Weight *A = (Weight *) tile.csr->A->data;
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    nnz_per_row = IA[i + 1] - IA[i];
                    for(uint32_t j = 0; j < nnz_per_row; j++)
                    {
                        y_data[i] += A[JA[k]] * x_data[JA[k]];
                        k++;
                    }
                }
            }
            else if(A->compression == Compression_type::_CSC_)    
            {
                uint32_t k = 0;
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                for(uint32_t i = 0; i < ncols_plus_one_minus_one; i++)
                {
                    nnz_per_col = COL_PTR[i + 1] - COL_PTR[i];
                    for(uint32_t j = 0; j < nnz_per_col; j++)
                    {
                        y_data[ROW_INDEX[k]] += A[ROW_INDEX[k]] * x_data[ROW_INDEX[k]];
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
                Integer_Type v_nitems = v_seg.D->n;
                Integer_Type v_nbytes = v_seg.D->nbytes;
                
                
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
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;
                        MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &status);
                    }
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                    {
                        yj = A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Y->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
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
            //uint32_t vi = V->owned_segment;
            //auto &v_seg = V->segments[vi];
            //print(v_seg);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::print(Segment<Weight,
                            Integer_Type, Fractional_Type> &segment)
{
    Triple<Weight, Integer_Type> pair;
    Triple<Weight, Integer_Type> pair1;
    auto *data = (Fractional_Type *) segment.D->data;
    Integer_Type nitems = segment.D->n;
    
    for(uint32_t i = 0; i < nitems; i++)
    {
        pair.row = i;
        pair.col = 0;
        auto pair1 = A->base(pair, segment.rg, segment.cg);
        printf("R(%d),S[%d]=%f\n",  Env::rank, pair1.row, data[i]);
    } 
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::free()
{
    X->del_vec();
    Y->del_vec();
    V->del_vec();
    S->del_vec();
}



/* Duplicate code for Graphs with Empty weights
   Note, the better way is to use inheritance. */

template<typename Integer_Type, typename Fractional_Type>
class Vertex_Program<Empty, Integer_Type, Fractional_Type>
{
    public:
        Vertex_Program();
        Vertex_Program(Graph<Empty, Integer_Type, Fractional_Type> &Graph);
        ~Vertex_Program();
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s);
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Vertex_Program<Empty,
                  Integer_Type, Fractional_Type> *VProgram);
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void free();
    
    protected:
        void print(Segment<Empty, Integer_Type, Fractional_Type> &segment);
        void apply();
        void populate(Fractional_Type value, Segment<Empty, Integer_Type, Fractional_Type> &segment);
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
    
        Matrix<Empty, Integer_Type, Fractional_Type> *A;
        Vector<Empty, Integer_Type, Fractional_Type> *X;
        Vector<Empty, Integer_Type, Fractional_Type> *Y;
        Vector<Empty, Integer_Type, Fractional_Type> *V;
        Vector<Empty, Integer_Type, Fractional_Type> *S;
};

template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::Vertex_Program() {};
                
template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Empty,
                       Integer_Type, Fractional_Type> &Graph)
                       : X(nullptr), Y(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
}

template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    delete X;
    delete Y;
    delete V;
    delete S;   
};

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::populate(Fractional_Type value,
            Segment<Empty, Integer_Type, Fractional_Type> &segment)
{
    auto *data = (Fractional_Type *) segment.D->data;
    Integer_Type nitems = segment.D->n;
    Integer_Type nbytes = segment.D->nbytes;
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

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::init( 
        Fractional_Type x, Fractional_Type y, Fractional_Type v, 
        Vertex_Program<Empty, Integer_Type, Fractional_Type> *VProgram)
{
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
          std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
    X = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->local_col_segments);
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
    }    
    
    Y = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    }   

    V = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    populate(v, v_seg);
    
    Vertex_Program<Empty, Integer_Type, Fractional_Type> *VP = VProgram;
    uint32_t vi_ = VP->V->owned_segment;
    auto &v_seg_ = VP->V->segments[vi_];
    auto *v_data_ = (Fractional_Type *) v_seg_.D->data;
    Integer_Type v_nitems_ = v_seg_.D->n;
    Integer_Type v_nbytes_ = v_seg_.D->nbytes;
    
    S = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
               A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
    assert(vi_ == si);
    assert(v_nitems_ == s_nitems);
    assert(v_nbytes_ == s_nbytes);
    
    for(uint32_t i = 0; i < s_nitems; i++)
    {
        s_data[i] = v_data_[i];
    }
}

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::init(Fractional_Type x, 
                           Fractional_Type y, Fractional_Type v, Fractional_Type s)
{    
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
              std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
              
    X = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->local_col_segments);
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
    }                    

    Y = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    } 
    
    V = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    populate(v, v_seg);
                   
    S = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                   A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks);
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    populate(s, s_seg);
}

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t xi = X->owned_segment;
    auto &x_seg = X->segments[xi];
    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    uint32_t vi = V->owned_segment;
    auto &v_seg = V->segments[vi];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    
    uint32_t si = S->owned_segment;
    auto &s_seg = S->segments[si];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
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
            //if(!Env::rank)
              //  printf("%d --> %d\n", Env::rank, other_rank);
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

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::gather()
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
                Integer_Type xj_nbytes = xj_seg.D->nbytes;
                MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, xj_seg.leader_rank, xj_seg.cg, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
                //if(!Env::rank)
                  //  printf("%d <-- %d\n", Env::rank, xj_seg.leader_rank);
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
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    Env::barrier();   
}

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
        Integer_Type x_nitems = x_seg.D->n;
        Integer_Type x_nbytes = x_seg.D->nbytes;
        
        yi = Y->owned_segment;
        auto &y_seg = Y->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        // Local computation, no need to put it on X
        if(tile.allocated)
        {
            if(A->compression == Compression_type::_CSR_)
            {
            
                uint32_t k = 0;
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
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
            else if(A->compression == Compression_type::_CSC_)
            {
                uint32_t k = 0;
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                for(uint32_t i = 0; i < ncols_plus_one_minus_one; i++)
                {
                    nnz_per_col = COL_PTR[i + 1] - COL_PTR[i];
                    for(uint32_t j = 0; j < nnz_per_col; j++)
                    {
                        y_data[ROW_INDEX[k]] += x_data[ROW_INDEX[k]];
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
                Integer_Type v_nitems = v_seg.D->n;
                Integer_Type v_nbytes = v_seg.D->nbytes;
                
                
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
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;
                        MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &status);
                    }
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                    {
                        yj = A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Y->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
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
            //uint32_t vi = V->owned_segment;
            //auto &v_seg = V->segments[vi];
            //print(v_seg);
        }
    }
}

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::print(Segment<Empty,
                            Integer_Type, Fractional_Type> &segment)
{
    Triple<Empty, Integer_Type> pair;
    Triple<Empty, Integer_Type> pair1;
    auto *data = (Fractional_Type *) segment.D->data;
    Integer_Type nitems = segment.D->n;
    
    for(uint32_t i = 0; i < nitems; i++)
    {
        pair.row = i;
        pair.col = 0;
        auto pair1 = A->base(pair, segment.rg, segment.cg);
        printf("R(%d),S[%d]=%f\n",  Env::rank, pair1.row, data[i]);
    } 
}

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::free()
{
    X->del_vec();
    Y->del_vec();
    V->del_vec();
    S->del_vec();
}


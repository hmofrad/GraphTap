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
        Vector<Weight, Integer_Type, Fractional_Type> *V;
        Vector<Weight, Integer_Type, Fractional_Type> *S;
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};
                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight,
                       Integer_Type, Fractional_Type> &Graph)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    delete X;
    delete V;
    delete S;   
    for (uint32_t i = 0; i < A->tiling->rank_nrowgrps; i++)
    {
        delete Y[i];
    }
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
    
    Vector<Weight, Integer_Type, Fractional_Type> *YY;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        if(((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0)
        {
            if(pair.row == owned_diag_segment)
            {
                YY = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                            A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
                for(uint32_t yi : YY->local_segments)
                {
                    auto &y_seg = YY->segments[yi];
                    populate(y, y_seg);
                } 
            }
            else
            {
                YY = new Vector<Weight, Integer_Type, Fractional_Type>
                    (A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, A->tile_height, A->tile_width,
                     owned_diag_segment, A->leader_ranks);
                uint32_t yi = YY->owned_segment;  
                auto &y_seg = YY->segments[yi];
                populate(y, y_seg);   
            }
            Y.push_back(YY);
        }
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
    
    Vector<Weight, Integer_Type, Fractional_Type> *YY;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        if(((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0)
        {
            if(pair.row == owned_diag_segment)
            {
                YY = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                            A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
                for(uint32_t yi : YY->local_segments)
                {
                    auto &y_seg = YY->segments[yi];
                    populate(y, y_seg);
                } 
            }
            else
            {
                YY = new Vector<Weight, Integer_Type, Fractional_Type>
                    (A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, A->tile_height, A->tile_width,
                     owned_diag_segment, A->leader_ranks);
                uint32_t yi = YY->owned_segment;  
                auto &y_seg = YY->segments[yi];
                populate(y, y_seg);   
            }
            Y.push_back(YY);
        }
    }     
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
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        //MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        //out_requests.clear();
        //Env::barrier();
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    MPI_Request request;
    uint32_t xi, y = 0, yi, yj, yk, si, vi;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = X->segments[xi];
        auto *x_data = (Fractional_Type *) x_seg.D->data;
        Integer_Type x_nitems = x_seg.D->n;
        Integer_Type x_nbytes = x_seg.D->nbytes;
        
        auto *Yp = Y[y];
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        if(tile.allocated)
        {
            if(A->compression == Compression_type::_CSR_)
            {
                Weight *A = (Weight *) tile.csr->A->data;
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {
                        y_data[i] += A[j] * x_data[JA[j]];
                    }
                }
            }
            else if(A->compression == Compression_type::_CSC_)    
            {
                Weight *VAL = (Weight *) tile.csc->VAL->data;
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        y_data[ROW_INDEX[i]] += VAL[i] * x_data[j];   
                    }
                }
            }            
        }
        
        bool communication = (((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0);
        if(communication)
        {            
            uint32_t leader = Yp->segments[tile.rg].leader_rank;
            if(Env::rank == leader)
            {
                if(A->tiling->tiling_type == Tiling_type::_1D_ROW)
                {   
                    uint32_t vi = V->owned_segment;
                    auto &v_seg = V->segments[vi];
                    auto *v_data = (Fractional_Type *) v_seg.D->data;
                    Integer_Type v_nitems = v_seg.D->n;
                    Integer_Type v_nbytes = v_seg.D->nbytes;
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
                        auto &yj_seg = Yp->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;
                        //MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &status);
                        //printf("MPI_Irecv %d %d\n", Env::rank, t);
                        MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                    }
                }
                yk = y;
            }
            else
            {
                //MPI_Send(y_data, y_nbytes, MPI_BYTE, leader, pair.row, Env::MPI_WORLD);
                MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair.row, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            y++;
        }
    }
    
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_COL))
    {
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        
        auto *Yp = Y[yk];
        //auto *Yp = Y;
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
        {
            yj = A->rowgrp_ranks_accu_seg[j];
            auto &yj_seg = Yp->segments[yj];
            auto *yj_data = (Fractional_Type *) y_seg.D->data;
            Integer_Type yj_nitems = yj_seg.D->n;
            Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
            for(uint32_t i = 0; i < yj_nitems; i++)
            {
                y_data[i] += yj_data[i];
            }
        }
        
        vi = V->owned_segment;
        auto &v_seg = V->segments[vi];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        Integer_Type v_nbytes = v_seg.D->nbytes;
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            v_data[i] = (*f)(0, y_data[i], 0, 0); 
        }
        memset(y_data, 0, y_nbytes);
        
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        //Env::barrier();
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
    V->del_vec();
    S->del_vec();
    
    for (uint32_t i = 0; i < A->tiling->rank_nrowgrps; i++)
    {
        Y[i]->del_vec();
    }
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
        
        uint32_t rank_nrowgrps;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
    
        Matrix<Empty, Integer_Type, Fractional_Type> *A;
        Vector<Empty, Integer_Type, Fractional_Type> *X;
        //Vector<Empty, Integer_Type, Fractional_Type> *Y;
        Vector<Empty, Integer_Type, Fractional_Type> *V;
        Vector<Empty, Integer_Type, Fractional_Type> *S;
        std::vector<Vector<Empty, Integer_Type, Fractional_Type> *> Y;
};

template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::Vertex_Program() {};
                
template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Empty,
                       Integer_Type, Fractional_Type> &Graph)
                       : X(nullptr), V(nullptr), S(nullptr) // Y(nullptr), 
{
    A = Graph.A;
}

template<typename Integer_Type, typename Fractional_Type>
Vertex_Program<Empty, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    delete X;
    //delete Y;
    delete V;
    delete S;
    for (uint32_t i = 0; i < A->tiling->rank_nrowgrps; i++)
    {
        delete Y[i];
    }
    
    
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
    /*
    Y = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    }   
    */
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

    Vector<Empty, Integer_Type, Fractional_Type> *YY;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        //printf("t=%d row=%d col=%d ith=%d jth=%d nth=%d > %d >> %d\n", t, pair.row, pair.col, tile.ith,
        //tile.jth, tile.nth, (((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0), pair.row == owned_diag_segment);// tile.rg / A->tiling->rank_ncolgrps, tile.cg % A->tiling->rank_nrowgrps);
        
        if(((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0)
        {
            if(pair.row == owned_diag_segment)
            {
                //printf("Owner, %d %d\n", pair.row, owned_diag_segment);
                YY = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                            A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
                for(uint32_t yi : YY->local_segments)
                {
                    auto &y_seg = YY->segments[yi];
                    populate(y, y_seg);
                } 
            }
            else
            {
                //printf("diag, %d\n", owned_diag_segment);
                YY = new Vector<Empty, Integer_Type, Fractional_Type>
                    (A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, A->tile_height, A->tile_width,
                     owned_diag_segment, A->leader_ranks);
                uint32_t yi = YY->owned_segment;  
                auto &y_seg = YY->segments[yi];
                populate(y, y_seg);   
            }
            Y.push_back(YY);
        }
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
    /*
    Y = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                    A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
    for(uint32_t yi : Y->local_segments)
    {
        auto &y_seg = Y->segments[yi];
        populate(y, y_seg);
    } 
    */
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
    
    Vector<Empty, Integer_Type, Fractional_Type> *YY;
    for(uint32_t t: A->local_tiles)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        //printf("t=%d row=%d col=%d ith=%d jth=%d nth=%d > %d >> %d\n", t, pair.row, pair.col, tile.ith,
        //tile.jth, tile.nth, (((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0), pair.row == owned_diag_segment);// tile.rg / A->tiling->rank_ncolgrps, tile.cg % A->tiling->rank_nrowgrps);
        
        if(((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0)
        {
            if(pair.row == owned_diag_segment)
            {
                //if(Env::rank == 10)
                //printf("Owner, %d %d %d\n", pair.row, pair.col, owned_diag_segment);
                YY = new Vector<Empty, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                            A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks, A->rowgrp_ranks_accu_seg);
                for(uint32_t yi : YY->local_segments)
                {
                    auto &y_seg = YY->segments[yi];
                    populate(y, y_seg);
                } 
            }
            else
            {
                //if(Env::rank == 10)
                //printf("other, %d %d %d\n", pair.row, pair.col, owned_diag_segment);
                YY = new Vector<Empty, Integer_Type, Fractional_Type>
                    (A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, A->tile_height, A->tile_width,
                     owned_diag_segment, A->leader_ranks);
                uint32_t yi = YY->owned_segment;  
                auto &y_seg = YY->segments[yi];
                populate(y, y_seg);   
            }
            Y.push_back(YY);
        }
    }


    /*
    for (uint32_t i = 0; i < A->tiling->rank_nrowgrps; i++)
    {
        Vector<Empty, Integer_Type, Fractional_Type> *ZZ = new Vector<Empty, Integer_Type, Fractional_Type>
            (A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, A->tile_height, A->tile_width,
            owned_diag_segment, A->leader_ranks);
        uint32_t zi = Y->owned_segment;  
        auto &z_seg = ZZ->segments[zi];
        populate(y, z_seg);   
        Z.push_back(ZZ);            
    }
    */
    
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
            //out_requests.push_back(request);
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
        double t1 = Env::clock();
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        double t2 = Env::clock();
        if(!Env::rank)
            printf("Gather MPI_Waitall for in_req: %f\n", t2 - t1);
        //MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        //out_requests.clear();
        //Env::barrier();
        
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

    
    //Env::barrier();   
}

template<typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Empty, Integer_Type, Fractional_Type>::combine(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    MPI_Request request;
    uint32_t xi, y = 0, yi, yj, yk, si, vi;
    uint32_t zi = 0, zzi;
    uint32_t ii = -1;
    Vector<Empty, Integer_Type, Fractional_Type> *Yp = nullptr;
    //Segment<Empty, Integer_Type, Fractional_Type> &y_seg;
    //Fractional_Type *y_data = nullptr;
    //Integer_Type y_nitems = 0;
    //Integer_Type y_nbytes = 0;
    
    for(uint32_t t: A->local_tiles)
    {
        //printf("rank=%d, tile=%d\n", Env::rank, t);
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = X->segments[xi];
        auto *x_data = (Fractional_Type *) x_seg.D->data;
        Integer_Type x_nitems = x_seg.D->n;
        Integer_Type x_nbytes = x_seg.D->nbytes;
        
        
        
        
        
        //if(pair.row == pair.col)
        //{
          //  yk = y;
        //}

        Yp = Y[y];
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;   
        
        /*
        auto *Yp = Z[zi];
        //auto *Yp = Y;
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        */
        
        
        //if(!Env::rank)
        //{
            //printf(">>>tile=%d rowgrp_nranks=%d\n", t, A->tiling->rowgrp_nranks);
            //printf(">>>%d %d %d\n", t, zi, A->tiling->rank_nrowgrps);
            //printf("SEG=%d\n", ZZ->owned_segment);
        //}
        /*
        Vector<Empty, Integer_Type, Fractional_Type> *ZZ = Z[zi];
        zzi = ZZ->owned_segment;
        auto &zz_seg = ZZ->segments[zzi];
        auto *zz_data = (Fractional_Type *) zz_seg.D->data;
        Integer_Type zz_nitems = zz_seg.D->n;
        Integer_Type zz_nbytes = zz_seg.D->nbytes;
        */
        
        
        
        // Local computation, no need to put it on X
        if(tile.allocated)
        {
            if(A->compression == Compression_type::_CSR_)
            {
            
                //uint32_t k = 0;
                //uint32_t l = 0;
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
                //printf("START %d\n", Env::rank);
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    //nnz_per_row = IA[i + 1] - IA[i];
                    //for(uint32_t j = 0; j < nnz_per_row; j++)
                    //{
                    //    l = JA[k];
                    //    y_data[i] += x_data[l];
                    //   k++;
                    //}
                    
                    for(uint32_t j = IA[i]; j < IA[i +1]; j++)
                    {
                        y_data[i] += x_data[JA[j]];
                    }
                }
                //printf("END %d\n", Env::rank);
            }
            else if(A->compression == Compression_type::_CSC_)
            {
                //uint32_t k = 0;
                //uint32_t l = 0;
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    //for(uint32_t i = 0; i < ncols_plus_one_minus_one; i++)
                    //{
                    //nnz_per_col = COL_PTR[i + 1] - COL_PTR[i];
                    //for(uint32_t j = 0; j < nnz_per_col; j++)
                    //{
                    //    l = ROW_INDEX[k];
                    //    y_data[l] += x_data[l];
                    //    k++;
                    //}
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        y_data[ROW_INDEX[i]] += x_data[j];
                    }
                }
            }
        }
        //printf("communication= rank=%d yk=%d\n", Env::rank, yk);
        bool communication = (((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0); 
        //if(!Env::rank)
          //      printf("tile=%d, nth=%d, rowgrp_nranks=%d comm=%d\n", t, tile.nth, A->tiling->rank_ncolgrps, ((tile.nth + 1) % A->tiling->rank_ncolgrps) == 0);
        if(communication)
        {
            uint32_t leader = Yp->segments[tile.rg].leader_rank;
            if(Env::rank == leader)
            {
                if(A->tiling->tiling_type == Tiling_type::_1D_ROW)
                {   
                    vi = V->owned_segment;
                    auto &v_seg = V->segments[vi];
                    auto *v_data = (Fractional_Type *) v_seg.D->data;
                    Integer_Type v_nitems = v_seg.D->n;
                    Integer_Type v_nbytes = v_seg.D->nbytes;
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
                        auto &yj_seg = Yp->segments[yj];
                        auto *yj_data = (Fractional_Type *) y_seg.D->data;
                        Integer_Type yj_nitems = yj_seg.D->n;
                        Integer_Type yj_nbytes = yj_seg.D->nbytes;
                        //MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &status);
                        MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, other_rank, pair.row, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                    }
                    /*
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                    {
                        yj = A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Yp->segments[yj];
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
                    */
                }
                yk = y;
            }
            else
            {
                //MPI_Send(y_data, y_nbytes, MPI_BYTE, leader, pair.row, Env::MPI_WORLD);
                MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair.row, Env::MPI_WORLD, &request);
                //out_requests.push_back(request);
                
                //MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
                //in_requests.clear();
            }
            //memset(y_data, 0, y_nbytes);
            //ydi++;
            //uint32_t vi = V->owned_segment;
            //auto &v_seg = V->segments[vi];
            //print(v_seg);
            y++;
        }
    }
    
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_COL))
    {
        
        Yp = Y[yk];
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        int32_t incount = in_requests.size();
        int32_t outcount = 0;
        int32_t incounts = A->tiling->rowgrp_nranks - 1;
        std::vector<MPI_Status> statuses(incounts);
        std::vector<int> indices(incounts);
        uint32_t received = 0;
        while(received < incount)
        {
            MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
            assert(outcount != MPI_UNDEFINED);
            if(outcount > 0)
            {
                /*
                if(Env::rank == 5)
                {
                    printf("%d sends completed\n", outcount);
                    for(uint32_t i : indices)
                        printf("%d ", i);
                    printf("\n");
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                        printf("%d ", j);
                    printf("\n");
                    for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
                        printf("%d ", A->rowgrp_ranks_accu_seg[j]);
                    printf("\n");
                    
                }
                */
                /*
                if(Env::rank == 0)
                {
                    printf("<\n");
                    for(uint32_t j = 0; j < outcount; j++)
                    {
                        printf("%d %d %d\n", j, indices[j], A->rowgrp_ranks_accu_seg[indices[j]]);
                    }
                    printf(">\n");
                }
                */
                
                for(uint32_t j = 0; j < outcount; j++)
                {
                    yj = A->rowgrp_ranks_accu_seg[indices[j]];
                    auto &yj_seg = Yp->segments[yj];
                    auto *yj_data = (Fractional_Type *) y_seg.D->data;
                    Integer_Type yj_nitems = yj_seg.D->n;
                    Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
                    for(uint32_t i = 0; i < yj_nitems; i++)
                    {
                        y_data[i] += yj_data[i];
                    }
                }
                received += outcount;
            }
        }
        
        in_requests.clear();
        /*
        double t1 = Env::clock();
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        double t2 = Env::clock();
        if(!Env::rank)
            printf("Combine MPI_Waitall for in_req: %f\n", t2 - t1);
        
        
        Yp = Y[yk];
        //Yp = Y
        //auto *Yp = Y;
        yi = Yp->owned_segment;
        auto &y_seg = Yp->segments[yi];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        
        for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
        {
            yj = A->rowgrp_ranks_accu_seg[j];
            auto &yj_seg = Yp->segments[yj];
            auto *yj_data = (Fractional_Type *) y_seg.D->data;
            Integer_Type yj_nitems = yj_seg.D->n;
            Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
            for(uint32_t i = 0; i < yj_nitems; i++)
            {
                y_data[i] += yj_data[i];
            }
        }
        */
        vi = V->owned_segment;
        auto &v_seg = V->segments[vi];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        Integer_Type v_nbytes = v_seg.D->nbytes;
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            v_data[i] = (*f)(0, y_data[i], 0, 0); 
        }
        memset(y_data, 0, y_nbytes);
        
        //MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        //out_requests.clear();
        //Env::barrier();
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
    //Y->del_vec();
    V->del_vec();
    S->del_vec();
    
    
    for (uint32_t i = 0; i < A->tiling->rank_nrowgrps; i++)
    {
        Y[i]->del_vec();
    }
}


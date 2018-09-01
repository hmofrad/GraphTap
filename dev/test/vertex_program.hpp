/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP
 
#include "vector.hpp"
 
enum Order_type
{
  _ROW_,
  _COL_
}; 
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vertex_Program
{
    public:
        Vertex_Program();
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph, Order_type = _ROW_);
        ~Vertex_Program();
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s,
                            Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram = nullptr);                  
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void bcast(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void checksum();
        void checksumPR();
        void free();
        void filter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
    
    protected:
        void spmv(Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
                  Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void print(Segment<Weight, Integer_Type, Fractional_Type> &segment);
        void apply();
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src);
        void clear(Vector<Weight, Integer_Type, Fractional_Type> *vec);
        Order_type order;
        Tiling_type tiling_type;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nrowgrps, rank_ncolgrps;
        
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
                       Integer_Type, Fractional_Type> &Graph, Order_type order_)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
    order = order_;
    
    tiling_type = A->tiling->tiling_type;
    rowgrp_nranks = A->tiling->rowgrp_nranks;
    colgrp_nranks = A->tiling->colgrp_nranks;
    rank_nrowgrps = A->tiling->rank_nrowgrps;
    rank_ncolgrps = A->tiling->rank_ncolgrps;
    
    if(order == _ROW_)
    {
        ;
    }
    else if (order == _COL_)
    {
        fprintf(stderr, "Has not been implemented\n");
        Env::exit(1);
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    delete X;
    delete V;
    delete S;   
    
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        delete Y[i];
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::free()
{
    X->del_vec();
    V->del_vec();
    S->del_vec();
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        Y[i]->del_vec();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::clear(
                               Vector<Weight, Integer_Type, Fractional_Type> *vec)
{
    for(uint32_t i = 0; i < vec->vector_length; i++)
    {
        auto &seg = vec->segments[i];
        auto *data = (Fractional_Type *) seg.D->data;
        Integer_Type nitems = seg.D->n;
        Integer_Type nbytes = seg.D->nbytes;
        memset(data, 0, nbytes);
    }
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(
        Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value)
{
    for(uint32_t i = 0; i < vec->vector_length; i++)
    {
        auto &seg = vec->segments[i];
        auto *data = (Fractional_Type *) seg.D->data;
        Integer_Type nitems = seg.D->n;
        Integer_Type nbytes = seg.D->nbytes;

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
}         

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src)
{
    for(uint32_t i = 0; i < vec_src->vector_length; i++)
    {
        auto &seg_src = vec_src->segments[i];
        auto *data_src = (Fractional_Type *) seg_src.D->data;
        Integer_Type nitems_src = seg_src.D->n;
        Integer_Type nbytes_src = seg_src.D->nbytes;
        
        auto &seg_dst = vec_dst->segments[i];
        auto *data_dst = (Fractional_Type *) seg_dst.D->data;
        Integer_Type nitems_dst = seg_dst.D->n;
        Integer_Type nbytes_dst = seg_dst.D->nbytes;
        memcpy(data_dst, data_src, nbytes_src);
        //printf("%d %f %f\n", Env::rank, data_dst[0], data_src[0]);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, Fractional_Type y, 
     Fractional_Type v, Fractional_Type s, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height,  A->local_col_segments);
    populate(X, x);
    
    if(Env::comm_split)
        V = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->accu_segment_cg_vec);
    else
        V = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->owned_segment_vec);
    populate(V, v);
    /*
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        printf("r=%d i=%d v=%f v_=%f\n", Env::rank, i, v_data[i], v);
    }
    printf("***********************************************\n");
    */
    if(Env::comm_split)
        S = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->accu_segment_cg_vec);
    else
        S = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->owned_segment_vec);
    
    if(VProgram)
    {
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
        Vector<Weight, Integer_Type, Fractional_Type> *V_ = VP->V;
        populate(S, V_);
    }
    else
    {
        populate(S, s);
    }
    
    Vector<Weight, Integer_Type, Fractional_Type> *Y_;
    for(uint32_t t: A->local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        uint32_t tile_th = tile.nth;
        uint32_t pair_idx = pair.row;        
        bool vec_add = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(vec_add)
        {
            bool vec_owner = (pair_idx == A->owned_segment);
            
            if(vec_owner)
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->all_rowgrp_ranks_accu_seg);
            }
            else
            {
                if(Env::comm_split)
                    Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->accu_segment_rg_vec);
                else
                    Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(A->tile_height, A->owned_segment_vec);
            }
            Y.push_back(Y_);
        }
        
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t xo = A->accu_segment_cg;
    auto &x_seg = X->segments[xo];
    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    
    uint32_t so = 0;
    auto &s_seg = S->segments[so];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    
    for(uint32_t i = 0; i < x_nitems; i++)
    {
        x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
    }
    
    MPI_Request request;
    uint32_t follower, accu;
    if((tiling_type == Tiling_type::_2D_)
        or (tiling_type == Tiling_type::_1D_ROW))
    {
        for(uint32_t i = 0; i < colgrp_nranks - 1; i++)
        {
            if(Env::comm_split)
            {
               follower = A->follower_colgrp_ranks_cg[i];
               MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, x_seg.g, Env::colgrps_comm, &request);
               out_requests.push_back(request);
            }
            else
            {
                follower = A->follower_colgrp_ranks[i];
                MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, x_seg.g, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
                //printf("Send r=%d i=%d f=%d g=%d\n", Env::rank, i, follower, x_seg.g);
            }
            
        }
    }
    else if(tiling_type == Tiling_type::_1D_COL)
    {
        ;
    }
    else
    {
        fprintf(stderr, "Invalid tiling\n");
        Env::exit(1);
    }
    Env::barrier();   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::gather()
{   
    uint32_t leader;
    MPI_Request request;
    if((tiling_type == Tiling_type::_2D_)
        or (tiling_type == Tiling_type::_1D_ROW))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            auto& xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nitems = xj_seg.D->n;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;

            if(Env::comm_split)
            {
                leader = A->leader_ranks_cg[xj_seg.g];
                if(leader != Env::rank_cg)
                {
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, leader, xj_seg.g, Env::colgrps_comm, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                leader = A->leader_ranks[xj_seg.g];
                if(leader != Env::rank)
                {
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, leader, xj_seg.g, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                    //printf("Recv r=%d i=%d l=%d g=%d\n", Env::rank, i, leader, xj_seg.g);
                }
            }
            
        }
        
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        //Env::barrier();
    }
    else if(tiling_type == Tiling_type::_1D_COL)
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t leader;
    uint32_t xo = A->accu_segment_cg;
    auto &x_seg = X->segments[xo];
    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    
    uint32_t so = 0;
    auto &s_seg = S->segments[so];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
    for(uint32_t i = 0; i < x_nitems; i++)
    {
        x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
    }
    
    if((tiling_type == Tiling_type::_2D_)
        or (tiling_type == Tiling_type::_1D_ROW))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            leader = A->leader_ranks_cg[A->local_col_segments[i]];
            auto &xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
            if(Env::comm_split)
                MPI_Bcast(xj_data, xj_nbytes, MPI_BYTE, leader, Env::colgrps_comm);
            else
            {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
        }
    }
    else if(tiling_type == Tiling_type::_1D_COL)
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
            Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    
    auto *y_data = (Fractional_Type *) y_seg.D->data;
    Integer_Type y_nitems = y_seg.D->n;
    Integer_Type y_nbytes = y_seg.D->nbytes;

    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    if(tile.allocated)
    {
        if(A->compression == Compression_type::_CSR_)
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A->data;
            #endif
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
            Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
            Integer_Type nnz_per_row;
            for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
            {
                for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                {       
                    #ifdef HAS_WEIGHT
                    if(x_data[JA[j]] and A[j])
                        y_data[i] += A[j] * x_data[JA[j]];
                    #else
                    if(x_data[JA[j]])
                        y_data[i] += x_data[JA[j]];
                    #endif                        
                }
            }
        }
        else if(A->compression == Compression_type::_CSC_)    
        {
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            Integer_Type nnz_per_col;
            for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
            {
                for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                {
                    #ifdef HAS_WEIGHT
                    if(x_data[j] and VAL[i])
                        y_data[ROW_INDEX[i]] += VAL[i] * x_data[j];   
                    #else
                    if(x_data[j])
                    {
                        y_data[ROW_INDEX[i]] += x_data[j];
                    }
                    #endif
                }
            }
        }            
    }    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::filter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    MPI_Comm communicator;
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, y = 0, yi = 0, yk = 0, yo = 0;
    double t1, t2;
    t1 = Env::clock();
    for(uint32_t t: A->local_tiles_col_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        tile_th = tile.mth;
        pair_idx = pair.col;
        vec_owner = (pair_idx == A->owned_segment);
        
        if(vec_owner)
            yo = A->accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        auto &x_seg = X->segments[xi];
        
        spmv(y_seg, x_seg, tile);
        yi++;
        
        communication = (((tile_th + 1) % rank_nrowgrps) == 0);
        
        if(!Env::rank)
            printf("t=%d, com=%d\n", t, communication);
        
        
        if(communication)
        {
            if(!Env::rank)
                printf("rank=%d t=%d xi=%d yi=%d l=%d m=%d\n", Env::rank, t, xi, yi, leader, my_rank);            
            xi++;
            yi = 0;
        }
        
        /*
        if(communication and xi == rank_ncolgrps)
        {
            if(!Env::rank)
                printf("COMM NOW %d \n", A->accu_segment_rg);
            for(uint32_t i = 0; i < rank_nrowgrps; i++)
            {
                
                auto &tile = A->tile[i][i];
                if(Env::comm_split)
                {           
                    leader = tile.leader_rank_rg_rg;
                    my_rank = Env::rank_rg;
                }
                else
                {
                    leader = tile.leader_rank_rg;
                    my_rank = Env::rank;
                }
                
                
                if(leader == my_rank)
                {
                    follower = A->follower_rowgrp_ranks[xi];
                    if(!Env::rank)
                        printf("Recv: tile=%d l_rank=%d <-- follower=%d g=%d\n", t, leader, follower, y_seg.g);
                }
                else
                {
                    if(!Env::rank)
                        printf("Send: tile=%d rank=%d --> leader=%d g=%d\n", t, Env::rank, leader, y_seg.g);
                }
                    
                
                
            }
        }
        */   
    }
    t2 = Env::clock();
    yi = 0, xi = 0, yk = 0;
    for(uint32_t t: A->local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        tile_th = tile.nth;
        pair_idx = pair.row;
        vec_owner = (pair_idx == A->owned_segment);
        
        if(vec_owner)
            yo = A->accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            if(Env::comm_split)
            {
                leader = tile.leader_rank_rg_rg;
                my_rank = Env::rank_rg;
                communicator = Env::rowgrps_comm;
            }
            else
            {
                leader = tile.leader_rank_rg;
                my_rank = Env::rank;
                communicator = Env::MPI_WORLD;
            }
            
            if(leader == my_rank)
            {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                {
                    if(Env::comm_split)
                    {
                        follower = A->follower_rowgrp_ranks_rg[j];
                        accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
                    }
                    else
                    {
                        follower = A->follower_rowgrp_ranks[j];
                        accu = A->follower_rowgrp_ranks_accu_seg[j];
                    }
                    
                    auto &yj_seg = Yp->segments[accu];
                    auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                    Integer_Type yj_nitems = yj_seg.D->n;
                    Integer_Type yj_nbytes = yj_seg.D->nbytes;
                    MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                    if(!Env::rank)
                        printf("Recv: tile=%d l_rank=%d <-- follower=%d g=%d\n", t, leader, follower, y_seg.g);
                }
                yk = yi;
            }
            else
            {
                MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
                if(!Env::rank)
                    printf("Send: tile=%d rank=%d --> leader=%d g=%d\n", t, Env::rank, leader, y_seg.g);
            }
            xi = 0;
            yi++;
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();


    auto *Yp = Y[yk];
    yo = A->accu_segment_rg;
    auto &y_seg = Yp->segments[yo];
    auto *y_data = (Fractional_Type *) y_seg.D->data;
    Integer_Type y_nitems = y_seg.D->n;
    Integer_Type y_nbytes = y_seg.D->nbytes;
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = A->follower_rowgrp_ranks_accu_seg[j];
        auto &yj_seg = Yp->segments[accu];
        auto *yj_data = (Fractional_Type *) yj_seg.D->data;
        Integer_Type yj_nitems = yj_seg.D->n;
        Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
        for(uint32_t i = 0; i < yj_nitems; i++)
        {
            if(yj_data[i])
                y_data[i] += yj_data[i];
        }
    }
    
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    //Fractional_Type tol = 1e-5;
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        //Fractional_Type tmp = y_data[i];
        
        v_data[i] = (*f)(0, y_data[i], 0, 0); 
        
        //if(fabs(v_data[i] - tmp) > tol)
        //    printf("Converged\n");
        //printf("%d %d %f\n", Env::rank, i, (fabs(v_data[i] - tmp) > tol));
        //nedges_local += nedges_local;
    }
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        clear(Y[i]);
    
    
    
    
}

                   
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    
    //std::vector<uint32_t> tags;
    //uint64_t nedges_local = 0;
    //uint64_t nedges_global = 0;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, y = 0, yi = 0, yo = 0, o = 0, xo = 0, yj, yk, si, vi;
    double t1, t2;
    t1 = Env::clock();
    for(uint32_t t: A->local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        tile_th = tile.nth;
        pair_idx = pair.row;
        vec_owner = (pair_idx == A->owned_segment);

        if(vec_owner)
            yo = A->accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        auto &x_seg = X->segments[xi];
        //auto *x_data = (Fractional_Type *) x_seg.D->data;
        //Integer_Type x_nitems = x_seg.D->n;
        //Integer_Type x_nbytes = x_seg.D->nbytes;
        
        spmv(y_seg, x_seg, tile);
        
        /*
        if(tile.allocated)
        {
            if(A->compression == Compression_type::_CSR_)
            {
                #ifdef HAS_WEIGHT
                Weight *A = (Weight *) tile.csr->A->data;
                #endif
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {
                        
                        #ifdef HAS_WEIGHT
                        if(x_data[JA[j]] and A[j])
                            y_data[i] += A[j] * x_data[JA[j]];
                        #else
                        if(x_data[JA[j]])
                            y_data[i] += x_data[JA[j]];
                        #endif                        
                    }
                }
            }
            else if(A->compression == Compression_type::_CSC_)    
            {
                #ifdef HAS_WEIGHT
                Weight *VAL = (Weight *) tile.csc->VAL->data;
                #endif
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        #ifdef HAS_WEIGHT
                        if(x_data[j] and VAL[i])
                            y_data[ROW_INDEX[i]] += VAL[i] * x_data[j];   
                        #else
                        //if(x_data[ROW_INDEX[i]])
                        //if(((t == 0) or (t == 1) or (t == 2) or (t == 3)) and ROW_INDEX[i] == 0)
                          //  printf("<<<<<r=%d t=%d ri=%d yd=%f %f\n", Env::rank, t, ROW_INDEX[i], y_data[0], x_data[j]);   
                        if(x_data[j])
                        {
                            y_data[ROW_INDEX[i]] += x_data[j]; //x_data[ROW_INDEX[i]];   

                            //nedges_local +=x_data[ROW_INDEX[i]];
                        }
                        #endif
                    }
                }
            }            
        }
        //if(t == 0)
        //{
            //for(uint32_t i = 0; i < y_nitems; i++)
                //if(!t and i == 0)
               //printf(">>>>>%d y=%f\n", Env::rank, y_data[0]);
        //}
        */
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            if(Env::comm_split)
            {
                leader = tile.leader_rank_rg_rg;
                my_rank = Env::rank_rg;
            }
            else
            {
                leader = tile.leader_rank_rg;
                my_rank = Env::rank;
            }
            if(leader == my_rank)
            {
                if(tiling_type == Tiling_type::_1D_ROW)
                {   
                    uint32_t vo = 0;
                    auto &v_seg = V->segments[o];
                    auto *v_data = (Fractional_Type *) v_seg.D->data;
                    Integer_Type v_nitems = v_seg.D->n;
                    Integer_Type v_nbytes = v_seg.D->nbytes;
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        v_data[i] = (*f)(0, y_data[i], 0, 0);
                    }
                }
                else if((tiling_type == Tiling_type::_2D_)
                     or (tiling_type == Tiling_type::_1D_COL))
                {
                    MPI_Request request;
                    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                    {                        
                        if(Env::comm_split)
                        {   
                            follower = A->follower_rowgrp_ranks_rg[j];
                            accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
                            auto &yj_seg = Yp->segments[accu];
                            auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                            Integer_Type yj_nitems = yj_seg.D->n;
                            Integer_Type yj_nbytes = yj_seg.D->nbytes;
                            MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, Env::rowgrps_comm, &request);
                            in_requests.push_back(request);
                        }
                        else
                        {
                            follower = A->follower_rowgrp_ranks[j];
                            accu = A->follower_rowgrp_ranks_accu_seg[j];
                            auto &yj_seg = Yp->segments[accu];
                            auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                            Integer_Type yj_nitems = yj_seg.D->n;
                            Integer_Type yj_nbytes = yj_seg.D->nbytes;
                            MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, Env::MPI_WORLD, &request);
                            in_requests.push_back(request);
                            //printf("Recv t=%d l=%d me=%d f=%d a=%d la=%d\n", t, leader, my_rank, follower, accu, A->accu_segment_rg);
                        }
                    }
                }
                yk = yi;
            }
            else
            {
                MPI_Request request;
                if(Env::comm_split)
                {
                    MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, Env::rowgrps_comm, &request);
                    out_requests.push_back(request);
                }
                else
                {
                    MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                    //printf("Send t=%d l=%d me=%d\n", t, leader, my_rank);
                }
            }
            xi = 0;
            yi++;
        }
    }
    t2 = Env::clock();
    if(!Env::rank)
        printf("Combine Tile processing time: %f\n", t2 - t1);
    //Env::barrier();
    
    if((tiling_type == Tiling_type::_2D_)
        or (tiling_type == Tiling_type::_1D_COL))
    {
        /* Better way of waiting for ncoming messages using MPI_Waitsome */
        /*
        t1 = Env::clock();
        auto *Yp = Y[yk];
        yo = A->accu_segment_rg;
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        
        int32_t incount = in_requests.size();
        int32_t outcount = 0;
        int32_t incounts = rowgrp_nranks - 1;
        std::vector<MPI_Status> statuses(incounts);
        std::vector<int32_t> indices(incounts);
        uint32_t received = 0;
        uint32_t r = 0;
        int32_t index = 0;
        
        while(received < incount)
        {
            MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
            assert(outcount != MPI_UNDEFINED);
            //MPI_Waitany(in_requests.size(), in_requests.data(), &index, statuses.data());
            //outcount = 1;
            for(uint32_t i = 0; i < outcount; i++)
            {
                
                uint32_t j = indices[i];
                //MPI_Status status;
                //MPI_Wait(&in_requests[j], &status);
                 //if(!Env::rank)
                 //   printf(">> %d %d\n", i, j);   
                if(Env::comm_split)
                    accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
                else
                    accu = A->follower_rowgrp_ranks_accu_seg[j];
                

                auto &yj_seg = Yp->segments[accu];
                auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                Integer_Type yj_nitems = yj_seg.D->n;
                Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
               // MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, tags[j], Env::rowgrps_comm, &in_requests[j]);
                
                //MPI_Recv(yj_data, yj_nbytes, MPI_BYTE, follower, tags[j], Env::rowgrps_comm, &status);
                
                for(uint32_t k = 0; k < yj_nitems; k++)
                {
                    if(yj_data[k])
                        y_data[k] += yj_data[k];
                }

                
            }
            received += outcount;
        }
        in_requests.clear();
        t2 = Env::clock();
        if(!Env::rank)
            printf("Combine MPI_Waitsome for in_req: %f\n", t2 - t1);
        
        t1 = Env::clock();
        uint32_t vo = 0;
        auto &v_seg = V->segments[vo];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        Integer_Type v_nbytes = v_seg.D->nbytes;
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            
            v_data[i] = (*f)(0, y_data[i], 0, 0);
        }
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
        
        
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        Env::barrier();
        t2 = Env::clock();
        if(!Env::rank)
            printf("Combine Copy v to y time: %f\n", t2 - t1);
        */
        /*
        auto *Yp = Y[yk];
        yo = A->accu_segment_rg;
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        
        int32_t incount = in_requests.size();
        int32_t outcount = 0;
        int32_t incounts = rowgrp_nranks - 1;
        std::vector<MPI_Status> statuses(incounts);
        std::vector<int32_t> indices(incounts);
        uint32_t received = 0;
        uint32_t r = 0;
        t1 = Env::clock();
        while(received < incount)
        {
            
            MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
            assert(outcount != MPI_UNDEFINED);
            //Env::barrier();
            if(outcount > 0)
            {
                for(uint32_t j = 0; j < outcount; j++)
                {
                    if(Env::comm_split)
                        accu = A->follower_rowgrp_ranks_accu_seg_rg[indices[j]];
                    else
                        accu = A->follower_rowgrp_ranks_accu_seg[indices[j]];
                    
                    follower = accu = A->follower_rowgrp_ranks_rg[indices[j]];
                    auto &yj_seg = Yp->segments[accu];
                    auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                    Integer_Type yj_nitems = yj_seg.D->n;
                    Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
                    //MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, tags[indices[j]], Env::rowgrps_comm, &in_requests[indices[j]]);
                    for(uint32_t i = 0; i < yj_nitems; i++)
                    {
                        if(yj_data[i])
                            y_data[i] += yj_data[i];
                    }
                }
                received += outcount;
            }
        }
        in_requests.clear();
        t2 = Env::clock();
        if(!Env::rank)
            printf("Combine MPI_Waitsome for in_req: %f\n", t2 - t1);
        

        
        t1 = Env::clock();
        uint32_t vo = 0;
        auto &v_seg = V->segments[vo];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        Integer_Type v_nbytes = v_seg.D->nbytes;
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            
            v_data[i] = (*f)(0, y_data[i], 0, 0);
        }
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
        
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        //Env::barrier();
        t2 = Env::clock();
        if(!Env::rank)
            printf("Combine Copy v to y time: %f\n", t2 - t1);
        */
        
        
        
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        Env::barrier();
        auto *Yp = Y[yk];
        yo = A->accu_segment_rg;
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        {
            if(Env::comm_split)
                accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = A->follower_rowgrp_ranks_accu_seg[j];
            auto &yj_seg = Yp->segments[accu];
            auto *yj_data = (Fractional_Type *) yj_seg.D->data;
            Integer_Type yj_nitems = yj_seg.D->n;
            Integer_Type yj_nbytes = yj_seg.D->nbytes;                        
            for(uint32_t i = 0; i < yj_nitems; i++)
            {
                if(yj_data[i])
                    y_data[i] += yj_data[i];
            }
        }
        
        uint32_t vo = 0;
        auto &v_seg = V->segments[vo];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        Integer_Type v_nbytes = v_seg.D->nbytes;
        //Fractional_Type tol = 1e-5;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            //Fractional_Type tmp = y_data[i];
            
            v_data[i] = (*f)(0, y_data[i], 0, 0); 
            
            //if(fabs(v_data[i] - tmp) > tol)
            //    printf("Converged\n");
            //printf("%d %d %f\n", Env::rank, i, (fabs(v_data[i] - tmp) > tol));
            //nedges_local += nedges_local;
        }
        
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
        

//        Env::barrier();         
        
        //Env::barrier();

        //print(v_seg);
        
        
        //Env::barrier();
        //MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        //if(!Env::rank)
          //  printf("VP: %lu\n", nedges_global);
        
        
        
        
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::checksum()
{
    uint64_t v_sum_local = 0, v_sum_gloabl = 0;
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
        
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        v_sum_local += v_data[i];
        //if(i == 0)
        //printf("%d %f\n", Env::rank, v_data[i]);
    }
    
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Degree checksum: %lu\n", v_sum_gloabl);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::checksumPR()
{
    double v_local = 0, v_gloabl = 0;
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
        
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        v_local += v_data[i];
    }
    
    MPI_Allreduce(&v_local, &v_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Value checksum: %f\n", v_gloabl); 
    
    
    uint64_t s_local = 0, s_gloabl = 0;
    uint32_t so = 0;
    auto &s_seg = S->segments[so];
    auto *s_data = (Fractional_Type *) s_seg.D->data;
    Integer_Type s_nitems = s_seg.D->n;
    Integer_Type s_nbytes = s_seg.D->nbytes;
    
    for(uint32_t i = 0; i < s_nitems; i++)
    {
        s_local += s_data[i];
    }
    
    
    MPI_Allreduce(&s_local, &s_gloabl, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Score checksum: %lu\n", s_gloabl);
    

    uint32_t NUM = 31;
    uint32_t count = v_nitems < NUM ? v_nitems : NUM;
    //std::vector<double> v_nums(count);
    //std::vector<double> v_sum_nums(count);
    //std::vector<double> nums(count);
    //std::vector<double> snums(count);
    //for (uint32_t i = 0; i < nrowgrps; i++)
    //{
        /*
        for (uint32_t j = 0; j < A->tiling->ncolgrps; j++)  
        {
            auto &tile = A->tiles[0][j];
            if(tile.rank == Env::rank)
            {
                //uint32_t vo = 0;
                for(uint32_t i = 0; i < count; i++)
                {
                    //nums[i] += v_data[i];
                    printf("r=%d i=%d v=%f\n", Env::rank, i, v_data[i]);
                }
                
            }
        }
        */
    //}
    //MPI_Allreduce(nums.data(), snums.data(), count, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(!Env::rank)
    {
        for(uint32_t i = 0; i < count; i++)
        {
            //MPI_Allreduce(&v_local, &v_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
            
            //pair.row = i;
            //pair.col = 0;
            //auto pair1 = A->base(pair, A->owned_segment, A->owned_segment);
            printf("Rank[%d],Value[%d]=%f,Score[%d]=%f\n",  Env::rank, i, v_data[i], i, s_data[i]);
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
        auto pair1 = A->base(pair, A->owned_segment, A->owned_segment);
        printf("R(%d),V[%d]=%f\n",  Env::rank, pair1.row, data[i]);
    } 
}

#endif
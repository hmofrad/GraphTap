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
        
        //void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s);
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Vertex_Program<Weight,
                  Integer_Type, Fractional_Type> *VProgram);
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s,
                            Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram = nullptr);                  
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void bcast(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void checksum();
        void free();
    
    protected:
        void print(Segment<Weight, Integer_Type, Fractional_Type> &segment);
        void apply();
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src);
        
        Order_type order;
        Integer_Type nelems;
        uint32_t rank_ngrps, grp_nranks;
        std::vector<int32_t> accu_segment_vec, all_ranks, all_accu_segment_vec;
        std::vector<int32_t> follower_grp_ranks_cg, follower_grp_ranks_accu_seg_cg;
        std::vector<int32_t> follower_grp_ranks_rg, follower_grp_ranks_accu_seg_rg;
        int32_t owned_segment, accu_segment;
        std::vector<uint32_t> local_tiles_order;
        std::vector<int32_t> local_segments;
        
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
    owned_segment = A->owned_segment;
    order = order_;

    if(order == _ROW_)
    {
        nelems = A->nrows;
        rank_ngrps = A->tiling->rank_nrowgrps;
        grp_nranks = A->tiling->rowgrp_nranks;
        accu_segment = A->accu_segment_rg;
        accu_segment_vec = A->accu_segment_rg_vec;
        
        all_ranks = A->all_rowgrp_ranks;
        all_accu_segment_vec = A->all_rowgrp_ranks_accu_seg;
        /*
        all_ranks_ = A->all_colgrp_ranks;
        all_accu_segment_vec_ = A->all_colgrp_ranks_accu_seg;
        follower_grp_ranks_rg = A->follower_rowgrp_ranks_rg;
        follower_grp_ranks_accu_seg_rg = A->follower_rowgrp_ranks_accu_seg_rg;
        follower_grp_ranks_accu_seg_cg = A->follower_rowgrp_ranks_accu_seg_cg;
        */
        local_tiles_order = A->local_tiles_row_order;
        local_segments = A->local_col_segments;
    }
    else if (order == _COL_)
    {
        nelems = A->ncols;
        rank_ngrps = A->tiling->rank_ncolgrps;
        grp_nranks = A->tiling->colgrp_nranks;
        accu_segment = A->accu_segment_cg;
        accu_segment_vec = A->accu_segment_cg_vec;
        all_ranks = A->all_colgrp_ranks;
        all_accu_segment_vec = A->all_colgrp_ranks_accu_seg;

        /*
        
        follower_grp_ranks_rg = A->follower_colgrp_ranks_rg;
        follower_grp_ranks_accu_seg_rg = A->follower_colgrp_ranks_accu_seg_rg;
        
        follower_grp_ranks_cg = A->follower_colgrp_ranks_cg;
        follower_grp_ranks_accu_seg_cg = A->follower_colgrp_ranks_accu_seg_cg;
        */
        local_tiles_order = A->local_tiles_col_order;
        local_segments = A->local_col_segments;
        //If we wanted to work with the transpose
        //local_segments = A->local_row_segments;
    }   
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

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(
        Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value)
{
    for(uint32_t i; i < vec->vector_length; i++)
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
    for(uint32_t i; i < vec_src->vector_length; i++)
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
    }
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, Fractional_Type y, 
     Fractional_Type v, Fractional_Type s, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows,  A->local_col_segments);
    populate(X, x);
    
    V = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->accu_segment_rg_vec);
    populate(V, v);
    
    S = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->accu_segment_rg_vec);
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
        bool vec_add = (((tile_th + 1) % A->tiling->rank_ncolgrps) == 0);
        if(vec_add)
        {
            bool vec_owner = (pair_idx == A->owned_segment);
            
            if(vec_owner)
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->all_rowgrp_ranks_accu_seg);
            }
            else
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->accu_segment_rg_vec);
            }
            Y.push_back(Y_);
        }
        
    }
    
    /*
    for(uint32_t t: local_tiles_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        uint32_t tile_th;
        uint32_t pair_idx;
        if(order == _ROW_)
        {
            tile_th = tile.nth;
            pair_idx = pair.row;
        }
        else if(order == _COL_)
        {
            tile_th = tile.mth;
            pair_idx = pair.col;
        }
        
        bool vec_add = (((tile_th + 1) % rank_ngrps) == 0);
        if(vec_add)
        {
            bool vec_owner = (pair_idx == owned_segment);
            
            if(vec_owner)
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(nelems, all_accu_segment_vec);
            }
            else
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(nelems, accu_segment_vec);
            }
            Y.push_back(Y_);
        }
    }
    */
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
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_ROW))
    {
        for(uint32_t j = 0; j < A->tiling->colgrp_nranks - 1; j++)
        {
            uint32_t follower = A->follower_colgrp_ranks[j];
            if(Env::comm_split)
            {
                follower = A->follower_colgrp_ranks_cg[j];
                MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, 0, Env::colgrps_comm, &request);
            }
            else
            {
                follower = A->follower_colgrp_ranks[j];
                MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, 0, Env::MPI_WORLD, &request);
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
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::gather()
{   
    uint32_t o = 0;
    uint32_t leader;
    MPI_Request request;
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_ROW))
    {    
        for(uint32_t i = 0; i < A->tiling->rank_ncolgrps; i++)
        {
            auto& xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nitems = xj_seg.D->n;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;

            
           // if(Env::rank == 4)
            //{
                //printf("[%d %d %d %d %d] ", i, xj_seg.g, A->leader_ranks[xj_seg.g], (A->leader_ranks[xj_seg.g] == Env::rank), A->owned_segment);
                //printf("[%d %d %d %d] ", i, xj_seg.g, A->leader_ranks_cg[xj_seg.g], (A->leader_ranks_cg[xj_seg.g] == Env::rank_cg));
            //}
            
            if(Env::comm_split)
            {
                leader = A->leader_ranks_cg[A->local_col_segments[i]];
                if(leader != Env::rank_cg)
                //if(A->leader_ranks_cg[xj_seg.g] != Env::rank_cg)
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks_cg[xj_seg.g], 0, Env::colgrps_comm, &request);
            }
            else
            {
                leader = A->leader_ranks[A->local_col_segments[i]];
                if(leader != Env::rank)
                //if(A->leader_ranks[xj_seg.g] != Env::rank)
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks[xj_seg.g], 0, Env::MPI_WORLD, &request);
            }
            
            //if(((Env::rank_cg == 1) or (Env::rank_cg == 2) or (Env::rank_cg == 3)))
                //printf("<< cg=%d rcg=%d lrcg=%d\n", i, Env::rank_cg, A->leader_ranks_cg[xj_seg.g]);
        }
          //  if(Env::rank == 4)
            //    printf("\n");
        
        //for(uint32_t s: A->local_col_segments)
       // {
            //auto& xj_seg = X->segments[i];
            
            //if(Env::rank == 4)
            //    printf("[%d %d] ", s, A->leader_ranks[s]);
            
            
            //if(xj_seg.g != accu_segment)
            //{
                //auto *xj_data = (Fractional_Type *) xj_seg.D->data;
               // Integer_Type xj_nbytes = xj_seg.D->nbytes;
                /*
                if(Env::comm_split)
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks_cg[xj_seg.g], 0, Env::colgrps_comm, &request);
                else
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks[xj_seg.g], 0, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
                */
            //}
        //}
        //if(Env::rank == 4)
          //  printf("\n");
        
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        Env::barrier();
    /*    
    if(!Env::rank)
    {
        for(uint32_t i = 0; i < A->tiling->rank_ncolgrps; i++)
        {
            auto& xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nitems = xj_seg.D->n;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
            //if(A->leader_ranks_cg[xj_seg.g] != Env::rank_cg)
            //{
                printf("%d %f\n", i, xj_data[0]);
            //for(uint32_t j = 0; j < xj_nitems; j++)
              //  printf("%d %f\n", i, xj_data[j]);
            //}
        }
    }
    */
        
        
        
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t leader;
    uint32_t xo = A->accu_segment_rg;
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
    
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_ROW))
    {
        //uint32_t leader = x_seg.leader_rank;
        //uint32_t leader_cg = x_seg.leader_rank_cg;
        //uint32_t xi = X->owned_segment;
    //if(Env::rank == 0){
        //accu_segment_rg_vec
        
        for(uint32_t i = 0; i < A->tiling->rank_ncolgrps; i++)
        {
            leader = A->leader_ranks_cg[A->local_col_segments[i]];
            xo = A->accu_segment_cg;
            auto &xj_seg = X->segments[xo];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
            if(Env::comm_split)
                MPI_Bcast(xj_data, xj_nbytes, MPI_BYTE, leader, Env::colgrps_comm);
            else
            {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
            //A->leader_ranks_cg[xj_seg.g]
            
            
            //if(Env::rank == 0)
              //  printf("%d %d %d %d %d\n", i, xo, A->leader_ranks_cg[xj_seg.g], A->local_col_segments[i], A->leader_ranks_cg[A->local_col_segments[i]]);
            /*
            auto& xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nitems = xj_seg.D->n;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
        
            if(Env::comm_split)
            {
                if(A->leader_ranks_cg[xj_seg.g] != Env::rank_cg)
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks_cg[xj_seg.g], 0, Env::colgrps_comm, &request);
            }
            else
            {
                if(A->leader_ranks[xj_seg.g] != Env::rank)
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, A->leader_ranks[xj_seg.g], 0, Env::MPI_WORLD, &request);
            }
            */
        }
            
        
        
        
        /*
        for(uint32_t s: A->local_col_segments)
        {
            auto& xj_seg = X->segments[];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
            //printf("rank=%d rank_cg=%d leader_cg=%d seg=%d\n", Env::rank, Env::rank_cg, xj_seg.leader_rank_cg, s);
            MPI_Bcast(xj_data, xj_nbytes, MPI_BYTE, xj_seg.leader_rank_cg, Env::colgrps_comm);
            //Env::barrier();
            //
            //
            //if(xi == s)
            //    printf("send: %d ", s);
            //else
            //    printf("recv: %d ", s);
        }
        */
        //printf(" %d %d %d %d\n", Env::rank, xi, Env::rank_cg, leader_cg);
    //}
        //for(uint32_t j = 0; j < A->tiling->colgrp_nranks - 1; j++)
        //{
            //uint32_t other_rank = A->follower_colgrp_ranks[j];
            //MPI_Bcast(x_data, x_nbytes, MPI_BYTE, Env::rank, Env::colgrps_comm)
        //}
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
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    bool vec_add = (((tile_th + 1) % A->tiling->rank_ncolgrps) == 0);
    MPI_Request request;
    uint32_t xi, y = 0, yi = 0, yo = 0, o = 0, xo = 0, yj, yk, si, vi;
    for(uint32_t t: A->local_tiles_row_order)
    {
       // if(!Env::rank)
         //   printf("%d \n", t);
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
        
        uint32_t xo = A->accu_segment_rg;
        auto &x_seg = X->segments[xo];
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
                            y_data[ROW_INDEX[i]] += x_data[j];   
                        #endif
                    }
                }
            }            
        }

        communication = (((tile_th + 1) % A->tiling->rank_ncolgrps) == 0);
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

                if(A->tiling->tiling_type == Tiling_type::_1D_ROW)
                {   
                    auto &v_seg = V->segments[o];
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
                        if(Env::comm_split)
                        {   
                            follower = A->follower_rowgrp_ranks_rg[j];
                            accu = A->follower_rowgrp_ranks_accu_seg_rg[j];
                            //if(!Env::rank)
                              //  printf("t=%d, f=%d a=%d\n", t, follower, accu);
                            auto &yj_seg = Yp->segments[accu];
                            auto *yj_data = (Fractional_Type *) y_seg.D->data;
                            Integer_Type yj_nitems = yj_seg.D->n;
                            Integer_Type yj_nbytes = yj_seg.D->nbytes;
                            MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, Env::rowgrps_comm, &request);
                        }
                        else
                        {
                            follower = A->follower_rowgrp_ranks[j];
                            accu = A->follower_rowgrp_ranks_accu_seg[j];
                            auto &yj_seg = Yp->segments[accu];
                            auto *yj_data = (Fractional_Type *) y_seg.D->data;
                            Integer_Type yj_nitems = yj_seg.D->n;
                            Integer_Type yj_nbytes = yj_seg.D->nbytes;
                            MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, Env::MPI_WORLD, &request);
                        }
                        in_requests.push_back(request);
                    }
                }
                yk = yi;
            }
            else
            {
                if(Env::comm_split)
                    MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, Env::rowgrps_comm, &request);
                else
                    MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            yi++;
        }
    }
    
    if((A->tiling->tiling_type == Tiling_type::_2D_)
        or (A->tiling->tiling_type == Tiling_type::_1D_COL))
    {
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        
        auto *Yp = Y[yk];
        yo = A->accu_segment_rg;
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        int32_t incount = in_requests.size();
        int32_t outcount = 0;
        int32_t incounts = A->tiling->rowgrp_nranks - 1;
        std::vector<MPI_Status> statuses(incounts);
        std::vector<int> indices(incounts);
        uint32_t received = 0;
        
        double t1 = Env::clock();
        while(received < incount)
        {
            MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
            assert(outcount != MPI_UNDEFINED);
            if(outcount > 0)
            {
                for(uint32_t j = 0; j < outcount; j++)
                {
                    if(Env::comm_split)
                        accu = A->follower_rowgrp_ranks_accu_seg_rg[indices[j]];
                    else
                        accu = A->follower_rowgrp_ranks_accu_seg[indices[j]];
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
                received += outcount;
            }
        }
        in_requests.clear();
        double t2 = Env::clock();
        if(!Env::rank)
            printf("Combine MPI_Waitsome for in_req: %f\n", t2 - t1);
        
        
        
        /*
        for(uint32_t j = 0; j < A->tiling->rowgrp_nranks - 1; j++)
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
                y_data[i] += yj_data[i];
            }
            memset(yj_data, 0, yj_nbytes);
        }
        */
        uint32_t so = 0;
        auto &v_seg = V->segments[so];
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
        Env::barrier();
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
    }
    
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Degree checksum: %lu\n", v_sum_gloabl);
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

#endif
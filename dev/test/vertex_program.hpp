/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP
 
#include "vector.hpp"
 
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
        void combine();
        void bcast(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void checksum();
        void checksumPR();
        void free();
        void filter();
        //void combine1();
        void apply(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
    
    protected:
        void spmv(Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
                  Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &c_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &j_seg,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void print(Segment<Weight, Integer_Type, Fractional_Type> &segment);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src);
        void clear(Vector<Weight, Integer_Type, Fractional_Type> *vec);
        void wait_for_all();
        void optimized_1d_row();
        void optimized_1d_col();
        void optimized_2d();
        
        struct Triple<Weight, Integer_Type> tile_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                       struct Triple<Weight, Integer_Type> &pair);
        
        struct Triple<Weight, Integer_Type> leader_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);        
        
        Order_type order;
        Tiling_type tiling_type;
        Compression_type compression;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nrowgrps, rank_ncolgrps;
        Integer_Type tile_height, tile_width;
        int32_t owned_segment, accu_segment_rg, accu_segment_cg, accu_segment_row, accu_segment_col;
        std::vector<int32_t> local_col_segments;
        std::vector<int32_t> accu_segment_col_vec;
        std::vector<int32_t> accu_segment_row_vec;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        std::vector<int32_t> accu_segment_rg_vec;
        std::vector<int32_t> local_row_segments;
        std::vector<int32_t> follower_rowgrp_ranks_rg;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> follower_colgrp_ranks_cg;
        std::vector<int32_t> follower_colgrp_ranks;
        std::vector<uint32_t> leader_ranks;
        std::vector<uint32_t> leader_ranks_cg;
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;
        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        MPI_Comm rowgrps_communicator;
        MPI_Comm colgrps_communicator;
        MPI_Comm communicator;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        Vector<Weight, Integer_Type, Fractional_Type> *X;
        Vector<Weight, Integer_Type, Fractional_Type> *V;
        Vector<Weight, Integer_Type, Fractional_Type> *S;
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y;
        Vector<Weight, Integer_Type, Integer_Type> *C;
        Vector<Weight, Integer_Type, Integer_Type> *J;
        Vector<Weight, Integer_Type, Integer_Type> *R;
        Vector<Weight, Integer_Type, Integer_Type> *I;
        
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight,
                       Integer_Type, Fractional_Type> &Graph, Order_type order_)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
    //E = static_cast<Vector<Weight, Integer_Type, Integer_Type> *> (Graph.A->E);
    C = Graph.A->C;
    J = Graph.A->J;
    order = order_;
    tiling_type = A->tiling->tiling_type;
    compression = A->compression;
    owned_segment = A->owned_segment;
    leader_ranks = A->leader_ranks;

    if(order == _ROW_)
    {
        rowgrp_nranks = A->tiling->rowgrp_nranks;
        colgrp_nranks = A->tiling->colgrp_nranks;
        rank_nrowgrps = A->tiling->rank_nrowgrps;
        rank_ncolgrps = A->tiling->rank_ncolgrps;
        tile_height = A->tile_height;
        local_row_segments = A->local_row_segments;
        local_col_segments = A->local_col_segments;
        accu_segment_col = A->accu_segment_col;
        accu_segment_row = A->accu_segment_row;
        accu_segment_row_vec = A->accu_segment_row_vec;
        accu_segment_col_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_rowgrp_ranks_accu_seg;
        accu_segment_rg_vec = A->accu_segment_rg_vec;
        accu_segment_rg = A->accu_segment_rg;
        accu_segment_cg = A->accu_segment_cg;
        follower_rowgrp_ranks_rg = A->follower_rowgrp_ranks_rg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_rowgrp_ranks_accu_seg_rg;
        leader_ranks_cg = A->leader_ranks_cg;
        follower_colgrp_ranks_cg = A->follower_colgrp_ranks_cg;
        follower_colgrp_ranks = A->follower_colgrp_ranks;
        local_tiles_row_order = A->local_tiles_row_order;
        local_tiles_col_order = A->local_tiles_col_order;
        follower_rowgrp_ranks = A->follower_rowgrp_ranks;
        follower_rowgrp_ranks_accu_seg = A->follower_rowgrp_ranks_accu_seg;
        rowgrps_communicator = Env::rowgrps_comm;
        colgrps_communicator = Env::colgrps_comm;
    }
    else if (order == _COL_)
    {
        
        rowgrp_nranks = A->tiling->colgrp_nranks;
        colgrp_nranks = A->tiling->rowgrp_nranks;
        rank_nrowgrps = A->tiling->rank_ncolgrps;
        rank_ncolgrps = A->tiling->rank_nrowgrps;
        tile_height = A->tile_width;
        local_row_segments = A->local_col_segments;
        local_col_segments = A->local_row_segments;
        accu_segment_col = A->accu_segment_row;
        accu_segment_row = A->accu_segment_col;
        accu_segment_col_vec = A->accu_segment_row_vec;
        accu_segment_row_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_colgrp_ranks_accu_seg;
        accu_segment_rg_vec = A->accu_segment_cg_vec;
        accu_segment_rg = A->accu_segment_cg;
        accu_segment_cg = A->accu_segment_rg;
        follower_rowgrp_ranks_rg = A->follower_colgrp_ranks_cg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_colgrp_ranks_accu_seg_cg;
        leader_ranks_cg = A->leader_ranks_rg;
        follower_colgrp_ranks_cg = A->follower_rowgrp_ranks_rg;
        follower_colgrp_ranks = A->follower_rowgrp_ranks;
        local_tiles_row_order = A->local_tiles_col_order;
        local_tiles_col_order = A->local_tiles_row_order;
        follower_rowgrp_ranks = A->follower_colgrp_ranks;
        follower_rowgrp_ranks_accu_seg = A->follower_colgrp_ranks_accu_seg;
        rowgrps_communicator = Env::colgrps_comm;
        colgrps_communicator = Env::rowgrps_comm;
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
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, Fractional_Type y, 
     Fractional_Type v, Fractional_Type s, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    //printf("**************\n");
    //nnz_col_sizes_loc
    //accu_segment_col
    /*
    if(-1)
    {
        for(uint32_t j = 0; j < rank_ncolgrps; j++)
        {
            printf("%d %d %d %d\n", j, A->nnz_col_sizes_loc[j], accu_segment_col, A->nnz_col_sizes_loc[accu_segment_col]);
        }
        
        for(uint32_t j = 0; j < rowgrp_nranks; j++)
        {
            printf("[%d %d %d %d] ", A->all_rowgrp_ranks[j], all_rowgrp_ranks_accu_seg[j], j, accu_segment_row);
        }
        printf("\n");
       
        
    }
    */
    
    //X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nnz_row_sizes_loc,  local_row_segments);
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nnz_col_sizes_loc,  local_col_segments);
    populate(X, x);
    
    std::vector<Integer_Type> tile_height_sizes(1, tile_height);
    V = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height_sizes, accu_segment_col_vec);
    //std::vector<Integer_Type> nnz_col_size_accu_segment_col(1, A->nnz_col_sizes_loc[accu_segment_col]);
    //V = new Vector<Weight, Integer_Type, Fractional_Type>(nnz_col_size_accu_segment_col, accu_segment_col_vec);
    populate(V, v);
    S = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height_sizes, accu_segment_col_vec);
    //S = new Vector<Weight, Integer_Type, Fractional_Type>(nnz_col_size_accu_segment_col, accu_segment_col_vec);
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
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        if(local_row_segments[j] == owned_segment)
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height, all_rowgrp_ranks_accu_seg);
        else
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height, accu_segment_row_vec);
            //Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height, accu_segment_rg_vec);
        Y.push_back(Y_);
    }
    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    //uint32_t xo = A->accu_segment_cg;
    uint32_t xo = accu_segment_col;
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
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and order == _COL_))
    {
        for(uint32_t i = 0; i < colgrp_nranks - 1; i++)
        {
            if(Env::comm_split)
            {
               follower = follower_colgrp_ranks_cg[i];
               MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, x_seg.g, colgrps_communicator, &request);
               out_requests.push_back(request);
            }
            else
            {
                follower = follower_colgrp_ranks[i];
                MPI_Isend(x_data, x_nbytes, MPI_BYTE, follower, x_seg.g, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
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
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and order == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            auto& xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nitems = xj_seg.D->n;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;

            if(Env::comm_split)
            {
                leader = leader_ranks_cg[xj_seg.g];
                if(leader != Env::rank_cg)
                {
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, leader, xj_seg.g, colgrps_communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                leader = leader_ranks[xj_seg.g];
                if(leader != Env::rank)
                {
                    MPI_Irecv(xj_data, xj_nbytes, MPI_BYTE, leader, xj_seg.g, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
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
    //uint32_t xo = A->accu_segment_cg;
    uint32_t xo = accu_segment_col;
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
    
    uint32_t co = accu_segment_col;
    auto &c_seg = C->segments[co];
    auto *c_data = (Integer_Type *) c_seg.D->data;
    Integer_Type c_nitems = c_seg.D->n;
    Integer_Type c_nbytes = c_seg.D->nbytes;
    
    uint32_t jo = accu_segment_col;
    auto &j_seg = J->segments[jo];
    auto *j_data = (Integer_Type *) j_seg.D->data;
    Integer_Type j_nitems = j_seg.D->n;
    Integer_Type j_nbytes = j_seg.D->nbytes;
    
    
    for(uint32_t i = 0; i < c_nitems; i++)
    {
        x_data[i] = (*f)(0, 0, v_data[c_data[i]], s_data[c_data[i]]);
    }
    
    //printf("%d %d %d %d\n", Env::rank, c_nitems, tile_height, x_nitems );
    
    //if(!Env::rank)
    //    printf(">>>>>>>%d %d\n", owned_segment, x_nitems);
    
    if((tiling_type == Tiling_type::_2D_)
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and order == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            
            leader = leader_ranks_cg[local_col_segments[i]];
            auto &xj_seg = X->segments[i];
            auto *xj_data = (Fractional_Type *) xj_seg.D->data;
            Integer_Type xj_nbytes = xj_seg.D->nbytes;
            if(Env::comm_split)
                MPI_Bcast(xj_data, xj_nbytes, MPI_BYTE, leader, colgrps_communicator);
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
    
    
    /*
    if(Env::rank == 0)
    {
        for(uint32_t j = 0; j < rank_ncolgrps; j++)
        {
            auto &ej_seg = E->segments[j];
            auto *ej_data = (Integer_Type *) ej_seg.D->data;
            //Integer_Type *ej_data = (Integer_Type *) ej_seg.D->data;
            
            Integer_Type ej_nitems = ej_seg.D->n;
            Integer_Type ej_nbytes = ej_seg.D->nbytes;  
            
            auto &x_seg = X->segments[j];
            auto *x_data = (Fractional_Type *) x_seg.D->data;
            Integer_Type x_nitems = x_seg.D->n;
            Integer_Type x_nbytes = x_seg.D->nbytes;
            
            //printf("%d %d %d\n", j, x_nitems, ej_nitems);
            
         //   printf("%d %d %d %d\n", j, A->nnz_col_sizes_loc[j], accu_segment_col, A->nnz_col_sizes_loc[accu_segment_col]);
            for(uint32_t i = 0; i < ej_nitems; i++)
            {
               printf("a[%d]=%f\n", ej_data[i], x_data[i]);
            }
        }
        
        
        
    }    
    */
    
}                   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
            Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
            Segment<Weight, Integer_Type, Integer_Type> &c_seg,
            Segment<Weight, Integer_Type, Integer_Type> &j_seg,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    
    auto *y_data = (Fractional_Type *) y_seg.D->data;
    Integer_Type y_nitems = y_seg.D->n;
    Integer_Type y_nbytes = y_seg.D->nbytes;

    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    auto *c_data = (Integer_Type *) c_seg.D->data;
    Integer_Type c_nitems = c_seg.D->n;
    Integer_Type c_nbytes = c_seg.D->nbytes;
    
    auto *j_data = (Integer_Type *) j_seg.D->data;
    Integer_Type j_nitems = j_seg.D->n;
    Integer_Type j_nbytes = j_seg.D->nbytes;
    
    //printf("c=%d j=%d x=%d y=%d\n", c_nitems, j_nitems, x_nitems, y_nitems);
    /*
    printf("??\n");
    
    if(Env::rank == 0)
    {
        if(tile.allocated)
        {
            Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            Integer_Type nnz_per_col;
            Integer_Type k = 0;
            for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
            {
                
                for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                {
                    //ROW_INDEX[i] == e_data[k];
                    printf("col=%d i=%d row=%d e=%f\n", j, i, ROW_INDEX[i], x_data[i_data[ROW_INDEX[i]]]);
                    //k++;
                    
                    //if(x_data[ROW_INDEX[i]])
                    //y_data[j] += x_data[ROW_INDEX[i]];
                }    
                if(j == e_data[k])
                    k++;                
            }
        }
    }
    
    
    Env::finalize();
    exit(0); 
    */
    if(tile.allocated)
    {
        if(compression == Compression_type::_CSR_)
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A->data;
            #endif
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
            Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
            Integer_Type nnz_per_row;
            if(order == _ROW_)
            {
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {       
                        #ifdef HAS_WEIGHT
                        if(x_data[JA[j]] and A[j])
                            y_data[i] += A[j] * x_data[JA[j]];
                        #else
                        //if(x_data[JA[j]])
                        //    y_data[i] += x_data[JA[j]];
                        if(x_data[j_data[JA[j]]])
                            y_data[i] += x_data[j_data[JA[j]]];
                    
                        #endif                        
                    }
                }
            }
            else if(order == _COL_)
            {
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {       
                        #ifdef HAS_WEIGHT
                        if(x_data[i] and A[j])
                            y_data[JA[j]] += A[j] * x_data[i];
                        #else
                        if(x_data[i])
                            y_data[JA[j]] += x_data[i];
                        #endif                        
                    }
                }
            }
        }
        else if(compression == Compression_type::_CSC_)    
        {
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            Integer_Type nnz_per_col;
            if(order == _ROW_)
            {
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        #ifdef HAS_WEIGHT
                        if(x_data[j] and VAL[i])
                            y_data[ROW_INDEX[i]] += VAL[i] * x_data[j];   
                        #else
                        //if(x_data[j])
                        //    y_data[ROW_INDEX[i]] += x_data[j];
                        
                        if( j != c_data[j_data[j]])
                            printf("r=%d j=%d jd=%d c_d=%d ?=%d\n", Env::rank, j, j_data[j], c_data[j_data[j]], j == c_data[j_data[j]]);
                        assert(j == c_data[j_data[j]]);
                        if(x_data[j_data[j]])
                            y_data[ROW_INDEX[i]] += x_data[j_data[j]];
                        
                        
                        #endif
                    }
                }
            }
            else if(order == _COL_)
            {
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        #ifdef HAS_WEIGHT
                        if(x_data[ROW_INDEX[i]] and VAL[i])
                            y_data[j] += VAL[i] * x_data[ROW_INDEX[i]];   
                        #else
                        //if(x_data[ROW_INDEX[i]])
                            //y_data[j] += x_data[ROW_INDEX[i]];
                        if(x_data[j_data[ROW_INDEX[i]]])
                            y_data[j] += x_data[j_data[ROW_INDEX[i]]];
                        
                        #endif
                    }
                }
            }
        }            
    }    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine()
{
    if(tiling_type == Tiling_type::_1D_ROW)
    {  
        if(order == _ROW_)
        {
            optimized_1d_row();
        }
        else if(order == _COL_)
        {
            optimized_1d_col();
        }
    }
    else if(tiling_type == Tiling_type::_1D_COL)
    {
        if(order == _ROW_)
        {
            optimized_1d_col();    
        }
        else if(order == _COL_)
        {
            optimized_1d_row();    
        }
    }
    else if(tiling_type == Tiling_type::_2D_)
    {
        optimized_2d();    
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_row()
{
    uint32_t yi = accu_segment_row;
    uint32_t xi = 0;
    uint32_t yo = accu_segment_rg;
  
    /*
    if(order == _COL_)
    {
        yi = accu_segment_col;
        yo = accu_segment_cg;
    }
    
    */
    // SPMV
    for(uint32_t t: local_tiles_row_order)
    {
        if(!Env::rank)
        printf("rank=%d t=%d xi =%d yi=%d yo=%d \n", Env::rank, t, xi, yi, yo);

        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        //printf("%d %d %d %d %d %d\n", Env::rank, t, yi, yo, accu_segment_col, accu_segment_cg);    
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        auto &x_seg = X->segments[xi];
        
        auto &c_seg = C->segments[xi];
        
        auto &j_seg = J->segments[xi];
        
        spmv(y_seg, x_seg, c_seg, j_seg, tile);

        xi++;
    }
    //printf("!!!!\n");
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_col()
{
    //MPI_Comm communicator;
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    //double t1, t2;
    //t1 = Env::clock();
    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        
        //tile_th = tile.mth;
        //pair_idx = pair.row;
        //vec_owner = (pair_idx == owned_segment);
        
        
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        auto &x_seg = X->segments[xi];
        
        auto &c_seg = C->segments[xi];
        
        auto &j_seg = J->segments[xi];
        
        spmv(y_seg, x_seg, c_seg, j_seg, tile);
        
        yi++;
        
        //if(!Env::rank)
        //    printf(">>t=%d, com=%d as=%d\n", t, communication, A->accu_segment_rg);
    }
    //t2 = Env::clock();
    
    yi = 0, xi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        
        //tile_th = tile.nth;
        //pair_idx = pair.row;
        //vec_owner = (pair_idx == owned_segment);
        
        
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;
        
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            /*
            if(Env::comm_split)
            {
                leader = tile.leader_rank_rg_rg;
                my_rank = Env::rank_rg;
                communicator = rowgrps_communicator;
            }
            else
            {
                leader = tile.leader_rank_rg;
                my_rank = Env::rank;
                communicator = Env::MPI_WORLD;
            }
            */
            if(leader == my_rank)
            {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                {
                    if(Env::comm_split)
                    {
                        follower = follower_rowgrp_ranks_rg[j];
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    }
                    else
                    {
                        follower = follower_rowgrp_ranks[j];
                        accu = follower_rowgrp_ranks_accu_seg[j];
                    }
                    
                    auto &yj_seg = Yp->segments[accu];
                    auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                    Integer_Type yj_nitems = yj_seg.D->n;
                    Integer_Type yj_nbytes = yj_seg.D->nbytes;
                    MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
            }
            yi++;
        }
    }
    
    wait_for_all();
}
                   
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_2d()
{
    //struct Triple<Weight, Integer_Type> pair, pair1;
    //MPI_Comm communicator;
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    //double t1, t2;
    //t1 = Env::clock();
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        
        /*
        tile_th = tile.nth;
        pair_idx = pair.row;
        vec_owner = (pair_idx == owned_segment);
        */

        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        auto &x_seg = X->segments[xi];
        
        auto &c_seg = C->segments[xi];
        
        auto &j_seg = J->segments[xi];
        
        spmv(y_seg, x_seg, c_seg, j_seg, tile);

        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            
            /*
            if(Env::comm_split)
            {
                leader = tile.leader_rank_rg_rg;
                my_rank = Env::rank_rg;
                communicator = rowgrps_communicator;
            }
            else
            {
                leader = tile.leader_rank_rg;
                my_rank = Env::rank;
                communicator = Env::MPI_WORLD;
            }
            */
            if(leader == my_rank)
            {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                {                        
                    if(Env::comm_split)
                    {   
                        follower = follower_rowgrp_ranks_rg[j];
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    }
                    else
                    {
                        follower = follower_rowgrp_ranks[j];
                        accu = follower_rowgrp_ranks_accu_seg[j];
                    }                
                    
                    auto &yj_seg = Yp->segments[accu];
                    auto *yj_data = (Fractional_Type *) yj_seg.D->data;
                    Integer_Type yj_nitems = yj_seg.D->n;
                    Integer_Type yj_nbytes = yj_seg.D->nbytes;
                    MPI_Irecv(yj_data, yj_nbytes, MPI_BYTE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                }
                //yk = yi;
            }
            else
            {
                MPI_Isend(y_data, y_nbytes, MPI_BYTE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
            }
            xi = 0;
            yi++;
        }
    }
    wait_for_all();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::apply(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    auto *Yp = Y[yi];
    uint32_t yo = accu_segment_rg;
    auto &y_seg = Yp->segments[yo];
    auto *y_data = (Fractional_Type *) y_seg.D->data;
    Integer_Type y_nitems = y_seg.D->n;
    Integer_Type y_nbytes = y_seg.D->nbytes;
    
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];
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
    
    uint32_t co = accu_segment_col;
    auto &c_seg = C->segments[co];
    auto *c_data = (Integer_Type *) c_seg.D->data;
    Integer_Type c_nitems = c_seg.D->n;
    Integer_Type c_nbytes = c_seg.D->nbytes;
    
    uint32_t jo = accu_segment_col;
    auto &j_seg = J->segments[jo];
    auto *j_data = (Integer_Type *) j_seg.D->data;
    Integer_Type j_nitems = j_seg.D->n;
    Integer_Type j_nbytes = j_seg.D->nbytes;
    
    uint32_t vo = 0;
    auto &v_seg = V->segments[vo];
    auto *v_data = (Fractional_Type *) v_seg.D->data;
    Integer_Type v_nitems = v_seg.D->n;
    Integer_Type v_nbytes = v_seg.D->nbytes;
    
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        v_data[i] = (*f)(0, y_data[i], 0, 0); 
        //printf("%d %d\n", i, e_data[i]);
    }
    //printf("%d %d %d\n", Env::rank, v_nitems, e_nitems);
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        clear(Y[i]);
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_all()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type>::
        tile_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  struct Triple<Weight, Integer_Type> &pair)
{
    Integer_Type item1, item2;
    if(order == _ROW_)
    {
        item1 = tile.nth;
        item2 = pair.row;
        //return{tile.nth, pair.row};
    }
    else if(order == _COL_)
    {
        item1 = tile.mth;
        item2 = pair.col;
        //return{tile.mth, pair.col};
    }    
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type>::
        leader_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    Integer_Type item1, item2;
    if(order == _ROW_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_rg_rg;
            item2 = Env::rank_rg;
            communicator = rowgrps_communicator;
            //return{tile.leader_rank_rg_rg, Env::rank_rg};
        }
        else
        {
            item1 = tile.leader_rank_rg;
            item2 = Env::rank;
            communicator = Env::MPI_WORLD;
            //return{tile.leader_rank_rg, Env::rank};
        }
    }
    else if(order == _COL_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_cg_cg;
            item2 = Env::rank_cg;
            communicator = rowgrps_communicator;
            //return{tile.leader_rank_rg_rg, Env::rank_rg};
        }
        else
        {
            item1 = tile.leader_rank_cg;
            item2 = Env::rank;
            communicator = Env::MPI_WORLD;
            //return{tile.leader_rank_rg, Env::rank};
        }
    }
    return{item1, item2};
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
    
    uint32_t co = accu_segment_col;
    auto &c_seg = C->segments[co];
    auto *c_data = (Integer_Type *) c_seg.D->data;
    Integer_Type c_nitems = c_seg.D->n;
    Integer_Type c_nbytes = c_seg.D->nbytes;
    
    uint32_t jo = accu_segment_col;
    auto &j_seg = J->segments[jo];
    auto *j_data = (Integer_Type *) j_seg.D->data;
    Integer_Type j_nitems = j_seg.D->n;
    Integer_Type j_nbytes = j_seg.D->nbytes;
    
    v_sum_local = 0;  
if(c_seg.allocated)
{    
    for(uint32_t i = 0; i < v_nitems; i++)
    {
       v_sum_local += v_data[i];
       // if(!Env::rank)
            //printf("i=%d v=%f j=%d c=%d v=%f\n", i, v_data[i], j_data[i], c_data[j_data[i]], v_data[c_data[j_data[i]]]);
/*
        if(i == c_data[j_data[i]])
        {
            //printf("i=%d v=%f j=%d c=%d v=%f\n", i, v_data[i], j_data[i], c_data[j_data[i]], v_data[c_data[j_data[i]]]);
            assert(v_data[i] == v_data[c_data[j_data[i]]]);
            v_sum_local += v_data[c_data[j_data[i]]];
        }
        */
        //if(i == 0)
            
        //printf("%d %f\n", Env::rank, v_data[i]);
    }
}
    //printf("%d %lu\n", Env::rank, v_sum_local);
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Degree checksum: %lu\n", v_sum_gloabl);
    /*
    v_sum_local = 0, v_sum_gloabl = 0;
    
    

    
    
    for(uint32_t i = 0; i < c_nitems; i++)
    {
        v_sum_local += v_data[c_data[i]];
        //if(!Env::rank)
        //    printf("%d %d %f\n", i, c_data[i], v_data[c_data[i]]);
    }
    
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Degree checksum1: %lu\n", v_sum_gloabl);
    */
    
    
    
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
    if(!Env::rank)
    {
        Triple<Weight, Integer_Type> pair, pair1;
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            printf("Rank[%d],Value[%d]=%f,Score[%d]=%f\n",  Env::rank, pair1.row, v_data[i], pair1.row, s_data[i]);
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
        auto pair1 = A->base(pair, owned_segment, owned_segment);
        printf("R(%d),V[%d]=%f\n",  Env::rank, pair1.row, data[i]);
    } 
}

#endif
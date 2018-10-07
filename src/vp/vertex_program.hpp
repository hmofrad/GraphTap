/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP
 
#include "ds/vector.hpp"
#include "mpi/types.hpp" 
#include "mpi/comm.hpp" 

enum Ordering_type
{
  _ROW_,
  _COL_
};   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vertex_Program
{
    public:
        Vertex_Program();
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph, bool stationary_ = false, Ordering_type = _ROW_);
        ~Vertex_Program();
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s,
                            Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram = nullptr);                  
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine();
        void bcast(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void checksum();
        void display();
        void free();
        void apply(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
    
    protected:
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  std::vector<std::vector<Integer_Type>> &z_data);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value);
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src);
        void clear(Vector<Weight, Integer_Type, Fractional_Type> *vec);
        void wait_for_all();
        void wait_for_sends();
        void wait_for_recvs();
        void optimized_1d_row();
        void optimized_1d_col();
        void optimized_2d();
        
        struct Triple<Weight, Integer_Type> tile_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                       struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> leader_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);  
        MPI_Comm communicator_info();  
                       
        bool stationary;
        Ordering_type ordering_type;
        Tiling_type tiling_type;
        Compression_type compression_type;
        Filtering_type filtering_type;
        Integer_Type nrows, ncols;
        uint32_t nrowgrps, ncolgrps;
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
        std::vector<int32_t> all_rowgrp_ranks;
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
        //MPI_Comm communicator;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        std::vector<MPI_Request> out_requests_;
        std::vector<MPI_Request> in_requests_;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
        
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        
        Vector<Weight, Integer_Type, Fractional_Type> *X;
        Vector<Weight, Integer_Type, Fractional_Type> *V;
        Vector<Weight, Integer_Type, Fractional_Type> *S;
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y;

        Vector<Weight, Integer_Type, char> *I;
        Vector<Weight, Integer_Type, Integer_Type> *IV;
        Vector<Weight, Integer_Type, char> *J;
        Vector<Weight, Integer_Type, Integer_Type> *JV;
        
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        
        MPI_Datatype TYPE_DOUBLE;
        MPI_Datatype TYPE_INT;
        
        //std::vector<std::vector<Integer_Type>> Q;
        std::vector<std::vector<Integer_Type>> W;
        std::vector<std::vector<Integer_Type>> R;
        std::vector<std::vector<std::vector<Integer_Type>>> D;
        std::vector<std::vector<Integer_Type>> D_SIZE;
        std::vector<std::vector<std::vector<Integer_Type>>> Z;
        std::vector<std::vector<std::vector<Integer_Type>>> Z_SIZE;
        std::vector<std::vector<Integer_Type>> inboxes;
        std::vector<std::vector<Integer_Type>> outboxes;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight,
                       Integer_Type, Fractional_Type> &Graph, bool stationary_, Ordering_type ordering_type_)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
    stationary = stationary_;
    ordering_type = ordering_type_;
    tiling_type = A->tiling->tiling_type;
    compression_type = A->compression_type;
    filtering_type = A->filtering_type;
    owned_segment = A->owned_segment;
    leader_ranks = A->leader_ranks;

    if(ordering_type == _ROW_)
    {
        nrows = A->nrows;
        ncols = A->ncols;
        nrowgrps = A->nrowgrps;
        ncolgrps = A->ncolgrps;
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
        all_rowgrp_ranks = A->all_rowgrp_ranks;
        nnz_row_sizes_loc = A->nnz_row_sizes_loc;
        nnz_col_sizes_loc = A->nnz_col_sizes_loc;
        I = Graph.A->I;
        IV= Graph.A->IV;
        J = Graph.A->J;
        JV= Graph.A->JV;
    }
    else if (ordering_type == _COL_)
    {
        nrows = A->ncols;
        ncols = A->nrows;
        nrowgrps = A->ncolgrps;
        ncolgrps = A->nrowgrps;
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
        all_rowgrp_ranks = A->all_colgrp_ranks;
        nnz_row_sizes_loc = A->nnz_col_sizes_loc;
        nnz_col_sizes_loc = A->nnz_row_sizes_loc;
        I = Graph.A->J;
        IV= Graph.A->JV;
        J = Graph.A->I;
        JV= Graph.A->IV;
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::free()
{
    delete X;
    delete V;
    delete S;   
    
    for (uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        delete Y[j];
    }   
    
    Y.clear();
    Y.shrink_to_fit();
    
    
    if(not stationary)
    {
        for (uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            Z[j].clear();
            Z[j].shrink_to_fit();
        }   
        Z.clear();
        Z.shrink_to_fit();
        
        W.clear();
        W.shrink_to_fit();
        
        R.clear();
        R.shrink_to_fit();
        
        for (uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            Z_SIZE[j].clear();
            Z_SIZE[j].shrink_to_fit();
        }
        Z_SIZE.clear();
        Z_SIZE.shrink_to_fit();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::clear(
                               Vector<Weight, Integer_Type, Fractional_Type> *vec)
{
    for(uint32_t i = 0; i < vec->vector_length; i++)
    {
        if(vec->allocated[i])
        {
            Fractional_Type *data = (Fractional_Type *) vec->data[i];
            Integer_Type nitems = vec->nitems[i];
            uint64_t nbytes = nitems * sizeof(Fractional_Type);
            memset(data, 0, nbytes);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(
        Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value)
{
    for(uint32_t i = 0; i < vec->vector_length; i++)
    {
        if(vec->allocated[i])
        {
            Fractional_Type *data = (Fractional_Type *) vec->data[i];
            Integer_Type nitems = vec->nitems[i];
            uint64_t nbytes = nitems * sizeof(Fractional_Type);            

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
}         

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src)
{
    for(uint32_t i = 0; i < vec_src->vector_length; i++)
    {        
        Fractional_Type *data_src = (Fractional_Type *) vec_src->data[i];
        Integer_Type nitems_src = vec_src->nitems[i];
        uint64_t nbytes_src = nitems_src * sizeof(Fractional_Type);
        
        Fractional_Type *data_dst = (Fractional_Type *) vec_dst->data[i];
        Integer_Type nitems_dst = vec_dst->nitems[i];
        uint64_t nbytes_dst = nitems_dst * sizeof(Fractional_Type);
        
        memcpy(data_dst, data_src, nbytes_src);
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, Fractional_Type y, 
     Fractional_Type v, Fractional_Type s, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    double t1, t2;
    t1 = Env::clock();
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    std::vector<Integer_Type> x_sizes;
    
    if(filtering_type == _NONE_)
    {
        x_sizes.resize(rank_ncolgrps, tile_height);
    }
    else if(filtering_type == _SOME_)
    {
        x_sizes = nnz_col_sizes_loc;
    }
    
    X = new Vector<Weight, Integer_Type, Fractional_Type>(x_sizes,  local_col_segments);
    populate(X, x);

    std::vector<Integer_Type> v_s_size = {tile_height};

    V = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    populate(V, v);

    S = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
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
    
    std::vector<Integer_Type> y_size;
    std::vector<Integer_Type> y_sizes;
    std::vector<Integer_Type> yy_sizes;
    Vector<Weight, Integer_Type, Fractional_Type> *Y_;  

    if(filtering_type == _NONE_)
    {
        y_sizes.resize(rank_nrowgrps, tile_height);
    }
    else if(filtering_type == _SOME_) 
    {
        y_sizes = nnz_row_sizes_loc;
    }     
    
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {  
        if(local_row_segments[j] == owned_segment)
        {
            yy_sizes.resize(rowgrp_nranks, y_sizes[j]);
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(yy_sizes, all_rowgrp_ranks_accu_seg);
            yy_sizes.clear();
        }
        else
        {
            
            y_size = {y_sizes[j]};
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
            y_size.clear();
        }
        populate(Y_, y);
        Y.push_back(Y_);
    }    
    

    if(not stationary)
    {
        Z.resize(rank_nrowgrps);
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {  
            Z[j].resize(tile_height);
        }  
        
        W.resize(tile_height);
        
        if(VProgram)
        {
            Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
            std::vector<std::vector<Integer_Type>> W_ = VP->W;
            R = W_;
            //D.resize(tile_height * Env::nranks); 
            D.resize(nrowgrps);
            for(uint32_t i = 0; i < nrowgrps; i++)
                D[i].resize(tile_height);
            
            D_SIZE.resize(nrowgrps);
            for(uint32_t i = 0; i < nrowgrps; i++)
                D_SIZE[i].resize(tile_height + 1);
        }

        Z_SIZE.resize(rank_nrowgrps);
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            if(local_row_segments[j] == owned_segment)
            {
                yy_sizes.resize(rowgrp_nranks, y_sizes[j]);
                Z_SIZE[j].resize(rowgrp_nranks);
                for(uint32_t i = 0; i < rowgrp_nranks; i++)
                    Z_SIZE[j][i].resize(yy_sizes[i] + 1); // 0 + nitems
            }
            else
            {
                y_size = {y_sizes[j]};
                Z_SIZE[j].resize(1);
                Z_SIZE[j][0].resize(y_size[0] + 1); // 0 + nitems
            }        
        }
        inboxes.resize(rowgrp_nranks - 1);
        outboxes.resize(rowgrp_nranks - 1);
    }

    t2 = Env::clock();
    Env::print_time("Init", t2 - t1);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    double t1, t2;
    t1 = Env::clock();
     
    uint32_t xo = accu_segment_col;
    Fractional_Type *x_data = (Fractional_Type *) X->data[xo];
    Integer_Type x_nitems = X->nitems[xo];
    int32_t col_group = X->local_segments[xo];

    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];

    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];

    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
        }
    }
    else if(filtering_type == _SOME_)
    {
        
        char *j_data = (char *) J->data[xo];
        Integer_Type j_nitems = J->nitems[xo];

        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                x_data[j] = (*f)(0, 0, v_data[i], s_data[i]);
                j++;
            }               
        }
    }
    
    MPI_Request request;
    uint32_t follower, accu;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < colgrp_nranks - 1; i++)
        {
            if(Env::comm_split)
            {
               follower = follower_colgrp_ranks_cg[i];
               MPI_Isend(x_data, x_nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
               out_requests.push_back(request);
            }
            else
            {
                follower = follower_colgrp_ranks[i];
                MPI_Isend(x_data, x_nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
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
    
    t2 = Env::clock();
    Env::print_time("Scatter", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::gather()
{  
    double t1, t2;
    t1 = Env::clock();
     
    uint32_t leader;
    MPI_Request request;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            Fractional_Type *xj_data = (Fractional_Type *) X->data[i];
            Integer_Type xj_nitems = X->nitems[i];
            int32_t col_group = X->local_segments[i];
            if(Env::comm_split)
            {
                leader = leader_ranks_cg[col_group];
                if(leader != Env::rank_cg)
                {
                    MPI_Irecv(xj_data, xj_nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                leader = leader_ranks[col_group];
                if(leader != Env::rank)
                {
                    MPI_Irecv(xj_data, xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }
            
        }
        
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
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
    
    t2 = Env::clock();
    Env::print_time("Gather", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    double t1, t2;
    t1 = Env::clock();
    
    uint32_t leader;
    uint32_t xo = accu_segment_col;
    Fractional_Type *x_data = (Fractional_Type *) X->data[xo];
    Integer_Type x_nitems = X->nitems[xo];

    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];

    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];
    
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
        }
    }
    else if(filtering_type == _SOME_)
    {
        char *j_data = (char *) J->data[xo];
        Integer_Type j_nitems = J->nitems[xo];
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                x_data[j] = (*f)(0, 0, v_data[i], s_data[i]);
                j++;
            }               
        }
    }
    
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            leader = leader_ranks_cg[local_col_segments[i]];
            Fractional_Type *xj_data = (Fractional_Type *) X->data[i];
            Integer_Type xj_nitems = X->nitems[i];
            if(Env::comm_split)
            {
                MPI_Bcast(xj_data, xj_nitems, TYPE_DOUBLE, leader, colgrps_communicator);
            }
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
    
    t2 = Env::clock();
    Env::print_time("Bcast", t2 - t1);
}                   


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Fractional_Type *y_data, Fractional_Type *x_data,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<std::vector<Integer_Type>> &z_data)
{
    Triple<Weight, Integer_Type> pair;
    if(tile.allocated)
    {
        if(compression_type == Compression_type::_CSC_)    
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csc->A;
            #endif
            
            Integer_Type *IA = (Integer_Type *) tile.csc->IA; // ROW_INDEX
            Integer_Type *JA   = (Integer_Type *) tile.csc->JA; // COL_PTR
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
            {
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                {
                    #ifdef HAS_WEIGHT
                    if(x_data[j] and A[i])
                        y_data[IA[i]] += A[i] * x_data[j];   
                    #else
                    if(x_data[j])    
                        y_data[IA[i]] += x_data[j];
                    #endif
                    pair = A->base({IA[i] + (owned_segment * tile_height), j}, tile.rg, tile.cg);
                    z_data[IA[i]].push_back(pair.col);
                    //printf("<%d %d>\n", Env::rank, z_data[IA[i]].back());
                }
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Fractional_Type *y_data, Fractional_Type *x_data,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    if(tile.allocated)
    {
        if(compression_type == Compression_type::_CSR_)
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A;
            #endif
            Integer_Type *IA = (Integer_Type *) tile.csr->IA;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA;
            Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
            if(ordering_type == _ROW_)
            {
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
            else if(ordering_type == _COL_)
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
        else if(compression_type == Compression_type::_CSC_)    
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csc->A;
            #endif
            
            Integer_Type *IA = (Integer_Type *) tile.csc->IA; // ROW_INDEX
            Integer_Type *JA   = (Integer_Type *) tile.csc->JA; // COL_PTR
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            if(ordering_type == _ROW_)
            {
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                    {
                        #ifdef HAS_WEIGHT
                        if(x_data[j] and A[i])
                            y_data[IA[i]] += A[i] * x_data[j];   
                        #else
                        if(x_data[j])    
                            y_data[IA[i]] += x_data[j];
                        #endif
                    }
                }
            }
            else if(ordering_type == _COL_)
            {
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                    {
                        #ifdef HAS_WEIGHT
                        if(x_data[IA[i]] and A[i])
                            y_data[j] += A[i] * x_data[IA[i]];   
                        #else
                        if(x_data[IA[i]])
                            y_data[j] += x_data[IA[i]];
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
    double t1, t2;
    t1 = Env::clock();
    if(tiling_type == Tiling_type::_1D_ROW)
    {  
        if(ordering_type == _ROW_)
        {
            optimized_1d_row();
        }
        else if(ordering_type == _COL_)
        {
            optimized_1d_col();
        }
    }
    else if(tiling_type == Tiling_type::_1D_COL)
    {
        if(ordering_type == _ROW_)
        {
            optimized_1d_col();    
        }
        else if(ordering_type == _COL_)
        {
            optimized_1d_row();    
        }
    }
    else if((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
    {
        optimized_2d();    
    }
    t2 = Env::clock();
    Env::print_time("Combine", t2 - t1);
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_row()
{
    uint32_t yi = accu_segment_row;
    uint32_t xi = 0;
    uint32_t yo = accu_segment_rg;
  
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);
        xi++;
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_col()
{
    MPI_Request request;
    MPI_Status status;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);
        yi++;
    }
    
    yi = 0, xi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];

        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            auto pair2 = leader_info(tile);
            MPI_Comm communicator = communicator_info();
            leader = pair2.row;
            my_rank = pair2.col;
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
                    Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                    Integer_Type yj_nitems = Yp->nitems[accu];
                    MPI_Irecv(yj_data, yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);                   
                }
            }
            else
            {
                MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
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
    MPI_Request request;
    MPI_Status status;
    int flag, count;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0, oi = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        std::vector<std::vector<Integer_Type>> &z_data = Z[yi];  
        
        if(stationary)
            spmv(y_data, x_data, tile);
        else
            spmv(y_data, x_data, tile, z_data);
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
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
                    
                    if(stationary)
                    {
                        Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                        Integer_Type yj_nitems = Yp->nitems[accu];
                        MPI_Irecv(yj_data, yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                        in_requests.push_back(request);
                    }
                    else
                    {
                        std::vector<Integer_Type> &zj_size = Z_SIZE[yi][accu];
                        Integer_Type sj_s_nitems = zj_size.size();
                        //auto &inbox = inboxes[j];
                        
                        
                        MPI_Recv(zj_size.data(), sj_s_nitems, TYPE_INT, follower, pair_idx, communicator, &status);
                        //Comm<Weight, Integer_Type, Fractional_Type>::unpack_adjacency(zj_size, inbox);
                        auto &inbox = inboxes[j];
                        Integer_Type inbox_nitems = zj_size[sj_s_nitems - 1];
                        inbox.resize(inbox_nitems);
                        MPI_Irecv(inbox.data(), inbox.size(), TYPE_INT, follower, pair_idx, communicator, &request);
                        in_requests_.push_back(request);
                    }
                }
            }
            else
            {
                if(stationary)
                {
                    MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                    out_requests.push_back(request);
                }
                else
                {
                    std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
                    std::vector<Integer_Type> &z_size = Z_SIZE[yi][yo];
                    Integer_Type z_s_nitems = z_size.size();
                    auto &outbox = outboxes[oi];
                    
                    Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(z_size, z_data, outbox);
                    /*
                    
                    
                    
                    z_size[0] = 0;
                    for(Integer_Type i = 1; i < z_s_nitems; i++)
                    {
                        z_size[i] = z_size[i-1] + z_data[i-1].size();
                    } 
                    */
                    MPI_Send(z_size.data(), z_s_nitems, TYPE_INT, leader, pair_idx, communicator);
                    /*
                    Integer_Type outbox_nitems = z_size[z_s_nitems - 1];    
                    auto &outbox = outboxes[oi];
                    outbox.resize(outbox_nitems);
                    Integer_Type k = 0;
                    //std::vector<std::vector<Integer_Type>> &z_data = Z[yi]; 
                    for(Integer_Type i = 0; i < z_s_nitems - 1; i++)
                    {
                        for(auto j: z_data[i])
                        {
                            outbox[k] = j;
                            k++;
                        }
                    }
                    */
                    MPI_Isend(outbox.data(), outbox.size(), TYPE_INT, leader, pair_idx, communicator, &request);
                    out_requests_.push_back(request);  
                    oi++;
                }
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
    double t1, t2;
    t1 = Env::clock();
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;

    if(stationary)
    {
        auto *Yp = Y[yi];

        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
                   
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            
            Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
            Integer_Type yj_nitems = Yp->nitems[accu];

            for(uint32_t i = 0; i < yj_nitems; i++)
            {
                if(yj_data[i])
                    y_data[i] += yj_data[i];
            }    
        }
        
        uint32_t vo = 0;
        Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
        Integer_Type v_nitems = V->nitems[vo];

        if(filtering_type == _NONE_)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                v_data[i] = (*f)(0, y_data[i], 0, 0); 
            }
        }
        else if(filtering_type == _SOME_)
        {
            char *i_data = (char *) I->data[yi];        
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                if(i_data[i])
                {
                    v_data[i] = (*f)(0, y_data[j], 0, 0);
                    j++;
                }
                else
                    v_data[i] = (*f)(0, 0, 0, 0);
            }
        }
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
    }
    else
    {
        std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            
            
            std::vector<Integer_Type> &zj_size = Z_SIZE[yi][accu];
            //Integer_Type zj_nitems = zj_size.size() - 1;
            std::vector<Integer_Type> &inbox = inboxes[j];
            
            Comm<Weight, Integer_Type, Fractional_Type>::unpack_adjacency(zj_size, z_data, inbox);
            /*   
            for(uint32_t i = 0; i < zj_nitems; i++)
            {
                for(uint32_t j = zj_size[i]; j < zj_size[i+1]; j++)
                {
                    z_data[i].push_back(inbox[j]);  
                }
            }
            */
            inbox.clear();
            inbox.shrink_to_fit();
        }
        
        //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
        std::vector<std::vector<Integer_Type>> &w_data = W;
        w_data = z_data;        
        
        /*
        if(filtering_type == _NONE_)
        {
            std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
            std::vector<std::vector<Integer_Type>> &w_data = W;
            w_data = z_data;
        }
        else if(filtering_type == _SOME_)
        {
            fprintf(stderr, "Invalid filtering type\n");
            Env::exit(1);
        }
        */
        /*
        Env::barrier();
        if(Env::rank == -1)
        {
            std::vector<std::vector<Integer_Type>> &w_data = W;
            Integer_Type w_nitems = W.size();
            for(uint32_t i = 0; i < w_nitems; i++)
            {
                printf("wi=%d: ", i);
                auto &w_d = w_data[i];
                for(uint32_t j: w_d)
                    printf("%d ", j);
                printf("\n");
            }
            printf("\n");
            
            std::vector<std::vector<Integer_Type>> &r_data = R;
            Integer_Type r_nitems = R.size();
            for(uint32_t i = 0; i < r_nitems; i++)
            {
                printf("ri=%d: ", i);
                auto &r_d = r_data[i];
                for(uint32_t j: r_d)
                    printf("%d ", j);
                printf("\n");
            }
        }
        Env::barrier();  
        */

        

        if(R.size())
        {
            
            std::vector<std::vector<Integer_Type>> &w_data = W;
            Integer_Type w_nitems = w_data.size();
            std::vector<std::vector<Integer_Type>> &r_data = R;
            Integer_Type r_nitems = r_data.size();        
            D[owned_segment] = R;
            //yi = owned_segment;
            //std::vector<MPI_Request> out_requests;
            //std::vector<MPI_Request> in_requests;
            MPI_Request request;
            MPI_Status status;
            uint32_t my_rank, leader, follower, accu;
            Triple<Weight, Integer_Type> triple, triple1, pair;
            
            
            //D.resize(nrows);
            //printf("%d %d\n", Env::rank, owned_segment);
            //Env::barrier();
            //if(Env::rank == 1)
            //{
            
            //uint32_t my_row = Env::rank;
            
            
            
            
            //std::vector<std::vector<Integer_Type>> D_SIZE(nrowgrps);
            //for(uint32_t i = 0; i < nrowgrps; i++)
            //    D_SIZE[i].resize(r_nitems + 1);
            std::vector<std::vector<Integer_Type>> boxes(nrows);
            std::vector<Integer_Type> &d_size = D_SIZE[owned_segment];
            Integer_Type d_s_nitems = d_size.size();
            auto &outbox = boxes[owned_segment];
            Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(d_size, r_data, outbox);
            
            /*
            Integer_Type d_s_nitems = d_size.size();
            d_size[0] = 0;
            for(Integer_Type i = 1; i < d_s_nitems; i++)
            {
                d_size[i] = d_size[i-1] + r_data[i-1].size();
            }   
            
            Integer_Type outbox_nitems = d_size[d_s_nitems - 1];
            std::vector<std::vector<Integer_Type>> boxes(nrows);
            auto &outbox = boxes[owned_segment];
            outbox.resize(outbox_nitems);
            Integer_Type k = 0;
            for(Integer_Type i = 0; i < d_s_nitems - 1; i++)
            {
                for(auto j: r_data[i])
                {
                    outbox[k] = j;
                    k++;
                    //if(Env::rank == 1)
                    //    printf("%d ", j);
                }
            }
            */
            

            for(uint32_t i = 0; i < nrowgrps; i++)  
            {
                uint32_t r = (Env::rank + i) % Env::nranks;
                if(r != Env::rank)
                {
                    //printf("Send[%d --> %d]acc=%d size=%d num=%d\n", Env::rank, r, i, d_s_nitems, d_size[d_s_nitems - 1]);
                    MPI_Isend(d_size.data(), d_s_nitems, TYPE_INT, r, 0, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            
            for(uint32_t i = 0; i < nrowgrps; i++)  
            {
                leader = leader_ranks[i];
                my_rank = Env::rank;
                //uint32_t r = (Env::rank + i) % Env::nranks;
                //if(r != Env::rank)
                if(my_rank != leader)
                {
                    std::vector<Integer_Type> &dj_size = D_SIZE[i];
                    Integer_Type dj_s_nitems = dj_size.size();
                    //printf("Recv[%d <-- %d]accu=%d size=%d\n", Env::rank, leader, i, dj_s_nitems);
                    MPI_Irecv(dj_size.data(), dj_s_nitems, TYPE_INT, leader, 0, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }
            MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
            in_requests.clear();
            out_requests.clear();      
            

            //if(Env::rank == 1)
            //    printf("\n");
 
            
            //Env::barrier();
            for(uint32_t i = 0; i < nrowgrps; i++)  
            {
                uint32_t r = (Env::rank + i) % Env::nranks;
                if(r != Env::rank)
                {
                    //printf("Send[%d --> %d]acc=%d size=%lu\n", Env::rank, r, i, outbox.size());
                    MPI_Isend(outbox.data(), outbox.size(), TYPE_INT, r, 0, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            
            for(uint32_t i = 0; i < nrowgrps; i++)  
            {
                leader = leader_ranks[i];
                my_rank = Env::rank;
                //uint32_t r = (Env::rank + i) % Env::nranks;
                //if(r != Env::rank)
                if(my_rank != leader)
                {
                    std::vector<Integer_Type> &dj_size = D_SIZE[i];
                    Integer_Type dj_s_nitems = dj_size.size();
                    
                    auto &inbox = boxes[i];
                    Integer_Type inbox_nitems = dj_size[dj_s_nitems - 1];
                    inbox.resize(inbox_nitems);
                    //printf("Recv[%d <-- %d]accu=%d size=%d\n", Env::rank, leader, i, inbox_nitems);
                    MPI_Irecv(inbox.data(), inbox.size(), TYPE_INT, leader, 0, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);   
                }
            }
            MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
            in_requests.clear();
            out_requests.clear(); 
            //Env::barrier();
            
            
            //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
            //Env::barrier();
            //std::vector<std::vector<Integer_Type>> D(tile_height * Env::nranks); 
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                //if(Env::rank)
                //    printf("%d ", i);
                if(i != owned_segment)
                {
                    auto &box = boxes[i];
                    std::vector<Integer_Type> &dj_size = D_SIZE[i];
                    std::vector<std::vector<Integer_Type>> &dj_data = D[i];
                    Integer_Type dj_nitems = dj_size.size();
                    for(uint32_t j = 0; j < dj_nitems - 1; j++)
                    {
                        for(uint32_t k = dj_size[j]; k < dj_size[j+1]; k++)
                        {
                            Integer_Type row = j + (i * tile_height);
                            //if(Env::rank)
                            //    printf("[%d %d]", row, box[k]);
                            //D[row].push_back(box[k]);
                            dj_data[j].push_back(box[k]);
                        }
                        
                        
                    }
                    box.clear();
                    box.shrink_to_fit();
                }
                //if(Env::rank)
                //   printf("\n");
                

            }
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                D_SIZE[i].clear();
                D_SIZE[i].shrink_to_fit();
            }
            
            
            Env::barrier();

            
            
            uint64_t num_triangles_local = 0;
            uint64_t num_triangles_global = 0;
            //int c = 0;
            /*
            for(uint32_t i = 0; i < w_nitems; i++)
            {
                for(uint32_t j = 0; j < W[i].size(); j++)
                {
                    //uint32_t l = R[i][j];
                    //uint32_t s = W[l].size() > R[i].size
                    for(uint32_t k = 0; k < R[i].size(); k++)
                    {
                        for(uint32_t l = 0; l < D[W[i][j]].size(); l++)
                        {
                            if(R[i][k] == D[W[i][j]][l])
                            {
                                num_triangles_local++;
                                //printf("%d %d %lu \n", R[i][k], D[W[i][j]][l], num_triangles_local);
                            }
                        }
                    }
                }
            }
            */
            /*
            for(uint32_t i = 0; i < W.size(); i++)
            {
                if(W[i].size())
                    std::sort(W[i].begin(), W[i].end());
            }
            
            
            for(uint32_t i = 0; i < R.size(); i++)
            {
                if(R[i].size())
                    std::sort(R[i].begin(), R[i].end());
            }
            
            for(uint32_t i = 0; i < D.size(); i++)
            {
                if(D[i].size())
                    std::sort(D[i].begin(), D[i].end());
            }
            */
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                std::vector<std::vector<Integer_Type>> &dj_data = D[i];
                for(uint32_t j = 0; j < dj_data.size(); j++)
                {
                    if(dj_data[j].size())
                        std::sort(dj_data[j].begin(), dj_data[j].end());
                }
            }
            /*
            Env::barrier();
            if(!Env::rank)
            {
                for(uint32_t i = 0; i < nrowgrps; i++)
                {
                    //printf("i=%d\n", i);
                    for(uint32_t j = 0; j < D[i].size(); j++)
                    {
                        printf("j=%d ", j);
                        for(uint32_t k = 0; k < D[i][j].size(); k++)
                            printf("%d ", D[i][j][k]);
                        printf("\n");
                    }
                    printf("\n");
                }
            }
            Env::barrier();
            */
            
            for(uint32_t i = 0; i < w_nitems; i++)
            {
                
                //uint32_t ii = i + (owned_segment * tile_height);
                //uint32_t ii = i
                for(uint32_t j = 0; j < W[i].size(); j++)
                {
                    uint32_t other_segment = W[i][j] / tile_height;
                    uint32_t jj = W[i][j] % tile_height;
                    std::vector<Integer_Type> &in_neighbors = D[owned_segment][i];
                    std::vector<Integer_Type> &out_neighbors = D[other_segment][jj];
                    
                    
                    
                    //if(Env::rank)
                    //    printf("i=%d ii=%d j=%d jj=%d owns=%d others=%d\n", i, ii, W[i][j], jj, owned_segment, other_segment);
                    //triple = {ii, W[i][j]};
                    //pair = A->tile_of_triple(triple);
                    
                    uint32_t it1 = 0, it2 = 0;
                    uint32_t it1_end = in_neighbors.size(); // message.neighbors[it1]
                    //uint32_t it1_end = D[ii].size(); // message.neighbors[it1]
                    //int it1_end = R[i].size(); // message.neighbors[it1]
                    uint32_t it2_end = out_neighbors.size(); //vertexprop.neighbors[it2]
                    //uint32_t it2_end = D[W[i][j]].size(); //vertexprop.neighbors[it2]
                    while (it1 != it1_end && it2 != it2_end){
                      if (in_neighbors[it1] == out_neighbors[it2]) {
                      //if (D[ii][it1] == D[W[i][j]][it2]) {
                      //if (R[i][it1] == D[W[i][j]][it2]) {
                        num_triangles_local++;
                        ++it1; ++it2;
                      //} else if (R[i][it1] < D[W[i][j]][it2]) {
                      } else if (in_neighbors[it1] < out_neighbors[it2]) {
                      //} else if (D[ii][it1] < D[W[i][j]][it2]) {
                        ++it1;
                      } else {
                        ++it2;
                      }
                    }
                }
                if(Env::is_master)
                {
                    if ((i & ((1L << 10) - 1L)) == 0)
                    {
                        printf("|");
                    }
                }
            }
            

            
            
            /*
            for(uint32_t i = 0; i < w_nitems; i++)
            {
                int ii = i + (owned_segment * tile_height);
                for(uint32_t j = 0; j < W[i].size(); j++)
                {
                    int it1 = 0, it2 = 0;
                    int it1_end = D[ii].size(); // message.neighbors[it1]
                    //int it1_end = R[i].size(); // message.neighbors[it1]
                    int it2_end = D[W[i][j]].size(); //vertexprop.neighbors[it2]
                    while (it1 != it1_end && it2 != it2_end){
                      if (D[ii][it1] == D[W[i][j]][it2]) {
                      //if (R[i][it1] == D[W[i][j]][it2]) {
                        num_triangles_local++;
                        ++it1; ++it2;
                      //} else if (R[i][it1] < D[W[i][j]][it2]) {
                      } else if (D[ii][it1] < D[W[i][j]][it2]) {
                        ++it1;
                      } else {
                        ++it2;
                      }
                    }
                }
                if(Env::is_master)
                {
                    if ((i & ((1L << 10) - 1L)) == 0)
                    {
                        printf("|");
                    }
                }
            }
            */
            MPI_Allreduce(&num_triangles_local, &num_triangles_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
            if(!Env::rank)
                printf("Num_triangles = %lu\n", num_triangles_global);   
            
            
            for (uint32_t i = 0; i < rank_nrowgrps; i++)
            {
                D[i].clear();
                D[i].shrink_to_fit();
            }
            
        }
        else
            R = W;
        
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            std::vector<std::vector<Integer_Type>> &zj_size = Z_SIZE[j];
            if(local_row_segments[j] == owned_segment)
            {
                for(uint32_t i = 0; i < rowgrp_nranks; i++)
                {
                    zj_size[i].clear();
                    zj_size[i].shrink_to_fit();
                }
            }
            else
            {
                zj_size[0].clear();
                zj_size[0].shrink_to_fit();
            }
        }
        
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        {
            auto &outbox = outboxes[j];
            outbox.clear();
            outbox.shrink_to_fit();
        }
    }

    t2 = Env::clock();
    Env::print_time("Apply", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_all()
{
    if(stationary)
    {
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        out_requests.clear();
    }
    else
    {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
        out_requests_.clear();
    }
    
    
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_recvs()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_sends()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type>::
        tile_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  struct Triple<Weight, Integer_Type> &pair)
{
    Integer_Type item1, item2;
    if(ordering_type == _ROW_)
    {
        item1 = tile.nth;
        item2 = pair.row;
    }
    else if(ordering_type == _COL_)
    {
        item1 = tile.mth;
        item2 = pair.col;
    }    
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type>::
        leader_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    Integer_Type item1, item2;
    if(ordering_type == _ROW_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_rg_rg;
            item2 = Env::rank_rg;
            //communicator = rowgrps_communicator;
        }
        else
        {
            item1 = tile.leader_rank_rg;
            item2 = Env::rank;
            //communicator = Env::MPI_WORLD;
        }
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_cg_cg;
            item2 = Env::rank_cg;
            //communicator = rowgrps_communicator;
        }
        else
        {
            item1 = tile.leader_rank_cg;
            item2 = Env::rank;
            //communicator = Env::MPI_WORLD;
        }
    }
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
MPI_Comm Vertex_Program<Weight, Integer_Type, Fractional_Type>::communicator_info()
{
    MPI_Comm comm;
    if(ordering_type == _ROW_)
    {
        if(Env::comm_split)
        {
            comm = rowgrps_communicator;
        }
        else
        {
            comm = Env::MPI_WORLD;
        }
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
        {
            comm = rowgrps_communicator;
        }
        else
        {
            comm = Env::MPI_WORLD;
        }
    }
    return{comm};
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::checksum()
{
    double v_sum_local = 0, v_sum_gloabl = 0;
    
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    v_sum_local = 0;  
    for(uint32_t i = 0; i < v_nitems; i++)
       v_sum_local += v_data[i];
   
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Value checksum: %f\n", v_sum_gloabl);
    
    uint64_t s_local = 0, s_gloabl = 0;
    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];

    for(uint32_t i = 0; i < s_nitems; i++)
    {
        s_local += s_data[i];
    }
    
    MPI_Allreduce(&s_local, &s_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Score checksum: %lu\n", s_gloabl);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::display()
{
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];
    
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
            printf("Rank[%d],Value[%2d]=%f,Score[%2d]=%f\n",  Env::rank, pair1.row, v_data[i], pair1.row, s_data[i]);
        }  
    }
}
#endif

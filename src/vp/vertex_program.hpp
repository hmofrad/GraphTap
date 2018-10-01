/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP
 
#include "ds/vector.hpp"
#include "mpi/types.hpp" 

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
        void bcast1(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void checksum();
        void display();
        void free();
        void apply(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
    
    protected:
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);                  
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  std::vector<std::set<Integer_Type>> &z_data);
                  //std::vector<std::vector<Integer_Type>> &z_data);
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
        bool stationary;
        Ordering_type ordering_type;
        Tiling_type tiling_type;
        Compression_type compression_type;
        Filtering_type filtering_type;
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
        MPI_Comm communicator;
        
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
        
        //std::vector<std::vector<std::vector<std::vector<Integer_Type>>>> Z_;
        //std::vector<std::vector<std::vector<Integer_Type>>> Z;
        std::vector<std::vector<std::set<Integer_Type>>> Z;
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
    
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        delete Y[i];
    }   
    
    Y.clear();
    Y.shrink_to_fit();
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
    
    //TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
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
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_sizes, all_rowgrp_ranks_accu_seg);
            
            //Z[j].resize(rowgrp_nranks);
            //for(uint32_t i = 0; i < rowgrp_nranks; i++)
            //    Z[j][i].resize(yy_sizes[i]);
        }
        else
        {
            y_size = {y_sizes[j]};
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
            
            //Z[j].resize(1);
            //Z[j][0].resize(y_size[0]);
        }
        populate(Y_, y);
        Y.push_back(Y_);
    }    
 
    yy_sizes.clear();
    y_size.clear();
    
    Z.resize(rank_nrowgrps);
    //Z_.resize(rank_nrowgrps);
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {  
        ///Z[j].resize(1);
        Z[j].resize(y_sizes[j]);
        //Z_[j].resize(y_sizes[j]);
        /*
        if(local_row_segments[j] == owned_segment)
        {
            yy_sizes.resize(rowgrp_nranks, y_sizes[j]);
            Z[j].resize(rowgrp_nranks);
            for(uint32_t i = 0; i < rowgrp_nranks; i++)
                Z[j][i].resize(yy_sizes[i]);
        }
        else
        {
            y_size = {y_sizes[j]};
            Z[j].resize(1);
            Z[j][0].resize(y_size[0]);
        }
        */
    }  
    
    Z_SIZE.resize(rank_nrowgrps);
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        if(local_row_segments[j] == owned_segment)
        {
            yy_sizes.resize(rowgrp_nranks, y_sizes[j]);
            Z_SIZE[j].resize(rowgrp_nranks);
            for(uint32_t i = 0; i < rowgrp_nranks; i++)
                Z_SIZE[j][i].resize(yy_sizes[i] + 1); // 0 + items
        }
        else
        {
            y_size = {y_sizes[j]};
            Z_SIZE[j].resize(1);
            Z_SIZE[j][0].resize(y_size[0] + 1); // 0 + items
        }        
    }
    inboxes.resize(rowgrp_nranks - 1);
    outboxes.resize(rowgrp_nranks - 1);
    
    
    /*
    if(filtering_type == _NONE_)
    {
        y_size = {tile_height};
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {  
            if(local_row_segments[j] == owned_segment)
            {
                y_sizes.clear();
                y_sizes.resize(rowgrp_nranks, tile_height);
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_sizes, all_rowgrp_ranks_accu_seg);
            }
            else
            {
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
            }
            populate(Y_, y);
            Y.push_back(Y_);
        }
    }
    else if(filtering_type == _SOME_)
    {
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            if(local_row_segments[j] == owned_segment)
            {
                y_sizes.clear();
                y_sizes.resize(rowgrp_nranks, nnz_row_sizes_loc[j]);  
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_sizes, all_rowgrp_ranks_accu_seg);
            }
            else
            {
                y_size.clear();
                y_size = {nnz_row_sizes_loc[j]};
                Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
            }
            populate(Y_, y);
            Y.push_back(Y_);
        }
    }
    
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        y_size.clear();
        y_sizes.clear();        
        if(filtering_type == _NONE_)
        {
            y_size.resize(1, tile_height);
            y_sizes.resize(rowgrp_nranks, tile_height);            
        }
        else if(filtering_type == _SOME_)
        {
            y_size.resize(1, nnz_row_sizes_loc[j]);
            y_sizes.resize(rowgrp_nranks, nnz_row_sizes_loc[j]);    
        }
        
        if(local_row_segments[j] == owned_segment)
        {
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_sizes, all_rowgrp_ranks_accu_seg);
        }
        else
        {
            Y_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
        }
        populate(Y_, y);
        Y.push_back(Y_);
    }
    
    
    std::vector<Integer_Type> z_size;
    std::vector<Integer_Type> z_sizes;
    Vector<Weight, Integer_Type, Fractional_Type> *Z_;
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        z_size.clear();
        z_sizes.clear();        
        if(filtering_type == _NONE_)
        {
            z_size.resize(1, tile_height);
            z_sizes.resize(rowgrp_nranks, tile_height);
        }
        else if(filtering_type == _SOME_)
        {
            z_size.resize(1, nnz_row_sizes_loc[j]);
            z_sizes.resize(rowgrp_nranks, nnz_row_sizes_loc[j]);    
        }
        
        if(local_row_segments[j] == owned_segment)
        {
            Z_ = new Vector<Weight, Integer_Type, Fractional_Type>(z_sizes, all_rowgrp_ranks_accu_seg);
        }
        else
        {
            Z_ = new Vector<Weight, Integer_Type, Fractional_Type>(y_size, accu_segment_row_vec);
        }
        populate(Y_, y);
        Y.push_back(Y_);
    }
    
    
    if(filtering_type == _NONE_)
    {
        list.resize(rank_ncolgrps);
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
            list[i].resize(tile_height);
        
    }
    else if(filtering_type == _SOME_)
    {
        list.resize(rank_ncolgrps);
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
            list[i].resize(tile_height);
        x_sizes = nnz_col_sizes_loc;
    }
    */
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast1(Fractional_Type (*f)
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
            std::vector<std::set<Integer_Type>> &z_data)
            //std::vector<std::vector<Integer_Type>> &z_data)
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
            //std::vector<std::vector<Integer_Type>> list;
            //list.resize(tile_height);
            for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
            {
                //if(!Env::rank)
                //printf("%d: ", j);
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                {
                    #ifdef HAS_WEIGHT
                    if(x_data[j] and A[i])
                        y_data[IA[i]] += A[i] * x_data[j];   
                    #else
                    if(x_data[j])    
                        y_data[IA[i]] += x_data[j];
                    #endif
                    //list[IA[i]].push_back(j);
                    pair = A->base({IA[i], j}, tile.rg, tile.cg);
                    //z_data[IA[i]].push_back(pair.col);
                    z_data[IA[i]].insert(pair.col);
                    //if(!Env::rank)
                    //    printf("[%d %d]", pair.row, pair.col);
                    
                }
                //if(!Env::rank)
                //    printf("\n");
            }
            //struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp);
            ///struct Triple<Weight, Integer_Type> &pair
            //A->base(
            /*
            if(!Env::rank)
            {
            for(uint32_t i = 0; i < ncols_plus_one_minus_one; i++)
            {
                printf("%d: ",i);
                for(uint32_t j = 0; j < z_data[i].size(); j++)
                    printf("%f ", z_data[i][j]);
                printf("\n");
            }   
            }
            */
            //z_data[0].push_back(1);
           // printf(">>%lu\n", z_data.size());
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
    //std::vector<std::vector<Integer_Type>> inboxes;
    //inboxes.resize(rowgrp_nranks - 1);
    //std::vector<std::vector<Integer_Type>> outboxes(rowgrp_nranks - 1);
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
        
        std::vector<std::set<Integer_Type>> &z_data = Z[yi];  
        //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];  
        
        if(stationary)
            spmv(y_data, x_data, tile);
        else
            spmv(y_data, x_data, tile, z_data);
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
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
                        MPI_Recv(zj_size.data(), zj_size.size(), TYPE_INT, follower, pair_idx, communicator, &status);

                        auto &inbox = inboxes[j];
                        Integer_Type inbox_nitems = zj_size[y_nitems];
                        //printf("%d <-- %d, tag=%d and nitems=%d \n", my_rank, follower, pair_idx, inbox_nitems);
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
                    std::vector<Integer_Type> &z_size = Z_SIZE[yi][yo];
                    std::vector<std::set<Integer_Type>> &z_data = Z[yi];  
                    //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];  
                    
                    z_size[0] = 0;
                    for(Integer_Type i = 1; i < y_nitems + 1; i++)
                    {
                        z_size[i] = z_size[i-1] + z_data[i-1].size();
                    }
                    Integer_Type outbox_nitems = z_size[y_nitems];                    
                    MPI_Send(z_size.data(), z_size.size(), TYPE_INT, leader, pair_idx, communicator);

                    
                    //printf("%d --> %d, tag=%d and nitems=%d \n", my_rank, leader, pair_idx, outbox_nitems);
                    auto &outbox = outboxes[oi];
                    
                    //outbox.resize(outbox_nitems);
                    Integer_Type k = 0;
                    for(Integer_Type i = 0; i < y_nitems; i++)
                    {
                        //z_data[i].erase(unique(z_data[i].begin(), z_data[i].end()), z_data[i].end());
                        /*
                        for(uint32_t j = 0; j < z_data[i].size(); j++)
                        {
                            //outbox.push_back(z_data[i][j]);
                            outbox.push_back(z_data[i][j]);
                            //k++;
                        }
                        */
                        
                        for(auto j: z_data[i])
                        {
                            //printf("%d\n", i);
                            outbox.push_back(j);
                            //outbox.push_back(z_data[i][j]);
                            //k++;
                            
                        }
                        
                    }
                    //outbox.erase(unique(z_data[i].begin(), z_data[i].end()), z_data[i].end());
                    

                    
                    
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


    //t2 = Env::clock();
    //if(!Env::rank)
    //    printf("Combine recv: %f seconds\n", t2 - t1);

    //yi = accu_segment_row;
    //yo = accu_segment_rg;
    //auto *Yp = Y[yi];
    //Integer_Type y_nitems = Yp->nitems[yo];
    
    /*
    if(Env::rank == 0)
    {    
        for(uint32_t j = 0; j < rowgrp_nranks; j++)
        {
       //     printf("**************\n");
            std::vector<Integer_Type> &z_size = Z_SZ[yi][j];
            for(uint32_t i = 0; i < z_size.size(); i++)
            {
                printf("%d ", z_size[i]);
            }
            printf("\n");
            
            std::vector<std::vector<Integer_Type>> &z_data = Z[yi][j];
            printf("j=%d: ", j);
            for(uint32_t i = 0; i < z_data.size(); i++)
            {
                printf("i=%d: ", i);
                for(uint32_t j = 0; j < z_data[i].size(); j++)
                {
                    printf("z=%d ", z_data[i][j]);
                }
                printf("\n");
            }
            if(j == 1)
            {
                auto &inbox = inboxes[0];
                for(uint32_t i = 0; i < inbox.size(); i++)
                {
                    printf("i=%d ibox=%d\n", i, inbox[i]);
                }
            }
       }
       */
       //std::vector<std::vector<Integer_Type>> &z_data = Z[yi][yo];  
       //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
        //if(!Env::rank)
        //{
            
            /*
            std::vector<Integer_Type> &inbox = inboxes[0];
            
            for(uint32_t i = 0; i < inbox.size(); i++)
            {
                printf("i=%d inbox=%d", i, inbox[i]);
            }
            printf("\n**************\n");
            */
            
        //for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        //{
            /*
            if(Env::comm_split)
            {   
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            }
            else
            {
                accu = follower_rowgrp_ranks_accu_seg[j];
            }
            
            std::vector<Integer_Type> &z_size = Z_SIZE[yi][accu];
            std::vector<Integer_Type> &inbox = inboxes[j];
            */
            /*
            for(uint32_t i = 0; i < z_size.size(); i++)
            {
                printf("i=%d isz=%d ", i, z_size[i]);
            }
            printf("\n");
            for(uint32_t i = 0; i < inbox.size(); i++)
            {
                printf("i=%d ib=%d ", i, inbox[i]);
            }
            printf("\n");
            */
            //z_data[IA[i]].push_back(pair.col);
            //std::vector<Integer_Type> &z_data = Z_SZ[yi][yo][j]];
            //Integer_Type l = 0;
            /*
            for(uint32_t i = 0; i < z_size.size() - 1; i++)
            {
               // printf("%d %d: ", i, z_size[i]);  
                for(uint32_t k = z_size[i]; k < z_size[i+1]; k++)
                {
                    //printf("[%d %d]", k, inbox[k]);  
                    z_data[i].push_back(inbox[k]);
                    //l++;
                }
                //printf("\n");
            }
            */
        //}
        /*
        if(Env::rank == 3)
        {
            for(uint32_t i = 0; i < y_nitems; i++)
            {
                printf("i=%d: ", i);
                auto &z_d = z_data[i];
                for(uint32_t j: z_d)
                    printf("%d ", j);
                printf("\n");
            }
        }
        */
            
        //}
        

    
       
       
        /*
        if(!Env::rank)
        {
            std::vector<Integer_Type> z_sizes;
            //Triple<Weight, Integer_Type> pair;
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
            {
                //printf("j=%d, ", rowgrp_nranks);
                std::vector<std::vector<Integer_Type>> &z_data = Z[yi][yo];
                for(uint32_t i = 0; i < y_nitems; i++)
                {
                    z_sizes.push_back(z_data[i].size());
                    //printf("i=%d: ", i);
                    printf("j=%d i=%d sz=%lu\n", j, i, z_data[i].size());
                    //std::vector<Integer_Type> &z_d = z_data[i];
                    
                   // for(uint32_t k = 0 ; k < z_d.size(); k++)
                    //{
                        //pair = A->base({z_d[k], j}, 0, 1);
                      //  printf("k=%d ", z_d[k]);
                    //}
                    //printf("\n");
                }
                Fractional_Type *yj_data = (Fractional_Type *) Yp->data[j];
                Integer_Type yj_nitems = Yp->nitems[j];
            }
        }
        */
        
        
    
    


}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::apply(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    //printf("APPLY %d\n", Env::rank);
    double t1, t2;
    t1 = Env::clock();
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    auto *Yp = Y[yi];
    uint32_t yo = accu_segment_rg;
    
    Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
    Integer_Type y_nitems = Yp->nitems[yo];
    
    std::vector<std::set<Integer_Type>> &z_data = Z[yi];
    //std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
    
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];
        
        if(stationary)
        {
            Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
            Integer_Type yj_nitems = Yp->nitems[accu];

            for(uint32_t i = 0; i < yj_nitems; i++)
            {
                if(yj_data[i])
                    y_data[i] += yj_data[i];
            }
        }
        else
        {
            std::vector<Integer_Type> &zj_size = Z_SIZE[yi][accu];
            Integer_Type zj_nitems = zj_size.size() - 1;
            std::vector<Integer_Type> &inbox = inboxes[j];
            for(uint32_t i = 0; i < zj_nitems; i++)
            {
                for(uint32_t j = zj_size[i]; j < zj_size[i+1]; j++)
                    z_data[i].insert(inbox[j]);
                    //z_data[i].push_back(inbox[j]);  
            }
            /*
            if(Env::rank == 1)
            {
                
                for(uint32_t i = 0; i < zj_nitems; i++)
                {
                    printf("i=%d: ", i);
                    auto &z_d = z_data[i];
                    for(uint32_t j: z_d)
                        printf("%d ", j);
                    printf("\n");
                }
            }
            */
            inbox.clear();
            inbox.shrink_to_fit();
        }
    }
    
    if(not stationary)
    {
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
    t2 = Env::clock();
    Env::print_time("Apply", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_all()
{
    if(stationary)
    {
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        out_requests.clear();
    }
    else
    {
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
        in_requests_.clear();
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
            communicator = rowgrps_communicator;
        }
        else
        {
            item1 = tile.leader_rank_rg;
            item2 = Env::rank;
            communicator = Env::MPI_WORLD;
        }
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_cg_cg;
            item2 = Env::rank_cg;
            communicator = rowgrps_communicator;
        }
        else
        {
            item1 = tile.leader_rank_cg;
            item2 = Env::rank;
            communicator = Env::MPI_WORLD;
        }
    }
    return{item1, item2};
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
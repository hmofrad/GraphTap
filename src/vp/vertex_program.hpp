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

#define INF 2147483647

enum Ordering_type
{
  _ROW_,
  _COL_
};   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vertex_Program
{
    public:
        //Vertex_Program();
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph, 
                        bool stationary_ = false, bool gather_depends_on_apply_ = false, bool tc_family_ = false, Ordering_type = _ROW_);
        ~Vertex_Program();
        
        virtual bool initializer(Fractional_Type &v1, Fractional_Type &v2) { return(stationary);}
        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) { return(1);}
        virtual void combiner(Fractional_Type &y1, Fractional_Type &y2) { ; }
        virtual bool applicator(Fractional_Type &v, Fractional_Type &y) { return(true); }
        
        void execute(uint32_t niters = 0);
        
        //void init(std::function<bool(Fractional_Type&, Fractional_Type&)> initializer_,
        //     Fractional_Type v = 0, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram = nullptr);
        //void scatter_gather(std::function<Fractional_Type(Fractional_Type&, Fractional_Type&)> messenger_);
        //void combine(std::function<void(Fractional_Type&, Fractional_Type&)> combiner_);
        //void apply(std::function<bool(Fractional_Type&, Fractional_Type&)> applicator_);
        void initialize(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram = nullptr);
        void free();
        
        bool stationary;
        bool gather_depends_on_apply;
        bool tc_family;
        bool already_initialized = false;
        bool check_for_convergence = false;
    
    protected:        
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec, Fractional_Type value);
        
        void populate(Vector<Weight, Integer_Type, Fractional_Type> *vec_dst,
                      Vector<Weight, Integer_Type, Fractional_Type> *vec_src);
        void clear(Vector<Weight, Integer_Type, Fractional_Type> *vec);
        void specialized_nonstationary_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram);        
        void specialized_stationary_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram);
        void specialized_tc_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram);  
        void scatter_gather();
        void scatter();
        void gather();
        void bcast();
        void combine();
        void optimized_1d_row();
        void optimized_1d_col();
        void optimized_2d();
        void optimized_2d_for_tc();
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void spmv(std::vector<std::vector<Integer_Type>> &z_data, 
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void apply();                        
        void specialized_apply();
        void specialized_tc_apply();
        void wait_for_all();
        void wait_for_sends();
        void wait_for_recvs();
        bool has_converged();
        void checksum();
        void display();
        

        
        
        
        //std::function<bool(Fractional_Type&, Fractional_Type&)> initializer;
        //std::function<Fractional_Type(Fractional_Type&, Fractional_Type&)> messenger;
        //std::function<void(Fractional_Type&, Fractional_Type&)> combiner;
        //std::function<bool(Fractional_Type&, Fractional_Type&)> applicator;
        
        
        struct Triple<Weight, Integer_Type> tile_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                       struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> leader_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);  
        MPI_Comm communicator_info();  
                       
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
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A; // Adjacency list
        
        Vector<Weight, Integer_Type, Fractional_Type> *X; // Messages 
        Vector<Weight, Integer_Type, Fractional_Type> *V; // Values
        Vector<Weight, Integer_Type, Fractional_Type> *S; // Scores (States)
        Vector<Weight, Integer_Type, char> *C; // Convergence 
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y; //Accumulators
        Vector<Weight, Integer_Type, bool> *B; //Activity pattern bitvector
        
        std::vector<std::vector<char>> *I;
        //std::vector<std::vector<Integer_Type>> *IV;
        std::vector<std::vector<char>> *J;
        //std::vector<std::vector<Integer_Type>> *JV;
        
        
        /*
        unsigned char **I;
        Integer_Type **IV;
        unsigned char **J;
        Integer_Type **JV;
        */
        /*
        Vector<Weight, Integer_Type, char> *I;
        Vector<Weight, Integer_Type, Integer_Type> *IV;
        Vector<Weight, Integer_Type, char> *J;
        Vector<Weight, Integer_Type, Integer_Type> *JV;
        */
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        
        MPI_Datatype TYPE_DOUBLE;
        MPI_Datatype TYPE_INT;
        
        /* Specialized vectors for triangle counting */
        std::vector<std::vector<Integer_Type>> W; // Values (Outgoing edges)
        std::vector<std::vector<Integer_Type>> R; // Scores (Ingoing edges)
        std::vector<std::vector<std::vector<Integer_Type>>> D; // Data (All Ingoing edges)
        std::vector<std::vector<Integer_Type>> D_SIZE; 
        std::vector<std::vector<std::vector<Integer_Type>>> Z; // Accumulators
        std::vector<std::vector<std::vector<Integer_Type>>> Z_SIZE;
        std::vector<std::vector<Integer_Type>> inboxes; // Temporary buffers for deserializing the adjacency lists
        std::vector<std::vector<Integer_Type>> outboxes;// Temporary buffers for  serializing the adjacency lists
};

//template<typename Weight, typename Integer_Type, typename Fractional_Type>
//Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph,
         bool stationary_, bool gather_depends_on_apply_, bool tc_family_, Ordering_type ordering_type_)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;
    stationary = stationary_;
    gather_depends_on_apply = gather_depends_on_apply_;
    tc_family = tc_family_;
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
        nnz_row_sizes_all = A->nnz_row_sizes_all;
        nnz_col_sizes_all = A->nnz_col_sizes_all;
        I = &(Graph.A->I);
        //IV= &(Graph.A->IV);
        J = &(Graph.A->J);
        //JV= &(Graph.A->JV);
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
        nnz_row_sizes_all = A->nnz_col_sizes_all;
        nnz_col_sizes_all = A->nnz_row_sizes_all;
        I = &(Graph.A->J);
        //IV= &(Graph.A->JV);
        J = &(Graph.A->I);
        //JV= &(Graph.A->IV);
    }   
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::free()
{
    if(tc_family)
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
    else
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
            delete B;
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::execute(uint32_t niters)
{
    double t1, t2;
    t1 = Env::clock();

    if(not already_initialized)
        initialize();
    
    if(!niters)
    {
        check_for_convergence = true; 
        niters = INF;
    }

    uint32_t iter = 0;
    while(iter < niters)
    {
        scatter_gather();
        combine();
        apply();
        iter++;
        Env::print_me("Pagerank iteration: ", iter);
        if(check_for_convergence)
        {
            if(has_converged())
                break;
        }
    }
    
    t2 = Env::clock();
    Env::print_time("Execute", t2 - t1);
    checksum();
    display();
}



//template<typename Weight, typename Integer_Type, typename Fractional_Type>
//void Vertex_Program<Weight, Integer_Type, Fractional_Type>::initialize(
//        std::function<bool(Fractional_Type&, Fractional_Type&)> initializer_, Fractional_Type v, 
//        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::initialize(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    double t1, t2;
    t1 = Env::clock();

    //initializer = initializer_;
    if(stationary)
    {
        specialized_stationary_init(VProgram);
    }
    else
    {    
        if(tc_family)
            specialized_tc_init(VProgram);
        else
            specialized_nonstationary_init(VProgram);
    }

    t2 = Env::clock();
    Env::print_time("Init", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::specialized_stationary_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    /* Initialize messages */
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
    
    /* Initialize values and activity pattern */
    std::vector<Integer_Type> v_s_size = {tile_height};
    V = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);

    //populate(V, v);
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    Fractional_Type v = 0;
    for(uint32_t i = 0; i < v_nitems; i++)
        (void)initializer(v_data[i], v);
        //(void)initializer(v_data[i], v);

    /* Initialiaze scores/states */
    S = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    if(VProgram)
    {
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
        Vector<Weight, Integer_Type, Fractional_Type> *V_ = VP->V;
        populate(S, V_);
        already_initialized = true;
    }
    
    /* Array to look for convergence */ 
    C = new Vector<Weight, Integer_Type, char>(v_s_size, accu_segment_row_vec);
    
    /* Initialiaze accumulators */
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
        Y.push_back(Y_);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::specialized_nonstationary_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    /* Initialize messages */
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
    
    /* Initialize values and activity pattern */
    std::vector<Integer_Type> v_s_size = {tile_height};
    V = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    Fractional_Type v = 0;
    //populate(V, v);
    
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    B = new Vector<Weight, Integer_Type, bool>(x_sizes,  local_col_segments);    
    uint32_t bo = accu_segment_col;
    bool *b_data = (bool *) B->data[bo];
    Integer_Type b_nitems = B->nitems[bo];
    
    Fractional_Type tmp = 0;
    if(filtering_type == _NONE_)
    {
        if(gather_depends_on_apply)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                tmp = i + (owned_segment * tile_height);
                b_data[i] = initializer(v_data[i], tmp);
                //b_data[i] = initializer(v_data[i], tmp);
            }
        }
        else
        {
            for(uint32_t i = 0; i < v_nitems; i++)
                //b_data[i] = initializer(v_data[i], v);
                b_data[i] = initializer(v_data[i], v);
        }
    }
    else if(filtering_type == _SOME_)
    {
        auto &j_data = (*J)[bo];
        //char *j_data = (char *) J->data[bo];
        //Integer_Type j_nitems = J->nitems[bo];
        if(gather_depends_on_apply)
        {
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                if(j_data[i])
                {
                    tmp = i + (owned_segment * tile_height);    
                    //b_data[j] = initializer(v_data[i], tmp);
                    b_data[j] = initializer(v_data[i], tmp);
                    j++;
                }               
            }
        }
        else
        {
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                if(j_data[i])
                {
                    //b_data[j] = initializer(v_data[i], v);
                    b_data[j] = initializer(v_data[i], v);
                    j++;
                }               
            }
        }
    }

    /* Initialiaze scores/states */
    S = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    if(VProgram)
    {
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
        Vector<Weight, Integer_Type, Fractional_Type> *V_ = VP->V;
        populate(S, V_);
        already_initialized = true;
    }
    
    /* Initialiaze accumulators */
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
        Y.push_back(Y_);
    }    
    
    if(gather_depends_on_apply)
    {
        uint32_t yi = 0;
        uint32_t yo = 0;
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            yi = j;
            if(local_row_segments[j] == owned_segment)
                yo = accu_segment_rg;
            else
                yo = 0;
            
            auto *Yp = Y[yi];
            Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
            Integer_Type y_nitems = Yp->nitems[yo];
            if(filtering_type == _NONE_)
            {
                for(uint32_t i = 0; i < v_nitems; i++)
                    y_data[i] = v_data[i];
            }
            else if(filtering_type == _SOME_)
            {
                auto &i_data = (*I)[yi];
                //char *i_data = (char *) I->data[yi];        
                //Integer_Type j = 0;
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    if(i_data[i])
                    {
                        y_data[j] = v_data[i];
                        j++;
                    }
                }
            }
        }
    }    
}    

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::specialized_tc_init(Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    std::vector<Integer_Type> d_sizes;
    std::vector<Integer_Type> z_size;
    std::vector<Integer_Type> z_sizes;
    std::vector<Integer_Type> zz_sizes;
    
    if(filtering_type == _NONE_)
    {
        z_sizes.resize(rank_nrowgrps, tile_height);
        d_sizes.resize(nrowgrps, tile_height);
    }
    else if(filtering_type == _SOME_) 
    {
        //z_sizes = nnz_row_sizes_loc;
        //d_sizes = nnz_row_sizes_all;
        fprintf(stderr, "Invalid filtering type\n");
        Env::exit(1);
    }   

    /* Initialize values and activity pattern */
    W.resize(tile_height);
    
    if(VProgram)
    {
        /* Initialiaze scores/states */
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
        std::vector<std::vector<Integer_Type>> W_ = VP->W;
        R = W_;
        already_initialized = true;
        
        /* Initialiaze BIG 2D scores/states vector */
        D.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D[i].resize(d_sizes[i]);
        
        D_SIZE.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D_SIZE[i].resize(d_sizes[i] + 1); // 0 + nitems
    }

    /* Initialiaze accumulators */
    Z.resize(rank_nrowgrps);
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
        Z[j].resize(z_sizes[j]);
    Z_SIZE.resize(rank_nrowgrps);
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        if(local_row_segments[j] == owned_segment)
        {
            zz_sizes.resize(rowgrp_nranks, z_sizes[j]);
            Z_SIZE[j].resize(rowgrp_nranks);
            for(uint32_t i = 0; i < rowgrp_nranks; i++)
                Z_SIZE[j][i].resize(zz_sizes[i] + 1); // 0 + nitems
        }
        else
        {
            z_size = {z_sizes[j]};
            Z_SIZE[j].resize(1);
            Z_SIZE[j][0].resize(z_size[0] + 1); // 0 + nitems
        }        
    }
    inboxes.resize(rowgrp_nranks - 1);
    outboxes.resize(rowgrp_nranks - 1);
}        


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter_gather()
{
    double t1, t2;
    t1 = Env::clock();
    //messenger = messenger_;
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
           // printf("1.msg=%f %f %f\n", x_data[i], v_data[i], s_data[i]);
            x_data[i] = messenger(v_data[i], s_data[i]);
            //printf("2.msg=%f %f %f\n", x_data[i], v_data[i], s_data[i]);
            //x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
        }
    }
    else if(filtering_type == _SOME_)
    {
        auto &j_data = (*J)[xo];
        //char *j_data = (char *) J->data[xo];
        //Integer_Type j_nitems = J->nitems[xo];
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                x_data[j] = messenger(v_data[i], s_data[i]);
                //x_data[j] = (*f)(0, 0, v_data[i], s_data[i]);
                j++;
            }               
        }
    }
    
    if(Env::comm_split)
    {
        bcast();
    }
    else
    {
        scatter();
        gather();
    }
    t2 = Env::clock();
    Env::print_time("Scatter_gather", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter()
{
    uint32_t xo = accu_segment_col;
    Fractional_Type *x_data = (Fractional_Type *) X->data[xo];
    Integer_Type x_nitems = X->nitems[xo];
    int32_t col_group = X->local_segments[xo];
    
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
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::gather()
{  
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
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast()
{
    uint32_t leader;
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
}                   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            std::vector<std::vector<Integer_Type>> &z_data,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    Triple<Weight, Integer_Type> pair;
    if(tile.allocated)
    {
        if(compression_type == Compression_type::_CSC_)    
        {
            Integer_Type *IA = (Integer_Type *) tile.csc->IA; // ROW_INDEX
            Integer_Type *JA   = (Integer_Type *) tile.csc->JA; // COL_PTR
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            if(ordering_type == _ROW_)
            {
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                    {
                        pair = A->base({IA[i] + (owned_segment * tile_height), j}, tile.rg, tile.cg);
                        z_data[IA[i]].push_back(pair.col);
                    }
                }
            }
            else
            {
                fprintf(stderr, "Invalid ordering type\n");
                Env::exit(1);
            }                
        }
        else
        {
            fprintf(stderr, "Invalid compression type\n");
            Env::exit(1);
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
                        //if(x_data[JA[j]] and A[j])
                            combiner(y_data[i], A[j] * x_data[JA[j]]);
                            //y_data[i] += A[j] * x_data[JA[j]];
                        #else
                        //if(x_data[JA[j]])
                            combiner(y_data[i], x_data[JA[j]]);
                            //y_data[i] += x_data[JA[j]];
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
                       //if(x_data[i] and A[j])
                            combiner(y_data[JA[j]], A[j] * x_data[i]);
                            //y_data[JA[j]] += A[j] * x_data[i];
                        #else
                        //if(x_data[i])
                            combiner(y_data[JA[j]], x_data[i]);
                            //y_data[JA[j]] += x_data[i];
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
                        //if(x_data[j] and A[i])
                            combiner(y_data[IA[i]], A[i] * x_data[j]);
                            //y_data[IA[i]] += A[i] * x_data[j];
                        #else
                        //if(x_data[j])
                        ////{
                            //printf("1.%d %f %f\n", j, x_data[i], y_data[IA[i]]);
                            combiner(y_data[IA[i]], x_data[j]);
                            //printf("2.%d %f %f\n", j, x_data[i], y_data[IA[i]]);
                            //y_data[IA[i]] += x_data[j];
                        //}
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
                       /// if(x_data[IA[i]] and A[i])
                            combiner(y_data[j], A[i] * x_data[IA[i]]);   
                            //y_data[j] += A[i] * x_data[IA[i]];   
                        #else
                       // if(x_data[IA[i]])
                            combiner(y_data[j], x_data[IA[i]]);   
                            //y_data[j] += x_data[IA[i]];
                        #endif
                    }
                }
            }
        }            
    }    
}

//template<typename Weight, typename Integer_Type, typename Fractional_Type>
//void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine(
//        std::function<void(Fractional_Type&, Fractional_Type&)> combiner_)
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine()
{
    double t1, t2;
    t1 = Env::clock();
    
    //combiner = combiner_;
    if(tiling_type == Tiling_type::_1D_ROW and stationary)
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
    else if(tiling_type == Tiling_type::_1D_COL and stationary)
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
        if(tc_family)
            optimized_2d_for_tc();
        else
            optimized_2d();    
    }
    else
    {
        fprintf(stderr, "Invalid configuration\n");
        Env::exit(1);
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_2d_for_tc()
{
    MPI_Request request;
    MPI_Status status;
    int flag, count;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t yi = 0, yo = 0, oi = 0;
    Vector<Weight, Integer_Type, Fractional_Type> *Yp;
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
       
        std::vector<std::vector<Integer_Type>> &z_data = Z[yi];  
        spmv(z_data, tile);
        
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
                    
                    std::vector<Integer_Type> &zj_size = Z_SIZE[yi][accu];
                    Integer_Type sj_s_nitems = zj_size.size();
                    MPI_Recv(zj_size.data(), sj_s_nitems, TYPE_INT, follower, pair_idx, communicator, &status);
                    auto &inbox = inboxes[j];
                    Integer_Type inbox_nitems = zj_size[sj_s_nitems - 1];
                    inbox.resize(inbox_nitems);
                    MPI_Irecv(inbox.data(), inbox.size(), TYPE_INT, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
                std::vector<Integer_Type> &z_size = Z_SIZE[yi][yo];
                Integer_Type z_s_nitems = z_size.size();
                auto &outbox = outboxes[oi];
                
                Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(z_size, z_data, outbox);
                MPI_Send(z_size.data(), z_s_nitems, TYPE_INT, leader, pair_idx, communicator);
                MPI_Isend(outbox.data(), outbox.size(), TYPE_INT, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);  
                oi++;
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
    Vector<Weight, Integer_Type, Fractional_Type> *Yp;
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
        
        Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);
        
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
            xi = 0;
            yi++;
        }
    }
    wait_for_all();
}

//template<typename Weight, typename Integer_Type, typename Fractional_Type>
//void Vertex_Program<Weight, Integer_Type, Fractional_Type>::apply(
//                            std::function<bool(Fractional_Type&, Fractional_Type&)> applicator_)

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::apply()
{
    double t1, t2;
    t1 = Env::clock();
    //applicator = applicator_;

    if(tc_family)
    {
        specialized_tc_apply();
    }
    else
    {
        specialized_apply();
    }

    t2 = Env::clock();
    Env::print_time("Apply", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::specialized_apply()
{
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
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
    
    uint32_t co = 0;
    char *c_data = (char *) C->data[vo];
    Integer_Type c_nitems = C->nitems[co];

    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            //printf("%d y=%f v=%f\n", i, y_data[i], v_data[i]);
            //v_data[i] = 
            c_data[i] = applicator(v_data[i], y_data[i]);
            //printf("%d y=%f v=%f\n", i, y_data[i], v_data[i]);
            //v_data[i] = (*f)(0, y_data[i], 0, 0); 
        }
    }
    else if(filtering_type == _SOME_)
    {
        Fractional_Type tmp = 0;
        auto &i_data = (*I)[yi];
        
        //char *i_data = (char *) I->data[yi];        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(i_data[i])
            {
                c_data[i] = applicator(v_data[i], y_data[j]);
                //v_data[i] = (*f)(0, y_data[j], 0, 0);
                j++;
            }
            else
            {
                //printf("%f %d\n", v_data[i], fabs(v_data[i] - (0.15 + (1.0 - 0.15) * tmp)) > 1e-5);    
                c_data[i] = applicator(v_data[i], tmp);
                //v_data[i] = (*f)(0, 0, 0, 0);
                //printf("%d %f\n", ret, v_data[i]);    
                //return fabs(s.rank - tmp) > tol;
            }
        }
    }
    
    if(not gather_depends_on_apply)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
    }    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::specialized_tc_apply()
{
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    
    std::vector<std::vector<Integer_Type>> &z_data = Z[yi];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];
        
        std::vector<Integer_Type> &zj_size = Z_SIZE[yi][accu];
        std::vector<Integer_Type> &inbox = inboxes[j];
        Comm<Weight, Integer_Type, Fractional_Type>::unpack_adjacency(zj_size, z_data, inbox);
        inbox.clear();
        inbox.shrink_to_fit();
    }
    
    if(filtering_type == _NONE_)
    {
        std::vector<std::vector<Integer_Type>> &w_data = W;
        w_data = z_data;        
    }
    else if(filtering_type == _SOME_)
    {
        fprintf(stderr, "Invalid filtering type\n");
        Env::exit(1);
    }

    if(R.size())
    {
        std::vector<std::vector<Integer_Type>> &w_data = W;
        Integer_Type w_nitems = w_data.size();
        std::vector<std::vector<Integer_Type>> &r_data = R;
        Integer_Type r_nitems = r_data.size();        
        D[owned_segment] = R;

        MPI_Request request;
        MPI_Status status;
        uint32_t my_rank, leader, follower, accu;
        Triple<Weight, Integer_Type> triple, triple1, pair;
        
        std::vector<std::vector<Integer_Type>> boxes(nrows);
        std::vector<Integer_Type> &d_size = D_SIZE[owned_segment];
        Integer_Type d_s_nitems = d_size.size();
        auto &outbox = boxes[owned_segment];
        Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(d_size, r_data, outbox);

        for(uint32_t i = 0; i < nrowgrps; i++)  
        {
            uint32_t r = (Env::rank + i) % Env::nranks;
            if(r != Env::rank)
            {
                MPI_Isend(d_size.data(), d_s_nitems, TYPE_INT, r, 0, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        
        for(uint32_t i = 0; i < nrowgrps; i++)  
        {
            leader = leader_ranks[i];
            my_rank = Env::rank;
            if(my_rank != leader)
            {
                std::vector<Integer_Type> &dj_size = D_SIZE[i];
                Integer_Type dj_s_nitems = dj_size.size();
                MPI_Irecv(dj_size.data(), dj_s_nitems, TYPE_INT, leader, 0, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
        in_requests.clear();
        out_requests.clear();      
        
        for(uint32_t i = 0; i < nrowgrps; i++)  
        {
            uint32_t r = (Env::rank + i) % Env::nranks;
            if(r != Env::rank)
            {
                MPI_Isend(outbox.data(), outbox.size(), TYPE_INT, r, 0, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        
        for(uint32_t i = 0; i < nrowgrps; i++)  
        {
            leader = leader_ranks[i];
            my_rank = Env::rank;
            if(my_rank != leader)
            {
                std::vector<Integer_Type> &dj_size = D_SIZE[i];
                Integer_Type dj_s_nitems = dj_size.size();
                auto &inbox = boxes[i];
                Integer_Type inbox_nitems = dj_size[dj_s_nitems - 1];
                inbox.resize(inbox_nitems);
                MPI_Irecv(inbox.data(), inbox.size(), TYPE_INT, leader, 0, Env::MPI_WORLD, &request);
                in_requests.push_back(request);   
            }
        }
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
        in_requests.clear();
        out_requests.clear(); 
        
        for(uint32_t i = 0; i < nrowgrps; i++)
        {
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
                        dj_data[j].push_back(box[k]);
                    }
                }
                box.clear();
                box.shrink_to_fit();
            }
        }
        
        for(uint32_t i = 0; i < nrowgrps; i++)
        {
            D_SIZE[i].clear();
            D_SIZE[i].shrink_to_fit();
        }
        
        Env::barrier();

        for(uint32_t i = 0; i < nrowgrps; i++)
        {
            std::vector<std::vector<Integer_Type>> &dj_data = D[i];
            for(uint32_t j = 0; j < dj_data.size(); j++)
            {
                if(dj_data[j].size())
                    std::sort(dj_data[j].begin(), dj_data[j].end());
            }
        }

        /* adapted from https://github.com/narayanan2004/GraphMat/blob/master/src/TriangleCounting.cpp */
        uint64_t num_triangles_local = 0;
        uint64_t num_triangles_global = 0;
        uint32_t i_segment = owned_segment;
        uint32_t j_segment = 0;
        for(uint32_t i = 0; i < w_nitems; i++)
        {
            std::vector<Integer_Type> &i_neighbors = D[i_segment][i];
            for(uint32_t j = 0; j < W[i].size(); j++)
            {
                uint32_t j_segment = W[i][j] / tile_height;
                uint32_t jj = W[i][j] % tile_height;
                std::vector<Integer_Type> &j_neighbors = D[j_segment][jj];                    
                uint32_t it1 = 0, it2 = 0;
                uint32_t it1_end = i_neighbors.size(); // message.neighbors[it1]
                uint32_t it2_end = j_neighbors.size(); //vertexprop.neighbors[it2]
                while (it1 != it1_end && it2 != it2_end)
                {
                    if (i_neighbors[it1] == j_neighbors[it2]) 
                    {
                        num_triangles_local++;
                        it1++;
                        it2++;
                    } 
                    else if (i_neighbors[it1] < j_neighbors[it2])
                    {
                        it1++;
                    } 
                    else
                    {
                        it2++;
                    }
                }
            }
            
            if(Env::is_master)
            {
                if ((i & ((1L << 13) - 1L)) == 0)
                    printf("|");
            }
        }

        MPI_Allreduce(&num_triangles_local, &num_triangles_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(Env::is_master)
            printf("Num_triangles = %lu\n", num_triangles_global);   
        
        for (uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            D[i].clear();
            D[i].shrink_to_fit();
        }
    }
    
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

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_all()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
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
        }
        else
        {
            item1 = tile.leader_rank_rg;
            item2 = Env::rank;
        }
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_cg_cg;
            item2 = Env::rank_cg;
        }
        else
        {
            item1 = tile.leader_rank_cg;
            item2 = Env::rank;
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
            comm = rowgrps_communicator;
        else
            comm = Env::MPI_WORLD;
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
            comm = rowgrps_communicator;
        else
            comm = Env::MPI_WORLD;
    }
    return{comm};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type>::has_converged()
{
    bool converged = false;
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;

    uint32_t co = 0;
    char *c_data = (char *) C->data[co];
    Integer_Type c_nitems = C->nitems[co];    
    
    for(uint32_t i = 0; i < c_nitems; i++)
       c_sum_local += c_data[i];
   
    //printf("%lu %d\n", c_sum_local, tile_height);
    //if(c_sum_local == tile_height)
    //{
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(c_sum_gloabl == (tile_height * Env::nranks))
            converged = true;
    //}
    return(converged);   
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

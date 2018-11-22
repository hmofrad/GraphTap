/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include <type_traits>
#include <numeric>


#include "ds/vector.hpp"
#include "mpi/types.hpp" 
#include "mpi/comm.hpp" 
#include "mat/hashers.hpp"

struct State { State() {}; };

enum Ordering_type
{
  _ROW_,
  _COL_
};   

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
class Vertex_Program
{
    public:
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph,
                        bool stationary_ = false, bool activity_filtering_ = true, bool gather_depends_on_apply_ = false, 
                        bool apply_depends_on_iter_ = false, bool tc_family_ = false, Ordering_type = _ROW_);
        ~Vertex_Program();
        
        virtual bool initializer(Integer_Type vid, Vertex_State &state) { return(stationary);}
        virtual bool initializer(Integer_Type vid, Vertex_State &state, const State &other) { return(stationary);}
        virtual Fractional_Type messenger(Vertex_State &state) { return(1);}
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2, const Fractional_Type &w) { ; }
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) { ; }
        virtual bool applicator(Vertex_State &state, const Fractional_Type &y) { return(true); }
        virtual bool applicator(Vertex_State &state){ return(false); }
        virtual bool applicator(Vertex_State &state, const Fractional_Type &y, const Integer_Type iteration_) { return(true); }
        virtual Fractional_Type infinity() { return(0); }
        
        
        virtual bool initializer(Vertex_State &state, const Fractional_Type &v2) { return(stationary);}
        virtual bool initializer(Fractional_Type &v1, const Fractional_Type &v2) { return(stationary);}
        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) { return(1);}
        virtual bool applicator(Fractional_Type &v, const Fractional_Type &y) { return(true); }
        virtual bool applicator(Fractional_Type &v, const Fractional_Type &y, Integer_Type iteration_) { return(true); }
        
        void execute(Integer_Type num_iterations = 0);
        void initialize();
        template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_>
        void initialize(const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_> &VProgram);
        void free();
        void checksum();
        void display(Integer_Type count = 31);
        
        bool stationary = false;
        bool gather_depends_on_apply = false;
        bool apply_depends_on_iter = false;
        bool tc_family = false;
        Integer_Type iteration = 0;
        std::vector<Vertex_State> V;              // Values
        std::vector<std::vector<Integer_Type>> W; // Values (triangle counting)
    protected:
        bool already_initialized = false;
        bool check_for_convergence = false;
        void init_stationary();
        void init_nonstationary();
        void init_tc_family();
        void init_stationary_postprocess();
        void init_nonstationary_postprocess();
        void scatter_gather();
        void scatter_gather_stationary();
        void scatter_gather_nonstationary();
        void scatter_gather_nonstationary_activity_filtering();
        void scatter();
        void gather();
        void bcast();
        void scatter_stationary();
        void gather_stationary();
        void scatter_nonstationary();
        void gather_nonstationary();
        void bcast_stationary();
        void bcast_nonstationary();
        void combine();
        void combine_1d_row_stationary();
        void combine_1d_col_stationary();
        void combine_2d_stationary();
        void combine_2d_nonstationary();
        void combine_postprocess();
        void combine_postprocess_stationary_for_all();
        void combine_postprocess_nonstationary_for_all();
        void combine_postprocess_stationary_for_some();
        void combine_postprocess_nonstationary_for_some();
        void combine_2d_for_tc();
        void apply();                        
        void apply_stationary();
        void apply_nonstationary();
        void apply_tc();
        struct Triple<Weight, double> stats(std::vector<double> &vec);
        
        void spmv(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data); // Stationary spmv
                
        void spmv(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile, // Stationary spmv 
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data, 
                std::vector<Fractional_Type> &xv_data, 
                std::vector<Integer_Type> &xi_data,
                std::vector<char> &t_data);
                
        void spmv(std::vector<std::vector<Integer_Type>> &z_data, 
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile); // Triangle counting spmv
        
        
        
        void wait_for_all();
        void wait_for_sends();
        void wait_for_recvs();
        bool has_converged();
        Integer_Type get_vid(Integer_Type index);
        
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
        std::vector<int32_t> leader_ranks;
        std::vector<int32_t> leader_ranks_cg;
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;
        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        MPI_Comm rowgrps_communicator;
        MPI_Comm colgrps_communicator;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        std::vector<MPI_Request> out_requests_;
        std::vector<MPI_Request> in_requests_;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;          // Adjacency list
        /* Stationary */
        std::vector<std::vector<Fractional_Type>> X;               // Messages 
        std::vector<std::vector<std::vector<Fractional_Type>>> Y;  // Accumulators
        std::vector<char> C;                                       // Convergence vector
        /* Nonstationary */
        std::vector<std::vector<Integer_Type>> XI;                 // X Indices (Nonstationary)
        std::vector<std::vector<Fractional_Type>> XV;              // X Values  (Nonstationary)
        std::vector<std::vector<std::vector<Integer_Type>>> YI;    // Y Indices (Nonstationary)
        std::vector<std::vector<std::vector<Fractional_Type>>> YV; // Y Values (Nonstationary)
        std::vector<std::vector<char>> T;                          // Accumulators activity vectors
        std::vector<Integer_Type> msgs_activity_statuses;
        std::vector<Integer_Type> accus_activity_statuses;
        std::vector<Integer_Type> activity_statuses;
        /* Row/Col Filtering indices */
        std::vector<std::vector<char>> *I;
        //std::vector<std::vector<Integer_Type>> *IV;
        std::vector<std::vector<char>> *J;
        //std::vector<std::vector<Integer_Type>> *JV;
        std::vector<Integer_Type> *V2J;
        std::vector<Integer_Type> *J2V;
        std::vector<Integer_Type> *Y2V;
        std::vector<Integer_Type> *V2Y;
        std::vector<Integer_Type> *I2V;
        std::vector<Integer_Type> *V2I;

        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        
        MPI_Datatype TYPE_DOUBLE;
        MPI_Datatype TYPE_INT;
        MPI_Datatype TYPE_CHAR;
        
        /* Specialized vectors for triangle counting */
        std::vector<std::vector<Integer_Type>> R; // States (Ingoing edges)
        std::vector<std::vector<std::vector<Integer_Type>>> D; // Messages (All Ingoing edges)
        std::vector<std::vector<Integer_Type>> D_SIZE; // Messages sizes
        std::vector<std::vector<std::vector<Integer_Type>>> Z; // Accumulators
        std::vector<std::vector<std::vector<Integer_Type>>> Z_SIZE; // Accumulators sizes
        std::vector<std::vector<Integer_Type>> inboxes; // Temporary buffers for deserializing the adjacency lists
        std::vector<std::vector<Integer_Type>> outboxes;// Temporary buffers for  serializing the adjacency lists
        
        bool directed;
        bool transpose;
        double activity_filtering_ratio = 0.6;
        bool activity_filtering = true;
        bool accu_activity_filtering = false;
        bool msgs_activity_filtering = false;
        
        bool broadcast_communication = true;
        bool incremental_accumulation = true;
        
        
        std::vector<double> scatter_gather_time;
        std::vector<double> combine_time;
        std::vector<double> apply_time;
        
        
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph,
         bool stationary_, bool activity_filtering_, bool gather_depends_on_apply_, bool apply_depends_on_iter_, 
         bool tc_family_, Ordering_type ordering_type_)
{

    A = Graph.A;
    directed = A->directed;
    transpose = A->transpose;
    stationary = stationary_;
    activity_filtering = activity_filtering_;
    gather_depends_on_apply = gather_depends_on_apply_;
    apply_depends_on_iter = apply_depends_on_iter_;
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
        V2J = &(Graph.A->V2J);
        J2V = &(Graph.A->J2V);
        Y2V = &(Graph.A->Y2V);
        V2Y = &(Graph.A->V2Y);
        I2V = &(Graph.A->I2V);
        V2I = &(Graph.A->V2I);
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
        V2J = &(Graph.A->J2V);
        J2V = &(Graph.A->V2J);
        Y2V = &(Graph.A->V2Y);
        V2Y = &(Graph.A->Y2V);
        I2V = &(Graph.A->V2I);
        V2I = &(Graph.A->I2V);
    }   
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::free()
{
    if(tc_family)
    {
        
        for (uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            Z[i].clear();
            Z[i].shrink_to_fit();
        }   
        Z.clear();
        Z.shrink_to_fit();
        
        W.clear();
        W.shrink_to_fit();
        
        R.clear();
        R.shrink_to_fit();
        
        for (uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            Z_SIZE[i].clear();
            Z_SIZE[i].shrink_to_fit();
        }
        Z_SIZE.clear();
        Z_SIZE.shrink_to_fit();
    } 
    else
    {
        V.clear();
        V.shrink_to_fit();
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            X[i].clear();
            X[i].shrink_to_fit();
        }
        
        C.clear();
        C.shrink_to_fit();
        for (uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            if(local_row_segments[i] == owned_segment)
            {
                for(uint32_t j = 0; j < rowgrp_nranks; j++)
                {
                    Y[i][j].clear();
                    Y[i][j].shrink_to_fit();
                }
                    
            }
            else
            {
                Y[i][0].clear();
                Y[i][0].shrink_to_fit();
            }
            Y[i].clear();
            Y[i].shrink_to_fit();
        }   
        Y.clear();
        Y.shrink_to_fit();
        
        if(not stationary)
        {
            for(uint32_t i = 0; i < rank_ncolgrps; i++)
            {
                XV[i].clear();
                XV[i].shrink_to_fit();
                XI[i].clear();
                XI[i].shrink_to_fit();
            }
            
            for(uint32_t i = 0; i < rank_nrowgrps; i++)
            {
              if(local_row_segments[i] == owned_segment)
                {
                    for(uint32_t j = 0; j < rowgrp_nranks; j++)
                    {
                        YV[i][j].clear();
                        YV[i][j].shrink_to_fit();
                        YI[i][j].clear();
                        YI[i][j].shrink_to_fit();
                    }
                }
                else
                {
                    YV[i][0].clear();
                    YV[i][0].shrink_to_fit();
                    YI[i][0].clear();
                    YI[i][0].shrink_to_fit();
                }
                YV[i].clear();
                YV[i].shrink_to_fit();
                YI[i].clear();
                YI[i].shrink_to_fit();
            }
        }
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::execute(Integer_Type num_iterations)
{
    double t1, t2;
    t1 = Env::clock();

    if(not already_initialized)
        initialize();

    if(tc_family)
    {
        combine();
        apply();
    }
    else
    {
        if(!num_iterations)
            check_for_convergence = true; 

        while(true)
        {
            scatter_gather();
            
            combine();
            apply();
            
            iteration++;
            Env::print_me("Iteration: ", iteration);            
            if(check_for_convergence)
            {
                if(has_converged())
                    break;
            }
            else if(iteration >= num_iterations)
            {
                break;
            }
        }
    }
    t2 = Env::clock();
    Env::print_time("Execute", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize()
{
    double t1, t2;
    t1 = Env::clock();
    
    if(tc_family)
    {
        init_tc_family();
    }
    else
    {
        if(stationary)
        {
            init_stationary();
            init_stationary_postprocess();
        }
        else
        {
            init_stationary();
            init_nonstationary();
            init_nonstationary_postprocess();
        }
    }

    t2 = Env::clock();
    Env::print_time("Init", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize(
    const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_> &VProgram)
{
    double t1, t2;
    t1 = Env::clock();
    
    if(tc_family)
    {
        init_tc_family();
        // Initialiaze states
        std::vector<std::vector<Integer_Type>> W_ = VProgram.W;
        R = W_;
        std::vector<Integer_Type> d_sizes;
        if(filtering_type == _NONE_)
            d_sizes.resize(nrowgrps, tile_height);
        else if(filtering_type == _SOME_) 
        {
            //z_sizes = nnz_row_sizes_loc;
            //d_sizes = nnz_row_sizes_all;
            fprintf(stderr, "Invalid filtering type\n");
            Env::exit(1);
        }
        // Initialiaze BIG 2D vector of states
        D.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D[i].resize(d_sizes[i]);
        D_SIZE.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D_SIZE[i].resize(d_sizes[i] + 1); // 0 + nitems
    }
    else
    {
        if(stationary)
        {
            init_stationary();
            Integer_Type v_nitems = V.size();
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i]; 
                C[i] = initializer(get_vid(i), state, (const State&) VProgram.V[i]);
            }
            
            init_stationary_postprocess();
        }
        else
        {
            init_stationary();
            init_nonstationary();
            init_nonstationary_postprocess();
        }
    }
    already_initialized = true;
    t2 = Env::clock();
    Env::print_time("Init", t2 - t1);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_tc_family()
{
    //std::vector<Integer_Type> d_sizes;
    std::vector<Integer_Type> z_size;
    std::vector<Integer_Type> z_sizes;
    std::vector<Integer_Type> zz_sizes;
    
    // Initialize values and activity pattern
    W.resize(tile_height);
    
    if(filtering_type == _NONE_)
    {
        z_sizes.resize(rank_nrowgrps, tile_height);
       // d_sizes.resize(nrowgrps, tile_height);
    }
    else if(filtering_type == _SOME_) 
    {
        //z_sizes = nnz_row_sizes_loc;
        //d_sizes = nnz_row_sizes_all;
        fprintf(stderr, "Invalid filtering type\n");
        Env::exit(1);
    }

    // Initialiaze accumulators
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_stationary()
{
    // Initialize Values
    V.resize(tile_height);
    Integer_Type v_nitems = V.size();
    C.resize(tile_height);
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V[i]; 
        C[i] = initializer(get_vid(i), state);
    }
    
    // Initialize messages
    std::vector<Integer_Type> x_sizes;
    if(filtering_type == _NONE_)
        x_sizes.resize(rank_ncolgrps, tile_height);
    else if(filtering_type == _SOME_)
        x_sizes = nnz_col_sizes_loc;
    X.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        X[i].resize(x_sizes[i]);

    // Initialiaze accumulators
    std::vector<Integer_Type> y_sizes;
    if(filtering_type == _NONE_)
        y_sizes.resize(rank_nrowgrps, tile_height);
    else if(filtering_type == _SOME_) 
        y_sizes = nnz_row_sizes_loc;
    Y.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(local_row_segments[i] == owned_segment)
        {
            Y[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                Y[i][j].resize(y_sizes[i]);
        }
        else
        {
            Y[i].resize(1);
            Y[i][0].resize(y_sizes[i]);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_stationary_postprocess()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type v_nitems = V.size();
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            x_data[i] = messenger(state);
        }
    }
    else if(filtering_type == _SOME_)
    {
        auto &j_data = (*J)[xo];
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                Vertex_State &state = V[i];
                x_data[j] = messenger(state);
                j++;
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_nonstationary()
{
    Integer_Type v_nitems = V.size();
    // Initialize activity statuses for all column groups
    // Assuming ncolgrps == nrowgrps
    activity_statuses.resize(ncolgrps);
    
    std::vector<Integer_Type> x_sizes;
    if(filtering_type == _NONE_)
        x_sizes.resize(rank_ncolgrps, tile_height);
    else if(filtering_type == _SOME_)
        x_sizes = nnz_col_sizes_loc;
    // Initialize nonstationary messages values
    XV.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XV[i].resize(x_sizes[i]);
    // Initialize nonstationary messages indices
    XI.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XI[i].resize(x_sizes[i]);
    
    msgs_activity_statuses.resize(colgrp_nranks);

    std::vector<Integer_Type> y_sizes;
    if(filtering_type == _NONE_)
        y_sizes.resize(rank_nrowgrps, tile_height);
    else if(filtering_type == _SOME_) 
        y_sizes = nnz_row_sizes_loc;
    
    T.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        T[i].resize(y_sizes[i]);
    
    accus_activity_statuses.resize(rowgrp_nranks);
    
    // Initialiaze nonstationary accumulators values
    YV.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(local_row_segments[i] == owned_segment)
        {
            YV[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                YV[i][j].resize(y_sizes[i]);
        }
        else
        {
            YV[i].resize(1);
            YV[i][0].resize(y_sizes[i]);
        }
    }
    // Initialiaze nonstationary accumulators indices
    YI.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(local_row_segments[i] == owned_segment)
        {
            YI[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                YI[i][j].resize(y_sizes[i]);
        }
        else
        {
            YI[i].resize(1);
            YI[i][0].resize(y_sizes[i]);
        }
    }
    
    for(uint32_t k = 0; k < rank_nrowgrps; k++)
    {
        uint32_t yi = k;
        uint32_t yo = 0;
        if(local_row_segments[k] == owned_segment)
            yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];

        if(filtering_type == _NONE_)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
                y_data[i] = infinity();
        }
        else if(filtering_type == _SOME_)
        {
            auto &i_data = (*I)[yi];       
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                if(i_data[i])
                {
                    y_data[j] = infinity();
                    j++;
                }
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_nonstationary_postprocess()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    std::vector<Fractional_Type> &xv_data = XV[xo];
    std::vector<Integer_Type> &xi_data = XI[xo];
    
    Integer_Type v_nitems = V.size();
    Integer_Type k = 0;
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            if(C[i])
            {
                x_data[i] = messenger(state);
                xv_data[k] = x_data[i];
                xi_data[k] = i;
                k++;
            }
            else
                x_data[i] = infinity();
        }
    }
    else if(filtering_type == _SOME_)
    {
        if(not directed)
        {
            uint32_t yi = accu_segment_row;
            auto &i_data = (*I)[yi];
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {    
                if(i_data[i])
                {
                    Vertex_State &state = V[i];
                    if(C[i])
                    {
                        x_data[j] = messenger(state);
                        xv_data[k] = x_data[j];
                        xi_data[k] = j;
                        k++;
                    }
                    else
                        x_data[j] = infinity();
                    j++;
                }
            }
        }
        else
        {
            auto &j_data = (*J)[xo];
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                
                if(j_data[i])
                {
                    Vertex_State &state = V[i];
                    if(C[i])                            
                    {
                        x_data[j] = messenger(state);
                        xv_data[k] = x_data[j];
                        xi_data[k] = j;
                        k++;
                    }
                    else
                        x_data[j] = infinity();
                    j++;
                }
            }
        }
    }
    
    if(activity_filtering)
        msgs_activity_statuses[xo] = k;
    else
        msgs_activity_statuses[xo] = 0;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather()
{
    double t1, t2, elapsed_time;
    t1 = Env::clock();

    if(stationary)
    {
        if(iteration > 0)
            scatter_gather_stationary();
        
        if(Env::comm_split)
        {
            if(broadcast_communication)
                bcast_stationary();
            else
            {
                scatter_stationary();
                gather_stationary();
            }
        }
        else
        {
            scatter_stationary();
            gather_stationary();
        }
    }
    else
    {
        if(iteration > 0)
            scatter_gather_nonstationary();
        
        scatter_gather_nonstationary_activity_filtering();
        
        if(Env::comm_split)
        {
            if(broadcast_communication)
                bcast_nonstationary();
            else
            {   
                scatter_nonstationary();
                gather_nonstationary();
            }
        }
        else
        {
            scatter_nonstationary();
            gather_nonstationary();
        }
    }

    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Scatter_gather", elapsed_time);
    scatter_gather_time.push_back(elapsed_time);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_stationary()
{
    uint32_t xo = accu_segment_col;    
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type v_nitems = V.size();
    
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            x_data[i] = messenger(state);
        }
    }
    else if(filtering_type == _SOME_)
    {
        
        auto &v2j_data = (*V2J);
        auto &j2v_data = (*J2V);
        Integer_Type v2j_nitems = v2j_data.size();
        for(uint32_t i = 0; i < v2j_nitems; i++)
        {
            Vertex_State &state = V[v2j_data[i]];
            x_data[j2v_data[i]] = messenger(state);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_nonstationary()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    std::vector<Fractional_Type> &xv_data = XV[xo];
    std::vector<Integer_Type> &xi_data = XI[xo];
    
    Integer_Type v_nitems = V.size();
    Integer_Type k = 0;
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            if(C[i])
            {
                x_data[i] = messenger(state);
                xv_data[k] = x_data[i];
                xi_data[k] = i;
                k++;
            }
            else
                x_data[i] = infinity();
        }
    }
    else if(filtering_type == _SOME_)
    {
        auto &v2j_data = (*V2J);
        auto &j2v_data = (*J2V);
        Integer_Type v2j_nitems = v2j_data.size();
        if(not directed)
        {
            for(uint32_t i = 0; i < v2j_nitems; i++)
            {
                Vertex_State &state = V[v2j_data[i]];
                if(C[v2j_data[i]])
                {
                    x_data[j2v_data[i]] = messenger(state);
                    xv_data[k] = x_data[j2v_data[i]];
                    xi_data[k] = j2v_data[i];
                    k++;
                }
                else
                    x_data[j2v_data[i]] = infinity();
            }
        }
        else
        {
            auto &v2i_data = (*V2I);
            auto &i2v_data = (*I2V);
            Integer_Type v2i_nitems = v2i_data.size();
            for(uint32_t i = 0; i < v2i_nitems; i++)
            {
                Vertex_State &state = V[v2i_data[i]];
                if(C[v2i_data[i]])
                {
                    x_data[i2v_data[i]] = messenger(state);
                    xv_data[k] = x_data[i2v_data[i]];
                    xi_data[k] = i2v_data[i];
                    k++;
                }
                else
                    x_data[i2v_data[i]] = infinity();
            
            }
        }
    }
    
    if(activity_filtering)
        msgs_activity_statuses[xo] = k;
    else
        msgs_activity_statuses[xo] = 0;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_nonstationary_activity_filtering()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    if(activity_filtering)
    {
        int nitems = msgs_activity_statuses[xo];
        // 0 all, 1 nothing, else nitems
        double ratio = (double) nitems/x_nitems;
        if(ratio <= activity_filtering_ratio)
            nitems++;
        else
            nitems = 0;
        msgs_activity_statuses[xo] = nitems;
        
        activity_statuses[owned_segment] = msgs_activity_statuses[xo];
        
        Env::barrier();
        for(int32_t i = 0; i < Env::nranks; i++)
        {
            int32_t r = leader_ranks[i];
            if(r != Env::rank)
            {
                MPI_Sendrecv(&activity_statuses[owned_segment], 1, TYPE_INT, r, Env::rank, 
                             &activity_statuses[i], 1, TYPE_INT, r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
            }
        }
        Env::barrier();
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_stationary()
{
   // printf("SCATTER:r=%d colgrp_nranks=%d rowgrp_nranks=%d\n", Env::rank, colgrp_nranks, rowgrp_nranks);
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    int32_t col_group = local_col_segments[xo];
    
    MPI_Request request;
    uint32_t follower;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < colgrp_nranks - 1; i++)
        {
            if(Env::comm_split)
            {
               follower = follower_colgrp_ranks_cg[i];
               //printf("scatter:r=%d leader=%d --> follower=%d %d\n", Env::rank, Env::rank, follower, follower_rowgrp_ranks_rg.size());
               MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
               out_requests.push_back(request);
            }
            else
            {
                follower = follower_colgrp_ranks[i];
                //printf("scatter:r=%d leader=%d --> follower=%d\n", Env::rank, Env::rank, follower);
                MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_stationary()
{  
   // printf("GATHER:r=%d rank_ncolgrps=%d rank_nrowgrps=%d\n", Env::rank, rank_ncolgrps, rank_nrowgrps);
    int32_t leader, my_rank;
    MPI_Request request;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            std::vector<Fractional_Type> &xj_data = X[i];
            Integer_Type xj_nitems = xj_data.size();
            int32_t col_group = local_col_segments[i];
            if(Env::comm_split)
            {
                leader = leader_ranks_cg[col_group];
                if(ordering_type == _ROW_)
                    my_rank = Env::rank_cg;
                else
                    my_rank = Env::rank_rg;
                    
                if(leader != my_rank)
                {
                    MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                    //printf("gather:i=%d r=%d follower=%d <-- leader=%d\n", i, Env::rank, Env::rank, leader);
                }
            }
            else
            {
                leader = leader_ranks[col_group];
                if(leader != Env::rank)
                {
                    MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                 //   printf("gather:i=%d r=%d follower=%d <-- leader=%d\n", i, Env::rank, Env::rank, leader);
                }
            }
        }
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        //MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        //out_requests.clear();
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_stationary()
{
    MPI_Request request;
    int32_t leader;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            int32_t col_group = local_col_segments[i];
            leader = leader_ranks_cg[col_group];
            //leader = leader_ranks_cg[local_col_segments[i]];
            std::vector<Fractional_Type> &xj_data = X[i];
            Integer_Type xj_nitems = xj_data.size();
            if(Env::comm_split)
            {
                //MPI_Bcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicator);
                MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicator, &request);
                out_requests.push_back(request);
            }
            else
            {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
        }
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_nonstationary()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    std::vector<Fractional_Type> &xv_data = XV[xo];
    std::vector<Integer_Type> &xi_data = XI[xo];
    int nitems = msgs_activity_statuses[xo];
    int32_t col_group = local_col_segments[xo];
    
    MPI_Request request;
    uint32_t follower;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < colgrp_nranks - 1; i++)
        {
            if(Env::comm_split)
            {
                follower = follower_colgrp_ranks_cg[i];
                MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, colgrps_communicator);
                if(activity_filtering and nitems)
                {
                    if(nitems > 1)
                    {
                        MPI_Isend(xi_data.data(), nitems - 1, TYPE_INT, follower, col_group, colgrps_communicator, &request);
                        out_requests.push_back(request);
                       
                        MPI_Isend(xv_data.data(), nitems - 1, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
                        out_requests.push_back(request);
                    }
                }
                else
                {
                    MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                follower = follower_colgrp_ranks[i];
                MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
                if(activity_filtering and nitems)
                {
                    if(nitems > 1)
                    {
                        MPI_Isend(xi_data.data(), nitems - 1, TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
                        out_requests.push_back(request);
                        MPI_Isend(xv_data.data(), nitems - 1, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
                        out_requests.push_back(request);
                    }
                }
                else
                {
                    MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_nonstationary()
{  
    int32_t leader;
    MPI_Request request;
    MPI_Status status;
    int nitems = 0;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            std::vector<Fractional_Type> &xj_data = X[i];
            Integer_Type xj_nitems = xj_data.size();
            std::vector<Integer_Type> &xij_data = XI[i];
            std::vector<Fractional_Type> &xvj_data = XV[i];
            int32_t col_group = local_col_segments[i];
            if(Env::comm_split)
            {
                leader = leader_ranks_cg[col_group];
                if(leader != Env::rank_cg)
                {
                    
                    MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, colgrps_communicator, &status);
                    msgs_activity_statuses[i] = nitems;
                    if(activity_filtering and nitems)
                    {
                        if(nitems > 1)
                        {
                            MPI_Irecv(xij_data.data(), nitems - 1, TYPE_INT, leader, col_group, colgrps_communicator, &request);
                            in_requests.push_back(request);
                            MPI_Irecv(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                            in_requests.push_back(request);
                        }    
                    }
                    else
                    {
                        MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                        in_requests.push_back(request);
                    }
                }
            }
            else
            {
                leader = leader_ranks[col_group];
                if(leader != Env::rank)
                {
                    MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
                    msgs_activity_statuses[i] = nitems;
                    if(activity_filtering and nitems)
                    {
                        if(nitems > 1)
                        {
                            MPI_Irecv(xij_data.data(), nitems - 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
                            in_requests.push_back(request);
                            MPI_Irecv(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                            in_requests.push_back(request);
                        }
                    }
                    else
                    {
                        MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                    }
                }
            }
        }
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        //MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        //out_requests.clear();
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_nonstationary()
{
    MPI_Request request;
    int32_t leader_cg;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            leader_cg = leader_ranks_cg[local_col_segments[i]]; 
            std::vector<Fractional_Type> &xj_data = X[i];
            Integer_Type xj_nitems = xj_data.size();
            std::vector<Integer_Type> &xij_data = XI[i];
            std::vector<Fractional_Type> &xvj_data = XV[i];
            int nitems = 0;
            if(Env::rank_cg == leader_cg)
                nitems = msgs_activity_statuses[i];
            //MPI_Bcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicator);
            MPI_Ibcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicator, &request);
            MPI_Wait(&request, MPI_STATUSES_IGNORE);
            
            if(Env::rank_cg != leader_cg)
                msgs_activity_statuses[i] = nitems;
            
            if(activity_filtering and nitems)
            {
                if(Env::comm_split)
                {
                    if(nitems > 1)
                    {
                        //MPI_Bcast(xij_data.data(), nitems - 1, TYPE_INT, leader_cg, colgrps_communicator);
                        //MPI_Bcast(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicator);
                        
                        MPI_Ibcast(xij_data.data(), nitems - 1, TYPE_INT, leader_cg, colgrps_communicator, &request);
                        out_requests.push_back(request);
                        MPI_Ibcast(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
                        out_requests.push_back(request);
                    }
                }
                else
                {
                    fprintf(stderr, "Invalid communicator\n");
                    Env::exit(1);
                }
            }
            else
            {
                if(Env::comm_split)
                {
                    //MPI_Bcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicator);
                    MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
                    out_requests.push_back(request);
                }
                else
                {
                    fprintf(stderr, "Invalid communicator\n");
                    Env::exit(1);
                }
            }
        }
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




                  
/* specialized triangle countring spmv */
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv(
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
            else if(ordering_type == _COL_)
            {
                
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                    {
                        //pair = A->base({j + (owned_segment * tile_width), IA[i]}, tile.rg, tile.cg);
                        pair = A->base({IA[i] + (owned_segment * tile_height), j}, tile.rg, tile.cg);
                        z_data[j].push_back(pair.col);
                    }
                }
            }                
        }
        else
        {
            fprintf(stderr, "Invalid compression type\n");
            Env::exit(1);
        }
    }
}



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv(
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<Fractional_Type> &y_data,
            std::vector<Fractional_Type> &x_data,
            std::vector<Fractional_Type> &xv_data, 
            std::vector<Integer_Type> &xi_data, std::vector<char> &t_data)
{
    if(tile.allocated)
    {
        if(compression_type == Compression_type::_CSR_)
        {
            fprintf(stderr, "Invalid compression type for nonstationary algorithms\n");
            Env::exit(1);
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
                //if(activity_filtering and msgs_activity_statuses[tile.jth])
                if(activity_filtering and activity_statuses[tile.cg])
                {
                    Integer_Type s_nitems = msgs_activity_statuses[tile.jth] - 1;
                    Integer_Type j = 0;
                    for(Integer_Type k = 0; k < s_nitems; k++)
                    {
                        j = xi_data[k];
                        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                        {
                            #ifdef HAS_WEIGHT
                            combiner(y_data[IA[i]], xv_data[k], A[i]);
                            #else
                            combiner(y_data[IA[i]], xv_data[k]);
                            #endif
                            t_data[IA[i]] = 1;
                        }
                    }
                }
                else
                {
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        if(x_data[j] != infinity())
                        {
                            for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                            {
                                #ifdef HAS_WEIGHT
                                combiner(y_data[IA[i]], x_data[j], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[j]);
                                #endif
                                t_data[IA[i]] = 1;
                            }
                        }
                    }
                }
            }
            else if(ordering_type == _COL_)
            {
                fprintf(stderr, "Invalid compression type for nonstationary algorithms\n");
                Env::exit(1);
            }
        }
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv(
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<Fractional_Type> &y_data,
            std::vector<Fractional_Type> &x_data)
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
                        combiner(y_data[i], x_data[JA[j]], A[j]);
                        #else
                        combiner(y_data[i], x_data[JA[j]]);
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
                        combiner(y_data[JA[j]], x_data[i], A[j]);
                        #else
                        combiner(y_data[JA[j]], x_data[i]);
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
                        combiner(y_data[IA[i]], x_data[j], A[i]);
                        #else
                        combiner(y_data[IA[i]], x_data[j]);
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
                        combiner(y_data[j], x_data[IA[i]], A[i]);   
                        #else
                        combiner(y_data[j], x_data[IA[i]]);
                        #endif
                    }
                }
            }
        }            
    }    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine()
{
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    
    if(stationary)
    {
        if((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
            combine_2d_stationary();
        else if(tiling_type == Tiling_type::_1D_ROW)
        {  
            if(ordering_type == _ROW_)
                combine_1d_row_stationary();
            else if(ordering_type == _COL_)
                combine_1d_col_stationary();
        }
        else if(tiling_type == Tiling_type::_1D_COL)
        {
            if(ordering_type == _ROW_)
                combine_1d_col_stationary();    
            else if(ordering_type == _COL_)
                combine_1d_row_stationary();
        }
        else
        {
            fprintf(stderr, "Invalid combine configuration for a stationary algorithm\n");
            Env::exit(1);
        }
        
        if(not tiling_type == Tiling_type::_1D_ROW)
            combine_postprocess();
    }
    else
    {    
        if((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        {
            if(tc_family)
                combine_2d_for_tc();
            else
            {
                combine_2d_nonstationary();
                combine_postprocess();
            }
        }
        else
        {
            fprintf(stderr, "Invalid combine configuration for a nonstationary algorithm\n");
            Env::exit(1);
        }
    }
        
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Combine", elapsed_time);
    combine_time.push_back(elapsed_time);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_1d_row_stationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t xi = 0;
    uint32_t yo = accu_segment_rg;
  
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        std::vector<Fractional_Type> &x_data = X[xi];
        spmv(tile, y_data, x_data);
        xi++;
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_1d_col_stationary()
{
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
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
        
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        std::vector<Fractional_Type> &x_data = X[xi];
        spmv(tile, y_data, x_data);
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
        
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
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
                    
                    std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                    Integer_Type yj_nitems = yj_data.size();
                    MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);                   
                }
            }
            else
            {
                MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
                
            }
            yi++;
        }
    }
    //wait_for_all();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_for_tc()
{
    MPI_Request request;
    MPI_Status status;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t yi = 0, yo = 0, oi = 0;
    
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
                   
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_stationary()
{
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    
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
        
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        
        std::vector<Fractional_Type> &x_data = X[xi];
        spmv(tile, y_data, x_data);

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
                    std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                    Integer_Type yj_nitems = yj_data.size();
                    
                    MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                }
                
            }
            else
            {
                MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);

            }
            xi = 0;
            yi++;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_nonstationary()
{
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;

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
        
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();

        std::vector<Fractional_Type> &x_data = X[xi];
        std::vector<Fractional_Type> &xv_data = XV[xi];
        std::vector<Integer_Type> &xi_data = XI[xi];
        std::vector<char> &t_data = T[yi];
        
        spmv(tile, y_data, x_data, xv_data, xi_data, t_data);
        
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
                    if(activity_filtering and activity_statuses[tile.rg])
                    {
                        // 0 all / 1 nothing / else nitems 
                        int nitems = 0;
                        MPI_Status status;
                        MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, communicator, &status);
                        accus_activity_statuses[accu] = nitems;
                        
                        if(accus_activity_statuses[accu] > 1)
                        {
                            std::vector<Integer_Type> &yij_data = YI[yi][accu];
                            std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                            MPI_Irecv(yij_data.data(), accus_activity_statuses[accu] - 1, TYPE_INT, follower, pair_idx, communicator, &request);
                            in_requests.push_back(request);
                            MPI_Irecv(yvj_data.data(), accus_activity_statuses[accu] - 1, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                            in_requests_.push_back(request);
                        }
                    }
                    else
                    {                                
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = yj_data.size();
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                        in_requests.push_back(request);
                    }       
                    
                }   
            }
            else
            {
                std::vector<Integer_Type> &yi_data = YI[yi][yo];
                std::vector<Fractional_Type> &yv_data = YV[yi][yo];
                int nitems = 0;
                
                if(activity_filtering and activity_statuses[tile.rg])
                {
                    std::vector<char> &t_data = T[yi];
                    Integer_Type j = 0;
                    for(uint32_t i = 0; i < y_nitems; i++)
                    {
                        if(t_data[i])
                        {
                            yi_data[j] = i;
                            yv_data[j] = y_data[i];
                            j++;
                        }
                    }
                    nitems = j;
                    nitems++;
                    MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, communicator);
                    if(nitems > 1)
                    {
                        MPI_Isend(yi_data.data(), nitems - 1, TYPE_INT, leader, pair_idx, communicator, &request);
                        out_requests.push_back(request);
                        MPI_Isend(yv_data.data(), nitems - 1, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                        out_requests_.push_back(request);
                    }
                }
                else
                {
                    MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                    out_requests.push_back(request);
                }
            }
            xi = 0;
            yi++;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess()
{
    if(incremental_accumulation)
    {
        if(stationary)
            combine_postprocess_stationary_for_some();
        else
        {
            combine_postprocess_nonstationary_for_some();
            
            std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
            std::fill(accus_activity_statuses.begin(), accus_activity_statuses.end(), 0);
        }
    }
    else
    {
        if(stationary)
            combine_postprocess_stationary_for_all();
        else
        {
            combine_postprocess_nonstationary_for_all();
            
            std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
            std::fill(accus_activity_statuses.begin(), accus_activity_statuses.end(), 0);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_stationary_for_all()
{
    wait_for_recvs();
    
    uint32_t accu = 0;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];
    
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];

        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
        Integer_Type yj_nitems = yj_data.size();
        for(uint32_t i = 0; i < yj_nitems; i++)
            combiner(y_data[i], yj_data[i]);
    }
    //wait_for_sends();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_stationary_for_some()
{
    uint32_t accu = 0;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];
    
    int32_t incount = in_requests.size();
    int32_t outcount = 0;
    int32_t incounts = rowgrp_nranks - 1;
    std::vector<MPI_Status> statuses(incounts);
    std::vector<int32_t> indices(incounts);
    int32_t received = 0;

    while(received < incount)
    {
        MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
        assert(outcount != MPI_UNDEFINED);
        
        for(int32_t i = 0; i < outcount; i++)
        {
            uint32_t j = indices[i];
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            for(uint32_t i = 0; i < yj_nitems; i++)
                combiner(y_data[i], yj_data[i]);            
        }
        received += outcount;
    }
    in_requests.clear();
    //wait_for_sends();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_nonstationary_for_all()
{
    wait_for_recvs();
    uint32_t accu = 0;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];

    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];

        if(activity_filtering and accus_activity_statuses[accu])
        {
            if(accus_activity_statuses[accu] > 1)
            {
                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                for(uint32_t i = 0; i < accus_activity_statuses[accu] - 1; i++)
                {
                    Integer_Type k = yij_data[i];
                    combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else
        {
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            for(uint32_t i = 0; i < yj_nitems; i++)
                combiner(y_data[i], yj_data[i]);
        }
    }
    //wait_for_sends();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_nonstationary_for_some()
{
    Env::barrier();    
    if(activity_filtering and activity_statuses[owned_segment])
    {
        if(in_requests.size())
        {
            uint32_t accu = 0;
            uint32_t yi = accu_segment_row;
            uint32_t yo = accu_segment_rg;
            std::vector<Fractional_Type> &y_data = Y[yi][yo];
            
            int32_t outcount  = 0;
            int32_t outcount_ = 0;
            int32_t received = 0;
            int32_t received_ = 0;

            if((in_requests.size() + in_requests_.size()) == 2 * (rowgrp_nranks - 1))
            {
                int32_t incount  = rowgrp_nranks - 1;
                int32_t incounts = rowgrp_nranks - 1;
                std::vector<MPI_Status> statuses(incounts);
                std::vector<int32_t> indices(incounts);
                
                int32_t incount_  = rowgrp_nranks - 1;
                int32_t incounts_ = rowgrp_nranks - 1;
                std::vector<MPI_Status> statuses_(incounts_);
                std::vector<int32_t> indices_(incounts_);
                std::vector<int32_t> indices_all(incounts_);
                
                while((received + received_) < (incount + incount_))
                {
                    if(received < incount)
                    {
                        MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
                        assert(outcount != MPI_UNDEFINED);
                    }

                    if(outcount)
                    {
                        for(int32_t i = 0; i < outcount; i++)
                        {
                            uint32_t j = indices[i];
                            indices_all[j]++;
                            
                            if(indices_all[j] == 2)
                            {                
                                if(Env::comm_split)
                                    accu = follower_rowgrp_ranks_accu_seg_rg[j];
                                else
                                    accu = follower_rowgrp_ranks_accu_seg[j];
                                
                                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                                for(uint32_t k = 0; k < accus_activity_statuses[accu] - 1; k++)
                                {
                                    Integer_Type l = yij_data[k];
                                    combiner(y_data[l], yvj_data[k]);
                                }
                                
                                indices_all[j]++;
                            } 
                        }
                        received += outcount;
                    }
                    
                    if(received_ < incount_)
                    {
                        MPI_Waitsome(in_requests_.size(), in_requests_.data(), &outcount_, indices_.data(), statuses_.data());
                        assert(outcount_ != MPI_UNDEFINED);
                    }
                    
                    if(outcount_)
                    {
                        for(int32_t i = 0; i < outcount_; i++)
                        {
                            uint32_t j = indices_[i];
                            indices_all[j]++;

                            if(indices_all[j] == 2)
                            {                
                                if(Env::comm_split)
                                    accu = follower_rowgrp_ranks_accu_seg_rg[j];
                                else
                                    accu = follower_rowgrp_ranks_accu_seg[j];
                                
                                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                                for(uint32_t k = 0; k < accus_activity_statuses[accu] - 1; k++)
                                {
                                    Integer_Type l = yij_data[k];
                                    combiner(y_data[l], yvj_data[k]);
                                }
                                indices_all[j]++;
                            } 
                        }
                        received_ += outcount_;
                    }
                }
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                {
                    if(indices_all[j] == 2)
                    {               
                        if(Env::comm_split)
                            accu = follower_rowgrp_ranks_accu_seg_rg[j];
                        else
                            accu = follower_rowgrp_ranks_accu_seg[j];
                        
                        std::vector<Integer_Type> &yij_data = YI[yi][accu];
                        std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                        for(uint32_t i = 0; i < accus_activity_statuses[accu] - 1; i++)
                        {
                            Integer_Type k = yij_data[i];
                            combiner(y_data[k], yvj_data[i]);
                        }
                    }
                }
            }
            else
            {
                int32_t incount  = in_requests.size();
                int32_t incounts = in_requests.size();
                std::vector<MPI_Status> statuses(incounts);
                std::vector<int32_t> indices(incounts);
                
                int32_t incount_  = in_requests_.size();
                int32_t incounts_ = in_requests_.size();
                std::vector<MPI_Status> statuses_(incounts_);
                std::vector<int32_t> indices_(incounts_);
                
                std::vector<int32_t> indices_all(incounts);
                std::vector<int32_t> indices_accu(incounts);
                
                int32_t idx = 0;
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
                {
                    if(Env::comm_split)
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    else
                        accu = follower_rowgrp_ranks_accu_seg[j];
                    
                    if(accus_activity_statuses[accu] > 1)
                    {
                        indices_accu[idx] = accu;
                        idx++;
                    }
                }
                assert(idx == incount);
                
                while((received + received_) < (incount + incount_))
                {
                    if(received < incount)
                    {
                        MPI_Waitsome(in_requests.size(), in_requests.data(), &outcount, indices.data(), statuses.data());
                        assert(outcount != MPI_UNDEFINED);
                    }

                    if(outcount)
                    {
                        for(int32_t i = 0; i < outcount; i++)
                        {
                            uint32_t j = indices[i];
                            indices_all[j]++;
                            
                            if(indices_all[j] == 2)
                            {                
                                accu = indices_accu[j];

                                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];                                
                                for(uint32_t k = 0; k < accus_activity_statuses[accu] - 1; k++)
                                {
                                    Integer_Type l = yij_data[k];
                                    combiner(y_data[l], yvj_data[k]);
                                }
                                indices_all[j]++;
                            } 
                        }
                        received += outcount;
                        outcount = 0;
                    }
                    if(received_ < incount_)
                    {
                        MPI_Waitsome(in_requests_.size(), in_requests_.data(), &outcount_, indices_.data(), statuses_.data());
                        assert(outcount_ != MPI_UNDEFINED);
                    }
                    if(outcount_)
                    {
                        for(int32_t i = 0; i < outcount_; i++)
                        {
                            uint32_t j = indices_[i];
                            indices_all[j]++;
                            
                            if(indices_all[j] == 2)
                            {                
                                accu = indices_accu[j];

                                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                                for(uint32_t k = 0; k < accus_activity_statuses[accu] - 1; k++)
                                {
                                    Integer_Type l = yij_data[k];
                                    combiner(y_data[l], yvj_data[k]);
                                }
                                indices_all[j]++;
                            } 
                        }
                        received_ += outcount_;
                        outcount_ = 0;
                    }
                }
                
                for(int32_t j = 0; j < incount; j++)
                {
                    if(indices_all[j] == 2)
                    {
                        accu = indices_accu[j];
                        std::vector<Integer_Type> &yij_data = YI[yi][accu];
                        std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                        for(Integer_Type i = 0; i < accus_activity_statuses[accu] - 1; i++)
                        {
                            Integer_Type k = yij_data[i];
                            combiner(y_data[k], yvj_data[i]);
                        }
                    }
                }
            }
            in_requests.clear();
            in_requests_.clear();   
            //wait_for_sends();
        }
    }
    else
        combine_postprocess_stationary_for_some();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_all()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    if(not stationary and activity_filtering)
    {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
    }
    //Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_sends()
{
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    if(not stationary and activity_filtering)
    {
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_recvs()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    if(not stationary and activity_filtering)
    {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply()
{
    double t1, t2, elapsed_time;
    t1 = Env::clock();

    if(stationary)
    {
        apply_stationary();
    }
    else
    {
        if(tc_family)
            apply_tc();
        else
            apply_nonstationary();
    }

    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Apply", elapsed_time);
    apply_time.push_back(elapsed_time);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_stationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];
    
    Integer_Type v_nitems = V.size();
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            C[i] = applicator(state, y_data[i]);
        }
    }
    else if(filtering_type == _SOME_)
    {
        if(iteration == 0)
        {
            auto &i_data = (*I)[yi];
            Integer_Type j = 0;
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                if(i_data[i])
                {
                    C[i] = applicator(state, y_data[j]);
                    j++;
                }
                else
                    C[i] = applicator(state);
            }
        }
        else    
        {
            auto &y2v_data = (*Y2V);
            auto &v2y_data = (*V2Y);
            Integer_Type y2v_nitems = y2v_data.size();
            for(uint32_t i = 0; i < y2v_nitems; i++)
            {
                Vertex_State &state = V[v2y_data[i]];
                C[v2y_data[i]] = applicator(state, y_data[y2v_data[i]]);
            }
        }
    }

    wait_for_sends();
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        for(uint32_t j = 0; j < Y[i].size(); j++)
            std::fill(Y[i][j].begin(), Y[i][j].end(), 0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_nonstationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];

    Integer_Type v_nitems = V.size();
    if(filtering_type == _NONE_)
    {
        if(apply_depends_on_iter)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                C[i] = applicator(state, y_data[i], iteration);
            }
        }
        else
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                C[i] = applicator(state, y_data[i]);
            }
        }
    }
    else if(filtering_type == _SOME_)
    {
        auto &i_data = (*I)[yi];
        Integer_Type j = 0;
        if(apply_depends_on_iter)
        {
            if(iteration == 0)
            {
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                    {
                        C[i] = applicator(state, y_data[j], iteration);
                        j++;
                    }
                    else
                        C[i] = applicator(state);    
                }  
               
            }
            else
            {
                auto &y2v_data = (*Y2V);
                auto &v2y_data = (*V2Y);
                Integer_Type y2v_nitems = y2v_data.size();
                for(uint32_t i = 0; i < y2v_nitems; i++)
                {
                    Vertex_State &state = V[v2y_data[i]];
                    C[v2y_data[i]] = applicator(state, y_data[y2v_data[i]], iteration);
                }
            }
            
        }
        else
        {
            if(iteration == 0)
            {
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                    {
                        C[i] = applicator(state, y_data[j]);
                        j++;
                    }
                    else
                        C[i] = applicator(state);
                }
               
            }
            else
            {
                auto &y2v_data = (*Y2V);
                auto &v2y_data = (*V2Y);
                Integer_Type y2v_nitems = y2v_data.size();
                for(uint32_t i = 0; i < y2v_nitems; i++)
                {
                    Vertex_State &state = V[v2y_data[i]];
                    C[v2y_data[i]] = applicator(state, y_data[y2v_data[i]]);
                }
            }
            /*
            for(uint32_t i = 0; i < v_nitems; i++)
                printf("%d ", i);
            printf("\n");
            for(uint32_t i = 0; i < v_nitems; i++)
                printf("%d ", C[i]);
            printf("\n");
            */
        }
    }
    
    wait_for_sends();
    
    if(not gather_depends_on_apply and not apply_depends_on_iter)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            for(uint32_t j = 0; j < Y[i].size(); j++)
                std::fill(Y[i][j].begin(), Y[i][j].end(), 0);
        }
    } 

    if(activity_filtering)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            std::fill(T[i].begin(), T[i].end(), 0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_tc()
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
        D[owned_segment] = R;

        MPI_Request request;
        uint32_t my_rank, leader;
        
        std::vector<std::vector<Integer_Type>> boxes(nrows);
        std::vector<Integer_Type> &d_size = D_SIZE[owned_segment];
        Integer_Type d_s_nitems = d_size.size();
        auto &outbox = boxes[owned_segment];
        Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(d_size, r_data, outbox);

        for(uint32_t i = 0; i < nrowgrps; i++)  
        {
            int32_t r = (Env::rank + i) % Env::nranks;
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
            int32_t r = (Env::rank + i) % Env::nranks;
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
            if(i != (uint32_t) owned_segment)
            {
                auto &box = boxes[i];
                std::vector<Integer_Type> &dj_size = D_SIZE[i];
                std::vector<std::vector<Integer_Type>> &dj_data = D[i];
                Integer_Type dj_nitems = dj_size.size();
                for(uint32_t j = 0; j < dj_nitems - 1; j++)
                {
                    for(uint32_t k = dj_size[j]; k < dj_size[j+1]; k++)
                    {
                        //Integer_Type row = j + (i * tile_height);
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
            std::cout << "\nNum_triangles: " << num_triangles_global << std::endl;

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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Integer_Type Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::get_vid(Integer_Type index)
{
    return(index + (owned_segment * tile_height));
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
MPI_Comm Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::communicator_info()
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::has_converged()
{
    bool converged = false;
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;

    Integer_Type c_nitems = C.size();    
    
    for(uint32_t i = 0; i < c_nitems; i++)
    {
        if(not C[i]) 
            c_sum_local++;
    }
    //printf("%d %lu\n", iteration, (tile_height * Env::nranks) - c_sum_local);
   
    MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(c_sum_gloabl == (tile_height * Env::nranks))
        converged = true;
    
    return(converged);   
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::checksum()
{
    uint64_t v_sum_local = 0, v_sum_global = 0;
    Integer_Type v_nitems = V.size();
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V[i];
        if((state.get_state() != infinity()) and (get_vid(i) < nrows))    
                v_sum_local += state.get_state();
            
    }
    MPI_Allreduce(&v_sum_local, &v_sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
    {
        std::cout << "Iterations: " << iteration << std::endl;
        std::cout << std::fixed << "Value checksum: " << v_sum_global << std::endl;
    }

    //if(apply_depends_on_iter or gather_depends_on_apply)
    //{

        uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            if((state.get_state() != infinity()) and (get_vid(i) < nrows)) 
                v_sum_local_++;
        }

        MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(Env::is_master)
            std::cout << std::fixed << "Reachable vertices: " << v_sum_global_ << std::endl;
    //}
    Env::barrier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::display(Integer_Type count)
{
    Integer_Type v_nitems = V.size();
    count = (v_nitems < count) ? v_nitems : count;
    Env::barrier();
    Triple<Weight, Integer_Type> pair, pair1;
    Triple<Weight, double> stats_pair;
    if(!Env::rank)
    {
        stats_pair = stats(scatter_gather_time);
        std::cout << "Scatter_gather time (avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_time);
        std::cout << "Combine time        (avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(apply_time);
        std::cout << "Apply time          (avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            Vertex_State &state = V[i];
            std::cout << std::fixed <<  "vertex[" << A->hasher->unhash(pair1.row) << "]:" << state.print_state() << std::endl;
            //std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
        }
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, double> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::stats(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
    return{mean, std_dev};
}
#endif

/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include <type_traits>
#include <numeric>

#include "mpi/types.hpp" 

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
                        bool stationary_ = false, bool gather_depends_on_apply_ = false, 
                        bool apply_depends_on_iter_ = false, Ordering_type = _ROW_);
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
        Integer_Type iteration = 0;
        std::vector<Vertex_State> V;              // Values
        std::vector<std::vector<Integer_Type>> W; // Values (triangle counting)
    protected:
        bool already_initialized = false;
        bool check_for_convergence = false;
        void init_stationary();
        void init_nonstationary();
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
        void combine_2d_stationary();
        void combine_2d_nonstationary();
        void combine_postprocess();
        void combine_postprocess_stationary_for_all();
        void combine_postprocess_nonstationary_for_all();
        void apply();                        
        void apply_stationary();
        void apply_nonstationary();
        struct Triple<Weight, double> stats(std::vector<double> &vec);
        void stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev);
        
        void spmv(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data); // Stationary spmv/spmspv
                
        void spmv(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile, // Stationary spmv/spmspv 
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data, 
                std::vector<Fractional_Type> &xv_data, 
                std::vector<Integer_Type> &xi_data,
                std::vector<char> &t_data);
        
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
        //std::vector<int32_t> accu_segment_row_vec;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        //std::vector<int32_t> accu_segment_rg_vec;
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
        std::vector<std::vector<Integer_Type>> *IV;
        std::vector<std::vector<char>> *J;
        std::vector<std::vector<Integer_Type>> *JV;
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
        
        bool directed;
        bool transpose;
        double activity_filtering_ratio = 0.6;
        bool activity_filtering = true;
        bool accu_activity_filtering = false;
        bool msgs_activity_filtering = false;
        
        bool broadcast_communication = true;
        bool incremental_accumulation = false;
        
        #ifdef TIMING
        std::vector<double> init_time;
        std::vector<double> scatter_gather_time;
        std::vector<double> combine_time;
        std::vector<double> combine_comp_time;
        std::vector<double> combine_comm_time;
        std::vector<double> apply_time;
        std::vector<double> execute_time;
        #endif
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph, bool stationary_, 
         bool gather_depends_on_apply_, bool apply_depends_on_iter_, Ordering_type ordering_type_)
{

    A = Graph.A;
    directed = A->directed;
    transpose = A->transpose;
    stationary = stationary_;
    gather_depends_on_apply = gather_depends_on_apply_;
    apply_depends_on_iter = apply_depends_on_iter_;
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
        //accu_segment_row_vec = A->accu_segment_row_vec;
        //accu_segment_col_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_rowgrp_ranks_accu_seg;
        //accu_segment_rg_vec = A->accu_segment_rg_vec;
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
        IV= &(Graph.A->IV);
        J = &(Graph.A->J);
        JV= &(Graph.A->JV);
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
        //accu_segment_col_vec = A->accu_segment_row_vec;
        ///accu_segment_row_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_colgrp_ranks_accu_seg;
        //accu_segment_rg_vec = A->accu_segment_cg_vec;
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
        IV= &(Graph.A->JV);
        J = &(Graph.A->I);
        JV= &(Graph.A->IV);
        V2J = &(Graph.A->J2V);
        J2V = &(Graph.A->V2J);
        Y2V = &(Graph.A->V2Y);
        V2Y = &(Graph.A->Y2V);
        I2V = &(Graph.A->V2I);
        V2I = &(Graph.A->I2V);
        if(filtering_type == _SNKS_)
            filtering_type = _SRCS_;
        else if(filtering_type == _SRCS_)
            filtering_type = _SNKS_;
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::execute(Integer_Type num_iterations)
{
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    
    if(not already_initialized)
        initialize();

    if(!num_iterations)
        check_for_convergence = true; 

    while(true)
    {
        scatter_gather();
        
        combine();
        apply();
        
        iteration++;
        Env::print_num("Iteration: ", iteration);            
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
    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Execute", elapsed_time);
    #ifdef TIMING
    execute_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize()
{
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
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
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize(
    const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_> &VProgram)
{
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary)
    {
        init_stationary();
        Integer_Type v_nitems = V.size();
        //#pragma omp parallel for schedule(static)
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
    already_initialized = true;
    
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif
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
    if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
        x_sizes.resize(rank_ncolgrps, tile_height);
    else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
        x_sizes = nnz_col_sizes_loc;
    X.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        X[i].resize(x_sizes[i]);

    // Initialiaze accumulators
    std::vector<Integer_Type> y_sizes;
    if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
        y_sizes.resize(rank_nrowgrps, tile_height);
    else if((filtering_type == _SOME_) or (filtering_type == _SRCS_))
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
        if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
        {
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                x_data[i] = messenger(state);
            }

        }
        else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
        {
            
            auto &j_data = (*J)[xo];
            auto &jv_data = (*JV)[xo];
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                if(j_data[i])
                {
                    Vertex_State &state = V[i];
                    x_data[jv_data[i]] = messenger(state);
                }
            }
            /*
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
            */
        }
    //}
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_nonstationary()
{
    Integer_Type v_nitems = V.size();
    // Initialize activity statuses for all column groups
    // Assuming ncolgrps == nrowgrps
    activity_statuses.resize(ncolgrps);
    
    std::vector<Integer_Type> x_sizes;
    if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
        x_sizes.resize(rank_ncolgrps, tile_height);
    else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
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
    if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
        y_sizes.resize(rank_nrowgrps, tile_height);
    else if((filtering_type == _SOME_) or (filtering_type == _SRCS_))
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

        if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
        {
            for(uint32_t i = 0; i < v_nitems; i++)
                y_data[i] = infinity();
        }
        else if((filtering_type == _SOME_) or (filtering_type == _SRCS_))
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
    if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
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
    else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
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
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif

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

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Scatter_gather", elapsed_time);
    scatter_gather_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_stationary()
{
    uint32_t xo = accu_segment_col;    
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type v_nitems = V.size();
        if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
        {
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                x_data[i] = messenger(state);
            }
        }
        else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
        {
            
            auto &v2j_data = (*V2J);
            auto &j2v_data = (*J2V);
            Integer_Type v2j_nitems = v2j_data.size();
            //#pragma omp parallel for schedule(static)
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
    if((filtering_type == _NONE_) or (filtering_type == _SRCS_))
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
    else if((filtering_type == _SOME_) or (filtering_type == _SNKS_))
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
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_stationary()
{  
   // printf("GATHER:r=%d rank_ncolgrps=%d rank_nrowgrps=%d\n", Env::rank, rank_ncolgrps, rank_nrowgrps);
    int32_t leader, my_rank;
    MPI_Request request;
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_stationary()
{
    MPI_Request request;
    int32_t leader;
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_nonstationary()
{  
    int32_t leader;
    MPI_Request request;
    MPI_Status status;
    int nitems = 0;    
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_nonstationary()
{
    MPI_Request request;
    int32_t leader_cg;
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
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
    if(stationary)
    {
        combine_2d_stationary();
        combine_postprocess();
    }
    else
    {    
        combine_2d_nonstationary();
        combine_postprocess();
    }

    #ifdef TIMING    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    combine_comm_time.push_back(elapsed_time - combine_comp_time.back());
    Env::print_time("Combine communication", combine_comm_time.back());
    Env::print_time("Combine all", elapsed_time);
    combine_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_stationary()
{
    #ifdef TIMING
    double t1, t2, elapsed_time = 0;
    t1 = Env::clock();
    #endif
    
    MPI_Request request;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        uint32_t tile_th = pair1.row;
        uint32_t pair_idx = pair1.col;
        
        bool vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        
        std::vector<Fractional_Type> &x_data = X[xi];
        
        #ifdef TIMING
        t1 = Env::clock();
        #endif
        spmv(tile, y_data, x_data);
        #ifdef TIMING
        t2 = Env::clock();
        elapsed_time += (t2-t1);
        #endif
        
        xi++;
        bool communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication)
        {
            MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            int32_t leader = pair2.row;
            int32_t my_rank = pair2.col;
            int32_t follower, accu;
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
    
    #ifdef TIMING
    Env::print_time("Combine computation", elapsed_time);
    combine_comp_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_nonstationary()
{
    #ifdef TIMING
    double t1, t2, elapsed_time = 0;
    #endif
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
        
        #ifdef TIMING
        t1 = Env::clock();
        #endif
        spmv(tile, y_data, x_data, xv_data, xi_data, t_data);
        #ifdef TIMING
        t2 = Env::clock();
        elapsed_time += (t2 - t1);
        #endif
        
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
    #ifdef TIMING
    Env::print_time("Combine computation", elapsed_time);
    combine_comp_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess()
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
        //#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < yj_nitems; i++)
            combiner(y_data[i], yj_data[i]);
    }
    wait_for_sends();
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
                //#pragma omp parallel for schedule(static)
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
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                combiner(y_data[i], yj_data[i]);
        }
    }
    wait_for_sends();
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
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
    if(stationary)
    {
        apply_stationary();
    }
    else
    {
        apply_nonstationary();
    }

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Apply", elapsed_time);
    apply_time.push_back(elapsed_time);
    #endif
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_stationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];
    Integer_Type v_nitems = V.size();
    if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
    {
        //#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i];
            C[i] = applicator(state, y_data[i]);
        }
    }
    else if((filtering_type == _SOME_) or (filtering_type == _SRCS_))
    {
        if(iteration == 0)
        {
            auto &i_data = (*I)[yi];
            auto &iv_data = (*IV)[yi];
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                if(i_data[i])
                {
                    C[i] = applicator(state, y_data[iv_data[i]]);
                }
                else
                    C[i] = applicator(state);
            }
           /*
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
            */
        }
        else    
        {
            auto &y2v_data = (*Y2V);
            auto &v2y_data = (*V2Y);
            Integer_Type y2v_nitems = y2v_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < y2v_nitems; i++)
            {
                Vertex_State &state = V[v2y_data[i]];
                C[v2y_data[i]] = applicator(state, y_data[y2v_data[i]]);
            }
        }
    }
//}
    //wait_for_sends();
    
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        for(uint32_t j = 0; j < Y[i].size(); j++)
        {
            std::vector<Fractional_Type> &y_data = Y[i][j];
            Integer_Type y_nitems = y_data.size();
            //#pragma omp parallel for schedule(static)
            //for(uint32_t k = 0; k < y_nitems; k++)
            //    y_data[k] = 0;        
          
            std::fill(y_data.begin(), y_data.end(), 0);
        }
    }
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_nonstationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];

    Integer_Type v_nitems = V.size();
    if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
    {
        if(apply_depends_on_iter)
        {
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                C[i] = applicator(state, y_data[i], iteration);
            }
        }
        else
        {
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                C[i] = applicator(state, y_data[i]);
            }
        }
    }
    else if((filtering_type == _SOME_) or (filtering_type == _SRCS_))
    {
        if(apply_depends_on_iter)
        {
            if(iteration == 0)
            {
                auto &i_data = (*I)[yi];
                auto &iv_data = (*IV)[yi];
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                    {
                        C[i] = applicator(state, y_data[iv_data[i]], iteration);
                    }
                    else
                    {
                        C[i] = applicator(state);    
                    }
                } 
                /*
                auto &i_data = (*I)[yi];
                Integer_Type j = 0;
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
               */
            }
            else
            {
                auto &y2v_data = (*Y2V);
                auto &v2y_data = (*V2Y);
                Integer_Type y2v_nitems = y2v_data.size();
                //#pragma omp parallel for schedule(static)
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
                auto &i_data = (*I)[yi];
                auto &iv_data = (*IV)[yi];
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                    {
                        C[i] = applicator(state, y_data[iv_data[i]]);
                    }
                    else
                    {
                        C[i] = applicator(state);
                    }
                }
                /*
                auto &i_data = (*I)[yi];
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
                */
            }
            else
            {
                auto &y2v_data = (*Y2V);
                auto &v2y_data = (*V2Y);
                Integer_Type y2v_nitems = y2v_data.size();
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < y2v_nitems; i++)
                {
                    Vertex_State &state = V[v2y_data[i]];
                    C[v2y_data[i]] = applicator(state, y_data[y2v_data[i]]);
                }
            }
        }
    }
    
    //wait_for_sends();
    
    if(not gather_depends_on_apply and not apply_depends_on_iter)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            for(uint32_t j = 0; j < Y[i].size(); j++)
            {
                std::vector<Fractional_Type> &y_data = Y[i][j];
                Integer_Type y_nitems = y_data.size();
                //#pragma omp parallel for schedule(static)
                //for(uint32_t k = 0; k < y_nitems; k++)
                 //   y_data[k] = 0;      
                
                std::fill(y_data.begin(), y_data.end(), 0);
            }
        }
    }

    if(activity_filtering)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            std::vector<char> &t_data = T[i];
            Integer_Type t_nitems = t_data.size();
            //#pragma omp parallel for schedule(static)
            //for(uint32_t j = 0; j < t_nitems; j++)
            //        t_data[j] = 0;  
                
            std::fill(t_data.begin(), t_data.end(), 0);
        }
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
    
    Triple<Weight, double> stats_pair;
    if(!Env::rank)
    {
        
        #ifdef TIMING
        double sum = 0.0, mean = 0.0, std_dev = 0.0;
        std::cout << "Init           time: " << init_time[0] * 1e3 << " ms" << std::endl;
        stats(scatter_gather_time, sum, mean, std_dev);
        std::cout << "Scatter_gather time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        stats(combine_time, sum, mean, std_dev);
        std::cout << "Combine        time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        stats(apply_time, sum, mean, std_dev);
        std::cout << "Apply          time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        std::cout << "Execute        time: " << execute_time[0] * 1e3 << " ms" << std::endl;
        
        std::cout << "TIMING " << init_time[0] * 1e3;
        stats(scatter_gather_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(combine_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(apply_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        std::cout << " " << execute_time[0] * 1e3 << std::endl;
        
        /*
        stats_pair = stats(scatter_gather_time);
        std::cout << "Scatter_gather time  (sum: avg +/- std_dev): " <<  stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_time);
        std::cout << "Combine all     time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_comp_time);
        std::cout << "Combine compute time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_comm_time);
        std::cout << "Combine communi time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(apply_time);
        std::cout << "Apply time           (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        */
        #endif
        
        
        Triple<Weight, Integer_Type> pair, pair1;
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            Vertex_State &state = V[i];
            //std::cout << std::fixed <<  "vertex[" << A->hasher->unhash(pair1.row) << "]:" << state.print_state() << std::endl;
            std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
        }
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev)
{
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
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

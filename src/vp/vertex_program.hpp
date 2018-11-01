/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include <type_traits>

#include "ds/vector.hpp"
#include "mpi/types.hpp" 
#include "mpi/comm.hpp" 

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
        
        bool stationary;
        bool gather_depends_on_apply;
        bool tc_family;
        bool already_initialized = false;
        bool check_for_convergence = false;
        bool apply_depends_on_iter = false;
        Integer_Type iteration = 0;
        std::vector<Vertex_State> V2; // Values
        std::vector<std::vector<Integer_Type>> W; // Values for triangle counting
    protected:        
        void clear(Vector<Weight, Integer_Type, Fractional_Type> *vec);
        void specialized_tc_init(Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State> *VProgram);  
        void scatter_gather();
        void scatter();
        void gather();
        void bcast();
        void specialized_stationary_scatter();
        void specialized_stationary_gather();
        void specialized_nonstationary_scatter();
        void specialized_nonstationary_gather();
        void specialized_stationary_bcast();
        void specialized_nonstationary_bcast();
        void combine();
        void optimized_1d_row();
        void optimized_1d_col();
        void optimized_2d();
        void optimized_2d_for_tc();
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void spmv(Fractional_Type *y_data, std::vector<Fractional_Type> &x_data, 
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                std::vector<Integer_Type> &s_data, std::vector<char> &t_data);//, std::unordered_set<Integer_Type> &p_data);        
                
        void spmv(std::vector<std::vector<Integer_Type>> &z_data, 
                struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
        void apply();                        
        void specialized_stationary_apply();
        void specialized_nonstationary_apply();
        void specialized_tc_apply();
        void wait_for_all();
        void wait_for_sends();
        void wait_for_recvs();
        bool has_converged();
        
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
        std::vector<MPI_Request> out_requests_;
        std::vector<MPI_Request> in_requests_;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A; // Adjacency list
        Vector<Weight, Integer_Type, Fractional_Type> *X; // Messages 
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y; //Accumulators
        //Vector<Weight, Integer_Type, Fractional_Type> *Yt; // Accumulators (temp)

        Vector<Weight, Integer_Type, char> *C; // Convergence vector
        //Vector<Weight, Integer_Type, char> *B; //Activity vector
        std::vector<std::vector<Fractional_Type>> X2; // New X
        std::vector<std::vector<Integer_Type>> S; // Reduced activity vector for X
        std::vector<std::vector<std::vector<Fractional_Type>>> Y2; // New Y
        std::vector<std::vector<std::vector<Integer_Type>>> P; // Reduced activity vector for Y
        //std::vector<std::vector<std::unordered_set<Integer_Type>>> P;
        std::vector<std::vector<char>> T;
        std::vector<Integer_Type> x2_nitems_vec;
        std::vector<Integer_Type> y2_nitems_vec;
        std::vector<std::vector<char>> *I;
        //std::vector<std::vector<Integer_Type>> *IV;
        std::vector<std::vector<char>> *J;
        //std::vector<std::vector<Integer_Type>> *JV;

        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        
        MPI_Datatype TYPE_DOUBLE;
        MPI_Datatype TYPE_INT;
        MPI_Datatype TYPE_CHAR;
        
        /* Specialized vectors for triangle counting */
        std::vector<std::vector<Integer_Type>> R; // Scores (Ingoing edges)
        std::vector<std::vector<std::vector<Integer_Type>>> D; // Data (All Ingoing edges)
        std::vector<std::vector<Integer_Type>> D_SIZE; 
        std::vector<std::vector<std::vector<Integer_Type>>> Z; // Accumulators
        std::vector<std::vector<std::vector<Integer_Type>>> Z_SIZE;
        std::vector<std::vector<Integer_Type>> inboxes; // Temporary buffers for deserializing the adjacency lists
        std::vector<std::vector<Integer_Type>> outboxes;// Temporary buffers for  serializing the adjacency lists
        
        Integer_Type get_vid(Integer_Type index){return(index + (owned_segment * tile_height));};
        bool directed;
        bool transpose;
        double activity_filtering_ratio = 0.6;
        bool activity_filtering = true;
        bool accu_activity_filtering = false;
        bool msgs_activity_filtering = false;
        uint64_t num_row_touches = 0;
        uint64_t num_row_edges = 0;
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph,
         bool stationary_, bool gather_depends_on_apply_, bool apply_depends_on_iter_, 
         bool tc_family_, Ordering_type ordering_type_)
                       : X(nullptr)
{

    A = Graph.A;
    directed = A->directed;
    transpose = A->transpose;
    stationary = stationary_;
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
    TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::free()
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
        V2.clear();
        V2.shrink_to_fit();
        delete X;
        //delete V;
        //delete S;   
        delete C;   
        for (uint32_t j = 0; j < rank_nrowgrps; j++)
        {
            delete Y[j];
        }   
        Y.clear();
        Y.shrink_to_fit();
        
        if(not stationary)
        {
            for(uint32_t j = 0; j < rank_ncolgrps; j++)
            {
                X2[j].clear();
                X2[j].shrink_to_fit();
            }
    
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
            {
                Y2[j].clear();
                Y2[j].shrink_to_fit();
                P[j].clear();
                P[j].shrink_to_fit();
            }
        }
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::clear(
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
            //exit(0);
            combine();
            //exit(0);
            apply();
            iteration++;
            Env::print_me("Iteration: ", iteration);            
            if(check_for_convergence)
            {
                if(has_converged())
                    break;
            }
            else if(iteration >= num_iterations)
                break;
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
        std::vector<Integer_Type> d_sizes;
        std::vector<Integer_Type> z_size;
        std::vector<Integer_Type> z_sizes;
        std::vector<Integer_Type> zz_sizes;
        
        /* Initialize values and activity pattern */
        W.resize(tile_height);
        
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
    else
    {
        /* Initialize Values (States)*/
        V2.resize(tile_height);
        Integer_Type v_nitems = V2.size();
        
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
        
        /* Array to look for convergence */ 
        std::vector<Integer_Type> v_s_size = {tile_height};
        C = new Vector<Weight, Integer_Type, char>(v_s_size, accu_segment_row_vec);
        uint32_t co = 0;
        char *c_data = (char *) C->data[co];
        Integer_Type c_nitems = C->nitems[co];
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V2[i]; 
            c_data[i] = initializer(get_vid(i), state);
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
        
        if(not stationary)
        {
            X2.resize(rank_ncolgrps);
            for(uint32_t j = 0; j < rank_ncolgrps; j++)
                X2[j].resize(x_sizes[j]);
            S.resize(rank_ncolgrps);
            for(uint32_t j = 0; j < rank_ncolgrps; j++)
                S[j].resize(x_sizes[j]);
            x2_nitems_vec.resize(colgrp_nranks);
            
            T.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
                T[j].resize(y_sizes[j]);
            y2_nitems_vec.resize(rowgrp_nranks);
            Y2.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
            {
                if(local_row_segments[j] == owned_segment)
                {
                    Y2[j].resize(rowgrp_nranks);
                    
                    for(uint32_t i = 0; i < rowgrp_nranks; i++)
                    {
                        if(i != accu_segment_rg)
                            Y2[j][i].resize(y_sizes[j]);
                    }
                }
                else
                {
                    Y2[j].resize(1);
                    Y2[j][0].resize(y_sizes[j]);
                }
            }
            P.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
            {
                if(local_row_segments[j] == owned_segment)
                {
                    P[j].resize(rowgrp_nranks);
                    
                    for(uint32_t i = 0; i < rowgrp_nranks; i++)
                    {
                        if(i != accu_segment_rg)
                            P[j][i].resize(y_sizes[j]);
                    }
                }
                else
                {
                    P[j].resize(1);
                    P[j][0].resize(y_sizes[j]);
                }
            }
            
            uint32_t yi = 0;
            uint32_t yo = 0;
            for(uint32_t k = 0; k < rank_nrowgrps; k++)
            {
                yi = k;
                auto *Yp = Y[yi];
                if(local_row_segments[k] == owned_segment)
                    yo = accu_segment_rg;
                else
                    yo = 0;
                Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
                Integer_Type y_nitems = Yp->nitems[yo];
                if(filtering_type == _NONE_)
                {
                    for(uint32_t i = 0; i < v_nitems; i++)
                        y_data[i] = V2[i].get_inf();
                }
                else if(filtering_type == _SOME_)
                {
                    auto &i_data = (*I)[yi];       
                    Integer_Type j = 0;
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        if(i_data[i])
                        {
                            y_data[j] = V2[i].get_inf();
                            j++;
                        }
                    }
                }
            }
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
        std::vector<Integer_Type> d_sizes;
        std::vector<Integer_Type> z_size;
        std::vector<Integer_Type> z_sizes;
        std::vector<Integer_Type> zz_sizes;
        
        /* Initialize values and activity pattern */
        W.resize(tile_height);
        
        /* Initialiaze scores/states */
        std::vector<std::vector<Integer_Type>> W_ = VProgram.W;
        R = W_;
        
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
        
        /* Initialiaze BIG 2D scores/states vector */
        D.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D[i].resize(d_sizes[i]);
        
        D_SIZE.resize(nrowgrps);
        for(uint32_t i = 0; i < nrowgrps; i++)
            D_SIZE[i].resize(d_sizes[i] + 1); // 0 + nitems

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
    else
    {
        /* Initialize Values (States)*/
        V2.resize(tile_height);
        Integer_Type v_nitems = V2.size();

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
        
        /* Array to look for convergence */ 
        std::vector<Integer_Type> v_s_size = {tile_height};
        C = new Vector<Weight, Integer_Type, char>(v_s_size, accu_segment_row_vec);
        uint32_t co = 0;
        char *c_data = (char *) C->data[co];
        Integer_Type c_nitems = C->nitems[co];
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V2[i]; 
            c_data[i] = initializer(get_vid(i), state, (const State&) VProgram.V2[i]);
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
        
        if(not stationary)
        {
            X2.resize(rank_ncolgrps);
            for(uint32_t j = 0; j < rank_ncolgrps; j++)
                X2[j].resize(x_sizes[j]);
            S.resize(rank_ncolgrps);
            for(uint32_t j = 0; j < rank_ncolgrps; j++)
                S[j].resize(x_sizes[j]);
            x2_nitems_vec.resize(colgrp_nranks);
            
            T.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
                T[j].resize(y_sizes[j]);
            y2_nitems_vec.resize(rowgrp_nranks);
            Y2.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
            {
                if(local_row_segments[j] == owned_segment)
                {
                    Y2[j].resize(rowgrp_nranks);
                    
                    for(uint32_t i = 0; i < rowgrp_nranks; i++)
                    {
                        if(i != accu_segment_rg)
                            Y2[j][i].resize(y_sizes[j]);
                    }
                }
                else
                {
                    Y2[j].resize(1);
                    Y2[j][0].resize(y_sizes[j]);
                }
            }
            P.resize(rank_nrowgrps);
            for(uint32_t j = 0; j < rank_nrowgrps; j++)
            {
                if(local_row_segments[j] == owned_segment)
                {
                    P[j].resize(rowgrp_nranks);
                    
                    for(uint32_t i = 0; i < rowgrp_nranks; i++)
                    {
                        if(i != accu_segment_rg)
                            P[j][i].resize(y_sizes[j]);
                    }
                }
                else
                {
                    P[j].resize(1);
                    P[j][0].resize(y_sizes[j]);
                }
            }
            
            uint32_t yi = 0;
            uint32_t yo = 0;
            for(uint32_t k = 0; k < rank_nrowgrps; k++)
            {
                yi = k;
                auto *Yp = Y[yi];
                if(local_row_segments[k] == owned_segment)
                    yo = accu_segment_rg;
                else
                    yo = 0;
                Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
                Integer_Type y_nitems = Yp->nitems[yo];
                
                if(filtering_type == _NONE_)
                {
                    for(uint32_t i = 0; i < v_nitems; i++)
                        y_data[i] = V2[i].get_inf();
                }
                else if(filtering_type == _SOME_)
                {
                    auto &i_data = (*I)[yi];       
                    Integer_Type j = 0;
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        if(i_data[i])
                        {
                            y_data[j] = V2[i].get_inf();
                            j++;
                        }
                    }
                }
            }
        }
    }
    already_initialized = true;
    t2 = Env::clock();
    Env::print_time("Init", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather()
{
    double t1, t2;
    t1 = Env::clock();

    uint32_t leader;
    uint32_t xo = accu_segment_col;
    Fractional_Type *x_data = (Fractional_Type *) X->data[xo];
    Integer_Type x_nitems = X->nitems[xo];

    Integer_Type v_nitems = V2.size();
    
    if(stationary)
    {    
        if(filtering_type == _NONE_)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V2[i];
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
                    Vertex_State &state = V2[i];
                    x_data[j] = messenger(state);
                    j++;
                }
            }
        }
    }
    else
    {   
        uint32_t co = 0;
        char *c_data = (char *) C->data[co];
        Integer_Type c_nitems = C->nitems[co];
        
        std::vector<Fractional_Type> &x2_data = X2[xo];
        std::vector<Integer_Type> &s_data = S[xo];
        Integer_Type k = 0;
        if(filtering_type == _NONE_)
        {
            for(uint32_t i = 0; i < c_nitems; i++)
            {
                Vertex_State &state = V2[i];
                x_data[i] = messenger(state);
                if(c_data[i])
                {
                    x2_data[k] = x_data[i];
                    s_data[k] = i;
                    k++;
                }
            }
        }
        else if(filtering_type == _SOME_)
        {
            if(not directed)
            {
                uint32_t yi = accu_segment_row;
                auto &i_data = (*I)[yi];
                Integer_Type j = 0;
                for(uint32_t i = 0; i < c_nitems; i++)
                {
                    if(i_data[i])
                    {
                        Vertex_State &state = V2[i];
                        x_data[j] = messenger(state);
                        if(c_data[i])
                        {
                            x2_data[k] = x_data[j];
                            s_data[k] = j;
                            k++;
                        }                    
                        j++;
                    }
                }
            }
            else
            {
                if(transpose)
                {                
                    auto &j_data = (*J)[xo];
                    Integer_Type j = 0;
                    for(uint32_t i = 0; i < c_nitems; i++)
                    {
                        if(j_data[i])
                        {
                            Vertex_State &state = V2[i];
                            x_data[j] = messenger(state);
                            if(c_data[i])                            
                            {
                                x2_data[k] = x_data[j];
                                s_data[k] = j;
                                k++;
                            }
                            j++;
                        }
                    }
                }
                else
                {
                    fprintf(stderr, "Directed graphs for non-stationary algorithms should be transposed\n");
                    Env::exit(1);
                }
            }
            x2_nitems_vec[xo] = k;
            //printf("rank=%d nnzsize=%d\n", Env::rank, x2_nitems_vec[xo]);
        }
    }

    if(stationary)
    {
        if(Env::comm_split)
        {
            specialized_stationary_bcast();
        }
        else
        {
            specialized_stationary_scatter();
            specialized_stationary_gather();
        }
    }
    else
    {
        int nitems = x2_nitems_vec[xo];
        /* 0 all / 1 nothing / else nitems  */
        double ratio = (double) nitems/x_nitems;
        if(ratio <= activity_filtering_ratio)
        {
            msgs_activity_filtering = true;                            
            if(nitems)
                nitems++;
            else
                nitems = 1;
        }
        else
        {
            msgs_activity_filtering = false;
            nitems = 0;
        }  
        x2_nitems_vec[xo] = nitems;
        //if(!Env::rank)
        //{
            //printf("rank=%d num=%d all_num=%d filter=%d\n", Env::rank, x2_nitems_vec[xo], x_nitems, msgs_activity_filtering);
        
        
        if(Env::comm_split)
        {
            specialized_nonstationary_bcast();
        }
        else
        {
            specialized_nonstationary_scatter();
            specialized_nonstationary_gather();
        }
    }
    t2 = Env::clock();
    Env::print_time("Scatter_gather", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_stationary_scatter()
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_stationary_gather()
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_stationary_bcast()
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_nonstationary_scatter()
{
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X2[xo];
    Integer_Type x_nitems = x_data.size();
    
    std::vector<Integer_Type> &s_data = S[xo];
    Integer_Type s_nitems = s_data.size();
    
    int nitems = x2_nitems_vec[xo];
    
    int32_t col_group = local_col_segments[xo];
    
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
               
               MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, colgrps_communicator);
               
               MPI_Isend(x_data.data(), nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
               out_requests.push_back(request);
               
               MPI_Isend(s_data.data(), nitems, TYPE_INT, follower, col_group, colgrps_communicator, &request);
               out_requests.push_back(request);
               
            }
            else
            {
                
                follower = follower_colgrp_ranks[i];
                
                MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
                
                MPI_Isend(x_data.data(), nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
                
                MPI_Isend(s_data.data(), nitems, TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
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

/* Buggy */
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_nonstationary_gather()
{  
    uint32_t leader;
    MPI_Request request;
    MPI_Status status;
    int nitems = 0;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            std::vector<Fractional_Type> &xj_data = X2[i];
            Integer_Type xj_nitems = xj_data.size();
            
            std::vector<Integer_Type> &sj_data = S[i];
            Integer_Type sj_nitems = sj_data.size();
            
            int32_t col_group = local_col_segments[i];
            if(Env::comm_split)
            {
                leader = leader_ranks_cg[col_group];
                if(leader != Env::rank_cg)
                {
                    
                    MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, colgrps_communicator, &status);
                    x2_nitems_vec[i] = nitems;
                    
                    MPI_Irecv(xj_data.data(), nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                    
                    MPI_Irecv(sj_data.data(), nitems, TYPE_INT, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                leader = leader_ranks[col_group];
                if(leader != Env::rank)
                {
                    MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
                    x2_nitems_vec[i] = nitems;
                    
                    MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                    
                    MPI_Irecv(sj_data.data(), sj_nitems, TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }
        }
        MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
        in_requests.clear();
        //MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        //in_requests_.clear();
        MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
        out_requests.clear();
        //MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        //out_requests_.clear();        
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_nonstationary_bcast()
{
    uint32_t leader, leader_cg;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            leader_cg = leader_ranks_cg[local_col_segments[i]];  
            std::vector<Fractional_Type> &xj_data = X2[i];
            std::vector<Integer_Type> &sj_data = S[i];
            
            Integer_Type nitems = 0;
            if(Env::rank_cg == leader_cg)
                nitems = x2_nitems_vec[i];
            MPI_Bcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicator);
            if(Env::rank_cg != leader_cg)
                x2_nitems_vec[i] = nitems;
            
            if(x2_nitems_vec[i])
                msgs_activity_filtering = true;
            else
                msgs_activity_filtering = false;
            
            if(Env::comm_split)
            {
                if(msgs_activity_filtering)
                {
                    if(x2_nitems_vec[i] > 1)
                    {
                        MPI_Bcast(xj_data.data(), x2_nitems_vec[i] - 1, TYPE_DOUBLE, leader_cg, colgrps_communicator);
                        MPI_Bcast(sj_data.data(), x2_nitems_vec[i] - 1, TYPE_INT, leader_cg, colgrps_communicator);
                    }
                }
                else
                {
                    MPI_Bcast(xj_data.data(), xj_data.size(), TYPE_DOUBLE, leader_cg, colgrps_communicator);
                    x2_nitems_vec[i] = xj_data.size();
                }
            }
            else
            {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
        }
        /*
        if(!Env::rank)
        {
            for(uint32_t i = 0; i < rank_ncolgrps; i++)
                printf("%d ", x2_nitems_vec[i]);
            printf(".0.\n");
        }
        Env::barrier();
        if(Env::rank == 2)
        {
            for(uint32_t i = 0; i < rank_ncolgrps; i++)
                printf("%d ", x2_nitems_vec[i]);
            printf(".2.\n");
        }
        */
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



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv(
            Fractional_Type *y_data, std::vector<Fractional_Type> &x_data, 
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<Integer_Type> &s_data, std::vector<char> &t_data)
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
                if(x2_nitems_vec[tile.jth])
                {
                    Integer_Type s_nitems = x2_nitems_vec[tile.jth] - 1;
                    Integer_Type j = 0;
                    for(Integer_Type k = 0; k < s_nitems; k++)
                    {
                        j = s_data[k];
                        for(uint32_t i = JA[j]; i < JA[j + 1]; i++)
                        {
                            #ifdef HAS_WEIGHT
                            combiner(y_data[IA[i]], x_data[k], A[i]);
                            #else
                            combiner(y_data[IA[i]], x_data[k]);
                            #endif
                            t_data[IA[i]] = 1;
                        }
                    }
                }
                else
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
                            t_data[IA[i]] = 1;
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
                        combiner(y_data[i], A[j] * x_data[JA[j]]);
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
                        combiner(y_data[JA[j]], A[j] * x_data[i]);
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
                        combiner(y_data[IA[i]], A[i] * x_data[j]);
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
                        combiner(y_data[j], A[i] * x_data[IA[i]]);   
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
    double t1, t2;
    t1 = Env::clock();
    
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::optimized_1d_row()
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::optimized_1d_col()
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::optimized_2d_for_tc()
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
                   
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::optimized_2d()
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

        if(stationary)
        {
            Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
            spmv(y_data, x_data, tile);
        }
        else
        {
            std::vector<Fractional_Type> &x_data = X2[xi];
            std::vector<Integer_Type> &s_data = S[xi];
            std::vector<char> &t_data = T[yi];
            spmv(y_data, x_data, tile, s_data, t_data);
        }
        
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
                if(stationary)
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
                    if(activity_filtering)
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
                            
                            /* 0 all / 1 nothing / else nitems  */
                            int nitems = 0;
                            MPI_Status status;
                            MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, communicator, &status);
                            y2_nitems_vec[accu] = nitems;
                            
                           // if(pair_idx == 3)
                                //printf("<l=%d t=%d ni=%d\n", Env::rank, pair_idx, nitems);
                            
                            if(y2_nitems_vec[accu])
                                accu_activity_filtering = true;
                            else
                                accu_activity_filtering = false;
                            
                            if(accu_activity_filtering)
                            {
                                if(y2_nitems_vec[accu] > 1)
                                {
                                    std::vector<Fractional_Type> &y2j_data = Y2[yi][accu];
                                    std::vector<Integer_Type> &pj_data = P[yi][accu];
                                    MPI_Irecv(y2j_data.data(), y2_nitems_vec[accu] - 1, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                                    in_requests.push_back(request);
                                    MPI_Irecv(pj_data.data(), y2_nitems_vec[accu] - 1, TYPE_INT, follower, pair_idx, communicator, &request);
                                    in_requests.push_back(request);
                                }
                            }
                            else
                            {                                
                                Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                                Integer_Type yj_nitems = Yp->nitems[accu];
                                MPI_Irecv(yj_data, yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                                in_requests.push_back(request);
                            }
                        } 
                    }
                    else
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
                    if(activity_filtering)
                    {
                        std::vector<Fractional_Type> &y2_data = Y2[yi][yo];
                        std::vector<Integer_Type> &p_data = P[yi][yo];
                        std::vector<char> &t_data = T[yi];
                        Integer_Type j = 0;
                        for(uint32_t i = 0; i < y_nitems; i++)
                        {
                            if(t_data[i])
                            {
                                p_data[j] = i;
                                y2_data[j] = y_data[i];
                                j++;
                            }
                        }
                        int nitems = j;
                        //y2_nitems_vec[yo] = nitems;           
                        
                        /* 0 all / 1 nothing / else nitems  */
                        double ratio = (double) nitems/y_nitems;
                        if(ratio <= activity_filtering_ratio)
                        {
                            accu_activity_filtering = true;                            
                            if(nitems)
                                nitems++;
                            else
                                nitems = 1;
                        }
                        else
                        {
                            nitems = 0;
                            accu_activity_filtering = false;
                        }         
                        //y2_nitems_vec[yo] = nitems; 
                        //if(Env::rank == 3)
                          //  printf("rate=%f ni=%d\n", ratio, nitems);
                        MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, communicator);
                        if(accu_activity_filtering)
                        {
                            if(nitems > 1)
                            {
                                MPI_Isend(y2_data.data(), nitems - 1, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                                out_requests.push_back(request);
                                MPI_Isend(p_data.data(), nitems - 1, TYPE_INT, leader, pair_idx, communicator, &request);
                                out_requests.push_back(request);
                            }
                        }
                        else
                        {
                            MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                            out_requests.push_back(request);
                        }
                    }    
                    else
                    {
                        std::vector<Fractional_Type> &y2_data = Y2[yi][yo];
                        std::vector<Integer_Type> &p_data = P[yi][yo];
                        std::vector<char> &t_data = T[yi];
                        
                        Integer_Type j = 0;
                        for(uint32_t i = 0; i < y_nitems; i++)
                        {
                            if(t_data[i])
                            {
                                p_data[j] = i;
                                y2_data[j] = y_data[i];
                                j++;
                            }
                        }
                        int nitems = j;
                        //y2_nitems_vec[yo] = nitems; 
                        
                        MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                        out_requests.push_back(request);
                    }

                   // if(!Env::rank)
                       // printf(">r=%d t=%d yn=%d n=%d tf=%d,%d\n", Env::rank, pair_idx, y_nitems, y2_nitems_vec[yo], activity_filtering, accu_activity_filtering);
                }
            }
            xi = 0;
            yi++;
        }
    }
    
    wait_for_all();
    
    

    yi = accu_segment_row;
    yo = accu_segment_rg;
    Yp = Y[yi];
    Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
    Integer_Type y_nitems = Yp->nitems[yo];
    
    if(activity_filtering)
    {
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
        {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            
            if(y2_nitems_vec[accu])
            {
                if(y2_nitems_vec[accu] > 1)
                {
                    std::vector<Fractional_Type> &y2j_data = Y2[yi][accu];
                    std::vector<Integer_Type> &pj_data = P[yi][accu];
                    //Integer_Type i = y2_nitems_vec[accu] - 1;
                    for(uint32_t i = 0; i < y2_nitems_vec[accu] - 1; i++)
                    {
                        Integer_Type k = pj_data[i];
                        combiner(y_data[k], y2j_data[i]);
                      //  if(Env::rank == 2)
                        //    printf("%d %d\n", i, y2j_data[i]);
                    }
                }
            }
            else
            {
                Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                Integer_Type yj_nitems = Yp->nitems[accu];
                for(uint32_t i = 0; i < yj_nitems; i++)
                    combiner(y_data[i], yj_data[i]);
            }
        }
    }     
    else
    {
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
                combiner(y_data[i], yj_data[i]);
                //if(Env::rank == 2)
                  //  printf("r=%d j=%d i=%d yi=%d yji=%d\n", Env::rank, j, i, y_data[i], yj_data[i]);
            }
        }        
    }
    /*
    Env::barrier();
    for(uint32_t j = 0; j < Env::nranks; j++)
    {
        if(Env::rank == j)
        {
            //printf("rrrrr=%d %d \n", Env::rank, y_nitems);
            for(uint32_t i = 0; i < y_nitems; i++)
            {
                if(y_data[i] != 2147483647)
                    printf("r=%d i=%d yi=%d\n", Env::rank, i, y_data[i]);
            }
            for(uint32_t i =0; i < rank_nrowgrps; i++)
                printf("%d ", y2_nitems_vec[i]);
            printf("\n");
        }
    
    }
    Env::barrier();
    */
    //printf("rrrrr=%d\n", Env::rank);
   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply()
{
    double t1, t2;
    t1 = Env::clock();

    if(stationary)
    {
        specialized_stationary_apply();
    }
    else
    {
        if(tc_family)
            specialized_tc_apply();
        else
            specialized_nonstationary_apply();
    }

    t2 = Env::clock();
    Env::print_time("Apply", t2 - t1);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_stationary_apply()
{
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    auto *Yp = Y[yi];
    Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
    Integer_Type y_nitems = Yp->nitems[yo];
    
    uint32_t co = 0;
    char *c_data = (char *) C->data[co];
    Integer_Type c_nitems = C->nitems[co];
    Integer_Type v_nitems = V2.size();
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V2[i];
            c_data[i] = applicator(state, y_data[i]);
        }
    }
    else if(filtering_type == _SOME_)
    {
        Fractional_Type tmp = 0;
        auto &i_data = (*I)[yi];
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V2[i];
            if(i_data[i])
            {
                c_data[i] = applicator(state, y_data[j]);
                j++;
            }
            else
            {
                c_data[i] = applicator(state);
            }
        }
    }

    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        clear(Y[i]);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_nonstationary_apply()
{
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    auto *Yp = Y[yi];
    Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
    Integer_Type y_nitems = Yp->nitems[yo];
    
    Integer_Type v_nitems = V2.size();
    
    uint32_t co = 0;
    char *c_data = (char *) C->data[co];
    Integer_Type c_nitems = C->nitems[co];

    if(filtering_type == _NONE_)
    {
        if(apply_depends_on_iter)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V2[i];
                c_data[i] = applicator(state, y_data[i], iteration);
            }
        }
        else
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V2[i];
                c_data[i] = applicator(state, y_data[i]);
            }
        }
    }
    else if(filtering_type == _SOME_)
    {
        Fractional_Type tmp = 0;
        auto &i_data = (*I)[yi];
        Integer_Type j = 0;
        if(apply_depends_on_iter)
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V2[i];
                if(i_data[i])
                {
                    c_data[i] = applicator(state, y_data[j], iteration);
                    j++;
                }
                else
                {
                    c_data[i] = applicator(state);    
                }
            }
            
        }
        else
        {
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V2[i];
                if(i_data[i])
                {
                    c_data[i] = applicator(state, y_data[j]);
                    //if(!Env::rank)
                    //{
                       // if(c_data[i])
                       // printf("r=%d i=%d c=%d y=%d \n", Env::rank, i, c_data[i], y_data[j]);
                    //}
                    j++;
                }
                else
                {
                    c_data[i] = applicator(state);
                }
            }
        }
    }
    
    if(not gather_depends_on_apply and not apply_depends_on_iter)
    {
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
            clear(Y[i]);
    } 

    if(activity_filtering)
    {
        for(uint32_t j = 0; j < rank_nrowgrps; j++)
            std::fill(T[j].begin(), T[j].end(), 0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::specialized_tc_apply()
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_all()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_recvs()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_sends()
{
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
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

    uint32_t co = 0;
    char *c_data = (char *) C->data[co];
    Integer_Type c_nitems = C->nitems[co];    
    
    for(uint32_t i = 0; i < c_nitems; i++)
    {
        if(not c_data[i]) 
            c_sum_local++;
    }
   
    MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(c_sum_gloabl == (tile_height * Env::nranks))
        converged = true;

    return(converged);   
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::checksum()
{

    /*
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    
    
    if(apply_depends_on_iter)
    {
        uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(v_data[i] != INF) 
                v_sum_local_++;
        }

        MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(Env::is_master)
            printf("Reachable Vertices: %lu\n", v_sum_global_);
    }
    */
    
    uint64_t v_sum_local = 0, v_sum_global = 0;
    Integer_Type v_nitems = V2.size();
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V2[i];
        //if(get_vid(i) < nrows)
        if((state.get_state() != state.get_inf()) and (get_vid(i) < nrows))    
        {
                v_sum_local += state.get_state();
                //printf("%d %d\n", get_vid(i), state.get_state());
        }
            
    }
    MPI_Allreduce(&v_sum_local, &v_sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        std::cout << std::fixed << "Value checksum: " << v_sum_global << std::endl;

    if(apply_depends_on_iter or gather_depends_on_apply)
    {

        uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V2[i];
            if((state.get_state() != state.get_inf()) and (get_vid(i) < nrows)) 
            {
                v_sum_local_++;
            }
        }

        MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(Env::is_master)
            std::cout << std::fixed << "Reachable Vertices: " << v_sum_global_ << std::endl;
    }
    Env::barrier();
    
    /*
    uint64_t s_local = 0, s_global = 0;
    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];

    for(uint32_t i = 0; i < s_nitems; i++)
        s_local += s_data[i];
    
    MPI_Allreduce(&s_local, &s_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        std::cout << std::fixed << "Score checksum:" << s_global << std::endl;
    */
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::display(Integer_Type count)
{
    Integer_Type v_nitems = V2.size();
    count = (v_nitems < count) ? v_nitems : count;
    Env::barrier();
    if(!Env::rank)
    {
        Triple<Weight, Integer_Type> pair, pair1;
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            Vertex_State &state = V2[i];
            std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
        }  
    }
    Env::barrier();
    
    //Env::barrier();
    
    //for(int i = 0; i < 4; i++)
    //{
        
    if(Env::rank == -2)
    {
        Triple<Weight, Integer_Type> pair, pair1;
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            Vertex_State &state = V2[i];
            std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
        }  
    }    
    //}
    
    
}
#endif

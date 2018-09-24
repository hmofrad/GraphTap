/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP
 
#include "vector.hpp"
#include "types.hpp" 

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
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph, Ordering_type = _ROW_);
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
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);
                  
        void spmv(Fractional_Type *y_data, Fractional_Type *x_data,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);                  
                  
/*                  
        void spmv(Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
                  Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  Segment<Weight, Integer_Type, Integer_Type> &t_seg,
                  Segment<Weight, Integer_Type, char> &k_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &kv_seg);
                  
        void spmv(Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
                  Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
                  struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  Segment<Weight, Integer_Type, Integer_Type> &r_seg,
                  Segment<Weight, Integer_Type, char> &i_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &iv_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &c_seg,
                  Segment<Weight, Integer_Type, char> &j_seg,
                  Segment<Weight, Integer_Type, Integer_Type> &jv_seg);
*/        

                  
        void print(Segment<Weight, Integer_Type, Fractional_Type> &segment);
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

        //MPI_Datatype get_data_type();
        
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
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
        
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        Vector<Weight, Integer_Type, Fractional_Type> *X;
        Vector<Weight, Integer_Type, Fractional_Type> *V;
        Vector<Weight, Integer_Type, Fractional_Type> *S;
        std::vector<Vector<Weight, Integer_Type, Fractional_Type> *> Y;
        Vector<Weight, Integer_Type, Integer_Type> *R;
        Vector<Weight, Integer_Type, char> *I;
        Vector<Weight, Integer_Type, Integer_Type> *IV;
        Vector<Weight, Integer_Type, Integer_Type> *C;
        Vector<Weight, Integer_Type, char> *J;
        Vector<Weight, Integer_Type, Integer_Type> *JV;
        
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        
        MPI_Datatype MPI_TYPE;
        
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program() {};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::Vertex_Program(Graph<Weight,
                       Integer_Type, Fractional_Type> &Graph, Ordering_type ordering_type_)
                       : X(nullptr), V(nullptr), S(nullptr)
{
    A = Graph.A;

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
        R = Graph.A->R;
        I = Graph.A->I;
        IV= Graph.A->IV;
        C = Graph.A->C;
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
        R = Graph.A->C;
        I = Graph.A->J;
        IV= Graph.A->JV;
        C = Graph.A->R;
        J = Graph.A->I;
        JV= Graph.A->IV;
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vertex_Program<Weight, Integer_Type, Fractional_Type>::~Vertex_Program()
{
    
    /*
    delete X;
    delete V;
    delete S;   
    
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        delete Y[i];
    }
    */
}

/* Borrowed from https://github.com/thu-pacman/GeminiGraph/blob/master/core/mpi.hpp */
/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
MPI_Datatype Vertex_Program<Weight, Integer_Type, Fractional_Type>::get_data_type()
{
    if (std::is_same<Fractional_Type, char>::value)
    {
        return MPI_CHAR;
    }
    else if (std::is_same<Fractional_Type, unsigned char>::value)
    {
        return MPI_UNSIGNED_CHAR;
    }
    else if (std::is_same<Fractional_Type, int>::value)
    {
        return MPI_INT;
    }
    else if (std::is_same<Fractional_Type, unsigned int>::value)
    {
        return MPI_UNSIGNED;
    }
    else if (std::is_same<Fractional_Type, long>::value)
    {
        return MPI_UNSIGNED_LONG;
    }
    else if (std::is_same<Fractional_Type, unsigned long>::value)
    {
        return MPI_UNSIGNED_LONG;
    }
    else if (std::is_same<Fractional_Type, float>::value)
    {
        return MPI_FLOAT;
    }
    else if (std::is_same<Fractional_Type, double>::value)
    {
        return MPI_DOUBLE;
    }
    else 
    {
        fprintf(stderr, "Type not supported\n");
        Env::exit(1);
    }   
}
*/

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::free()
{
    /*
    X->del_vec();
    V->del_vec();
    S->del_vec();
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        Y[i]->del_vec();
    }
    */
    delete X;
    delete V;
    delete S;   
    
    for (uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        delete Y[i];
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::clear(
                               Vector<Weight, Integer_Type, Fractional_Type> *vec)
{
    for(uint32_t i = 0; i < vec->vector_length; i++)
    {
        //auto &seg = vec->segments[i];
        //if(seg.allocated)
        if(vec->allocated[i])
        {
            //auto *data = (Fractional_Type *) seg.D;
            //Integer_Type nitems = seg.n;
            //uint64_t nbytes = seg.nbytes;
            //memset(data, 0, nbytes);
            
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
        //auto &seg = vec->segments[i];
        //if(seg.allocated)
        if(vec->allocated[i])
        {
            //auto *data = (Fractional_Type *) seg.D;
            //Integer_Type nitems = seg.n;
            //uint64_t nbytes = seg.nbytes;
            
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
    //if(filtering_type == _NONE_)
    //{
        for(uint32_t i = 0; i < vec_src->vector_length; i++)
        {
            //auto &seg_src = vec_src->segments[i];
            //auto *data_src = (Fractional_Type *) seg_src.D;
            //Integer_Type nitems_src = seg_src.n;
            //uint64_t nbytes_src = seg_src.nbytes;
            
            //auto &seg_dst = vec_dst->segments[i];
            //auto *data_dst = (Fractional_Type *) seg_dst.D;
            //Integer_Type nitems_dst = seg_dst.n;
            //uint64_t nbytes_dst = seg_dst.nbytes;
            
            Fractional_Type *data_src = (Fractional_Type *) vec_src->data[i];
            Integer_Type nitems_src = vec_src->nitems[i];
            uint64_t nbytes_src = nitems_src * sizeof(Fractional_Type);
            
            Fractional_Type *data_dst = (Fractional_Type *) vec_dst->data[i];
            Integer_Type nitems_dst = vec_dst->nitems[i];
            uint64_t nbytes_dst = nitems_dst * sizeof(Fractional_Type);
            
            memcpy(data_dst, data_src, nbytes_src);
        }
    //}
    //else if(filtering_type == _SOME_)
    //{
        
        
    //}
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::init(Fractional_Type x, Fractional_Type y, 
     Fractional_Type v, Fractional_Type s, Vertex_Program<Weight, Integer_Type, Fractional_Type> *VProgram)
{
    MPI_TYPE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    //printf("X %d\n", Env::rank);//
    std::vector<Integer_Type> x_sizes;
    //x_sizes.resize(rank_ncolgrps, tile_height);
    
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
    
    //if(!Env::rank)
    //    printf("X: %lu %lu\n", x_sizes.size(), local_col_segments.size());

    std::vector<Integer_Type> v_s_size = {tile_height};
    //std::vector<Integer_Type> v_s_size(1,tile_height);
    /*
    std::vector<Integer_Type> v_s_size;
    if(filtering_type == _NONE_)
    {
        v_s_size.resize(1, tile_height);     
    }
    else if(filtering_type == _SOME_)
    {
        v_s_size.resize(1, A->nnz_row_sizes_loc[accu_segment_row]);
    }
    */
    
    V = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    populate(V, v);
    //printf("S %d\n", Env::rank);
    S = new Vector<Weight, Integer_Type, Fractional_Type>(v_s_size, accu_segment_row_vec);
    if(VProgram)
    {
        Vertex_Program<Weight, Integer_Type, Fractional_Type> *VP = VProgram;
        Vector<Weight, Integer_Type, Fractional_Type> *V_ = VP->V;
        populate(S, V_);
        /*
        uint32_t so = 0;
        auto &s_seg = S->segments[so];
        auto *s_data = (Fractional_Type *) s_seg.D->data;
        Integer_Type s_nitems = s_seg.D->n;
        
        uint32_t vo = 0;
        auto &v_seg = VP->V->segments[so];
        auto *v_data = (Fractional_Type *) v_seg.D->data;
        Integer_Type v_nitems = v_seg.D->n;
        
        uint32_t yo = accu_segment_row;
        
        auto &r_seg = R->segments[yo];
        auto *r_data = (Integer_Type *) r_seg.D->data;
        Integer_Type r_nitems = r_seg.D->n;
        
        auto &i_seg = I->segments[yo];
        auto *i_data = (char *) i_seg.D->data;
        Integer_Type i_nitems = i_seg.D->n;
        
        auto &iv_seg = IV->segments[yo];
        auto *iv_data = (Integer_Type *) iv_seg.D->data;
        Integer_Type iv_nitems = iv_seg.D->n;
        
        uint32_t xo = accu_segment_col;
        
        auto &c_seg = C->segments[xo];
        auto *c_data = (Integer_Type *) c_seg.D->data;
        Integer_Type c_nitems = c_seg.D->n;
        
        auto &j_seg = J->segments[xo];
        auto *j_data = (char *) j_seg.D->data;
        Integer_Type j_nitems = j_seg.D->n;
        
        auto &jv_seg = JV->segments[xo];
        auto *jv_data = (Integer_Type *) jv_seg.D->data;
        Integer_Type jv_nitems = jv_seg.D->n;
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            printf("%f ", v_data[i]);
        }
        printf(" %d %d\n", v_nitems, r_nitems);
        
        Integer_Type k = 0;
        for(uint32_t i = 0; i < s_nitems; i++)
        {
            //printf("[%d %d %d %d]", i, c_data[i], i_data[c_data[i]], iv_data[c_data[i]]);//, j_data[r_data[i]]);
            printf("[%d %d %d %d]", i, r_data[i], i_data[r_data[i]], j_data[r_data[i]]);//, j_data[c_data[i]], jv_data[c_data[i]]);//, j_data[r_data[i]]);
            
            if(j_data[r_data[i]])
            {
               // printf("????\n");
                k = jv_data[r_data[i]];
                    
                s_data[i] = v_data[k];
            }
            else
               s_data[i] = s;
           
            //x_data[i] = 1;
        }
        */
        
    }
    else
    {
        populate(S, s);
    }
    
    //if(!Env::rank)
    //    printf("V: %lu %lu\n", v_s_size.size(), accu_segment_col_vec.size());
    
    //printf("Y %d\n", Env::rank);
    std::vector<Integer_Type> y_size;
    std::vector<Integer_Type> y_sizes;

    Vector<Weight, Integer_Type, Fractional_Type> *Y_;
    for(uint32_t j = 0; j < rank_nrowgrps; j++)
    {
        y_size.clear();
        y_sizes.clear();
        //y_size = {tile_height};
        //y_sizes.resize(rowgrp_nranks, tile_height);
        
        if(filtering_type == _NONE_)
        {
            y_size.resize(1, tile_height);
            //y_size = {tile_height};
            y_sizes.resize(rowgrp_nranks, tile_height);
        }
        else if(filtering_type == _SOME_)
        {
            y_size.resize(1, nnz_row_sizes_loc[j]);
            //y_size = {A->nnz_row_sizes_loc[j]};
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
        //if(!Env::rank)
        //    printf("Y: %d %lu %lu %lu %lu\n", local_row_segments[j] == owned_segment, y_size.size(), accu_segment_row_vec.size(), y_sizes.size(), all_rowgrp_ranks_accu_seg.size());
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::scatter(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    //MPI_Datatype T = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    
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
        
    
    
    
    //auto &x_seg = X->segments[xo];
    //auto *x_data = (Fractional_Type *) x_seg.D;
    //Integer_Type x_nitems = x_seg.n;
    
    //uint32_t vo = 0;
    //auto &v_seg = V->segments[vo];
    //auto *v_data = (Fractional_Type *) v_seg.D;
    //Integer_Type v_nitems = v_seg.n;
    
    //uint32_t so = 0;
    //auto &s_seg = S->segments[so];
    //auto *s_data = (Fractional_Type *) s_seg.D;
    //Integer_Type s_nitems = s_seg.n;
 
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
        //auto &j_seg = J->segments[xo];
        //char *j_data = (char *) j_seg.D;
        //Integer_Type j_nitems = j_seg.n;
        
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
               MPI_Isend(x_data, x_nitems, MPI_TYPE, follower, col_group, colgrps_communicator, &request);
               out_requests.push_back(request);
            }
            else
            {
                follower = follower_colgrp_ranks[i];
                MPI_Isend(x_data, x_nitems, MPI_TYPE, follower, col_group, Env::MPI_WORLD, &request);
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
    //MPI_Datatype T = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    uint32_t leader;
    MPI_Request request;
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW)
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {    
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            //auto& xj_seg = X->segments[i];
            //auto *xj_data = (Fractional_Type *) xj_seg.D;
            //Integer_Type xj_nitems = xj_seg.n;
            Fractional_Type *xj_data = (Fractional_Type *) X->data[i];
            Integer_Type xj_nitems = X->nitems[i];
            int32_t col_group = X->local_segments[i];
            

            if(Env::comm_split)
            {
                leader = leader_ranks_cg[col_group];
                if(leader != Env::rank_cg)
                {
                    MPI_Irecv(xj_data, xj_nitems, MPI_TYPE, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                }
            }
            else
            {
                leader = leader_ranks[col_group];
                if(leader != Env::rank)
                {
                    MPI_Irecv(xj_data, xj_nitems, MPI_TYPE, leader, col_group, Env::MPI_WORLD, &request);
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::bcast(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
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
    
    
    //uint32_t xo = accu_segment_col;
    //auto &x_seg = X->segments[xo];
    //auto *x_data = (Fractional_Type *) x_seg.D;
    //Integer_Type x_nitems = x_seg.n;
    
    //uint32_t vo = 0;
    //auto &v_seg = V->segments[vo];
    //auto *v_data = (Fractional_Type *) v_seg.D;
    //Integer_Type v_nitems = v_seg.n;
    
    //uint32_t so = 0;
    //auto &s_seg = S->segments[so];
    //auto *s_data = (Fractional_Type *) s_seg.D;
    //Integer_Type s_nitems = s_seg.n;
    
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
            //printf("[%f %f %f]", x_data[i], v_data[i], s_data[i]);
        }
        //printf("\n");
    }
    else if(filtering_type == _SOME_)
    {

        /*                
        uint32_t xo = accu_segment_col;
        uint32_t yo = accu_segment_row;
        
        auto &r_seg = R->segments[yo];
        auto *r_data = (Integer_Type *) r_seg.D;
        Integer_Type r_nitems = r_seg.n;
        
        auto &i_seg = I->segments[yo];
        auto *i_data = (char *) i_seg.D;
        Integer_Type i_nitems = i_seg.n;

        auto &iv_seg = IV->segments[yo];
        auto *iv_data = (Integer_Type *) iv_seg.D;
        Integer_Type iv_nitems = iv_seg.n;
        
        auto &c_seg = C->segments[xo];
        auto *c_data = (Integer_Type *) c_seg.D;
        Integer_Type c_nitems = c_seg.n;
        */
        //auto &j_seg = J->segments[xo];
        //char *j_data = (char *) j_seg.D;
        //Integer_Type j_nitems = j_seg.n;
        
        char *j_data = (char *) J->data[xo];
        Integer_Type j_nitems = J->nitems[xo];

        //auto &jv_seg = JV->segments[xo];
        //auto *jv_data = (Integer_Type *) jv_seg.D;
        //Integer_Type jv_nitems = jv_seg.n;
        
        /*
        Integer_Type j = 0;
        for(uint32_t i = 0; i < x_nitems; i++)
        {
            //printf("[%d %d %d %d]", i, c_data[i], i_data[c_data[i]], iv_data[c_data[i]]);//, j_data[r_data[i]]);
            //if(i_data[c_data[i]])
            //{
                //if(i_data[r_data[i]])
                //{
                    //k = iv_data[c_data[i]];
                    //x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
                    //assert(jv_data[c_data[i]] != NA);
                    x_data[i] = (*f)(0, 0, v_data[c_data[i]], s_data[c_data[i]]);
                    //assert(iv_data[i] == c_data[i]);
                    //j++;
                //}
            //}
            //else
              //  x_data[i] = (*f)(0, 0, 0, 0);
            //x_data[i] = 1;
        }
        */
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                //assert(jv_data[i] == j);
                x_data[j] = (*f)(0, 0, v_data[i], s_data[i]);
                j++;
            }               
        }
        
        
        //printf("\n");
        
        /*
        for(uint32_t i = 0; i < x_nitems; i++)
        {
            x_data[i] = 1;
        }
        */
    }
    
    /*
    if(filtering_type == _NONE_)
    {
        
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
        }
        //printf("%d %d %d\n", x_nitems, tile_height, v_nitems);
    }
    else if(filtering_type == _SOME_)
    {
        uint32_t jo = accu_segment_col;
        auto &j_seg = J->segments[jo];
        auto *j_data = (char *) j_seg.D->data;
        Integer_Type j_nitems = j_seg.D->n;
        Integer_Type j_nbytes = j_seg.D->nbytes;
        */
        /*
        for(uint32_t i = 0; i < c_nitems; i++)
        {
            x_data[i] = (*f)(0, 0, v_data[c_data[i]], s_data[c_data[i]]);
        }
        */
        /*
        Integer_Type j  = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                x_data[j] = (*f)(0, 0, v_data[i], s_data[i]);
                j++;
            }
        }
        */
        /*
        Integer_Type j  = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(j_data[i])
            {
                x_data[i] = (*f)(0, 0, v_data[i], s_data[i]);
            }
        }
        */
    //}
    
    if(((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_NUMA_))
        or (tiling_type == Tiling_type::_1D_ROW) 
        or (tiling_type == Tiling_type::_1D_COL and ordering_type == _COL_))
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            leader = leader_ranks_cg[local_col_segments[i]];
            //auto &xj_seg = X->segments[i];
            //auto *xj_data = (Fractional_Type *) xj_seg.D;
            //Integer_Type xj_nitems = xj_seg.n;
            
            Fractional_Type *xj_data = (Fractional_Type *) X->data[i];
            Integer_Type xj_nitems = X->nitems[i];
            
            
            if(Env::comm_split)
            {
                MPI_Bcast(xj_data, xj_nitems, MPI_TYPE, leader, colgrps_communicator);
                //MPI_Bcast(xj_data, xj_nbytes, MPI_BYTE, leader, colgrps_communicator);
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
    //printf("bcast %d\n", Env::rank);
    //int retval = MPI_Type_free(&T);
}                   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
            Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    ;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Fractional_Type *y_data, Fractional_Type *x_data,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    //Fractional_Type *y_data = (Fractional_Type *) y_seg.D;
    //Integer_Type y_nitems = y_seg.n;

    //Fractional_Type *x_data = (Fractional_Type *) x_seg.D;
    //Integer_Type x_nitems = x_seg.n;

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
            //uint32_t n = 0;
            if(ordering_type == _ROW_)
            {
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    //printf("%d: ", i);
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {
                        //printf("[%f %f", x_data[JA[j]], y_data[i]);
                        #ifdef HAS_WEIGHT
                        if(x_data[JA[j]] and A[j])
                            y_data[i] += A[j] * x_data[JA[j]];
                        #else
                        if(x_data[JA[j]])
                            y_data[i] += x_data[JA[j]];
                        #endif   
                        //printf("-%f %f]", x_data[JA[j]], y_data[i]);
                        //n++;
                        //if(!x_data[JA[j]])
                        //{
                            //printf("[%d %d %f]", j, JA[j], x_data[JA[j]]);
                          //  Env::exit(1);
                        //}
                    
                    }
                    //printf("\n");                        
                }
                //printf("\n");
                //printf("%d %d %d\n", Env::rank, tile.kth, n);  
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
/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
            Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            Segment<Weight, Integer_Type, Integer_Type> &r_seg,
            Segment<Weight, Integer_Type, char> &i_seg,
            Segment<Weight, Integer_Type, Integer_Type> &iv_seg,
            Segment<Weight, Integer_Type, Integer_Type> &c_seg,
            Segment<Weight, Integer_Type, char> &j_seg,
            Segment<Weight, Integer_Type, Integer_Type> &jv_seg)
{
    ;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::spmv(
            Segment<Weight, Integer_Type, Fractional_Type> &y_seg,
            Segment<Weight, Integer_Type, Fractional_Type> &x_seg,
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            Segment<Weight, Integer_Type, Integer_Type> &t_seg,
            Segment<Weight, Integer_Type, char> &k_seg,
            Segment<Weight, Integer_Type, Integer_Type> &kv_seg)
{
    auto *y_data = (Fractional_Type *) y_seg.D->data;
    Integer_Type y_nitems = y_seg.D->n;
    Integer_Type y_nbytes = y_seg.D->nbytes;

    auto *x_data = (Fractional_Type *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    auto *k_data = (char *) k_seg.D->data;
    Integer_Type k_nitems = k_seg.D->n;
    Integer_Type k_nbytes = k_seg.D->nbytes;
    
    Integer_Type *kv_data = (Integer_Type *) kv_seg.D->data;
    Integer_Type kv_nitems = kv_seg.D->n;
    Integer_Type kv_nbytes = kv_seg.D->nbytes;
  

    if(tile.allocated)
    {
        if(compression_type == Compression_type::_CSR_)
        {
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A->data;
            #endif
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
            Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
            
            if(ordering_type == _ROW_)
            {
                if(filtering_type == _SRCS_)
                {
                    Integer_Type k = 0;
                    //#pragma omp parallel for schedule(dynamic)
                    for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                    {
                        for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                        {       
                            #ifdef HAS_WEIGHT
                            if(x_data[JA[j]] and A[j])
                                y_data[k] += A[j] * x_data[JA[j]];
                            #else
                            if(x_data[JA[j]])
                                y_data[k] += x_data[JA[j]];
                            #endif                               
                        }
                        if(k_data[i])    
                            k++;
                    }
                    
                }
                else if (filtering_type == _SNKS_)
                {
                    Integer_Type k = 0;
                    //#pragma omp parallel for schedule(dynamic)
                    for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                    {
                        for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                        {       
                            #ifdef HAS_WEIGHT
                            if(x_data[kv_data[JA[j]]] and A[j])
                                y_data[i] += A[j] * x_data[kv_data[JA[j]]];
                            #else
                            if(x_data[kv_data[JA[j]]])
                                y_data[i] += x_data[kv_data[JA[j]]];
                            #endif                               
                        }   
                    }
                }
                
            }
            else if(ordering_type == _COL_)
            {
                fprintf(stderr, "Invalid filtering type\n");
                Env::exit(1);
            }
        }
        else if(compression_type == Compression_type::_CSC_)    
        {
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
            Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
            if(ordering_type == _ROW_)
            {
                if(filtering_type == _SRCS_)
                {
                    
                    //printf("RANK %d %d %d %d %d\n", Env::rank, k_nitems, k_nbytes, kv_nitems, kv_nbytes);
                    
                    Integer_Type k = 0;
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                        {
                            #ifdef HAS_WEIGHT
                            if(x_data[j] and VAL[i])
                                y_data[kv_data[ROW_INDEX[i]]] += VAL[i] * x_data[j];   
                            #else
                            if(x_data[j])   
                            {    
                                //printf("r=%d i=%d ri=%d kv_i=%d ta=%d ya=%d kva=%d yn=%d ", Env::rank, i, ROW_INDEX[i], kv_data[ROW_INDEX[i]], tile.allocated, y_seg.allocated, kv_seg.allocated, y_nitems);
                                //printf("r=%d y_kv_ri=%f\n", Env::rank, y_data[kv_data[ROW_INDEX[i]]]);
                                y_data[kv_data[ROW_INDEX[i]]] += x_data[j];
                            }
                            #endif
                            //if(j_data[ROW_INDEX[i]])
                            //    k++;
                        }
                    }
                }
                if(filtering_type == _SNKS_)
                {
                    Integer_Type k = 0;
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                        {
                            #ifdef HAS_WEIGHT
                            if(k_data[j] and VAL[i])
                                y_data[ROW_INDEX[i]] += VAL[i] * x_data[k];   
                            #else
                            if(k_data[j])    
                                y_data[ROW_INDEX[i]] += x_data[k];
                                //y_data[ROW_INDEX[i]] += x_data[j];
                            #endif
                        }
                        if(k_data[j])    
                            k++;
                    }
                }
            }
            else if(ordering_type == _COL_)
            {
                fprintf(stderr, "Invalid filtering type\n");
                Env::exit(1);
            }
        }            
    }    
}
*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::combine()
{
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
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_row()
{
    uint32_t yi = accu_segment_row;
    uint32_t xi = 0;
    uint32_t yo = accu_segment_rg;
  
    // SPMV
    for(uint32_t t: local_tiles_row_order)
    {
        //if(!Env::rank)
        //printf("rank=%d t=%d xi =%d yi=%d yo=%d \n", Env::rank, t, xi, yi, yo);

        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        //printf("%d %d %d %d %d %d\n", Env::rank, t, yi, yo, accu_segment_col, accu_segment_cg);    
        /*
        auto *Yp = Y[yi];
        

        
        auto &y_seg = Yp->segments[yo];
        auto &x_seg = X->segments[xi];
        spmv(y_seg, x_seg, tile);
        */
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);

        
        xi++;
    }
    //printf("!!!!\n");
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::optimized_1d_col()
{
    //MPI_Comm communicator;
    MPI_Request request;
    MPI_Status status;
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
        /*
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        
        auto &x_seg = X->segments[xi];
        spmv(y_seg, x_seg, tile);
        */
        
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);
        
        
        
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
        /*
        auto *Yp = Y[yi];
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D;
        Integer_Type y_nitems = y_seg.n;
        */
        
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        
        
        
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
                    
                    //auto &yj_seg = Yp->segments[accu];
                    //auto *yj_data = (Fractional_Type *) yj_seg.D;
                    //Integer_Type yj_nitems = yj_seg.n;
                    
                    Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                    Integer_Type yj_nitems = Yp->nitems[accu];
                    
                    
                    MPI_Irecv(yj_data, yj_nitems, MPI_TYPE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                    
                }
            }
            else
            {
                MPI_Isend(y_data, y_nitems, MPI_TYPE, leader, pair_idx, communicator, &request);
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
    //MPI_Datatype T;// = get_data_type();
    //MPI_Datatype T = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    //struct Triple<Weight, Integer_Type> pair, pair1;
    //MPI_Comm communicator;
    MPI_Request request;
    //MPI_Status status;
    uint32_t tile_th, pair_idx;
    uint32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    double t1, t2;
    t1 = Env::clock();
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        
        //if(Env::rank == 2)
         //   printf("tile=%2d, row=%d, col=%d\n", t, pair.row, pair.col);
        
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
        
        //auto *Yp = Y[yi];
        //auto &y_seg = Yp->segments[yo];
        //auto *y_data = (Fractional_Type *) y_seg.D;
        //Integer_Type y_nitems = y_seg.n;
        //auto &x_seg = X->segments[xi];
        //spmv(y_seg, x_seg, tile);
        
        auto *Yp = Y[yi];
        Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
        Integer_Type y_nitems = Yp->nitems[yo];
        
        Fractional_Type *x_data = (Fractional_Type *) X->data[xi];
        spmv(y_data, x_data, tile);
        
        
        //printf("x_seg %d\n", Env::rank);
        
        /*
        if(filtering_type == _NONE_)
        {
            spmv(y_seg, x_seg, tile);
        }
        else if(filtering_type == _SRCS_)
        {
            auto &r_seg = R->segments[yi];
            
            auto &i_seg = I->segments[yi];
            
            auto &iv_seg = IV->segments[yi];
            
            spmv(y_seg, x_seg, tile, r_seg, i_seg, iv_seg);
        }
        else if(filtering_type == _SNKS_)
        {
            auto &c_seg = C->segments[xi];
            
            auto &j_seg = J->segments[xi];
            
            auto &jv_seg = JV->segments[xi];
            
            spmv(y_seg, x_seg, tile, c_seg, j_seg, jv_seg);
        }
        else if(filtering_type == _BOTH_)
        {
            auto &r_seg = R->segments[yi];
            
            auto &i_seg = I->segments[yi];
            
            auto &iv_seg = IV->segments[yi];
            
            auto &c_seg = C->segments[xi];
            
            auto &j_seg = J->segments[xi];
            
            auto &jv_seg = JV->segments[xi];
            
            spmv(y_seg, x_seg, tile, r_seg, i_seg, iv_seg, c_seg, j_seg, jv_seg);
        }
        */
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
                //if(!Env::rank)
                //    printf("Recv me=%d leader=%d <-- follower=%d\n", my_rank, leader, follower);                    
                    
                    //auto &yj_seg = Yp->segments[accu];
                    //auto *yj_data = (Fractional_Type *) yj_seg.D;
                    //Integer_Type yj_nitems = yj_seg.n;
                    
                    Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
                    Integer_Type yj_nitems = Yp->nitems[accu];
                    
                    MPI_Irecv(yj_data, yj_nitems, MPI_TYPE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                    //in_statuses.push_back(status);
                }
                //yk = yi;
            }
            else
            {
                //if(!Env::rank)
                //    printf("Send me=%d --> leader=%d\n", my_rank, leader);
                MPI_Isend(y_data, y_nitems, MPI_TYPE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
                //out_statuses.push_back(status);
            }
            xi = 0;
            yi++;
        }
    }
     
    t2 = Env::clock();
    if(!Env::rank)
        printf("Combine spmv: %f seconds\n", t2 - t1);
    
    t1 = Env::clock();
    wait_for_all();
    //wait_for_recvs();
    t2 = Env::clock();
    if(!Env::rank)
        printf("Combine recv: %f seconds\n", t2 - t1);
    //int retval = MPI_Type_free(&T);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::apply(Fractional_Type (*f)
                   (Fractional_Type, Fractional_Type, Fractional_Type, Fractional_Type))
{
    //printf("Apply %d\n", Env::rank);
    uint32_t accu;
    uint32_t yi = accu_segment_row;
    auto *Yp = Y[yi];
    uint32_t yo = accu_segment_rg;
    
    Fractional_Type *y_data = (Fractional_Type *) Yp->data[yo];
    Integer_Type y_nitems = Yp->nitems[yo];
    
    //auto &y_seg = Yp->segments[yo];
    //auto *y_data = (Fractional_Type *) y_seg.D;
    //Integer_Type y_nitems = y_seg.n;
    
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];
        
        Fractional_Type *yj_data = (Fractional_Type *) Yp->data[accu];
        Integer_Type yj_nitems = Yp->nitems[accu];
        
        //auto &yj_seg = Yp->segments[accu];
        //auto *yj_data = (Fractional_Type *) yj_seg.D;
        //Integer_Type yj_nitems = yj_seg.n;

        for(uint32_t i = 0; i < yj_nitems; i++)
        {
            if(yj_data[i])
                y_data[i] += yj_data[i];
        }
    }
    /*
    uint32_t co = accu_segment_col;
    Segment<Weight, Integer_Type, int32_t> &c_seg = C->segments[co];
    int32_t *c_data = (int32_t *) c_seg.D->data;
    Integer_Type c_nitems = c_seg.D->n;
    Integer_Type c_nbytes = c_seg.D->nbytes;
    
    uint32_t jo = accu_segment_col;
    Segment<Weight, Integer_Type, int32_t> &j_seg = J->segments[jo];
    int32_t *j_data = (int32_t *) j_seg.D->data;
    Integer_Type j_nitems = j_seg.D->n;
    Integer_Type j_nbytes = j_seg.D->nbytes;
    */
    
    
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    //uint32_t vo = 0;
    //auto &v_seg = V->segments[vo];
    //auto *v_data = (Fractional_Type *) v_seg.D;
    //Integer_Type v_nitems = v_seg.n;
    
    
    
    if(filtering_type == _NONE_)
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            v_data[i] = (*f)(0, y_data[i], 0, 0); 
        }
    }
    else if(filtering_type == _SOME_)
    {
        /*
        yi = accu_segment_row;
        auto &r_seg = R->segments[yi];
        auto *r_data = (Integer_Type *) r_seg.D;
        Integer_Type r_nitems = r_seg.n;
        */
        //auto &i_seg = I->segments[yi];
        //auto *i_data = (char *) i_seg.D;
        //Integer_Type i_nitems = i_seg.n;
        /*
        auto &iv_seg = IV->segments[yi];
        auto *iv_data = (Integer_Type *) iv_seg.D;
        Integer_Type iv_nitems = iv_seg.n;
        */
        /*
        for(uint32_t i = 0; i < y_nitems; i++)
        {
                                
            //v_data[i] = (*f)(0, y_data[i], 0, 0);
            assert(iv_data[i] != NA);
            v_data[r_data[i]] = (*f)(0, y_data[i], 0, 0);
            
            //if(!Env::rank)
            //    printf("%d %d %d %d %f %f\n", i, r_data[i], tile_height, y_nitems, y_data[i], v_data[r_data[i]]);
        }
        */
       
        char *i_data = (char *) I->data[yi];
        //Integer_Type i_nitems = I->nitems[yi];
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(i_data[i])
            {
                //assert(iv_data[i] == j);
                v_data[i] = (*f)(0, y_data[j], 0, 0);
                j++;
            }
            else
                v_data[i] = (*f)(0, 0, 0, 0);
        }
        
        
    }
        
    /*
    if((filtering_type == _NONE_) or (filtering_type == _SNKS_))
    {
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            v_data[i] = (*f)(0, y_data[i], 0, 0); 
        }
    }
    else if((filtering_type == _SRCS_) or (filtering_type == _BOTH_))
    {
        //printf("%d\n", Env::rank);
        auto &i_seg = I->segments[yi];
        char *i_data = (char *) i_seg.D->data;
        Integer_Type i_nitems = i_seg.D->n;
        Integer_Type i_nbytes = i_seg.D->nbytes;
        
        Integer_Type j  = 0;
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            if(i_data[i])
            {
                //if(!Env::rank)
                    //printf("i=%d v=%f, y=%f\n", i_data[i], v_data[i], y_data[j]);
                v_data[i] = (*f)(0, y_data[j], 0, 0); 
                j++;
            }
        }
    }
    //printf("%d %d %d\n", Env::rank, v_nitems, i_nitems);
    */
    //wait_for_sends();
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        clear(Y[i]);
    //Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::wait_for_all()
{
    
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
    Env::barrier();
    /*
    MPI_Waitall(in_requests.size(), in_requests.data(), in_statuses.data());
    MPI_Datatype T = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    uint32_t i = 0;
    uint32_t this_rank = 0;
    uint32_t my_rank = Env::rank;
    for(uint32_t j = 0; j < rowgrp_nranks; j++)
    {            
        uint32_t yi = accu_segment_row;
        auto *Yp = Y[yi];
        uint32_t yo = accu_segment_rg;
        auto &y_seg = Yp->segments[yo];
        auto *y_data = (Fractional_Type *) y_seg.D->data;
        Integer_Type y_nitems = y_seg.D->n;
        Integer_Type y_nbytes = y_seg.D->nbytes;

        this_rank = all_rowgrp_ranks[j];
        if(this_rank != my_rank)
        {
            int count = 0;
            MPI_Get_count(&in_statuses[i], T, &count);
            assert(count == y_nitems);
            i++;
        }
    }
    
    
    
    in_requests.clear();
    in_statuses.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), out_statuses.data());
    out_requests.clear();
    out_statuses.clear();
    */
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
        //return{tile.nth, pair.row};
    }
    else if(ordering_type == _COL_)
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
    if(ordering_type == _ROW_)
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
    else if(ordering_type == _COL_)
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
    //Env::barrier();
    double v_sum_local = 0, v_sum_gloabl = 0;
    
    uint32_t vo = 0;
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    //uint32_t vo = 0;
    //auto &v_seg = V->segments[vo];
    //auto *v_data = (Fractional_Type *) v_seg.D;
    //Integer_Type v_nitems = v_seg.n;
    

    
    v_sum_local = 0;  
//if(c_seg.allocated)
//{    
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
//}
   // printf("r=%d r=%lu\n", Env::rank, v_sum_local);
   
    MPI_Allreduce(&v_sum_local, &v_sum_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Degree checksum: %f\n", v_sum_gloabl);
    //Env::barrier();
    /*  
    v_sum_local = 0, v_sum_gloabl = 0;
    
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
    Fractional_Type *v_data = (Fractional_Type *) V->data[vo];
    Integer_Type v_nitems = V->nitems[vo];
    
    //uint32_t vo = 0;
    //auto &v_seg = V->segments[vo];
    //auto *v_data = (Fractional_Type *) v_seg.D;
    //Integer_Type v_nitems = v_seg.n;
        
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        v_local += v_data[i];
    }
    
    MPI_Allreduce(&v_local, &v_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Value checksum: %f\n", v_gloabl); 
    
    
    uint64_t s_local = 0, s_gloabl = 0;
    
    uint32_t so = 0;
    Fractional_Type *s_data = (Fractional_Type *) S->data[so];
    Integer_Type s_nitems = S->nitems[so];
    
    //uint32_t so = 0;
    //auto &s_seg = S->segments[so];
    //auto *s_data = (Fractional_Type *) s_seg.D;
    //Integer_Type s_nitems = s_seg.n;
    
    for(uint32_t i = 0; i < s_nitems; i++)
    {
        s_local += s_data[i];
    }
    
    
    MPI_Allreduce(&s_local, &s_gloabl, 1, MPI_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        printf("Score checksum: %lu\n", s_gloabl);
    
    
    //if(filtering_type == _NONE_)
    //{
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
    //}
    /*
    else if(filtering_type == _SOME_)
    {
        
        uint32_t yo = accu_segment_row;
        auto &r_seg = R->segments[yo];
        auto *r_data = (Integer_Type *) r_seg.D->data;
        Integer_Type r_nitems = r_seg.D->n;
        
        auto &i_seg = I->segments[yo];
        char *i_data = (char *) i_seg.D->data;
        Integer_Type i_nitems = i_seg.D->n;

        auto &iv_seg = IV->segments[yo];
        auto *iv_data = (Integer_Type *) iv_seg.D->data;
        Integer_Type iv_nitems = iv_seg.D->n;
        
        uint32_t NUM = 31;
        uint32_t count = i_nitems < NUM ? i_nitems : NUM;
        if(!Env::rank)
        {
            Triple<Weight, Integer_Type> pair, pair1;
            for(uint32_t i = 0; i < count; i++)
            {
                //printf("%d %d\n", i, i_data[i]);
                pair.row = i;
                pair.col = 0;
                pair1 = A->base(pair, owned_segment, owned_segment);
                if(i_data[i])
                    printf("Rank[%d],Value[%d]=%f,Score[%d]=%f\n",  Env::rank, pair1.row, v_data[iv_data[i]], pair1.row, s_data[iv_data[i]]);
                else
                    printf("Rank[%d],Value[%d]=%f,Score[%d]=%f\n",  Env::rank, pair1.row, .0, pair1.row, .0);
            }  
        }
    }
    */
    
    
    
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vertex_Program<Weight, Integer_Type, Fractional_Type>::print(Segment<Weight,
                            Integer_Type, Fractional_Type> &segment)
{
    Triple<Weight, Integer_Type> pair;
    Triple<Weight, Integer_Type> pair1;
    auto *data = (Fractional_Type *) segment.D;
    Integer_Type nitems = segment.n;
    
    for(uint32_t i = 0; i < nitems; i++)
    {
        pair.row = i;
        pair.col = 0;
        auto pair1 = A->base(pair, owned_segment, owned_segment);
        printf("R(%d),V[%d]=%f\n",  Env::rank, pair1.row, data[i]);
    } 
}

#endif
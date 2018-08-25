/*
 * vector.hpp: Vector and segment implementations
 * A vector is consists of multiple segments.
 * The number of segments is equal to the number
 * of column gorups.
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
struct Segment
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vector;
    
    //template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    //friend class Vertex_Program;
    
    Segment();
    Segment(Integer_Type nrows_, Integer_Type ncols_, 
            uint32_t rg_, uint32_t cg_, uint32_t leader_rank_, 
            uint32_t leader_rank_rg_, uint32_t leader_rank_cg_, bool allocated_);
    ~Segment();
    void allocate();
    void allocate(Integer_Type n);
    void del_seg();
    
    struct Basic_Storage<Fractional_Type, Integer_Type> *D;

    Integer_Type nrows, ncols;
    uint32_t g;
    uint32_t rg, cg;
    uint32_t leader_rank;
    uint32_t leader_rank_rg;
    uint32_t leader_rank_cg;
    bool allocated;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(Integer_Type nrows_, Integer_Type ncols_,
                              uint32_t rg_, uint32_t cg_, uint32_t leader_rank_,
                              uint32_t leader_rank_rg_, uint32_t leader_rank_cg_, bool allocated_)
{
    nrows  = nrows_;
    ncols  = ncols_;
    rg     = rg_;
    cg     = cg_;
    leader_rank   = leader_rank_;       
    leader_rank_rg   = leader_rank_rg_;
    leader_rank_cg   = leader_rank_cg_;
    allocated = allocated_;
    D = nullptr;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::allocate()
{
    D = new struct Basic_Storage<Fractional_Type, Integer_Type>(nrows);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::allocate(Integer_Type n)
{
    D = new struct Basic_Storage<Fractional_Type, Integer_Type>(n);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::del_seg()
{
    delete D;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    //template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    //friend class Matrix;
    
    public:
        Vector();
        Vector(Integer_Type nrows_, Integer_Type ncols_, uint32_t nrowgrps_, uint32_t ncolgrps_,
               Integer_Type tile_height_, Integer_Type tile_width_, uint32_t owned_segment_,
               std::vector<uint32_t> &leader_ranks, std::vector<uint32_t> &leader_ranks_rg,
               std::vector<uint32_t> &leader_ranks_cg, std::vector<int32_t> &local_segments_);
        Vector(Integer_Type nrows_, Integer_Type ncols_, uint32_t nrowgrps_, uint32_t ncolgrps_,
               Integer_Type tile_height_, Integer_Type tile_width_, uint32_t owned_segment_,
               std::vector<uint32_t> &leader_ranks, std::vector<uint32_t> &leader_ranks_rg,
               std::vector<uint32_t> &leader_ranks_cg);
        Vector(Integer_Type nrows_, std::vector<int32_t> &local_segments_);
        ~Vector();
        void del_vec();
        void del_vec_1();
        
        std::vector<struct Segment<Weight, Integer_Type, Fractional_Type>> segments;
        std::vector<int32_t> local_segments;
        uint32_t owned_segment;
        uint32_t vector_length;
    private:
        Integer_Type nrows, ncols;
        uint32_t nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width;
        
        //void init_vec(std::vector<uint32_t> &diag_ranks, std::vector<int32_t>& local_segments);
        //void init_vec(std::vector<uint32_t> &diag_ranks);
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, std::vector<int32_t> &local_segments_)
{
    nrows = nrows_;
    local_segments = local_segments_;
    vector_length = local_segments.size();
    // Reserve the 1D vector of segments. 
    segments.resize(vector_length);
    for(uint32_t i = 0; i < vector_length; i++)
    {
        segments[i].allocate(nrows);
        segments[i].allocated = true;
        segments[i].g = local_segments[i];
        #ifdef PREFETCH
        madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
        #endif
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, Integer_Type ncols_,
               uint32_t nrowgrps_, uint32_t ncolgrps_, Integer_Type tile_height_, Integer_Type tile_width_,
               uint32_t owned_segment_, std::vector<uint32_t> &leader_ranks, 
               std::vector<uint32_t> &leader_ranks_rg, std::vector<uint32_t> &leader_ranks_cg,
               std::vector<int32_t> &local_segments_)
{
    nrows = nrows_;
    ncols = ncols_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    tile_height = tile_height_;
    tile_width = tile_width_;
    owned_segment = owned_segment_;
    
    // Reserve the 1D vector of segments. 
    segments.resize(ncolgrps);
    for (uint32_t i = 0; i < ncolgrps; i++)
    {
        segments[i].nrows  = tile_height;
        segments[i].ncols  = tile_width;
        segments[i].rg     = i;
        segments[i].cg     = i;
        segments[i].leader_rank = leader_ranks[i];
        segments[i].leader_rank_rg = leader_ranks_rg[i];
        segments[i].leader_rank_cg = leader_ranks_cg[i];
        segments[i].allocated = false;
        
        if(leader_ranks[i] == Env::rank)
        {
            segments[i].allocate();
            madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
            segments[i].allocated = true;
        }
    }
    
    local_segments = local_segments_;
    
    for(int32_t s: local_segments)
    {
        
        if(segments[s].leader_rank != Env::rank)
        {
            segments[s].allocate();
            madvise(segments[s].D->data, segments[s].D->nbytes, MADV_SEQUENTIAL);
            segments[s].allocated = true;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>    
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, Integer_Type ncols_,
               uint32_t nrowgrps_, uint32_t ncolgrps_, Integer_Type tile_height_, Integer_Type tile_width_,
               uint32_t owned_segment_, std::vector<uint32_t> &leader_ranks,
               std::vector<uint32_t> &leader_ranks_rg, std::vector<uint32_t> &leader_ranks_cg)
{
    nrows = nrows_;
    ncols = ncols_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    tile_height = tile_height_;
    tile_width = tile_width_;
    owned_segment = owned_segment_;
    
    // Reserve the 1D vector of segments. 
    segments.resize(ncolgrps);
    for (uint32_t i = 0; i < ncolgrps; i++)
    {
        segments[i].nrows  = tile_height;
        segments[i].ncols  = tile_width;
        segments[i].rg     = i;
        segments[i].cg     = i;
        segments[i].leader_rank   = leader_ranks[i];
        segments[i].leader_rank_rg = leader_ranks_rg[i];
        segments[i].leader_rank_cg = leader_ranks_cg[i];
        segments[i].allocated = false;
        
        if(leader_ranks[i] == Env::rank)
        {
            segments[i].allocate();
            madvise(segments[i].D->data, segments[i].D->nbytes, MADV_SEQUENTIAL);
            segments[i].allocated = true;
        }
    }
}
    
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vector<Weight, Integer_Type, Fractional_Type>::del_vec_1()
{
    //uint32_t i = 0;
    //f/or(int32_t s: local_segments)
        //if(!Env::rank)
          //  printf("len=%d\n", vector_length);
    for(uint32_t i = 0; i < vector_length; i++)
    //for(int32_t s: local_segments)
    //{
    {
        if(segments[i].allocated)
        {
            segments[i].del_seg();
        }
       // i++;
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Vector<Weight, Integer_Type, Fractional_Type>::del_vec()
{
    for (uint32_t i = 0; i < ncolgrps; i++)
    {
        if(segments[i].allocated)
        {
            segments[i].del_seg();
        }
    }
}
#endif
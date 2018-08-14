/*
 * vector.hpp: Vector and segment implementations
 * A vector is consists of multiple segments.
 * The number of segments is equal to the number
 * of column gorups.
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
struct Segment
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vector;
    
    //template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    //friend class Vertex_Program;
    
    Segment();
    Segment(Integer_Type nrows_, Integer_Type ncols_, 
            uint32_t rg_, uint32_t cg_, uint32_t leader_rank_, bool allocated_);
    ~Segment();
    void allocate();
    void del_seg();
    
    struct basic_storage<Fractional_Type, Integer_Type> *D;

    Integer_Type nrows, ncols;
    uint32_t rg, cg;
    uint32_t leader_rank;
    bool allocated;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(Integer_Type nrows_, Integer_Type ncols_,
                              uint32_t rg_, uint32_t cg_, uint32_t leader_rank_, bool allocated_)
{
    nrows  = nrows_;
    ncols  = ncols_;
    rg     = rg_;
    cg     = cg_;
    leader_rank   = leader_rank_;       
    allocated = allocated_;
    D = nullptr;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::allocate()
{
    D = new struct basic_storage<Fractional_Type, Integer_Type>(nrows);
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
    
    public:
        Vector(Integer_Type nrows_, Integer_Type ncols_, uint32_t nrowgrps_, uint32_t ncolgrps_,
               Integer_Type tile_height_, Integer_Type tile_width_, uint32_t owned_segment_,
               std::vector<uint32_t> &leader_ranks, std::vector<uint32_t> &local_segments_);
       Vector(Integer_Type nrows_, Integer_Type ncols_, uint32_t nrowgrps_, uint32_t ncolgrps_,
               Integer_Type tile_height_, Integer_Type tile_width_, uint32_t owned_segment_,
               std::vector<uint32_t> &leader_ranks);
        ~Vector();
        void del_vec();
        
        std::vector<struct Segment<Weight, Integer_Type, Fractional_Type>> segments;
        std::vector<uint32_t> local_segments;
        uint32_t owned_segment;
    private:
        Integer_Type nrows, ncols;
        uint32_t nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width;

        void init_vec(std::vector<uint32_t> &diag_ranks, std::vector<uint32_t>& local_segments);
        void init_vec(std::vector<uint32_t> &diag_ranks);
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, Integer_Type ncols_,
               uint32_t nrowgrps_, uint32_t ncolgrps_, Integer_Type tile_height_, Integer_Type tile_width_,
               uint32_t owned_segment_, std::vector<uint32_t> &leader_ranks, std::vector<uint32_t> &local_segments_)
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
        segments[i].allocated = false;
        
        if(leader_ranks[i] == Env::rank)
        {
            segments[i].allocate();
            segments[i].allocated = true;
        }
    }
    
    local_segments = local_segments_;
    
    for(uint32_t s: local_segments)
    {
        
        if(segments[s].leader_rank != Env::rank)
        {
            segments[s].allocate();
            segments[s].allocated = true;
        }
    }
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>    
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, Integer_Type ncols_,
               uint32_t nrowgrps_, uint32_t ncolgrps_, Integer_Type tile_height_, Integer_Type tile_width_,
               uint32_t owned_segment_, std::vector<uint32_t> &leader_ranks)
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
        segments[i].allocated = false;
        
        if(leader_ranks[i] == Env::rank)
        {
            segments[i].allocate();
            segments[i].allocated = true;
        }
    }
}
    
template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector() {}

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
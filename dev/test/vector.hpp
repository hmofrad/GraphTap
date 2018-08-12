/*
 * vector.hpp: Vector implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Segment
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vector;
    
    Segment();
    Segment(Integer_Type tile_height, Integer_Type tile_width, uint32_t rg, uint32_t cg, uint32_t rank_);
    ~Segment();
    
    struct basic_storage<Weight, Integer_Type> *D;

    uint32_t nrows, ncols;
    uint32_t rg, cg;
    uint32_t rank;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(Integer_Type tile_height,
                Integer_Type tile_width, uint32_t rg, uint32_t cg, uint32_t rank_)
{
    D = new struct basic_storage<Fractional_Type, Integer_Type>(tile_height);

    nrows  = tile_height;
    ncols  = tile_width;
    rg     = rg;
    cg     = cg;
    rank   = rank_;
        
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment()
{
    delete D;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    public:
        Vector(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, uint32_t owned_segment_,
               std::vector<uint32_t> &leader_ranks, std::vector<uint32_t> &local_segments);
        ~Vector();
    
    private:
        Integer_Type nrows, ncols;
        Integer_Type nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width; // == segment_height
        uint32_t owned_segment;
    
        std::vector<struct Segment<Weight, Integer_Type, Fractional_Type>> segments;
        std::vector<uint32_t> local_segments;
        
        void init_vec(std::vector<uint32_t> &diag_ranks, std::vector<uint32_t>& local_segments);
        void init_vec(std::vector<uint32_t> &diag_ranks);
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(Integer_Type nrows_, Integer_Type ncols_,
                uint32_t ntiles_, uint32_t owned_segment_, std::vector<uint32_t> &leader_ranks,
                std::vector<uint32_t> &local_segments)
    //: nrows(nrows_), ncols(ncols_), nrowgrps(sqrt(ntiles_)), ncolgrps(ntiles_/nrowgrps),
    //  tile_height((nrows_ / nrowgrps) + 1), tile_width((ncols_ / ncolgrps) + 1), owned_segment(owned_segment_)
{
    nrows = nrows_;
    ncols = ncols_;
    nrowgrps = sqrt(ntiles_);
    ncolgrps = ntiles_ / nrowgrps;
    tile_height = (nrows_ / nrowgrps) + 1;
    tile_width = (ncols_ / ncolgrps) + 1;
    owned_segment = owned_segment_;
    
    // Reserve the 1D vector of segments. 
    //segments.resize(ncolgrps);
    Segment<Weight, Integer_Type, Fractional_Type> segment;
    //for (uint32_t i = 0; i < ncolgrps; i++)
    //{
        //segment = new Segment<Weight, Integer_Type, Fractional_Type>(tile_height, tile_width, i, i, Env::rank);
    //}
     
    /*
    
    for (uint32_t i = 0; i < Vector<Weight, Integer_Type, Fractional_Type>::ncolgrps; i++)
    {
        segments[i].n      = tile_height;
        segments[i].nbytes = tile_height * sizeof(Integer_Type);
        segments[i].nrows  = tile_height;
        segments[i].ncols  = tile_width;
        segments[i].rg     = i;
        segments[i].cg     = i;
        segments[i].rank   = diag_ranks[i]; // leader_ranks
        
        if(diag_ranks[i] == rank)
        {
            segments[i] = new Segment<Weight, Integer_Type, Fractional_Type>();
            //segments[i].allocate();  
        }
    }
    
    Vector<Weight>::local_segments = local_segments;
    
    for(uint32_t s: Vector<Weight>::local_segments)
    {
        if(Vector<Weight>::segments[s].rank != rank)
        {
            Vector<Weight>::segments[s].allocate();
        }
    }
    */
    
};
    

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector() {};
/*
template<typename Weight>
void Vector<Weight>::init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments)
{
    // Reserve the 1D vector of segments. 
    Vector<Weight>::segments.resize(Vector<Weight>::ncolgrps);

    for (uint32_t i = 0; i < Vector<Weight>::ncolgrps; i++)
    {
        Vector<Weight>::segments[i].n = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].nbytes = Vector<Weight>::tile_height * sizeof(fp_t);
        Vector<Weight>::segments[i].nrows = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].ncols = Vector<Weight>::tile_width;
        Vector<Weight>::segments[i].rg = i;
        Vector<Weight>::segments[i].cg = i;
        Vector<Weight>::segments[i].rank = diag_ranks[i];
        
        if(diag_ranks[i] == rank)
        {
            Vector<Weight>::segments[i].allocate();  
        }
    }
    
    Vector<Weight>::local_segments = local_segments;
    
    for(uint32_t s: Vector<Weight>::local_segments)
    {
        if(Vector<Weight>::segments[s].rank != rank)
        {
            Vector<Weight>::segments[s].allocate();
        }
    }
}


template<typename Weight>
void Vector<Weight>::init_vec(std::vector<uint32_t>& diag_ranks)
{
    // Reserve the 1D vector of segments. 
    Vector<Weight>::segments.resize(Vector<Weight>::ncolgrps);

    for (uint32_t i = 0; i < Vector<Weight>::ncolgrps; i++)
    {
        Vector<Weight>::segments[i].n = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].nbytes = Vector<Weight>::tile_height * sizeof(fp_t);
        Vector<Weight>::segments[i].nrows = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].ncols = Vector<Weight>::tile_width;
        Vector<Weight>::segments[i].rg = i;
        Vector<Weight>::segments[i].cg = i;
        Vector<Weight>::segments[i].rank = diag_ranks[i];
        if(diag_ranks[i] == rank)
        {
            Vector<Weight>::segments[i].allocate();
            Vector<Weight>::local_segments.push_back(i);
        }
    }
}
*/
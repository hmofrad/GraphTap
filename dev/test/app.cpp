/*
 * Test app 
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>

#include <sys/mman.h>
#include <type_traits>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <mpi.h>
#include <sched.h>


using fp_t = double;

int sched_getcpu(void);

int nranks;
int rank;
int cpu_id;
bool is_master;

template <typename Weight>
struct Triple
{
    uint32_t row;
    uint32_t col;
    Weight weight;
    Triple(uint32_t row = 0, uint32_t col = 0, Weight weight = 0)
        : row(row), col(col), weight(weight) {}  
    void set_weight(Weight& w) {this->weight = w;};
    Weight get_weight() {return(this->weight);};
};

struct Empty {};

template <>
struct Triple <Empty>
{
    uint32_t row;
    union {
        uint32_t col;
        Empty weight;
    };
    void set_weight(Empty& w) {};
    bool get_weight() {return 1;};
};

template <typename Weight>
struct Functor
{
    bool operator()(const struct Triple<Weight>& a, const struct Triple<Weight>& b)
    {
        return((a.row == b.row) ? (a.col < b.col) : (a.row < b.row));
    }
};

template<typename Weight>
struct CSR
{
    uint32_t nnz;
    uint32_t nrows_plus_one;
    uint32_t* A;
    uint32_t* IA;
    uint32_t* JA;
    void allocate(uint32_t nnz, uint32_t nrows_plus_one);
    void free();
};

template<typename Weight>
void CSR<Weight>::allocate(uint32_t nnz, uint32_t nrows_plus_one)
{        
    CSR<Weight>::nnz = nnz;
    CSR<Weight>::nrows_plus_one = nrows_plus_one;
    CSR<Weight>::A = (uint32_t *) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    //(void *) -1)
    //{
      //  fprintf(stderr, "Error mapping memory\n");
        //exit(1);
    //}
    memset(CSR<Weight>::A, 0, CSR<Weight>::nnz * sizeof(uint32_t));
    CSR<Weight>::IA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nrows_plus_one) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(CSR<Weight>::IA, 0, CSR<Weight>::nrows_plus_one * sizeof(uint32_t));
    CSR<Weight>::JA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(CSR<Weight>::JA, 0, CSR<Weight>::nnz * sizeof(uint32_t));
}        

template<typename Weight>
void CSR<Weight>::free()
{
    munmap(CSR<Weight>::A, (this->nnz) * sizeof(uint32_t));
    //{
    //if(munmap(address, size) == -1) {
    //    fprintf(stderr, "Error unmapping memory\n");
      //  exit(1);
    //}
    
    
    
    munmap(CSR<Weight>::IA, (this->nrows_plus_one) * sizeof(uint32_t));
    munmap(CSR<Weight>::JA, (this->nnz) * sizeof(uint32_t));    
}

template <typename Weight>
struct Tile2D
{ 
    template <typename Weight_>
    friend class Matrix;
    std::vector<struct Triple<Weight>>* triples;
    struct CSR<Weight>* csr;
    uint32_t rg, cg;
    uint32_t ith, jth, nth;
    int32_t rank;
};

enum Tiling
{
  _1D_ROW,
  _1D_COL,
  _2D_
};

template<typename Weight>
class Partitioning
{    
    template<typename Weight_>
    friend class Matrix;

    template<typename Weight__>
    friend class Graph;
    
    public:    
        Partitioning(uint32_t nranks, uint32_t rank, uint32_t ntiles, uint32_t nrowgrps, uint32_t ncolgrps, Tiling tiling);
        ~Partitioning();
    
    private:
        const uint32_t nranks, rank;
        const uint32_t ntiles, nrowgrps, ncolgrps;
        const Tiling tiling;
        
        uint32_t rank_ntiles, rank_nrowgrps, rank_ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        
        void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
};

template<typename Weight>
Partitioning<Weight>::Partitioning(uint32_t nranks, uint32_t rank, uint32_t ntiles, uint32_t nrowgrps, uint32_t ncolgrps, Tiling tiling) 
    : nranks(nranks), rank(rank), ntiles(ntiles), nrowgrps(nrowgrps), ncolgrps(ncolgrps), rank_ntiles(ntiles / nranks), tiling(tiling)
{
    assert(Partitioning<Weight>::rank_ntiles * Partitioning<Weight>::nranks == ntiles);
    if (Partitioning<Weight>::tiling == Tiling::_1D_ROW)
    {
        Partitioning<Weight>::rowgrp_nranks = 1;
        Partitioning<Weight>::colgrp_nranks = nranks;
        assert(Partitioning<Weight>::rowgrp_nranks * Partitioning<Weight>::colgrp_nranks == Partitioning<Weight>::nranks);

        Partitioning<Weight>::rank_nrowgrps = 1;
        Partitioning<Weight>::rank_ncolgrps = Partitioning<Weight>::ncolgrps;
        assert(Partitioning<Weight>::rank_nrowgrps * Partitioning<Weight>::rank_ncolgrps == Partitioning<Weight>::rank_ntiles);
    }        
    else if (Partitioning<Weight>::tiling == Tiling::_2D_)
    {
        integer_factorize(Partitioning<Weight>::nranks, Partitioning<Weight>::rowgrp_nranks, Partitioning<Weight>::colgrp_nranks);
        assert(Partitioning<Weight>::rowgrp_nranks * Partitioning<Weight>::colgrp_nranks == Partitioning<Weight>::nranks);
    
        Partitioning<Weight>::rank_nrowgrps = Partitioning<Weight>::nrowgrps / Partitioning<Weight>::colgrp_nranks;
        Partitioning<Weight>::rank_ncolgrps = Partitioning<Weight>::ncolgrps / Partitioning<Weight>::rowgrp_nranks;
        assert(Partitioning<Weight>::rank_nrowgrps * Partitioning<Weight>::rank_ncolgrps == Partitioning<Weight>::rank_ntiles);
    }
};

template<typename Weight>
Partitioning<Weight>::~Partitioning() {};

template <class Weight>
void Partitioning<Weight>::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b)
{
  /* This approach is adapted from that of GraphPad. */
  a = b = sqrt(n);
  while (a * b != n)
  {
    b++;
    a = n / b;
  }
  assert(a * b == n);
}



template<typename Weight>
class Matrix
{
    template<typename Weight_>
    friend class Graph;
    
    public:    
        Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles, Tiling tiling);
        ~Matrix();

    private:
        const uint32_t nrows, ncols;
        const uint32_t ntiles, nrowgrps, ncolgrps;
        const uint32_t tile_height, tile_width;    
        
        Partitioning<Weight>* partitioning;

        std::vector<std::vector<struct Tile2D<Weight>>> tiles;
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_segments;
        std::vector<uint32_t> local_col_segments;
        
        std::vector<uint32_t> diag_ranks;
        std::vector<uint32_t> other_row_ranks_accu_seg;
        std::vector<uint32_t> other_col_ranks_accu_seg;

        std::vector<uint32_t> other_rowgrp_ranks;
        std::vector<uint32_t> rowgrp_ranks_accu_seg;
        std::vector<uint32_t> other_colgrp_ranks; 
        
        void init_mat();
        void del_triples();
        void init_csr();
        void del_csr();
        
        uint32_t local_tile_of_tile(const struct Triple<Weight>& pair);
        uint32_t segment_of_tile(const struct Triple<Weight>& pair);
        struct Triple<Weight> tile_of_triple(const struct Triple<Weight>& triple);
        struct Triple<Weight> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight> rebase(const struct Triple<Weight>& pair);
        struct Triple<Weight> base(const struct Triple<Weight>& pair, uint32_t rowgrp, uint32_t colgrp);
};

template<typename Weight>
Matrix<Weight>::Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles, Tiling tiling) 
    : nrows(nrows), ncols(ncols), ntiles(ntiles), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles / nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1) {};

template<typename Weight>
Matrix<Weight>::~Matrix() {};
      
template <typename Weight>
uint32_t Matrix<Weight>::local_tile_of_tile(const struct Triple<Weight>& pair)
{
  return((pair.row * Matrix<Weight>::ncolgrps) + pair.col);
}      
      
template <typename Weight>
struct Triple<Weight> Matrix<Weight>::tile_of_triple(const struct Triple<Weight>& triple)
{
  return{(triple.row / Matrix<Weight>::tile_height), (triple.col / Matrix<Weight>::tile_width)};
}

template <typename Weight>
struct Triple<Weight> Matrix<Weight>::tile_of_local_tile(const uint32_t local_tile)
{
  return{(local_tile - (local_tile % Matrix<Weight>::ncolgrps)) / Matrix<Weight>::ncolgrps, local_tile % Matrix<Weight>::ncolgrps};
}

template <typename Weight> 
uint32_t Matrix<Weight>::segment_of_tile(const struct Triple<Weight>& pair)
{
    return(pair.col);
}
template <typename Weight> 
struct Triple<Weight> Matrix<Weight>::rebase(const struct Triple<Weight>& pair)
{
    return{(pair.row % Matrix<Weight>::tile_height), (pair.col % Matrix<Weight>::tile_width)};
}

template <typename Weight>
struct Triple<Weight> Matrix<Weight>::base(const struct Triple<Weight>& pair, uint32_t rowgrp, uint32_t colgrp)
{
   return{(pair.row + (rowgrp * Matrix<Weight>::tile_height)), (pair.col + (colgrp * Matrix<Weight>::tile_width))};
}



template<typename Weight>
void Matrix<Weight>::del_triples()
{
    // Delete triples
    Triple<Weight> pair;
    for(uint32_t t: Matrix<Weight>::local_tiles)
    {
        pair = Matrix<Weight>::tile_of_local_tile(t);
        auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
        delete tile.triples;        
    }
    
}

template <typename Weight> 
void Matrix<Weight>::del_csr()
{
    Triple<Weight> pair;
    for(uint32_t t: Matrix<Weight>::local_tiles)
    {
        pair = Matrix<Weight>::tile_of_local_tile(t);
        auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
        tile.csr->free();
        delete tile.csr;
    }
}

template<typename Weight>
struct Segment
{
    template<typename Weight_>
    friend class Vector;
    
    //uint32_t *data1;    
    //
    void *data;    
    uint32_t n;
    uint64_t nbytes;
    uint32_t nrows, ncols;
    uint32_t rg, cg;
    uint32_t rank;
    void allocate();
    void free();
    uint64_t get_nbytes();
    
    void allocate1();
    void free1();
};

/*
template<typename Weight>
void Segment<Weight>::allocate1()
{
    Segment<Weight>::data = (uint32_t*) mmap(nullptr, (this->n) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(Segment<Weight>::data, 0, this->n * sizeof(uint32_t));
}

template<typename Weight>
void Segment<Weight>::free1()
{
    munmap(Segment<Weight>::data, (this->n) * sizeof(uint32_t));
}
*/

template<typename Weight>
void Segment<Weight>::allocate()
{
    if((Segment<Weight>::data = mmap(nullptr, this->nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1)
    //if((void *) Segment<Weight>::data1 == (void*) -1)
    {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
        
    memset(Segment<Weight>::data, 0, this->nbytes);
}

template<typename Weight>
void Segment<Weight>::free()
{
    if(munmap(Segment<Weight>::data, this->nbytes) == -1)
    {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    //munmap(Segment<Weight>::data1, this->nbytes);
}

template<typename Weight>
uint64_t Segment<Weight>::get_nbytes()
{
    return(this->nbytes);
}



template<typename Weight>
class Vector
{
    template<typename Weight_>
    friend class Graph;
    
    public:
        Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles, uint32_t diag_segment);
        ~Vector();
    
    private:
        const uint32_t nrows, ncols;
        const uint32_t nrowgrps, ncolgrps;
        const uint32_t tile_height, tile_width;    // == segment_height
        const uint32_t diag_segment;
    
        std::vector<struct Segment<Weight>> segments;
        std::vector<uint32_t> local_segments;
        
        //void init_vec1(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments);
        //void init_vec1(std::vector<uint32_t>& diag_ranks);
        void init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments);
        void init_vec(std::vector<uint32_t>& diag_ranks);
};

template<typename Weight>
Vector<Weight>::Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles, uint32_t diag_segment) 
    : nrows(nrows), ncols(ncols), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles/nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1), diag_segment(diag_segment) {};
    

template<typename Weight>
Vector<Weight>::~Vector() {};
/*
template<typename Weight>
void Vector<Weight>::init_vec1(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments)
{
    // Reserve the 1D vector of segments. 
    Vector<Weight>::segments.resize(Vector<Weight>::ncolgrps);

    for (uint32_t i = 0; i < Vector<Weight>::ncolgrps; i++)
    {
        Vector<Weight>::segments[i].n = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].nrows = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].ncols = Vector<Weight>::tile_width;
        Vector<Weight>::segments[i].rg = i;
        Vector<Weight>::segments[i].cg = i;
        Vector<Weight>::segments[i].rank = diag_ranks[i];
        if(diag_ranks[i] == rank)
        {
           Vector<Weight>::segments[i].allocate1();    
        }
    }
    
    Vector<Weight>::local_segments = local_segments;
    
    for(uint32_t s: Vector<Weight>::local_segments)
    {
        if(Vector<Weight>::segments[s].rank != rank)
            Vector<Weight>::segments[s].allocate1();
    }
}

template<typename Weight>
void Vector<Weight>::init_vec1(std::vector<uint32_t>& diag_ranks)
{
    // Reserve the 1D vector of segments. 
    Vector<Weight>::segments.resize(Vector<Weight>::ncolgrps);

    for (uint32_t i = 0; i < Vector<Weight>::ncolgrps; i++)
    {
        Vector<Weight>::segments[i].n = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].nrows = Vector<Weight>::tile_height;
        Vector<Weight>::segments[i].ncols = Vector<Weight>::tile_width;
        Vector<Weight>::segments[i].rg = i;
        Vector<Weight>::segments[i].cg = i;
        Vector<Weight>::segments[i].rank = diag_ranks[i];
        if(diag_ranks[i] == rank)
        {
           Vector<Weight>::segments[i].allocate1();    
           Vector<Weight>::local_segments.push_back(i);
        }
    }
}

*/

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
           //if(!rank)
          //printf("SIZEEEEEE=%d fp=%lu nb=%lu\n", Vector<Weight>::segments[i].n, sizeof(fp_t), Vector<Weight>::segments[i].nbytes);
        }
    }
    
    Vector<Weight>::local_segments = local_segments;
    
    for(uint32_t s: Vector<Weight>::local_segments)
    {
        if(Vector<Weight>::segments[s].rank != rank)
            Vector<Weight>::segments[s].allocate();
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
         // if(!rank)
           //   printf("SIZEEEEEE=%d fp=%lu nb=%lu d=%lu\n", Vector<Weight>::tile_height, sizeof(fp_t), Vector<Weight>::segments[i].nbytes, sizeof(Vector<Weight>::segments[i].data));

        }
    }
}



template<typename Weight>
class Graph
{
    public:    
        Graph();
        ~Graph();
        
        void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_ = true, bool transpose_ = false);
        void load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_ = true, bool transpose_ = false);

    private:
        std::string filepath;
        uint32_t nvertices;
        uint32_t mvertices;
        uint64_t nedges;
        bool directed;
        bool transpose;
        Matrix<Weight>* A;
        Vector<Weight>* X;
        Vector<Weight>* Y;
        Vector<Weight>* V;
        Vector<Weight>* S;
        
        void degree();
        void pagerank();
        void free();
};

template<typename Weight>
Graph<Weight>::Graph() : A(nullptr) {};

template<typename Weight>
Graph<Weight>::~Graph()
{
    delete Graph<Weight>::A->partitioning;
    delete Graph<Weight>::A;
    
    delete Graph<Weight>::X;
    delete Graph<Weight>::Y;
    delete Graph<Weight>::V;
    delete Graph<Weight>::S;
}

template<typename Weight>
void Graph<Weight>::free()
{
    Graph<Weight>::A->del_csr();
    
    for(uint32_t s: Graph<Weight>::X->local_segments)
    {
        auto& segment = Graph<Weight>::X->segments[s];
        segment.free();
    }
    
    for(uint32_t s: Graph<Weight>::Y->local_segments)
    {
        auto& segment = Graph<Weight>::Y->segments[s];
        segment.free();
    }

    for(uint32_t s: Graph<Weight>::V->local_segments)
    {
        auto& segment = Graph<Weight>::V->segments[s];
        segment.free();
    }
    
    for(uint32_t s: Graph<Weight>::S->local_segments)
    {
        auto& segment = Graph<Weight>::S->segments[s];
        segment.free();
    }
}

template<typename Weight>
void Graph<Weight>::load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_, bool transpose_)
{
    // Initialize graph data
    // Note we keep using Graph<Weight> format to avoid confusion
    Graph<Weight>::filepath = filepath_;
    Graph<Weight>::nvertices = nrows;
    Graph<Weight>::mvertices = ncols;
    Graph<Weight>::nedges = 0;
    Graph<Weight>::directed = directed_;
    Graph<Weight>::transpose = transpose_;
    
    Graph<Weight>::A = new Matrix<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, tiling);
    Graph<Weight>::A->partitioning = new Partitioning<Weight>(nranks, rank, nranks * nranks, Graph<Weight>::A->nrowgrps, Graph<Weight>::A->ncolgrps, tiling);
    Graph<Weight>::A->init_mat();

    // Open matrix file.
    std::ifstream fin(Graph<Weight>::filepath.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        std::cout << "Unable to open input file" << std::endl;
        exit(1); 
    }
    
    // Obtain filesize
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    struct Triple<Weight> triple;
    struct Triple<Weight> pair;
    while (offset < filesize)
    {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        
        if(Graph<Weight>::transpose)
        {
            std::swap(triple.row, triple.col);
        }
        
        if(fin.gcount() != sizeof(Triple<Weight>))
        {
            std::cout << "read() failure" << std::endl;
            exit(1);
        }
        
        pair = Graph<Weight>::A->tile_of_triple(triple);
        Graph<Weight>::nedges++;    

        if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)    
        {
            Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
        offset += sizeof(Triple<Weight>);
    }
    fin.close();
    assert(offset == filesize);
    
    /*
    
    if(rank == 3)
    {
        for(uint32_t t: Graph<Weight>::A->local_tiles)
        {
            pair = Graph<Weight>::A->tile_of_local_tile(t);
            auto& triples = *(Graph<Weight>::A->tiles[pair.row][pair.col].triples);
            for(auto& tt: triples)
                printf("%d[%d][%d]:%d %d\n", rank, pair.row, pair.col, tt.row, tt.col);
            
        }
    }
    
    */
    
    if(!rank)
    {
        printf("[x]Reading\n");
    }
    
    Graph<Weight>::A->init_csr();
    
    //std::vector<uint32_t> my_segment;
    uint32_t diag_segment = distance(Graph<Weight>::A->diag_ranks.begin(), find(Graph<Weight>::A->diag_ranks.begin(), Graph<Weight>::A->diag_ranks.end(), rank));
    //my_segment.push_back(diag_segment);
    
    //uint32_t index = distance(Graph<Weight>::A->rowgrp_ranks.begin(), find(Graph<Weight>::A->rowgrp_ranks.begin(), Graph<Weight>::A->rowgrp_ranks.end(), rank));
    //uint32_t accu_segment = Graph<Weight>::A->rowgrp_ranks[index];
    //Graph<Weight>::A->rowgrp_ranks_accu_seg[index] = diag_segment;
    
    
    
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    
    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->rowgrp_ranks_accu_seg);    
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks);//, my_segment);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks);
    

    
    
    
    //Graph<Weight>::XX = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    //Graph<Weight>::XX->init_vec1(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    
    //Graph<Weight>::T = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    //Graph<Weight>::T->init_vec1(Graph<Weight>::A->diag_ranks);    
    

    
    
    if(rank == 0)
    {
        //uint32_t index = distance(Graph<Weight>::A->rowgrp_ranks.begin(), find(Graph<Weight>::A->rowgrp_ranks.begin(), Graph<Weight>::A->rowgrp_ranks.end(), rank));
        //uint32_t accu_seg = Graph<Weight>::A->rowgrp_ranks[index];
        //printf("%d %d %d\n", index, accu_seg, diag_segment);
        //Graph<Weight>::A->rowgrp_ranks_accu_seg[index] = diag_segment;
        for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; i++)
        {
            printf("[%d %d] ", Graph<Weight>::A->other_rowgrp_ranks[i], Graph<Weight>::A->rowgrp_ranks_accu_seg[i]);
        }
        printf("\n");
        
        for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->colgrp_nranks - 1; i++)
        {
            printf("[%d] ", Graph<Weight>::A->other_colgrp_ranks[i]);
        }
        printf("\n");

        for(uint32_t i = 0; i < Graph<Weight>::A->local_col_segments.size(); i++)
        {
            printf("%d ", Graph<Weight>::A->local_col_segments[i]);
        }
        printf(" %d\n", diag_segment);
        
    }
    
    
    Graph<Weight>::degree();
    Graph<Weight>::pagerank();
    
    if(!rank)
    {
        printf("[x]DEGREE\n");
    }
    
    Graph<Weight>::free();
    
    
}





template<typename Weight>
void Graph<Weight>::load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_, bool transpose_)
{
    
    // Initialize graph data
    // Note we keep using Graph<Weight> format to avoid confusion
    Graph<Weight>::filepath  = filepath_;
    Graph<Weight>::nvertices = nrows;
    Graph<Weight>::mvertices = ncols;
    Graph<Weight>::nedges    = 0;
    Graph<Weight>::directed  = directed_;
    Graph<Weight>::transpose = transpose_;
    
    Graph<Weight>::A = new Matrix<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, tiling);
    Graph<Weight>::A->partitioning = new Partitioning<Weight>(nranks, rank, nranks * nranks, Graph<Weight>::A->nrowgrps, Graph<Weight>::A->ncolgrps, tiling);
    Graph<Weight>::A->init_mat();
    
    // Open matrix file.
    std::ifstream fin(Graph<Weight>::filepath.c_str());
    if(!fin.is_open())
    {
        std::cout << "Unable to open input file" << std::endl;
        exit(1); 
    }

    // Obtain filesize
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    // Skip comments
    std::string line;
    uint32_t position; // Fallback position
    do
    {
        position = fin.tellg();
        std::getline(fin, line);
    } while ((line[0] == '#') || (line[0] == '%'));
    fin.seekg(position, std::ios_base::beg);

    struct Triple<Weight> triple;
    struct Triple<Weight> pair;
    std::istringstream iss;
    while (std::getline(fin, line) && !line.empty())
    {
        iss.clear();
        iss.str(line);
        
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
        {
            std::cout << "read() failure" << std::endl;
            exit(1);
        }
        
        if(Graph<Weight>::transpose)
        {
            iss >> triple.col >> triple.row;
        }
        else
        {
            iss >> triple.row >> triple.col;
        }
        
        Graph<Weight>::nedges++;
    
        pair = Graph<Weight>::A->tile_of_triple(triple);
        if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)    
        {
            Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
        offset = fin.tellg();
    }
    fin.close();
    assert(offset == filesize);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(!rank)
    {
        printf("[x]Reading\n");
    }
    
    Graph<Weight>::A->init_csr();
    
    //std::vector<uint32_t> my_segment;
    uint32_t diag_segment = distance(Graph<Weight>::A->diag_ranks.begin(), find(Graph<Weight>::A->diag_ranks.begin(), Graph<Weight>::A->diag_ranks.end(), rank));
    //my_segment.push_back(diag_segment);

    
    //uint32_t index = distance(Graph<Weight>::A->rowgrp_ranks.begin(), find(Graph<Weight>::A->rowgrp_ranks.begin(), Graph<Weight>::A->rowgrp_ranks.end(), rank));
    //uint32_t accu_segment = Graph<Weight>::A->rowgrp_ranks[index];
    //Graph<Weight>::A->rowgrp_ranks_accu_seg[index] = diag_segment;
    
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    
    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
       // if(!rank)
       // {
         //   printf("+++++SB=%lu\n", Graph<Weight>::X->segments[diag_segment].nbytes);
       // }
    
    
    
    
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->rowgrp_ranks_accu_seg);    
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks);//, my_segment);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks);//, my_segment);

    
        
    
    
    //Graph<Weight>::XX = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    //Graph<Weight>::XX->init_vec1(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    
    //Graph<Weight>::T = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    //Graph<Weight>::T->init_vec1(Graph<Weight>::A->diag_ranks);    
    
    
    //if(!rank)
    //    printf("SZ=%lu\n", sizeof(Graph<Weight>::T->segments[diag_segment].nbytes));

    if(rank == 0)
    {
        //uint32_t index = distance(Graph<Weight>::A->rowgrp_ranks.begin(), find(Graph<Weight>::A->rowgrp_ranks.begin(), Graph<Weight>::A->rowgrp_ranks.end(), rank));
        //uint32_t accu_seg = Graph<Weight>::A->rowgrp_ranks[index];
        //printf("%d %d %d\n", index, accu_seg, diag_segment);
        //Graph<Weight>::A->rowgrp_ranks_accu_seg[index] = diag_segment;
        for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; i++)
        {
            printf("[%d %d] ", Graph<Weight>::A->other_rowgrp_ranks[i], Graph<Weight>::A->rowgrp_ranks_accu_seg[i]);
        }
        printf("\n");
        
        for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->colgrp_nranks - 1; i++)
        {
            printf("[%d] ", Graph<Weight>::A->other_colgrp_ranks[i]);
        }
        printf("\n");

        for(uint32_t i = 0; i < Graph<Weight>::A->local_col_segments.size(); i++)
        {
            printf("%d ", Graph<Weight>::A->local_col_segments[i]);
        }
        printf(" %d\n", diag_segment);
        
    }

    
    
    
    //printf("[%d]INIT\n", rank);    
    Graph<Weight>::degree();
    Graph<Weight>::pagerank();
    //Graph<Weight>::spmv();
    //printf("[%d]SPMV\n", rank);    
    Graph<Weight>::free();
    
    if(!rank)
    {
        printf("[x]SPMV\n");
    }
}

template<typename Weight>
void Matrix<Weight>::init_mat()
{    

    // Reserve the 2D vector of tiles. 
    Matrix<Weight>::tiles.resize(Matrix<Weight>::nrowgrps);
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
        Matrix<Weight>::tiles[i].resize(Matrix<Weight>::ncolgrps);
    
    
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
    {
        for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
        {
            Matrix<Weight>::tiles[i][j].rg = i;
            Matrix<Weight>::tiles[i][j].cg = j;
            if(Matrix<Weight>::partitioning->tiling == Tiling::_1D_ROW)
            {
                Matrix<Weight>::tiles[i][j].rank = i;
                Matrix<Weight>::tiles[i][j].ith = Matrix<Weight>::tiles[i][j].rg / Matrix<Weight>::partitioning->colgrp_nranks;
                Matrix<Weight>::tiles[i][j].jth = Matrix<Weight>::tiles[i][j].cg / Matrix<Weight>::partitioning->rowgrp_nranks;
                Matrix<Weight>::tiles[i][j].nth = Matrix<Weight>::tiles[i][j].ith * Matrix<Weight>::partitioning->rank_ncolgrps + Matrix<Weight>::tiles[i][j].jth;
            }
            else if(Matrix<Weight>::partitioning->tiling == Tiling::_2D_)
            {
                Matrix<Weight>::tiles[i][j].rank = (j % Matrix<Weight>::partitioning->rowgrp_nranks) * Matrix<Weight>::partitioning->colgrp_nranks
                                                 + (i % Matrix<Weight>::partitioning->colgrp_nranks);
                Matrix<Weight>::tiles[i][j].ith = Matrix<Weight>::tiles[i][j].rg / Matrix<Weight>::partitioning->colgrp_nranks;
                Matrix<Weight>::tiles[i][j].jth = Matrix<Weight>::tiles[i][j].cg / Matrix<Weight>::partitioning->rowgrp_nranks;
                Matrix<Weight>::tiles[i][j].nth = Matrix<Weight>::tiles[i][j].ith * Matrix<Weight>::partitioning->rank_ncolgrps + Matrix<Weight>::tiles[i][j].jth;
            }
        }
    }
    
    Matrix<Weight>::diag_ranks.resize(Matrix<Weight>::nrowgrps, -1);
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
    {
        for (uint32_t j = i; j < Matrix<Weight>::ncolgrps; j++)  
        {
            if(not (std::find(Matrix<Weight>::diag_ranks.begin(), Matrix<Weight>::diag_ranks.end(), Matrix<Weight>::tiles[j][i].rank) != Matrix<Weight>::diag_ranks.end()))
            {
                std::swap(Matrix<Weight>::tiles[j], Matrix<Weight>::tiles[i]);
                break;
            }
        }
        Matrix<Weight>::diag_ranks[i] = Matrix<Weight>::tiles[i][i].rank;
    }
    
    
    struct Triple<Weight> pair;
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
    {
        for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
        {
            Matrix<Weight>::tiles[i][j].rg = i;
            Matrix<Weight>::tiles[i][j].cg = j;
            if(Matrix<Weight>::tiles[i][j].rank == rank)
            {
                pair.row = i;
                pair.col = j;    
                Matrix<Weight>::local_tiles.push_back(Matrix<Weight>::local_tile_of_tile(pair));

                if (std::find(Matrix<Weight>::local_col_segments.begin(), Matrix<Weight>::local_col_segments.end(), pair.col) == local_col_segments.end())
                {
                    Matrix<Weight>::local_col_segments.push_back(pair.col);
                }
            }
        }
    }

    for(uint32_t t: Matrix<Weight>::local_tiles)
    {
        pair = Matrix<Weight>::tile_of_local_tile(t);
        if(pair.row == pair.col)
        {
            for(uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)
            {
                if((Matrix<Weight>::tiles[pair.row][j].rank != rank) 
                    and (std::find(other_rowgrp_ranks.begin(), other_rowgrp_ranks.end(), Matrix<Weight>::tiles[pair.row][j].rank) == other_rowgrp_ranks.end()))
                {
                    Matrix<Weight>::other_rowgrp_ranks.push_back(Matrix<Weight>::tiles[pair.row][j].rank);
                    Matrix<Weight>::rowgrp_ranks_accu_seg.push_back(Matrix<Weight>::tiles[pair.row][j].cg);
                }
            }
            
            for(uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
            {
                if((Matrix<Weight>::tiles[i][pair.col].rank != rank) 
                    and (std::find(other_colgrp_ranks.begin(), other_colgrp_ranks.end(), Matrix<Weight>::tiles[i][pair.col].rank) == other_colgrp_ranks.end()))
                {
                    Matrix<Weight>::other_colgrp_ranks.push_back(Matrix<Weight>::tiles[i][pair.col].rank);
                }
            }
        }
    }
    
    // Initialize triples 
    for(uint32_t t: Matrix<Weight>::local_tiles)
    {
        pair = Matrix<Weight>::tile_of_local_tile(t);
        auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
        tile.triples = new std::vector<struct Triple<Weight>>;
    }
        
    if(!rank)
    {    
        for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
        {
            for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
            {
                printf("%02d ", Matrix<Weight>::tiles[i][j].rank);
            }
            printf("\n");
        }
    }
}

template<typename Weight>
void Matrix<Weight>::init_csr()
{
    // Create the csr format
    // Allocate csr data structure
    // Sort triples and populate csr
    struct Triple<Weight> pair;
    for(uint32_t t: Matrix<Weight>::local_tiles)
    {

        pair = Matrix<Weight>::tile_of_local_tile(t);
        auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
        tile.csr = new struct CSR<Weight>;
        tile.csr->allocate(tile.triples->size(), Matrix<Weight>::tile_height + 1);
        
        Functor<Weight> f;
        std::sort(tile.triples->begin(), tile.triples->end(), f);

        uint32_t i = 0;
        uint32_t j = 1;
        tile.csr->IA[0] = 0;
        for (auto& triple : *(tile.triples))
        {
            pair = rebase(triple);
            while((j - 1) != pair.row)
            {
                j++;
                tile.csr->IA[j] = tile.csr->IA[j - 1];
            }            
            tile.csr->A[i] = triple.get_weight(); // In case weights are implemented
            tile.csr->IA[j]++;
            tile.csr->JA[i] = pair.col;    
            i++;
        }
        
        // Not necessary
        while(j < (Matrix<Weight>::tile_height))
        {
              j++;
           tile.csr->IA[j] = tile.csr->IA[j - 1];
        }
    }
   
    Matrix<Weight>::del_triples();
}

template<typename Weight>
void Graph<Weight>::degree()
{
    Triple<Weight> pair = {0, 0};
    Triple<Weight> pair1 = {0, 0};
    uint32_t xi, yi, yj, si;
    uint32_t nitems;
    
    // Initializing X and Y
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto &tile = Graph<Weight>::A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = Graph<Weight>::X->segments[xi];
        auto *x_data = (fp_t *) x_seg.data;
        
        yi = Graph<Weight>::Y->diag_segment;
        auto &y_seg = Graph<Weight>::Y->segments[yi];
        auto *y_data = (fp_t *) y_seg.data;
        
        nitems = x_seg.n;
        
        for(uint32_t i = 0; i < nitems; i++)
        {
            x_data[i] = 0;
            y_data[i] = 0;
        }
    }
    
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto &tile = Graph<Weight>::A->tiles[pair.row][pair.col];

        xi = pair.col;
        auto &x_seg = Graph<Weight>::X->segments[xi];
        auto *x_data = (fp_t *) x_seg.data;
        
        yi = Graph<Weight>::Y->diag_segment;
        auto &y_seg = Graph<Weight>::Y->segments[yi];
        auto *y_data = (fp_t *) y_seg.data;

        // Local computation, no need to put it on X
        uint32_t k = 0;
        for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
        {
            uint32_t nnz_per_row = tile.csr->IA[i + 1] - tile.csr->IA[i];
            for(uint32_t j = 0; j < nnz_per_row; j++)
            {
                y_data[i] += (fp_t) tile.csr->A[k];
                //if(!rank)
                //    std::cout << *((fp_t *) y_seg.data + i) << " " << ((fp_t *) y_seg.data + i) << " " << y_data << " " << *y_data << std::endl;
                k++;
            }
        }

        
        bool communication = ((tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks == 0);    
        if(communication)
        {
            uint32_t leader = Graph<Weight>::Y->segments[tile.rg].rank;
            if(rank == leader)
            {
                if(Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW)
                {   
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        uint32_t si = Graph<Weight>::S->diag_segment;
                        auto &s_seg = Graph<Weight>::S->segments[si];
                        auto *s_data = (fp_t *) s_seg.data;
                        s_data[i] = y_data[i];
                    }
                }
                else if(Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
                {
                    MPI_Status status;
                    for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                    {
                        uint32_t other_rank = Graph<Weight>::A->other_rowgrp_ranks[j];
                        yj = Graph<Weight>::A->rowgrp_ranks_accu_seg[j];
                        auto &yj_seg = Graph<Weight>::Y->segments[yj];
                        MPI_Recv(yj_seg.data, yj_seg.nbytes, MPI_BYTE, other_rank, pair.row, MPI_COMM_WORLD, &status);
                    }
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                        {
                            uint32_t other_rank = Graph<Weight>::A->other_rowgrp_ranks[j];
                            yj = Graph<Weight>::A->rowgrp_ranks_accu_seg[j];
                            auto &yj_seg = Graph<Weight>::Y->segments[yj];
                            auto *yj_data = (fp_t *) yj_seg.data;
                            y_data[i] += yj_data[i];
                        }
                        uint32_t si = Graph<Weight>::S->diag_segment;
                        auto &s_seg = Graph<Weight>::S->segments[si];
                        auto *s_data = (fp_t *) s_seg.data;
                        s_data[i] = y_data[i];
                    }
                }
                /*
                si = Graph<Weight>::S->diag_segment;
                auto& s_seg = Graph<Weight>::S->segments[si];
                auto *s_data = (fp_t *) s_seg.data;
                nitems = s_seg.n;
                
                for(uint32_t i = 0; i < nitems; i++)
                {
                    pair.row = i;
                    pair.col = 0;
                    pair1 = Graph<Weight>::A->base(pair, s_seg.rg, s_seg.cg);
                    printf("R(%d),S[%d]=%f\n",  rank, pair1.row, s_data[i]);
                }
                */
                
            }
            else
            {
                MPI_Send(y_seg.data, y_seg.nbytes, MPI_BYTE, leader, pair.row, MPI_COMM_WORLD);
            }
            memset(y_seg.data, 0, y_seg.nbytes);
        }
    }
}

template<typename Weight>
void Graph<Weight>::pagerank()
{
    Triple<Weight> pair;
    Triple<Weight> pair1;   
    Triple<Weight> pair2;   
    uint32_t xi, si, vi, yi, yj;
    uint32_t nitems;
    double alpha = 0.3;
    uint32_t niters = 5;
    static uint32_t iter = 0;
    

    // Init
    si = Graph<Weight>::S->diag_segment;
    //si = Graph<Weight>::S->local_segments[0];
    auto &s_seg = Graph<Weight>::S->segments[si];
    auto *s_data = (fp_t *) s_seg.data;
    // Populated by degree
    
    //vi = Graph<Weight>::V->local_segments[0];
    vi = Graph<Weight>::V->diag_segment;
    auto &v_seg = Graph<Weight>::V->segments[vi];
    auto *v_data = (fp_t *) v_seg.data;

    nitems = v_seg.n;
    //fp_t *v_data = (fp_t *) v_seg.data;
    for(uint32_t i = 0; i < nitems; i++)
    {
        //v_seg.data[i] = alpha;
        //v_data += i;
        v_data[i] = alpha;
        //*((fp_t *) v_seg.data + i) = alpha;
    }

    while(iter < niters)
    {
    iter++;
    // Scatter 
    xi = Graph<Weight>::X->diag_segment;
    //auto& x_seg = Graph<Weight>::X->segments[xi];
    auto &x_seg = Graph<Weight>::X->segments[xi];
    auto *x_data = (fp_t *) x_seg.data;
    nitems = x_seg.n;
    for(uint32_t i = 0; i < nitems; i++)
    {
        //if(!rank)
          //  printf("V=%d", *((uint32_t *) x_seg.data1 + i));
      
      x_data[i] = v_data[i] / s_data[i];
       if (std::isinf(x_data[i]))
           x_data[i] = 0;
      //*((fp_t *) x_seg.data + i) = *((fp_t *) v_seg.data + i) / *((fp_t *) s_seg.data + i); // v / s
       //if (std::isinf(*((fp_t *) x_seg.data + i)))
       //    *((fp_t *) x_seg.data + i) = 0;
           //printf("INF\n");
           
      //if(!rank)
        //  printf("X=%f\n", *((fp_t *) x_seg.data + i));
      //*((uint32_t *) x_seg.data1 + i) = v_seg.data[i] + s_seg.data[i];
        //x_seg.data1[i] = v_seg.data[i] + s_seg.data[i];
    }

    

    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;
        
    if((Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
        or (Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW))
    {
        uint32_t leader = x_seg.rank;
        for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->colgrp_nranks - 1; j++)
        {
            uint32_t other_rank = Graph<Weight>::A->other_colgrp_ranks[j];
            //if(other_rank != leader)
            //{
                MPI_Isend(x_seg.data, x_seg.nbytes, MPI_BYTE, other_rank, x_seg.cg, MPI_COMM_WORLD, &request);
                //MPI_Isend(x_seg.data, x_seg.n, MPI_INTEGER, other_rank, x_seg.cg, MPI_COMM_WORLD, &request);
                out_requests.push_back(request);
            //}
        }
    }
    else
    {
        std::cout << "Invalid tiling\n";
        exit(0);
    }
    //gather
    MPI_Status status;
    
   
    if((Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
        or (Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW))
    {    
        for(uint32_t s: Graph<Weight>::A->local_col_segments)
        {
            if(s != xi)
            {
                auto& xj_seg = Graph<Weight>::X->segments[s];
                //auto& xj_seg = Graph<Weight>::X->segments[s];
                MPI_Irecv(xj_seg.data, xj_seg.nbytes, MPI_BYTE, xj_seg.rank, xj_seg.cg, MPI_COMM_WORLD, &request);
                //MPI_Irecv(xj_seg.data, xj_seg.n, MPI_INTEGER, xj_seg.rank, xj_seg.cg, MPI_COMM_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    }
    else
    {
        std::cout << "Invalid tiling\n";
        exit(0);
    }
    
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    // Combine
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto &tile = Graph<Weight>::A->tiles[pair.row][pair.col];
 
        xi = pair.col;
        //auto& x_seg = Graph<Weight>::X->segments[xi];
        auto &x_seg = Graph<Weight>::X->segments[xi];
        auto *x_data = (fp_t *) x_seg.data;
        
        yi = Graph<Weight>::Y->diag_segment;
        auto &y_seg = Graph<Weight>::Y->segments[yi];
        auto *y_data = (fp_t *) y_seg.data;
        //memset(yi.data, 0, sizeof(yi.data);
        

        
        uint32_t k = 0;
        for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
        {
            //if(!rank)
            //    printf("X=%f, Y=%f\n", *((fp_t *) x_seg.data + i), *((fp_t *) y_seg.data + i));
            uint32_t nnz_per_row = tile.csr->IA[i + 1] - tile.csr->IA[i];
            for(uint32_t j = 0; j < nnz_per_row; j++)
            {
                //if((rank == 0) or (rank == 2))
                  //  printf("%d Y=%f, OFF=%d D=%f, ", rank, *((fp_t *) y_seg.data + i), tile.csr->JA[k], (*((fp_t *) x_seg.data + tile.csr->JA[k])));
                y_data[i] += tile.csr->A[k] * x_data[tile.csr->JA[k]];
                //*((fp_t *) y_seg.data + i) += tile.csr->A[k] * (*((fp_t *) x_seg.data + tile.csr->JA[k]));
                //printf("FY= %f\n", *((fp_t *) y_seg.data + i));
                //y_seg.data[i] += tile.csr->A[k] * (*((uint32_t *) x_seg.data1 + tile.csr->JA[k]));
                //y_seg.data[i] += tile.csr->A[k] * x_seg.data[tile.csr->JA[k]];
                k++;
            }
        }
        
        bool communication = ((tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks == 0);    
        if(communication)
        {
            uint32_t leader = Graph<Weight>::Y->segments[tile.rg].rank;
            if(rank == leader)
            {
                if(Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW)
                {
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        uint32_t vi = Graph<Weight>::V->diag_segment;
                        auto &v_seg = Graph<Weight>::V->segments[si];
                        auto *v_data = (fp_t *) v_seg.data;
                        v_data[i] = alpha + ((1- alpha) * y_data[i]);
                        //*((fp_t *) v_seg.data + i) = *((fp_t *) y_seg.data + i);
                        //*((fp_t *) v_seg.data + i) = alpha + (1- alpha) * *((fp_t *) y_seg.data + i);
                        //v_seg.data[i] = y_seg.data[i];
                    }
                }
                else if(Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
                {
                    MPI_Status status;
                    for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                    {
                        uint32_t other_rank = Graph<Weight>::A->other_rowgrp_ranks[j];
                        //if(other_rank != leader)
                        //{
                            yj = Graph<Weight>::A->rowgrp_ranks_accu_seg[j];
                            auto &yj_seg = Graph<Weight>::Y->segments[yj];
                            MPI_Recv(yj_seg.data, yj_seg.nbytes, MPI_BYTE, other_rank, pair.row, MPI_COMM_WORLD, &status);
                        //}
                    }
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                        {
                            uint32_t other_rank = Graph<Weight>::A->other_rowgrp_ranks[j];
                            //if(other_rank != leader)
                            //{
                                yj = Graph<Weight>::A->rowgrp_ranks_accu_seg[j];
                                auto &yj_seg  = Graph<Weight>::Y->segments[yj];
                                auto *yj_data = (fp_t *) yj_seg.data;
                                y_data[i] += yj_data[i];
                                //*((fp_t *) y_seg.data + i) += *((fp_t *) yj_seg.data + i);
                                //y_seg.data[i] += yj_seg.data[i];
                            //}
                        }
                            uint32_t vi = Graph<Weight>::V->diag_segment;
                            auto& v_seg = Graph<Weight>::V->segments[vi];
                            auto *v_data = (fp_t *) v_seg.data;
                            v_data[i] = alpha + ((1 - alpha) * y_data[i]);
                            //*((fp_t *) v_seg.data + i) = alpha + (1- alpha) * (*((fp_t *) y_seg.data + i));
                            //v_seg.data[i] = y_seg.data[i];
                    }
                }
                
                if(iter == niters)
                {
                vi = Graph<Weight>::V->diag_segment;
                auto& v_seg = Graph<Weight>::V->segments[vi];
                auto *v_data = (fp_t *) v_seg.data;
                nitems = v_seg.n;
                for(uint32_t i = 0; i < nitems; i++)
                {
                    pair.row = i;
                    pair.col = 0;
                    pair1 = Graph<Weight>::A->base(pair, v_seg.rg, v_seg.cg);
                    printf("R(%d),S[%d]=%f\n",  rank, pair1.row, v_data[i]);
                }
                }
                
                
            }
            else
            {
                MPI_Send(y_seg.data, y_seg.nbytes, MPI_BYTE, leader, pair.row, MPI_COMM_WORLD);
            }
            memset(y_seg.data, 0, y_seg.nbytes);
        }
        
    }
    }
    
    // Scatterp
}



using ew_t = Empty;

int main(int argc, char** argv) {
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;
      MPI_Init_thread(nullptr, nullptr, required, &provided);
      assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));


    nranks = -1;
      MPI_Comm_size(MPI_COMM_WORLD, &nranks);
      assert(nranks >= 0);

      rank   = -1;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);
    
    MPI_Get_processor_name(processor_name, &processor_name_len);

    cpu_id = sched_getcpu();
    //std::cout << "Rank " << rank << " of " << nranks << 
    //               ", hostname " << processor_name << ", CPU " << cpu_id << std::endl;

    is_master = (rank == 0);
    
    
    
    // Print usage
    // Should be moved later
    if(argc != 4)  {
        if(is_master) {
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<num_iterations>]\""
                      << std::endl;
        }    
        MPI_Barrier(MPI_COMM_WORLD);
         MPI_Finalize();
        std::exit(1);
    }
    
    std::string file_path = argv[1]; 
    uint32_t num_vertices = std::atoi(argv[2]);
    uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
    bool directed = true;
    bool transpose = false;
    if(!rank)
    {
        printf("[x]MAIN\n");
    }
    
    Graph<ew_t> G;
    G.load_binary(file_path, num_vertices, num_vertices, _2D_, directed, transpose);
    //G.load_text(file_path, num_vertices, num_vertices, Tiling::_2D_, directed, transpose);
    //G.A->spmv();
    
    

    
    //G.free();
    
    
    

    
    MPI_Finalize();

    return(0);

}


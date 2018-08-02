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
    CSR<Weight>::A = (uint32_t*) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
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
        std::vector<uint32_t> local_row_segments;
        
        std::vector<uint32_t> diag_ranks;
        std::vector<uint32_t> other_ranks;
        
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
    
    uint32_t* data;    
    uint32_t n;
    uint32_t nrows, ncols;
    uint32_t rg, cg;
    uint32_t rank;
    void allocate();
    void free();
};

template<typename Weight>
void Segment<Weight>::allocate()
{
    Segment<Weight>::data = (uint32_t*) mmap(nullptr, (this->n) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memset(Segment<Weight>::data, 0, this->n * sizeof(uint32_t));
}

template<typename Weight>
void Segment<Weight>::free()
{
    munmap(Segment<Weight>::data, (this->n) * sizeof(uint32_t));
}

template<typename Weight>
class Vector
{
    template<typename Weight_>
    friend class Graph;
    
    public:
        Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles);
        ~Vector();
    
    private:
        const uint32_t nrows, ncols;
        const uint32_t nrowgrps, ncolgrps;
        const uint32_t tile_height, tile_width;    // == segment_height
    
        std::vector<struct Segment<Weight>> segments;
        std::vector<uint32_t> local_segments;
        
        void init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments);
};

template<typename Weight>
Vector<Weight>::Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles) 
    : nrows(nrows), ncols(ncols), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles/nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1) {};
    

template<typename Weight>
Vector<Weight>::~Vector() {};

template<typename Weight>
void Vector<Weight>::init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments)
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
    }
    
    Vector<Weight>::local_segments = local_segments;
    
    for(uint32_t s: Vector<Weight>::local_segments)
    {
        Vector<Weight>::segments[s].allocate();
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
        Vector<Weight>* Z;
        Vector<Weight>* V;
        Vector<Weight>* S;
        
        void spmv();
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
    delete Graph<Weight>::Z;
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
    
    for(uint32_t s: Graph<Weight>::Z->local_segments)
    {
        auto& segment = Graph<Weight>::Z->segments[s];
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
    
    
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::Z = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    
    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_row_segments);
    Graph<Weight>::Z->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->other_ranks);
    std::vector<uint32_t> my_row;
    uint32_t row = distance(Graph<Weight>::A->diag_ranks.begin(), find(Graph<Weight>::A->diag_ranks.begin(), Graph<Weight>::A->diag_ranks.end(), rank));
    my_row.push_back(row);
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks, my_row);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks, my_row);
    
    
    
    //Graph<Weight>::spmv();
    Graph<Weight>::degree();
    Graph<Weight>::pagerank();
    Graph<Weight>::free();
    
    if(!rank)
    {
        printf("[x]SPMV\n");
    }
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
    //printf("[%d]CSR\n", rank);
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::Z = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);

    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_row_segments);
    Graph<Weight>::Z->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->other_ranks);
    std::vector<uint32_t> my_row;
    uint32_t row = distance(Graph<Weight>::A->diag_ranks.begin(), find(Graph<Weight>::A->diag_ranks.begin(), Graph<Weight>::A->diag_ranks.end(), rank));
    my_row.push_back(row);
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks, my_row);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    
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

                if (std::find(Matrix<Weight>::local_row_segments.begin(), Matrix<Weight>::local_row_segments.end(), pair.row) == local_row_segments.end())
                {
                    Matrix<Weight>::local_row_segments.push_back(pair.row);
                }    
                
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
                if((pair.row != j) and (Matrix<Weight>::tiles[pair.row][j].rank != rank))
                {
                    if (std::find(other_ranks.begin(), other_ranks.end(), Matrix<Weight>::tiles[pair.row][j].rank) == other_ranks.end())
                    {
                        Matrix<Weight>::other_ranks.push_back(Matrix<Weight>::tiles[pair.row][j].rank);
                    }
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
void Graph<Weight>::spmv()
{
    Triple<Weight> pair;
    Triple<Weight> pair1;
    uint32_t x_c = 0, y_r = 0;
    // Data initialization
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
        uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
        x_c = pair.col;
        y_r = pair.row;
        
        auto& x_seg = Graph<Weight>::X->segments[x_c];
        auto& y_seg = Graph<Weight>::Y->segments[y_r];
        
        uint32_t nitems = x_seg.n;
        for(uint32_t i = 0; i < nitems; i++)
        {
            x_seg.data[i] = 1;
            y_seg.data[i] = 0;
        }
        
    }    
    
    
    // Data processing 
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
        uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
        x_c = pair.col;
        y_r = pair.row;
         
        auto& x_seg = Graph<Weight>::X->segments[x_c];
        auto& y_seg = Graph<Weight>::Y->segments[y_r];

        uint32_t k = 0;
        for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
        {
            uint32_t nnz_per_row = tile.csr->IA[i + 1] - tile.csr->IA[i];
            for(uint32_t j = 0; j < nnz_per_row; j++)
            {
                y_seg.data[i] += tile.csr->A[k] * x_seg.data[i];
                k++;
            }
        }
        
        bool communication = ((tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks == 0);    
        if(communication)
        {
            //printf("COMM %d %d\n", rank, tile.nth);
            if(y_seg.rank == rank)
            {
                if(Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW)
                {
                    
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        uint32_t v = 0;
                        v += y_seg.data[i];
                        //printf(">>1D %lu\n", Graph<Weight>::V->local_segments.size());
                        uint32_t v_r = Graph<Weight>::V->local_segments[0];
                        //printf(">>1D %d\n", v_r);
                        auto& v_seg = Graph<Weight>::V->segments[v_r];
                        v_seg.data[i] = v;
                    }
                    
                }
                else if(Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
                {
                    MPI_Status status;
                    for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; i++)
                    {
                        uint32_t z_r = Graph<Weight>::A->other_ranks[i];
                        auto& z_seg = Graph<Weight>::Z->segments[z_r];
                        MPI_Recv(z_seg.data, z_seg.n, MPI_INTEGER, z_r, pair.row, MPI_COMM_WORLD, &status);
                    }

                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        uint32_t v = 0;
                        v += y_seg.data[i];
                    
                        for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                        {
                            uint32_t z_r = Graph<Weight>::A->other_ranks[j];
                            auto& z_seg = Graph<Weight>::Z->segments[z_r];
                            v += z_seg.data[i];
                        }
                    
                        uint32_t v_r = Graph<Weight>::V->local_segments[0];
                        auto& v_seg = Graph<Weight>::V->segments[v_r];
                        v_seg.data[i] = v;
                    }
                }
                
                uint32_t v_r = Graph<Weight>::V->local_segments[0];
                auto& v_seg = Graph<Weight>::V->segments[v_r];
                for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                {
                    //printf("R(%d),v[%d]=%d, rg=%d cg=%d, %d %d \n",  rank, i + tile.rg * Graph<Weight>::A->tile_width, v_seg.data[i], tile.rg, tile.cg, Graph<Weight>::A->tile_width, tile.rg * Graph<Weight>::A->tile_width);
                    pair.row = i;
                    pair.col = 0;
                    pair1 = Graph<Weight>::A->base(pair, tile.rg, tile.cg);
                    printf("R(%d),v[%d]=%d\n",  rank, pair1.row, v_seg.data[i]);
                    //printf("R(%d),v[%d]=%d, %d %d %d %d\n",  rank, i + tile.rg * Graph<Weight>::A->tile_width, v_seg.data[i], pair1.row, pair1.col, tile.rg, tile.cg);
                }
                
                
                /*
                if(rank == 11)
                {
                for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                {
                    uint32_t v_r = Graph<Weight>::Z->local_segments[0];
                    auto& v_seg = Graph<Weight>::Z->segments[v_r];
                    
                    printf("%d ", v_seg.data[i]);
                    
                }
                printf("\n");
                }
                */
                /*
                for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                {
                    
                    uint32_t z_r = Graph<Weight>::A->other_ranks[j];
                    auto& z_seg = Graph<Weight>::Z->segments[z_r];
                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        v += z_seg.data[i];
                    }
                }
                */
            ////    printf("<%d %d>\n", rank, Graph<Weight>::Z->local_segments[0]);
                
            }
            else
            {
                // MPI_Send
                MPI_Send(y_seg.data, y_seg.n, MPI_INTEGER, y_seg.rank, pair.row, MPI_COMM_WORLD);
                
                //printf("ROW=%d: send(%d) to (%d) of size [%d %d] %d\n",  pair.row, rank, y_seg.rank, Graph<Weight>::A->tile_width, y_seg.n, y_seg.data[65524]);
            }
            //printf("[%d %d] %d %d %d %d %d %d\n",x_seg.rank, y_seg.rank, tile.ith, tile.jth,  tile.nth, Graph<Weight>::A->partitioning->colgrp_nranks, (tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks, communication);    

        }
    }
}

template<typename Weight>
void Graph<Weight>::degree()
{
    Triple<Weight> pair = {0, 0};
    Triple<Weight> pair1 = {0, 0};
    uint32_t x_c = 0, y_r = 0, v_r = 0, s_r = 0;
    uint32_t nitems;
    // Initializing X and Y
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
        uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
        x_c = pair.col;
        y_r = pair.row;
        auto& x_seg = Graph<Weight>::X->segments[x_c];
        auto& y_seg = Graph<Weight>::Y->segments[y_r];
        auto& s_seg = Graph<Weight>::X->segments[x_c];
        
        nitems = x_seg.n;
        for(uint32_t i = 0; i < nitems; i++)
        {
            x_seg.data[i] = 0;
            y_seg.data[i] = 0;
            s_seg.data[i] = 0;
        }
        
        //uint32_t v_r = Graph<Weight>::V->local_segments[0];
        //auto& v_seg = Graph<Weight>::V->segments[v_r];
    }
    
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
        pair = Graph<Weight>::A->tile_of_local_tile(t);
        auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
        uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
        x_c = pair.col;
        y_r = pair.row;
        auto& x_seg = Graph<Weight>::X->segments[x_c];
        auto& y_seg = Graph<Weight>::Y->segments[y_r];

        // Local computation, no need to put it on X
        uint32_t k = 0;
        for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
        {
            uint32_t nnz_per_row = tile.csr->IA[i + 1] - tile.csr->IA[i];
            for(uint32_t j = 0; j < nnz_per_row; j++)
            {
                y_seg.data[i] += tile.csr->A[k];
                k++;
            }
        }
        
        bool communication = ((tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks == 0);    
        if(communication)
        {
            if(y_seg.rank == rank)
            {
                if(Graph<Weight>::A->partitioning->tiling == Tiling::_1D_ROW)
                {
                    printf("NOT added yet\n");
                    exit(0);
                }
                else if(Graph<Weight>::A->partitioning->tiling == Tiling::_2D_)
                {
                    MPI_Status status;
                    for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; i++)
                    {
                        uint32_t z_r = Graph<Weight>::A->other_ranks[i];
                        auto& z_seg = Graph<Weight>::Z->segments[z_r];
                        MPI_Recv(z_seg.data, z_seg.n, MPI_INTEGER, z_r, pair.row, MPI_COMM_WORLD, &status);
                    }

                    for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
                    {
                        uint32_t s = 0;
                        s += y_seg.data[i];
                    
                        for(uint32_t j = 0; j < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; j++)
                        {
                            uint32_t z_r = Graph<Weight>::A->other_ranks[j];
                            auto& z_seg = Graph<Weight>::Z->segments[z_r];
                            s += z_seg.data[i];
                        }
                        uint32_t s_r = Graph<Weight>::V->local_segments[0];
                        auto& s_seg = Graph<Weight>::S->segments[s_r];
                        s_seg.data[i] = s;
                    }
                }
                
                /*
                uint32_t v_r = Graph<Weight>::V->local_segments[0];
                auto& v_seg = Graph<Weight>::V->segments[v_r];
                nitems = v_seg.n;
                for(uint32_t i = 0; i < nitems; i++)
                {
                    pair.row = i;
                    pair.col = 0;
                    pair1 = Graph<Weight>::A->base(pair, tile.rg, tile.cg);
                    printf("R(%d),v[%d]=%d\n",  rank, pair1.row, v_seg.data[i]);
                }
                */
            }
            else
            {
                MPI_Send(y_seg.data, y_seg.n, MPI_INTEGER, y_seg.rank, pair.row, MPI_COMM_WORLD);
            }
            

        }
        
        
    }
    
    
    
    
    
}

template<typename Weight>
void Graph<Weight>::pagerank()
{
    Triple<Weight> pair;
    Triple<Weight> pair1;   
    uint32_t x_c = 0, y_r = 0, v_r = 0, s_r;
    uint32_t nitems;
    /*
    v_r = Graph<Weight>::V->local_segments[0];
    auto& v_seg = Graph<Weight>::V->segments[v_r];
    nitems = v_seg.n;
    for(uint32_t i = 0; i < nitems; i++)
    {
        pair.row = i;
        pair.col = 0;
        pair1 = Graph<Weight>::A->base(pair, v_seg.rg, v_seg.cg);
        printf("R(%d),v[%d]=%d\n",  rank, pair1.row, v_seg.data[i]);
    }
    */
    
    uint32_t alpha = 3;
    uint32_t niters = 2;
    static uint32_t iter = 0;
    

    // Init
    s_r = Graph<Weight>::S->local_segments[0];
    auto& s_seg = Graph<Weight>::S->segments[s_r];
    
    
    v_r = Graph<Weight>::V->local_segments[0];
    auto& v_seg = Graph<Weight>::V->segments[v_r];
    nitems = s_seg.n;
    for(uint32_t i = 0; i < nitems; i++)
    {
        v_seg.data[i] = alpha;
    }
    iter++;
    // Scatter
    // send ==>  v / s ==> send to others
    while(iter < niters)
    {
        for(uint32_t t: Graph<Weight>::A->local_tiles)
        {
            pair = Graph<Weight>::A->tile_of_local_tile(t);
            auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
            uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
            x_c = pair.col;
            y_r = pair.row;
            auto& x_seg = Graph<Weight>::X->segments[x_c];
            auto& y_seg = Graph<Weight>::Y->segments[y_r];
            nitems = x_seg.n;
            for(uint32_t i = 0; i < nitems; i++)
            {
                x_seg.data[i] = v_seg.data[i] + s_seg.data[i];
                //y_seg.data[i] = 0;
            }
        }
        iter++;    
    }
    
   

    //v_r = Graph<Weight>::V->local_segments[0];
    //auto& v_seg = Graph<Weight>::V->segments[v_r];
    //nitems = v_seg.n;
    for(uint32_t i = 0; i < nitems; i++)
    {
        pair.row = i;
        pair.col = 0;
        pair1 = Graph<Weight>::A->base(pair, v_seg.rg, v_seg.cg);
        printf("R(%d),v[%d]=%d\n",  rank, pair1.row, v_seg.data[i]);
    }   
    
    
    
    
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
    std::cout << "Rank " << rank << " of " << nranks << 
                   ", hostname " << processor_name << ", CPU " << cpu_id << std::endl;

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
    bool transpose = true;
    if(!rank)
    {
        printf("[x]MAIN\n");
    }
    
    Graph<ew_t> G;
    //G.load_binary(file_path, num_vertices, num_vertices, _2D_, directed, transpose);
    G.load_text(file_path, num_vertices, num_vertices, Tiling::_2D_, directed, transpose);
    //G.A->spmv();
    
    

    
    //G.free();
    
    
    

    
    MPI_Finalize();

    return(0);

}


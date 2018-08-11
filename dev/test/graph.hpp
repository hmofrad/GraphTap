/*
 * graph.hpp: Graph implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include "ds.hpp"
#include "matrix.hpp"



template<typename Weight = Empty, typename Integer_Type = uint32_t, typename Fractional_Type = float>
class Graph
{
    public:    
        Graph();
        ~Graph();
        
        void load_binary(std::string filepath, Integer_Type nrows, Integer_Type ncols,
                       bool directed = true, bool transpose = false, Tiling_type tiling_type = _2D_);
        void load_text(std::string filepath, Integer_Type nrows, Integer_Type ncols,
                       bool directed = true, bool transpose = false, Tiling_type tiling_type = _2D_);
        void degree();
        void pagerank(uint32_t niters, bool clear_state = false);
        void initialize(Graph<Weight> &G);
        void free(bool clear_state = true);

    private:
        std::string filepath;
        Integer_Type nrows;
        Integer_Type ncols;
        uint64_t nedges;
        bool directed;
        bool transpose;
        Matrix<Weight, Integer_Type, Fractional_Type>* A;
        //Vector<Weight>* X;
        //Vector<Weight>* Y;
        //Vector<Weight>* V;
        //Vector<Weight>* S;
        
        void init_graph(std::string filepath, Integer_Type nrows, Integer_Type ncols, 
               bool directed, bool transpose, Tiling_type tiling_type);
        void read_text();
        void read_binary();
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s, bool clear_state = true);
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void apply();
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::Graph() : A(nullptr) {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::~Graph()
{
    delete A;
    /*
    delete Graph<Weight>::A->partitioning;
    
    
    delete Graph<Weight>::X;
    delete Graph<Weight>::Y;
    delete Graph<Weight>::V;
    delete Graph<Weight>::S;
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_text(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,  Tiling_type tiling_type)
{
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_, tiling_type);
    // Read graph

    read_text();
    A->init_csr();
    
    uint32_t diag_segment = std::distance(A->leader_ranks.begin(), 
            std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
    
    if(!Env::rank)
    {
        
        //printf(">>>>>\n");
        //Integer_Type N = 10;
        //struct basic_storage<Weight, Integer_Type> st(N);
        
        //printf("%lu %lu\n", st.nbytes, st.n);
        
        //st.free();
        //st.free<Weight, Integer_Type>();
    }
    
    
    
    



    
    
    
    

    
    
    /*
    Graph<Weight>::A->init_csr();
    Graph<Weight>::A->init_bv();
    
    
    
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    
    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->rowgrp_ranks_accu_seg);
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks);
    */
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::init_graph(std::string filepath_, 
           Integer_Type nrows_, Integer_Type ncols_, bool directed_, 
           bool transpose_, Tiling_type tiling_type)
{
    
    filepath  = filepath_;
    nrows = nrows_ + 1; // In favor of vertex id 0
    ncols = ncols_ + 1; // In favor of vertex id 0
    nedges    = 0;
    directed  = directed_;
    transpose = transpose_;
    // Initialize matrix
    A = new Matrix<Weight, Integer_Type, Fractional_Type>(nrows, ncols, 
                                Env::nranks * Env::nranks, tiling_type);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::read_text()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str());
    if(!fin.is_open())
    {
        std::cout << "Unable to open input file" << std::endl;
        Env::exit(1);
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

    struct Triple<Weight, Integer_Type> triple;
    struct Triple<Weight, Integer_Type> pair;
    std::istringstream iss;
//    bool empty_weight = (std::is_same<Weight, Empty>::value) ? true : false;
    while (std::getline(fin, line) && !line.empty())
    {
        iss.clear();
        iss.str(line);
        
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 3)
        {
            std::cout << "read() failure" << std::endl;
            Env::exit(1);
        }

        if(transpose)
        {
            iss >> triple.col >> triple.row >> triple.weight;
        }
        else
        {
            iss >> triple.row >> triple.col >> triple.weight;
        }
        
        nedges++;
    
        pair = A->tile_of_triple(triple);
        assert((triple.row <= nrows) and (triple.col <= ncols));
        // A better but expensive way is to determine the file size beforehand
        if(A->tiles[pair.row][pair.col].rank == Env::rank)    
        {
            A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
        offset = fin.tellg();
    }
    
    fin.close();
    assert(offset == filesize);   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::read_binary()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str(), std::ios_base::binary);
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
        
        if(fin.gcount() != sizeof(Triple<Weight>))
        {
            std::cout << "read() failure" << std::endl;
            Env::exit(1);
        }
        
        if(Graph<Weight>::transpose)
        {
            std::swap(triple.row, triple.col);
        }
        Graph<Weight>::nedges++;    
        
        pair = Graph<Weight>::A->tile_of_triple(triple);
        assert((triple.row <= nrows) and (triple.col <= ncols));
        // A better but expensive way is to determine the file size beforehand
        if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == Env::rank)    
        {
            Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
        offset += sizeof(Triple<Weight>);
    }
    fin.close();
    assert(offset == filesize);
}


/* Class template specialization for Weight
 * We think of two ways for processing graphs with empty weights:
 * 1) Representing the adjacency matrix with a matrix of type char,
 * 2) Infer the adjacency matrix from the csr or csc.
 * We picked the 2nd approach and the following code path completely
 * remove the weight and adjacency matrix from the implementation.
 * Thus, We have saved Graph_size * sizeof(Weight) in space because
 * representing the adjacency matrix even with char will cost us
 * Graph_size * sizeof(char). At scale, this means if e.g. 
 * a graph with 1TB of edges, we're saving 1TB of memory.
 */
template<typename Integer_Type, typename Fractional_Type>
class Graph<Empty, Integer_Type, Fractional_Type>
{
    public:    
        Graph();
        ~Graph();
        
        void load_binary(std::string filepath, Integer_Type nrows, Integer_Type ncols,
                       bool directed = true, bool transpose = false, Tiling_type tiling_type = _2D_);
        void load_text(std::string filepath, Integer_Type nrows, Integer_Type ncols,
                       bool directed = true, bool transpose = false, Tiling_type tiling_type = _2D_);
        void degree();
        void pagerank(uint32_t niters, bool clear_state = false);
        void initialize(Graph<Empty> &G);
        void free(bool clear_state = true);

    private:
        std::string filepath;
        Integer_Type nrows;
        Integer_Type ncols;
        uint64_t nedges;
        bool directed;
        bool transpose;
        Matrix<Empty, Integer_Type, Fractional_Type>* A;
        //Vector<Weight>* X;
        //Vector<Weight>* Y;
        //Vector<Weight>* V;
        //Vector<Weight>* S;
        
        void init_graph(std::string filepath, Integer_Type nrows, Integer_Type ncols, 
               bool directed, bool transpose, Tiling_type tiling_type);
        void read_text();
        void read_binary();
        void parse_triple(std::istringstream &iss, struct Triple<Empty, Integer_Type> &triple, bool transpose);
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        
        void init(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s, bool clear_state = true);
        void scatter(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void gather();
        void combine(Fractional_Type (*f)(Fractional_Type x, Fractional_Type y, Fractional_Type v, Fractional_Type s));
        void apply();
};


template<typename Integer_Type, typename Fractional_Type>
Graph<Empty, Integer_Type, Fractional_Type>::Graph() : A(nullptr) {};

template<typename Integer_Type, typename Fractional_Type>
Graph<Empty, Integer_Type, Fractional_Type>::~Graph()
{
    delete A;
    /*
    delete Graph<Weight>::A->partitioning;
    
    
    delete Graph<Weight>::X;
    delete Graph<Weight>::Y;
    delete Graph<Weight>::V;
    delete Graph<Weight>::S;
    */
}

template<typename Integer_Type, typename Fractional_Type>
void Graph<Empty, Integer_Type, Fractional_Type>::load_text(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,  Tiling_type tiling_type)
{
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_, tiling_type);
    // Read graph
    read_text();
    A->init_csr();
    
    if(!Env::rank)
    {
        
        //printf(">>>>>\n");
        //Integer_Type N = 10;
        //struct basic_storage<Weight, Integer_Type> st(N);
        
        //printf("%lu %lu\n", st.nbytes, st.n);
        
        //st.free();
        //st.free<Weight, Integer_Type>();
    }
    
    
    
    



    
    
    
    

    
    
    /*
    Graph<Weight>::A->init_csr();
    Graph<Weight>::A->init_bv();
    
    uint32_t diag_segment = distance(Graph<Weight>::A->diag_ranks.begin(), find(Graph<Weight>::A->diag_ranks.begin(), Graph<Weight>::A->diag_ranks.end(), rank));
    
    Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    Graph<Weight>::S = new Vector<Weight>(Graph<Weight>::nvertices,  Graph<Weight>::mvertices, nranks * nranks, diag_segment);
    
    Graph<Weight>::X->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->local_col_segments);
    Graph<Weight>::Y->init_vec(Graph<Weight>::A->diag_ranks, Graph<Weight>::A->rowgrp_ranks_accu_seg);
    Graph<Weight>::V->init_vec(Graph<Weight>::A->diag_ranks);
    Graph<Weight>::S->init_vec(Graph<Weight>::A->diag_ranks);
    */
}

template<typename Integer_Type, typename Fractional_Type>
void Graph<Empty, Integer_Type, Fractional_Type>::init_graph(std::string filepath_, 
           Integer_Type nrows_, Integer_Type ncols_, bool directed_, 
           bool transpose_, Tiling_type tiling_type)
{
    
    filepath  = filepath_;
    nrows = nrows_ + 1; // In favor of vertex id 0
    ncols = ncols_ + 1; // In favor of vertex id 0
    nedges    = 0;
    directed  = directed_;
    transpose = transpose_;
    // Initialize matrix
    A = new Matrix<Empty, Integer_Type, Fractional_Type>(nrows, ncols, Env::nranks * Env::nranks, tiling_type);
}

template<typename Integer_Type, typename Fractional_Type>
void Graph<Empty, Integer_Type, Fractional_Type>::read_text()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str());
    if(!fin.is_open())
    {
        std::cout << "Unable to open input file" << std::endl;
        Env::exit(1);
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

    struct Triple<Empty, Integer_Type> triple;
    struct Triple<Empty, Integer_Type> pair;
    std::istringstream iss;
    while (std::getline(fin, line) && !line.empty())
    {
        iss.clear();
        iss.str(line);
        
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
        {
            std::cout << "read() failure" << std::endl;
            Env::exit(1);
        }


        if(transpose)
        {
            iss >> triple.col >> triple.row;
        }
        else
        {
            iss >> triple.row >> triple.col;
        }

        nedges++;
    
        pair = A->tile_of_triple(triple);
        if(A->tiles[pair.row][pair.col].rank == Env::rank)    
        {
            A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
        offset = fin.tellg();
    }
    fin.close();
    assert(offset == filesize);   
}




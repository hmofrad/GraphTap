/*
 * graph.hpp: Graph implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

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
        void init(std::string filepath, Integer_Type nrows, Integer_Type ncols, 
                       bool directed, bool transpose, Tiling_type tiling_type);
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
void Graph<Weight, Integer_Type, Fractional_Type>::init(std::string filepath_, 
           Integer_Type nrows_, Integer_Type ncols_, bool directed_, 
           bool transpose_, Tiling_type tiling_type)
{
    
    filepath  = filepath_;
    nrows = nrows_;
    ncols = ncols_;
    nedges    = 0;
    directed  = directed_;
    transpose = transpose_;
    // Initialize matrix
    A = new Matrix<Weight, Integer_Type, Fractional_Type>(nrows, ncols, Env::nranks * Env::nranks, tiling_type);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_text(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,  Tiling_type tiling_type)
{
    printf("[x]load_text\n");
    printf("%lu %lu\n", sizeof(Weight), sizeof(Graph<Weight, Integer_Type, Fractional_Type>::nrows));
    // Initialize graph
    init(filepath_, nrows_, ncols_, directed_, transpose_, tiling_type);
    //Graph<Weight>::A->init_mat();
    
    
    
    
    
    
    /*
    
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

/*
 * graph.hpp: Graph implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef GRAPH_HPP
#define GRAPH_HPP 
 
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>

#include "ds/triple.hpp"
#include "ds/compressed_storage.hpp"
#include "mat/matrix.hpp"

/* Class template specialization for Weight
 * We think of two ways for processing graphs with empty weights:
 * 1) Representing the adjacency matrix with a matrix of type char,
 * 2) Infer the adjacency matrix from the csr or csc.
 * We picked the 2nd approach and use C macros to completely
 * remove the weight and adjacency matrix for empty weight graphs.
 * Thus, We have saved Graph_size * sizeof(Weight) in space because
 * representing the adjacency matrix even with char will cost us
 * Graph_size * sizeof(char). At scale, this means if e.g. a graph 
 * with 1B edges, we're saving 1B bytes of memory. Instead of using
 * C macrots, a better way is to create a base class and inherit it.
 * #define HAS_WEIGHT macro will be defined by compiler at compile time.
 */

template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
class Graph
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Vertex_Program;
    
    public:    
        Graph();
        ~Graph();
        
        void load(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_ = true, bool transpose_ = false, bool acyclic_ = false, Tiling_type tiling_type_ = _2D_,
            Compression_type compression_type_ = _CSC_, Filtering_type filtering_type_ = _NONE_, bool parread_ = true);
        void load_binary(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_, bool transpose_, bool acycic_, Tiling_type tiling_type_, 
            Compression_type compression_type_, Filtering_type filtering_type_, bool parread_);
        void load_text(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_, bool transpose_, bool acycic_, Tiling_type tiling_type_, 
            Compression_type compression_type_, Filtering_type filtering_type_, bool parread_);
            
        void free();

    private:
        std::string filepath;
        Integer_Type nrows, ncols;
        uint64_t nedges;
        bool directed;
        bool transpose;
        bool acyclic;
        bool parread;
        Matrix<Weight, Integer_Type, Fractional_Type> *A;
        
        void init_graph(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_, 
               bool directed_, bool transpose_, bool acyclic_, Tiling_type tiling_type, 
               Compression_type compression_type_, Filtering_type filtering_type_, bool parread_);
        void read_text();
        void read_binary();
        void parread_text();
        void parread_binary();
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::Graph() : A(nullptr) {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::~Graph(){};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::free()
{

    A->del_compression();
 
    A->del_filtering();

    A->free_tiling();
    
    delete A;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::init_graph(std::string filepath_, 
           Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,
           bool acyclic_, Tiling_type tiling_type_, Compression_type compression_type_, 
           Filtering_type filtering_type_, bool parread_)
{
    
    filepath  = filepath_;
    nrows = nrows_ + 1; // In favor of vertex id 0
    ncols = ncols_ + 1; // In favor of vertex id 0
    nedges    = 0;
    directed  = directed_;
    transpose = transpose_;
    acyclic = acyclic_;
    parread = parread_;
    
    // Initialize matrix
    A = new Matrix<Weight, Integer_Type, Fractional_Type>(nrows, ncols, 
                                Env::nranks * Env::nranks, tiling_type_, compression_type_, filtering_type_, parread_);
}

/* The read_text and read_binary functions can be called indivdually from the user program.
   The reasin we put this wrapper in between is to identify the file type inside the engine.*/
   
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,
        bool acyclic_, Tiling_type tiling_type_, Compression_type compression_type_,
        Filtering_type filtering_type_, bool parread_)
{
    double t1, t2;
    t1 = Env::clock();
    int buffer_len = 100;
    char buffer[buffer_len];
    memset(buffer, '\n', buffer_len);
    
    int token_len = 10;
    char token[token_len];
    memset(token, '\n', token_len);
    
    FILE *fd = NULL;
    int len = 8 + filepath_.length() + 1; // file -b filepath\n
    char cmd[len];
    memset(cmd, '\0', len);
    sprintf(cmd, "file -b %s", filepath_.c_str());
    fd = popen(cmd, "r");
    while (fgets(buffer, buffer_len, fd) != NULL)
    { 
        ;
    }
    pclose(fd);
    
    std::istringstream iss (buffer);
    iss >> token;
    const char* text = "ASCII";
    const char* data = "data";
    const char* data1 = "Hitachi";
    if(!strcmp(token, text))
    {
        load_text(filepath_, nrows_, ncols_, directed_, transpose_, acyclic_, tiling_type_, compression_type_, filtering_type_, parread_);
    }
    else if(!strcmp(token, data) or !strcmp(token, data1))
    {
        load_binary(filepath_, nrows_, ncols_, directed_, transpose_, acyclic_, tiling_type_, compression_type_, filtering_type_, parread_);
    }
    else
    {
        fprintf(stderr, "Undefined file type %s\n", token);
        Env::exit(1);
    }
    t2 = Env::clock();
    Env::print_time("Ingress transpose", t2 - t1);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_text(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_, bool acyclic_,
        Tiling_type tiling_type_, Compression_type compression_type_, Filtering_type filtering_type_, bool parread_)
{
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_, acyclic_, tiling_type_, compression_type_, filtering_type_, parread_);
 
    // Read graph
    if(parread_)
    {
        if(Env::is_master)
            printf("%s: Distributed read using %d ranks\n", filepath_.c_str(), Env::nranks);
        parread_text();
    }
    else
    {
        if(not directed_)
        {
            fprintf(stderr, "Directed graphs can only be read by parallel read\n");
            Env::exit(1);
        }
        
        if(Env::is_master)
            printf("%s: Sequential read using %d ranks\n", filepath_.c_str(), Env::nranks);
        read_text();
    }
    
    // Initialize tiles
    A->init_tiles();
    
    // Filter the graph
    A->init_filtering();
    
    // Compress the graph
    A->init_compression();
    
    // Delete triples
    A->del_triples();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_binary(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_, bool acyclic_,
        Tiling_type tiling_type_, Compression_type compression_type_, Filtering_type filtering_type_, bool parread_)
{
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_, acyclic_, tiling_type_, compression_type_, filtering_type_, parread_);
 
    // Read graph
    if(parread_)
    {
        if(Env::is_master)
            printf("%s: Distributed read using %d ranks\n", filepath_.c_str(), Env::nranks);
        parread_binary();
    }
    else
    {
        if(not directed_)
        {
            fprintf(stderr, "Directed graphs can only be read by parallel read API\n");
            Env::exit(1);
        }
        
        if(Env::is_master)
            printf("%s: Sequential read using %d ranks\n", filepath_.c_str(), Env::nranks);        
        read_binary();
    }
    
    // Initialize tiles
    A->init_tiles();
    
    // Filter the graph
    A->init_filtering();
    
    // Compress the graph
    A->init_compression();
    
    // Delete triples
    A->del_triples();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::read_text()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str());
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file\n");
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
    while (std::getline(fin, line) && !line.empty())
    {
        iss.clear();
        iss.str(line);
        
        #ifdef HAS_WEIGHT
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 3)
        #else
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
        #endif
        {
            fprintf(stderr, "read() failure\n");
            Env::exit(1);
        }

        if(transpose)
        {
            #ifdef HAS_WEIGHT
            iss >> triple.col >> triple.row >> triple.weight;
            #else    
            iss >> triple.col >> triple.row;
            #endif
        }
        else
        {
            #ifdef HAS_WEIGHT
            iss >> triple.row >> triple.col >> triple.weight;
            #else
            iss >> triple.row >> triple.col;
            #endif
        }
        
        if(acyclic)
        {
          if(triple.row < triple.col)
            std::swap(triple.row, triple.col);
        }
        
        pair = A->tile_of_triple(triple);
        if(A->tiles[pair.row][pair.col].rank == Env::rank)    
        {
            A->test(triple);
            A->tiles[pair.row][pair.col].triples->push_back(triple);
        }

        nedges++;
        offset = fin.tellg();

        if(Env::is_master)
        {
            if ((offset & ((1L << 26) - 1L)) == 0)
            {
                printf("|");
            }
        }
    }
    
    fin.close();
    assert(offset == filesize);   
    
    if(!Env::rank)
    {
        printf("\n");
        printf("%s: Read %lu edges\n", filepath.c_str(), nedges);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::read_binary()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file");
        Env::exit(1); 
    }
    
    // Obtain filesize
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    struct Triple<Weight, Integer_Type> triple;
    struct Triple<Weight, Integer_Type> pair;
    while (offset < filesize)
    {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        
        if(fin.gcount() != sizeof(Triple<Weight, Integer_Type>))
        {
            fprintf(stderr, "read() failure\n");
            Env::exit(1);
        }

        if(transpose)
        {
            std::swap(triple.row, triple.col);
        }
                
        if(acyclic)
        {
          if(triple.row < triple.col)
            std::swap(triple.row, triple.col);
        }
        
        pair = A->tile_of_triple(triple);
        if(A->tiles[pair.row][pair.col].rank == Env::rank)    
        {
            A->test(triple);
            A->tiles[pair.row][pair.col].triples->push_back(triple);
        }
  
        nedges++;
        offset += sizeof(Triple<Weight, Integer_Type>);
        
        if(Env::is_master)
        {
            if ((offset & ((1L << 26) - 1L)) == 0)
                printf("|");
        }
    }
    
    fin.close();
    assert(offset == filesize);
    
    if(!Env::rank)
    {
        printf("\n");
        printf("%s: Read %lu edges\n", filepath.c_str(), nedges);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::parread_text()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str());
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file\n");
        Env::exit(1);
    }

    // Obtain filesize
    uint64_t filesize = 0, skip = 0,share = 0, offset = 0, endpos = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    // Skip comments
    std::string line;
    uint32_t position;  // Fallback position
    do
    {
        position = fin.tellg();
        std::getline(fin, line);
    } while ((line[0] == '#') || (line[0] == '%') || line.empty());
    
    // Calculate the number of edges
    // We assume there's no empty line 
    // in the middle of the file
    fin.clear();
    fin.seekg(position, std::ios_base::beg);
    while (std::getline(fin, line))
    {
        if(line.empty())
        {
            while(std::getline(fin, line))
            {
                if(fin.eof() or !line.empty())
                    break;
            }
            if(fin.eof())
                break;
        }
        nedges++;
    }
    fin.clear();
    fin.seekg(position, std::ios_base::beg);

    share = nedges / Env::nranks;
    offset = share * Env::rank;
    endpos = (Env::rank == Env::nranks - 1) ? nedges : offset + share;
    
    while(skip < offset)
    {
        std::getline(fin, line);
        skip++;
    }

    uint64_t nedges_local = 0;
    uint64_t nedges_global = 0;
    struct Triple<Weight, Integer_Type> triple;
    struct Triple<Weight, Integer_Type> pair;
    std::istringstream iss;
    while (std::getline(fin, line) && !line.empty() && offset < endpos)
    {
        iss.clear();
        iss.str(line);
        #ifdef HAS_WEIGHT
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 3)
        #else
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
        #endif
        {
            fprintf(stderr, "read() failure \"%s\"\n", line.c_str());
            Env::exit(1);
        }
        
        if(transpose)
        {
            #ifdef HAS_WEIGHT
            iss >> triple.col >> triple.row >> triple.weight;
            #else    
            iss >> triple.col >> triple.row;
            #endif
        }
        else
        {
            #ifdef HAS_WEIGHT
            iss >> triple.row >> triple.col >> triple.weight;
            #else
            iss >> triple.row >> triple.col;
            #endif
        }
        
        if(acyclic)
        {
          if((not transpose and triple.col < triple.row) or
             (transpose and triple.col > triple.row))
            std::swap(triple.row, triple.col);
        }
        
        A->insert(triple);
        
        if(not directed)
        {
            std::swap(triple.row, triple.col);
            A->insert(triple);
        }
        
        nedges_local++;
        offset++;
        if(Env::is_master)
        {
            if ((offset & ((1L << 26) - 1L)) == 0)
            {
                printf("|");
                fflush(stdout);
            }
        }
    }
    fin.close();
    assert(offset == endpos);   
    
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges == nedges_global);
    
    if(!Env::rank)
    {
        printf("%s: Read %lu edges\n", filepath.c_str(), nedges);
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::parread_binary()
{
    // Open graph file.
    std::ifstream fin(filepath.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file\n");
        Env::exit(1); 
    }
    
    // Obtain filesize
    uint64_t filesize = 0, skip = 0,share = 0, offset = 0, endpos = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    //fin.seekg(0, std::ios_base::beg);
    nedges = filesize / sizeof(Triple<Weight, Integer_Type>);
    
    
    share = (filesize / Env::nranks) / sizeof(Triple<Weight, Integer_Type>) * sizeof(Triple<Weight, Integer_Type>);
    assert(share % sizeof(Triple<Weight, Integer_Type>) == 0);

    offset += share * Env::rank;
    endpos = (Env::rank == Env::nranks - 1) ? filesize : offset + share;
    fin.seekg(offset, std::ios_base::beg);
    
    uint64_t nedges_local = 0;
    uint64_t nedges_global = 0;
    
    struct Triple<Weight, Integer_Type> triple;
    struct Triple<Weight, Integer_Type> pair;
    while (offset < endpos)
    {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        
        if(fin.gcount() != sizeof(Triple<Weight, Integer_Type>))
        {
            fprintf(stderr, "read() failure\n");
            Env::exit(1);
        }
        
        if(transpose)
        {
            std::swap(triple.row, triple.col);
        }
        
        if(acyclic)
        {
          if((not transpose and triple.col < triple.row) or
             (transpose and triple.col > triple.row))
            std::swap(triple.row, triple.col);
        }

        A->insert(triple);
        
        if(not directed)
        {
            std::swap(triple.row, triple.col);
            A->insert(triple);
        }
        
        nedges_local++;
        offset += sizeof(Triple<Weight, Integer_Type>);
        
        if(Env::is_master)
        {
            if ((offset & ((1L << 26) - 1L)) == 0)
            {
                printf("|");
                fflush(stdout);
            }
        }
    }
    
    fin.close();
    assert(offset == endpos);
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges == nedges_global);
    
    if(!Env::rank)
    {
        printf("\n");
        printf("%s: Read %lu edges\n", filepath.c_str(), nedges);
    }
}
#endif
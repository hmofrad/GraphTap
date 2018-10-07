/*
 * Degree.cpp: Degree benchmark (Outgoing)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"
#include "vp/vertex_program.hpp"

/* HAS_WEIGHT macro will be defined by compiler.
   So, you don't have to change this.   
   make WEIGHT="-DHASWEIGHT"         */
   
using em = Empty; // Weight (default is Empty)
#ifdef HAS_WEIGHT
using wp = uint32_t;
#else
using wp = em;
#endif

using ip = uint32_t; // Integer precision (default is uint32_t)
using fp = double;   // Fractional precision (default is float)

struct Generic_functions
{
    static fp ones(fp x, fp y, fp v, fp s)
    {
        return(1);
    }
    
    static fp assign(fp x, fp y, fp v, fp s)
    {
        return(y);
    }
};

int main(int argc, char **argv)
{
    bool comm_split = true;
    Env::init(comm_split);    
    
    if(argc != 4)  {
        if(Env::is_master) {
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<num_iterations>]\""
                      << std::endl;
        }    
        Env::exit(1);
    }
    
    std::string file_path = argv[1]; 
    ip num_vertices = std::atoi(argv[2]);
    uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
    bool directed = true;
    bool transpose = false;
    bool acyclic = false;
    bool parallel_edges = true;
    Tiling_type TT = _2D_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _SOME_;
    bool stationary = true;
    bool parread = true;
    Ordering_type OT = _ROW_;
    double time1 = 0;
    double time2 = 0;

    /* Degree execution */
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    Vertex_Program<wp, ip, fp> V(G, stationary, OT);    
    fp x = 0, y = 0, v = 0, s = 0;
    Generic_functions f;
    
    time1 = Env::clock();
    V.init(x, y, v, s);
    V.scatter_gather(f.ones);
    V.combine();
    V.apply(f.assign);  
    time2 = Env::clock();
    Env::print_time("Degree execution", time2 - time1);
    
    V.checksum();
    V.display();
    V.free();
    G.free();
    Env::finalize();
    return(0);
}
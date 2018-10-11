/*
 * Degree.cpp: Degree benchmark (Outgoing)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
#include <functional>

#include "deg.h"
 
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

/*  Integer precision (default is uint32_t)
    Controls the number of vertices,
    the engine can possibly process
*/
using ip = uint32_t;

/*
    Fractional precision (default is float)
    Controls the precision of values.
*/
using fp = double;

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
    bool parread = true;
    double time1 = 0, time2 = 0;
    
    /* Degree execution */
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    bool stationary = true;
    bool tc_family = false;
    bool gather_depends_on_apply = false;
    Ordering_type OT = _ROW_;
    // Register triangle counting function pointer handles
    DEG_state<wp, ip, fp> Deg_state;
    auto initializer = std::bind(&DEG_state<wp, ip, fp>::initializer, Deg_state, std::placeholders::_1, std::placeholders::_2);
    auto messenger   = std::bind(&DEG_state<wp, ip, fp>::messenger,   Deg_state, std::placeholders::_1, std::placeholders::_2);
    auto combiner    = std::bind(&DEG_state<wp, ip, fp>::combiner,    Deg_state, std::placeholders::_1, std::placeholders::_2);    
    auto applicator  = std::bind(&DEG_state<wp, ip, fp>::applicator,  Deg_state, std::placeholders::_1, std::placeholders::_2);
    // Run vertex program
    time1 = Env::clock();
    Vertex_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, tc_family, OT);
    V.init(initializer);
    V.scatter_gather(messenger);
    V.combine(combiner);
    V.apply(applicator);  
    time2 = Env::clock();
    Env::print_time("Degree execution", time2 - time1);
    
    V.checksum();
    V.display();
    V.free();
    G.free();
    Env::finalize();
    return(0);
}
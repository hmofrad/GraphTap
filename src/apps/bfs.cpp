/*
 * bfs.cpp: Breadth First Search (BFS) benchmark
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"
#include "vp/vertex_program.hpp"

#include "bfs.h"

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
using fp = uint32_t;

int main(int argc, char **argv)
{ 
    bool comm_split = true;
    Env::init(comm_split);    
    double time1 = Env::clock();   
    
    if(argc != 4)  {
        if(Env::is_master) {
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<root>]\""
                      << std::endl;
        }    
        Env::exit(1);
    }
    
    
    std::string file_path = argv[1]; 
    ip num_vertices = std::atoi(argv[2]);
    uint32_t root = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
    uint32_t num_iterations = 0;
    bool directed = false;
    bool transpose = false;
    bool acyclic = false;
    bool parallel_edges = false;
    Tiling_type TT = _2D_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _NONE_;
    bool parread = true;
    
    // Breadth First Search (BFS) execution 
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    
    bool stationary = false;
    bool tc_family  = false;
    bool gather_depends_on_apply = false;
    bool apply_depends_on_iter  = true;
    Ordering_type OT = _ROW_;
    BFS_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, apply_depends_on_iter, tc_family, OT);   
    V.root = root;
    V.execute(num_iterations);
    V.checksum();
    V.display();
    V.free();
    G.free();    

    double time2 = Env::clock();
    Env::print_time("Breadth First Search (BFS)", time2 - time1);
    Env::finalize();
    return(0);
}
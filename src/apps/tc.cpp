/*
 * tc.cpp: Triangle counting benchmark
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"
#include "vp/vertex_program.hpp"

#include "tc.h" 

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
    double time1 = Env::clock();    
    
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
    bool transpose = true;
    bool acyclic = true;
    bool parallel_edges = false;
    Tiling_type TT = _2D_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _NONE_; // Do not turn on
    bool parread = true;
    
    /* Triangle counting execution */
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    bool stationary = false;
    bool tc_family = true;
    bool gather_depends_on_apply = false;
    Ordering_type OT = _ROW_;
    // Run 1st vertex program and calculate ingoing adjacency list
    TC_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, tc_family, OT);
    V.execute(1);
    G.free();
    Env::barrier();
    
    transpose = false;
    Graph<wp, ip, fp> GR;    
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    // Run 2nd vertex program and calculate outgoing adjacency list
    TC_Program<wp, ip, fp> VR(GR, stationary, gather_depends_on_apply, tc_family, OT);  
    VR.initialize(&V);
    V.free();
    VR.execute(1);
    VR.free();
    GR.free();
    
    /*
    // Register triangle counting function pointer handles
    TC_state<wp, ip, fp> Tc_state;
    auto initializer = std::bind(&TC_state<wp, ip, fp>::initializer, Tc_state, std::placeholders::_1, std::placeholders::_2);
    auto combiner    = std::bind(&TC_state<wp, ip, fp>::combiner,    Tc_state, std::placeholders::_1, std::placeholders::_2);    
    auto applicator  = std::bind(&TC_state<wp, ip, fp>::applicator,  Tc_state, std::placeholders::_1, std::placeholders::_2);
    // Run 1st vertex program and calculate ingoing adjacency list
    time1 = Env::clock();
    Vertex_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, tc_family, OT);
    V.init(initializer);
    V.combine(combiner);
    V.apply(applicator);  
    G.free();
    
    fp v = 0;
    transpose = false;
    Graph<wp, ip, fp> GR;    
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    // Run 2nd vertex program and calculate outgoing adjacency list
	Vertex_Program<wp, ip, fp> VR(GR, stationary, gather_depends_on_apply, tc_family, OT);  
    VR.init(initializer, v, &V);
    V.free();
    VR.combine(combiner);
    VR.apply(applicator);  
    VR.free();
	GR.free();
    */

    double time2 = Env::clock();    
    Env::print_time("Triangle counting end-to-end", time2 - time1);
    Env::finalize();
    return(0);
}
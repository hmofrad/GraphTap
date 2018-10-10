/*
 * tc.cpp: Triangle counting benchmark
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
#include <functional>

#include "cc.hpp"
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"
#include "vp/vertex_program.hpp"

#define TRIANGLE_COUNTING

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
    bool directed = false;
    bool transpose = false;
    bool acyclic = false;
    bool parallel_edges = false;
    Tiling_type TT = _2D_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _NONE_;
    bool parread = true;
    double time1 = 0;
    double time2 = 0;
    
    /* Connected component execution */
    time1 = Env::clock();
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    
    
    CC_state<wp, ip, fp> Cc_state;
    auto init_func    = std::bind(&CC_state<wp, ip, fp>::init_func, Cc_state, std::placeholders::_1, std::placeholders::_2);    
    auto message_func = std::bind(&CC_state<wp, ip, fp>::message_func, Cc_state, std::placeholders::_1, std::placeholders::_2);
    auto combine_func = std::bind(&CC_state<wp, ip, fp>::combine_func, Cc_state, std::placeholders::_1, std::placeholders::_2);    
    auto apply_func   = std::bind(&CC_state<wp, ip, fp>::apply_func, Cc_state, std::placeholders::_1, std::placeholders::_2);

    bool stationary = true;
    bool gather_depends_on_apply = true;
    Ordering_type OT = _ROW_;
    Vertex_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, OT);    
    fp x = 0, y = 0, v = 0, s = 0;
    bool vid_flag = true;
    V.init(init_func);
    
    uint32_t iter = 0;
    uint32_t niters = num_iterations;
    while(iter < niters)
    {
        iter++;
        V.scatter_gather(message_func);
        V.combine(combine_func);   
        V.apply(apply_func);
        Env::print_me("Connected component, iteration: ", iter);
        //V.display(); 
    }
    time2 = Env::clock();    
    Env::print_time("Vertex execution", time2 - time1);
    
    //V.display(); 
    
    V.free();
    G.free();
	
    time2 = Env::clock();
    Env::print_time("Triangle couting", time2 - time1);
    Env::finalize();
    return(0);
}
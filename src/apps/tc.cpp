/*
 * tc.cpp: Triangle Counting (TC) benchmark main
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"

#include "tc.h" 

int main(int argc, char **argv)
{
    bool comm_split = true;
    Env::init(comm_split);  
    double time1 = Env::clock();    
    if(argc != 3 and argc != 4) {
        if(Env::is_master)
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices>\"" << std::endl;
        Env::exit(1);
    }
    std::string file_path = argv[1]; 
    ip num_vertices = std::atoi(argv[2]);
    bool directed = true;
    bool transpose = true;
    bool self_loops = false;
    bool acyclic = true;
    bool parallel_edges = false;
    Tiling_type TT = _2D_;
    Compression_type CT = _CSC_; // Only CSC is supported
    Filtering_type FT = _NONE_; // Do not turn on
    Hashing_type HT = BUCKET;
    bool parread = true;
    
    /* Triangle counting execution */
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, self_loops, acyclic, parallel_edges, TT, CT, FT, HT, parread);
    bool stationary = false;
    bool activity_filtering = false; // Has no effect (always on)
    bool tc_family = true;
    bool gather_depends_on_apply = false;
    bool apply_depends_on_iter  = false;
    Ordering_type OT = _ROW_;
    // Run 1st vertex program and calculate ingoing adjacency list
    TC_Program<wp, ip, fp> V(G, stationary, activity_filtering, gather_depends_on_apply, apply_depends_on_iter, tc_family, OT);
    V.execute(1);
    G.free();
    Env::barrier();
    
    transpose = false;
    //OT = _ROW_;
    Graph<wp, ip, fp> GR;    
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, self_loops, acyclic, parallel_edges, TT, CT, FT, HT, parread);
    // Run 2nd vertex program and calculate outgoing adjacency list
    TC_Program<wp, ip, fp> VR(GR, stationary, activity_filtering, gather_depends_on_apply, apply_depends_on_iter, tc_family, OT);  
    VR.initialize(V);
    V.free();
    VR.execute(1);
    VR.free();
    GR.free();
    
    double time2 = Env::clock();    
    Env::print_time("Triangle counting end-to-end", time2 - time1);
    Env::finalize();
    return(0);
}

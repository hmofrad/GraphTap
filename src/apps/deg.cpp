/*
 * deg.cpp: Degree benchmark main
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"
#include "deg.h"

int main(int argc, char** argv) {
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
    ip num_iterations = 1;
    bool directed = true;
    bool transpose = false;
    bool self_loops = true;
    bool acyclic = false;
    bool parallel_edges = true;
    Tiling_type TT = _2DT_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _SOME_;
    Hashing_type HT = NONE;
    bool parread = true;
    Graph<wp, ip, ip> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, self_loops, acyclic, parallel_edges, TT, CT, FT, HT, parread);
    bool stationary = true;
    bool activity_filtering = false; // Has no effect (Always off)
    bool tc_family = false;
    bool gather_depends_on_apply = false;
    bool apply_depends_on_iter  = false;
    Ordering_type OT = _ROW_;
    /* Degree execution */
    Deg_Program<wp, ip, ip> V(G, stationary, activity_filtering, gather_depends_on_apply, apply_depends_on_iter, tc_family, OT);
    V.execute(num_iterations);
    V.checksum();
    V.display();
    V.free();
    G.free();
    
    double time2 = Env::clock();
    Env::print_time("Degree end-to-end", time2 - time1);
    Env::finalize();
    return(0);
}
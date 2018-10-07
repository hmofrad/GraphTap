/*
 * pr.cpp: PageRank benchmark
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
    
    static fp div(fp x, fp y, fp v, fp s)
    {
        if(v and s)
        {
            return(v / s);
        }
        else
        {
            return(0);
        }
    }
    
    static fp assign(fp x, fp y, fp v, fp s)
    {
        return(y);
    }
    
    static fp rank(fp x, fp y, fp v, fp s)
    {
        //fp tol = 1e-5;
        fp alpha = 0.15;
        return(alpha + (1.0 - alpha) * y);
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
    G.free();
    
    /* Vertex execution */
    transpose = true;
    Graph<wp, ip, fp> GR;    
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, acyclic, parallel_edges, TT, CT, FT, parread);
    fp alpha = 0.15;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(GR, stationary, OT);   
    
    time1 = Env::clock();
    VR.init(x, y, v, s, &V);
    V.free();
    uint32_t iter = 0;
    uint32_t niters = num_iterations;
    while(iter < niters)
    {
        iter++;
        VR.scatter_gather(f.div);
        VR.combine();   
        VR.apply(f.rank);
        Env::print_me("Pagerank iteration: ", iter);
    }
    time2 = Env::clock();    
    Env::print_time("Vertex execution", time2 - time1);
    
    VR.checksum();
    VR.display(); 
    VR.free();
    GR.free();
    Env::finalize();
    return(0);   
}
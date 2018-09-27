/*
 * Degree.cpp: Degree benchmark
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "env.hpp"
#include "graph.hpp"
#include "vertex_program.hpp"
#include <iostream>
#include <unistd.h>

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
    Tiling_type TT = _2D_;
    Ordering_type OT = _ROW_;
    Compression_type CT = _CSC_;
    Filtering_type FT = _SOME_;
    bool parread = true;
    double time1 = 0;
    double time2 = 0;

    /* Degree execution */
    if(!Env::rank)
        printf("Computing Degree ...\n");
    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT, FT, parread);
    if(!Env::rank)
        Env::tock("Ingress");
    
    if(!Env::rank)
        Env::tick();
    fp x = 0, y = 0, v = 0, s = 0;
    Generic_functions f;
    Vertex_Program<wp, ip, fp> V(G, OT);
    V.init(x, y, v, s);
    if(!Env::rank)
        Env::tock("Init");
    
    if(!Env::rank)
    {
        printf("Degree execution\n");
        time1 = Env::clock();
    }
    if(comm_split)
    {
        if(!Env::rank)
            Env::tick();
        V.bcast(f.ones);
        if(!Env::rank)
            Env::tock("Bcast"); 
    }
    else
    {
        if(!Env::rank)
            Env::tick();
        V.scatter(f.ones);    
        if(!Env::rank)
            Env::tock("Scatter"); 
        
        if(!Env::rank)
            Env::tick();
        V.gather();
        if(!Env::rank)
            Env::tock("Gather"); 
    }
    
    if(!Env::rank)
        Env::tick();
    V.combine();
    if(!Env::rank)
        Env::tock("Combine stacked"); 

    if(!Env::rank)
        Env::tick();
    V.apply(f.assign);
    if(!Env::rank)
        Env::tock("Apply"); 
    
    if(!Env::rank)
    {
        time2 = Env::clock();
        printf("Degree time: %fseconds\n", time2 - time1);
    }

    V.checksum_degree();
    V.checksum();
    V.free();
    G.free();
    
    Env::finalize();
    return(0);
}
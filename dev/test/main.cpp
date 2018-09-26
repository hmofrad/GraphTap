/*
 * main.cpp: Main application
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

    V.checksum();
    V.checksumPR();
    //V.free();
    G.free();
    //Env::finalize();
    //return(1);

    Env::barrier(); 
    if(!Env::rank)
        printf("\n");
    
    /* Vertex execution */
    if(!Env::rank)
        printf("Computing PageRank ...\n");
    
    transpose = true;
    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> GR;
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT, FT, parread);
    Env::barrier();
    if(!Env::rank)
        Env::tock("Ingress transpose");
    
    fp alpha = 0.15;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(GR);
    
    
    /*
    OT = _COL_;
    fp alpha = 0.15;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(G, OT);
    */
    
    if(!Env::rank)
        Env::tick();
    VR.init(x, y, v, s, &V);
    V.free();
    if(!Env::rank)
        Env::tock("Init");
    
    uint32_t iter = 0;
    uint32_t niters = num_iterations;
    
    if(!Env::rank)
        time1 = Env::clock();
    while(iter < niters)
    {
        iter++;
        if(comm_split)
        {
            if(!Env::rank)
                Env::tick();
            VR.bcast(f.div);
            if(!Env::rank)
                Env::tock("Bcast"); 
        }
        else
        {        
            if(!Env::rank)
                Env::tick();
            VR.scatter(f.div);
            if(!Env::rank)
                Env::tock("Scatter"); 
            
            if(!Env::rank)
                Env::tick();
            VR.gather();
            if(!Env::rank)
                Env::tock("Gather"); 
        }
        
        if(!Env::rank)
            Env::tick();
        VR.combine();        
        if(!Env::rank)
            Env::tock("Combine"); 
        
        if(!Env::rank)
                Env::tick();
        VR.apply(f.rank);
        if(!Env::rank)
            Env::tock("Apply"); 
        
        if(!Env::rank)
            printf("Pagerank, iter: %d\n", iter);
    }
    
    if(!Env::rank)
    {
        time2 = Env::clock();
        printf("Pagerank time: %f seconds\n", time2 - time1);
    }
    
    VR.checksumPR();
    //VR.free();
    //V.free();
    //G.free();
    
    VR.free();
    GR.free();
    
    Env::finalize();
    return(0);
}
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
    //printf("rank=%d,nranks=%d,is_master=%d\n", Env::rank, Env::nranks, Env::is_master);
    
    
    
    // Print usage
    // Should be moved later
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
    //std::cout << file_path.c_str() << " " << num_vertices << " " << num_iterations << std::endl;
    bool directed = true;
    bool transpose = false;
    Tiling_type TT = _2D_;
    Ordering_type OT = _ROW_;
    Compression_type CT = _CSR_;
    Filtering_type FT = _SRCS_;
    bool parread = true;
    

    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> G;
    //Graph<> G;
    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT, FT, parread);
    
    if(!Env::rank)
        Env::tock("Ingress");



    
    Vertex_Program<wp, ip, fp> V(G, OT);
    
    
        
    
    
    
    fp x = 0, y = 0, v = 0, s = 0;
    V.init(x, y, v, s);
    
    /*
    Env::barrier();
    
    V.free();
    G.free();
    Env::finalize();
    return(0);    
*/
    
    Generic_functions f;
    
    if(comm_split)
    {
        if(!Env::rank)
            printf("bcast\n");
        V.bcast(f.ones);
    }
    else
    {
    
        if(!Env::rank)
            printf("scatter\n");
        V.scatter(f.ones);    
        if(!Env::rank)
            printf("gather\n");
        V.gather();
    }
    

    
    
    if(!Env::rank)
        printf("combine\n");
    V.combine();

/*    
    G.free();
    Env::barrier(); 
    //Env::barrier();
    V.free();
    
    Env::finalize();
    return(0);
*/
    if(!Env::rank)
        printf("apply\n");
    V.apply(f.assign);

    
    //V.checksumPR();
    

    if(!Env::rank)
        printf("Checksum\n");        
    V.checksum();
    //V.checksumPR();

    if(!Env::rank)
        Env::tock("Degree");
    
    /*
    G.free();
    Env::barrier(); 
    //Env::barrier();
    V.free();
    
    Env::finalize();
    return(0);
    */
    
    //sleep(3);
    G.free();
    Env::barrier(); 
    transpose = true;
    
    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> GR;
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT, FT, parread);

    if(!Env::rank)
        Env::tock("Ingress transpose");
    
    Env::barrier();
    
    fp alpha = 0.15;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(GR);
    //OT = _COL_;
    //Vertex_Program<wp, ip, fp> VR(G, OT);
    
    if(!Env::rank)
        Env::tick();
    VR.init(x, y, v, s, &V);
    if(!Env::rank)
        Env::tock("Init"); 
    
    V.free();
    
    uint32_t iter = 0;
    uint32_t niters = num_iterations;
    
    double time1 = 0;
    double time2 = 0;
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
            printf("Pagerank,iter=%d\n", iter);
    }
    
    if(!Env::rank)
    {
        time2 = Env::clock();
        printf("Pagerank time=%f\n", time2 - time1);
    }
    Env::barrier();
    VR.checksumPR();
    
    
    
    Env::barrier();
    VR.free();
    GR.free();
    //G.free();
    Env::finalize();
    return(0);
}


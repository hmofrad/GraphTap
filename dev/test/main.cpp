/*
 * main.cpp: Main application
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "env.hpp"
#include "graph.hpp"
#include "vertex_program.hpp"

using em = Empty;
using wp = Empty;   // Weight (default is Empty)
using ip = uint32_t; // Integer precision (default is uint32_t)
using fp = float;   // Fractional precision (default is float)




struct Generic_functions
{
    static fp ones(fp x, fp y, fp v, fp s)
    {
        return(1);
    }
    
    static fp div(fp x, fp y, fp v, fp s)
    {
        if(s)
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
        fp alpha = 0.1;
        return(alpha + (1 - alpha) * y);
    }
};




int main(int argc, char **argv)
{ 
    Env::init();
    ///if(!Env::rank)
       // Env::tick();
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
    Compression_type CT = _CSC_;

    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> G;
    //Graph<> G;
    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT);
    if(!Env::rank)
        Env::tock("Ingress");
    
    //if(!Env::rank);
      //  Env::tock("Test");
    
    //G.free();
    
    if(!Env::rank)
        Env::tick();


    Vertex_Program<wp, ip, fp> V(G);
    fp x = 0, y = 0, v = 0, s = 0;
    //printf("init\n");
    V.init(x, y, v, s);
    
    Generic_functions f;
    //printf("scatter\n");
    V.bcast(f.ones);
    //V.scatter(f.ones);    
    //printf("gather\n");
    //V.gather();
    //printf("combine %d\n", Env::rank);
    V.combine(f.assign);
    //printf("free %d\n",  Env::rank);
    V.free();
    G.free();
    if(!Env::rank)
        Env::tock("Degree");
    Env::barrier(); 
    Env::finalize();
    return(0);
    
    
    transpose = true;
    //Env::barrier();
    
    if(!Env::rank)
        Env::tick();
    Graph<wp, ip, fp> GR;
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, TT, CT);
    if(!Env::rank)
        Env::tock("Ingress transpose");
    
    fp alpha = 0.1;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(GR);
    
    if(!Env::rank)
        Env::tick();
    VR.init(x, y, v, &V);
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

        if(!Env::rank)
            Env::tick();

        VR.bcast(f.ones);
        
        if(!Env::rank)
            Env::tock("Bcast"); 
        
        /*
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
        */
        
        if(!Env::rank)
            Env::tick();
        
        VR.combine(f.rank);
        
        if(!Env::rank)
            Env::tock("Combine"); 
        
        if(!Env::rank)
            printf("Pagerank,iter=%d\n", iter);
    }
    
    if(!Env::rank)
    {
        time2 = Env::clock();
        printf("Pagerank time=%f\n", time2 - time1);
    }
    Env::barrier();
    VR.free();
    GR.free();
    
    
    
    
    //Graph<Weight>::combine(f.assign);
    
    //Env::barrier();
    Env::finalize();
    return(0);
}


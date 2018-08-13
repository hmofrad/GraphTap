/*
 * main.cpp: Main application
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "env.hpp"
#include "graph.hpp"
#include "vertex_program.hpp"
//#include "ds.hpp"

using em = Empty;
using wp = uint32_t;   // Weight (default is Empty)
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
    bool transpose = true;

    Graph<wp, ip, fp> G;
    //Graph<> G;
    G.load(file_path, num_vertices, num_vertices);
    //if(!Env::rank);
      //  Env::tock("Test");
    
    Vertex_Program<wp, ip, fp> V(G);
    //if(!Env::rank)
        //printf("MAIN DONE\n");
    fp x = 0, y = 0, v = 0, s = 0;
    V.init(x, y, v, s);
    
    Generic_functions f;
    V.scatter(f.ones);    
    V.gather();
    V.combine(f.assign);
    G.free();
    
    Graph<wp, ip, fp> GR;
    GR.load(file_path, num_vertices, num_vertices, directed, transpose, Tiling_type::_2D_);
    
    fp alpha = 0.1;
    x = 0, y = 0, v = alpha, s = 0;
    Vertex_Program<wp, ip, fp> VR(GR);
    //VR.init(x, y, v, V);
    
    
    
    GR.free();
    
    //Graph<Weight>::combine(f.assign);
    
    
    Env::finalize();
    return(0);
    
   // 
    
    //start = MPI_Wtime();
    //G.load_binary(file_path, num_vertices, num_vertices, Tiling::_2D_);
    
    //G.load_text(file_path, num_vertices, num_vertices, Tiling::_2D_);
    
    /*
    finish = MPI_Wtime();
    if(!rank)
        printf("Ingress: %f seconds\n", finish - start); 
    
    start = MPI_Wtime();
    G.degree();
    finish = MPI_Wtime();
    if(!rank)
        printf("Degree: %f seconds\n", finish - start); 
    G.free();
    */
    
    /*
    G.free(clear_state);
    
    transpose = true;
    Graph<ew_t> GR;
    start = MPI_Wtime();
    //GR.load_binary(file_path, num_vertices, num_vertices, Tiling::_2D_, directed, transpose);
    GR.load_text(file_path, num_vertices, num_vertices, Tiling::_2D_, directed, transpose);
    finish = MPI_Wtime();
    if(!rank)
        printf("Ingress T: %f seconds\n", finish - start); 


    
    GR.initialize(G);
    

    start = MPI_Wtime();  
    GR.pagerank(num_iterations);
    finish = MPI_Wtime();
    if(!rank)
        printf("Pagerank: %f seconds\n", finish - start); 
    
    GR.free();
    */
    //MPI_Finalize();

    

    

}


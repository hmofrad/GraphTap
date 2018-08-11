/*
 * main.cpp: Main application
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include "env.hpp"
#include "graph.hpp"

using wp = double;   // Weight (default is Empty)
using ip = uint64_t; // Integer precision (default is uint32_t)
using fp = double;   // Fractional precision (default is float)

int main(int argc, char** argv)
{ 
    Env::init();
    Env::tick();
    printf("rank=%d,nranks=%d,is_master=%d\n", Env::rank, Env::nranks, Env::is_master);
    
    
    
    // Print usage
    // Should be moved later
    if(argc != 4)  {
        if(Env::is_master) {
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<num_iterations>]\""
                      << std::endl;
        }    
        Env::exit(1);
    }
    
    double start, finish;
    
    std::string file_path = argv[1]; 
    ip num_vertices = std::atoi(argv[2]);
    uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
    std::cout << file_path.c_str() << " " << num_vertices << " " << num_iterations << std::endl;
    bool directed = true;
    bool transpose = false;
    bool clear_state = false;

    Graph<> G;
    G.load_text(file_path, num_vertices, num_vertices);
    Env::tock("Test");
    
    
    
    
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


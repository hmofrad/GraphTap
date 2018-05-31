/*
 * Test app 
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>

#include <mpi.h>


//void shuffle_ranks(int nranks) {
//	std::vector<int> rnaks(nranks);
//    int seed = 0;
//    srand(seed);

//}

int main(int argc, char ** argv) {
	
	int required = MPI_THREAD_MULTIPLE;
	int provided = -1;
  	MPI_Init_thread(nullptr, nullptr, required, &provided);
  	assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));


	int nranks = -1;
  	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  	assert(nranks >= 0);

  	int rank   = -1;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(nranks >= 0);

	bool is_master = (rank == 0);





  	//std::cout << nranks << "," << rank << "," << is_master << std::endl;

	std::vector<int> ranks(nranks);
  	if(is_master) {		
    	int seed = 0;
    	srand(seed);
    	//std::vector<int>::iterator it = ranks.begin();
    	//std::cout << *it  << std::endl;

    	//std::vector<int>::iterator it1 = ranks.end();
    	//std::cout << *it1  << std::endl;

	    std::iota(ranks.begin() + 1, ranks.end(), 1);
	    //for(int i = 0; i < nranks; i++) 
	    //	std::cout << ranks[i] << " ";
    	//std::cout << "\n";
        std::random_shuffle(ranks.begin() + 1, ranks.end());

		//for(int i = 0; i < nranks; i++) 
	    //	std::cout << ranks[i] << " ";
    	//std::cout << "\n";



  	}
  	MPI_Bcast(ranks.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);

  	//if(rank == 1) {		
  	//	std::vector<int>::iterator it;
	//	for(it = ranks.begin(); it < ranks.end(); it++) 
	//    	std::cout << *it << " ";
    //	std::cout << "\n";


  	//}

	
	MPI_Group world_group;
  	MPI_Comm_group(MPI_COMM_WORLD, &world_group); // Create the world group
  	MPI_Group reordered_group;
  	MPI_Group_incl(world_group, nranks, ranks.data(), &reordered_group); // Create the reordered group
  	MPI_Comm MPI_WORLD;
  	MPI_Comm_create(MPI_COMM_WORLD, reordered_group, &MPI_WORLD); // Create a communicator for the reordered group

  	MPI_Comm_rank(MPI_WORLD, &rank);

	
  	// Print usage
  	// Should be moved later
	if(argc != 4)  {
		if(is_master) {
			std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<num_iterations>]\""
	    			  << std::endl;
		}	
		MPI_Barrier(MPI_WORLD);
     	MPI_Finalize();
		std::exit(1);
	}

	std::string filepath = argv[1]; 
	uint32_t num_vertices = std::atoi(argv[2]);
  	uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
  	std::cout << num_vertices << " "<< num_iterations <<  std::endl;


	














	MPI_Finalize();

	return(0);
}

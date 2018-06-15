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

#include <sched.h>
int sched_getcpu(void);

//#include "hashers.h"

/*
uint32_t hash(long max_domain, long nbuckets, long v) {
	const long int multiplier = 128u;
	long nparts = 0;
	long height = 0;
	long max_range = 0;

	nparts = nbuckets * multiplier;
	height = max_domain / nparts;
	max_range = height / nparts;
	std::cout << "multiplier=" << multiplier << std::endl;
	std::cout << "nparts=" << nparts << std::endl;
	std::cout << "height=" << height << std::endl;
	std::cout << "max_range=" << max_range << std::endl;


    if(v >= max_range) return v;
    long col = (uint32_t) v % nparts;
    long row = v / nparts;
    return row + col * height;


	//return(0);


}


void load_binary(std::string filepath, uint32_t num_vertices, int nranks,bool is_master) {
	uint32_t num_rows = num_vertices;
	uint32_t num_cols = num_vertices;

	ReversibleHasher* hasher = new SimpleBucketHasher(num_vertices, nranks);
	if(is_master) {
		uint32_t i, j ,k;

		for(i = num_vertices - 100; i < num_vertices; i++) {
			j = hasher->hash(i);
			k = hasher->hash(j);
			//std::cout << "i=" << i << ",hash(i)=" << j << ",hash(j)=" << k << std::endl;
			if((i != j) && (j == k)) {
				std::cout << "i=" << i << ",hash(i)=" << j << ",hash(j)=" << k << std::endl;
			}

		}

	}
	


*/
	/* if(is_master) {
		long v = 10;
		std::cout << "hash(" << v  << ")=" << hash(num_vertices, nranks, v) <<  std::endl;

		v = 976560;
		std::cout << "hash(" << v  << ")=" << hash(num_vertices, nranks, v) <<  std::endl;
	} */

//}


int main(int argc, char** argv) {
	
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int processor_name_len;
	
	int required = MPI_THREAD_MULTIPLE;
	int provided = -1;
  	MPI_Init_thread(nullptr, nullptr, required, &provided);
  	assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));


	int nranks = -1;
  	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  	assert(nranks >= 0);

  	int rank   = -1;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(rank >= 0);
	
	MPI_Get_processor_name(processor_name, &processor_name_len);

    int cpu_id = sched_getcpu();
	std::cout << "Rank " << rank << " of " << nranks << 
	               ", hostname " << processor_name << ", CPU " << cpu_id << std::endl;

	bool is_master = (rank == 0);
	
	
	
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
/*


  	//std::cout << nranks << "," << rank << "," << is_master << std::endl;

	std::vector<int> ranks(nranks, 0);
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


	

  	load_binary(filepath, num_vertices, nranks, is_master);

*/












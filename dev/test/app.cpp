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
#include <fstream>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <mpi.h>
#include <sched.h>

int sched_getcpu(void);

int nranks;
int rank;
int cpu_id;
bool is_master;

//static const Weight uint32_t;

//template <class Weight>
struct Triple {
  uint32_t row;
  uint32_t col;
  //Weight weight;
  Triple(uint32_t row = 0, uint32_t col = 0)
	: row(row), col(col) {}  
};

void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols)
{
//  assert(A == nullptr);

  // Initialize graph meta.
  std::string filepath = filepath_;
  uint32_t nvertices = nrows;
  uint32_t mvertices = ncols;
  uint64_t  nedges = 0;  // TODO

  // Open matrix file.
  std::ifstream fin(filepath.c_str(), std::ios_base::binary);
  if(!fin.is_open())
  {
    std::cout << "Unable to open input file" << std::endl;
    exit(1); 
  }



  // Obtain filesize (minus header) and initial offset (beyond header).
  bool header_present = nvertices == 0;
  uint64_t orig_filesize, filesize, share, offset = 0, endpos;
  fin.seekg (0, std::ios_base::end);
  orig_filesize = filesize = (uint64_t) fin.tellg();
  fin.seekg(0, std::ios_base::beg);

if (header_present)
  {
    // Read header as 32-bit nvertices, 32-bit mvertices, 64-bit nedges (nnz)
    Triple header;

    uint32_t n, m;
    uint64_t nnz = 0;


    fin.read(reinterpret_cast<char *>(&header), sizeof(header));

    if(fin.gcount() != sizeof(header))
    {
      std::cout << "read() failure" << std::endl;
      exit(1);
    }

    nvertices = nrows = header.row + 1;  // HACK: (the "+ 1"; for one/zero-based)
    mvertices = ncols = header.col + 1;  // HACK: (the "+ 1"; for one/zero-based)
    //nedges = header.weight;

    //LOG.info("Read header: nvertices = %u, mvertices = %u, nedges (nnz) = %lu \n",
      //       nvertices, mvertices, nedges);

    //bipartite = nrows != ncols;
    //if (bipartite)
    //  nvertices = mvertices = nrows + ncols;

    // Let offset and filesize reflect the body of the file.
    offset += sizeof(header);
    filesize -= offset;
  }
	uint64_t ntriples = (orig_filesize - offset);
  //uint64_t ntriples = (orig_filesize - offset) / sizeof(Triple<Weight>);
  //LOG.info("File appears to have %lu edges (%u-byte weights). \n",
    //       ntriples, sizeof(Triple<Weight>) - sizeof(Triple<Empty>));

//  if (header_present and nedges != ntriples)
  //  LOG.info("[WARN] Number of edges in header does not match number of edges in file. \n");

  // Now with nvertices potentially changed, initialize the matrix object.
  //A = new Matrix(nvertices, mvertices, Env::nranks * Env::nranks, partitioning);

  // Determine current rank's offset and endpos in File.
  share = (filesize / nranks);
  assert(share % sizeof(struct Triple) == 0);

  offset += share * rank;
  endpos = (rank == nranks - 1) ? orig_filesize : offset + share;

  // Seek up to the offset for rank.
  fin.seekg(offset, std::ios_base::beg);

  if(!fin.good()) {
    std::cout << "seekg() failure \n" << std::endl;
    exit(1);
  }

  // Start reading from file and scattering to matrix.
  //LOG.info("Reading input file ... \n");

  //DistTimer read_timer("Reading Input File");

  //if(!Env::rank) {
    //std::cout << Env::rank << ":" << offset << "," << endpos << " " <<  (endpos - offset) / sizeof(Triple<Weight>) << std::endl;
  //}

  Triple triple;

  while (offset < endpos)
  {
    fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
	std::cout << triple.row << " " << triple.col << std::endl;
    if(fin.gcount() != sizeof(triple))
    {
      std::cout << "read() failure" << std::endl;
      exit(1);
    }

    if ((offset & ((1L << 26) - 1L)) == 0)
    {
      std::cout << "|" << std::endl;
      fflush(stdout);
    }
    offset += sizeof(Triple);

    //if (bipartite)
    //  triple.col += nrows;

    // Remove self-loops
    //if (triple.row == triple.col)
    //  continue;    

    // Flip the edges to transpose the matrix, since y = ATx => process messages along in-edges
    // (unless graph is to be reversed).
    //if (directed and not reverse_edges)
    //  std::swap(triple.row, triple.col);

   //if (remove_cycles)
    //{
     // if ((not reverse_edges and triple.col > triple.row)
      //    or (reverse_edges and triple.col < triple.row))
       // std::swap(triple.row, triple.col);
    //}

    //triple.row = (uint32_t) hasher->hash(triple.row);
    //triple.col = (uint32_t) hasher->hash(triple.col);

    // Insert edge.
   // A->insert(triple);

    //if (not directed)  // Insert mirrored edge.
    //{
     // std::swap(triple.row, triple.col);
      //A->insert(triple);
    //}


      


  }

  //read_timer.stop();

  //LOG.info<false, false>("[%d]", Env::rank);
 // Env::barrier();
//  LOG.info<true, false>("\n");

  assert(offset == endpos);
  fin.close();

}






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


	nranks = -1;
  	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  	assert(nranks >= 0);

  	rank   = -1;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(rank >= 0);
	
	MPI_Get_processor_name(processor_name, &processor_name_len);

    cpu_id = sched_getcpu();
	std::cout << "Rank " << rank << " of " << nranks << 
	               ", hostname " << processor_name << ", CPU " << cpu_id << std::endl;

	is_master = (rank == 0);
	
	
	
	// Print usage
  	// Should be moved later
	if(argc != 4)  {
		if(is_master) {
			std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> [<num_iterations>]\""
	    			  << std::endl;
		}	
		MPI_Barrier(MPI_COMM_WORLD);
     	MPI_Finalize();
		std::exit(1);
	}

	std::string filepath = argv[1]; 
	uint32_t num_vertices = std::atoi(argv[2]);
  	uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
  	std::cout << num_vertices << " "<< num_iterations <<  std::endl;
	
	load_binary(filepath, num_vertices, num_vertices);
	
	
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












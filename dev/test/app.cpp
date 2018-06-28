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

struct Tile2D
{
  //int test;
  std::vector<struct Triple >* triples;
  /*
  Tile2D() { allocate_triples(); }
  ~Tile2D() { free_triples(); }
  
  void allocate_triples()
  {
    if (!triples)
      triples = new std::vector<struct Triple>;
  }
  
  void free_triples()
  {
    delete triples;
    triples = nullptr;
  }
  */
  uint32_t rg, cg;
  uint32_t ith, jth, nth;
  int32_t rank;
};

std::vector<std::vector<struct Tile2D>> tiles;
std::vector<int> local_tiles;

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

    offset += sizeof(header);
    filesize -= offset;
  }
  
  uint64_t ntriples = (orig_filesize - offset) / sizeof(struct Triple);
  share = ((filesize / nranks) / sizeof(struct Triple)) * sizeof(struct Triple);
  assert(share % sizeof(struct Triple) == 0);

  //offset += share * rank;
  //endpos = (rank == nranks - 1) ? orig_filesize : offset + share;

  offset = offset;
  endpos = orig_filesize;
  
  // Seek up to the offset for rank.
  fin.seekg(offset, std::ios_base::beg);

  if(!fin.good()) {
    std::cout << "seekg() failure \n" << std::endl;
    exit(1);
  }
  
  
  //int nrows = nvertices;
  //int ncols = mvertices;
  int ntiles = nranks * nranks;
  int nrowgrps = sqrt(ntiles);
  int ncolgrps = ntiles / nrowgrps;
  int tile_height = (nrows / nrowgrps) + 1;
  int tile_width = (ncols / ncolgrps) + 1;
  
  if(!rank)
	  printf("nrows=%d ncols=%d tiles=%d nrowgrps=%d ncolgrps=%d tile_height=%d tile_width=%d endpos=%lu\n", nrows, ncols, ntiles, nrowgrps, ncolgrps, tile_height, tile_width, endpos);

  
  
  /* Reserve the 2D vector of tiles. */
  tiles.resize(nrowgrps);
  for (uint32_t i = 0; i < nrowgrps; i++)
    tiles[i].resize(ncolgrps);

  for (uint32_t i = 0; i < nrowgrps; i++)
  {
	for (uint32_t j = 0; j < ncolgrps; j++)  
	{
	  tiles[i][j].rg = i;
      tiles[i][j].cg = j;
	  tiles[i][j].rank = i;
	}
  }
 // if(rank == 2) {
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
	  for (uint32_t j = 0; j < ncolgrps; j++)  
	  {
		  if(tiles[i][j].rank == rank)
		  {
		    local_tiles.push_back((i * ncolgrps) + j);
          printf(">>>> %d %d %d %d\n", tiles[i][j].rank, i, j, (i * ncolgrps) + j);
		  }
	  }
	  //printf("\n");
    }
	
	/*
	for(uint32_t i = 0; i < local_tiles.size(); i++)
	{
		
		uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
		uint32_t col = local_tiles[i] % ncolgrps;
		printf("[%d][%d] = %d ", row, col, local_tiles[i]);
	}
    printf("\n");  
	*/
 //}
 
  
  
  
  
  for (auto& tile_r : tiles)
  {
	  for (auto& tile_c: tile_r)
	  {
		tile_c.triples = new std::vector<struct Triple>;
	  }
  }
  
  uint64_t sum = 0;
  Triple triple;

  while (offset < endpos)
  {
    fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
	uint32_t row_idx = triple.row / tile_height;
	uint32_t col_idx = triple.col / tile_width;
	
	
	uint32_t local_idx = (row_idx * ncolgrps) + col_idx;

    //assert(((row_idx * ncolgrps) + col_idx) == 
	if (std::find(local_tiles.begin(), local_tiles.end(), local_idx) != local_tiles.end())
	{
		tiles[row_idx][col_idx].triples->push_back(triple);
		sum++;
	//if(!rank)
	  //std::cout << tile_height << "," << tile_width << "," << "(" << triple.row << "," << triple.col << "):" << row_idx << " " << col_idx << " " << local_idx << std::endl;	
	  assert((row_idx >= 0) and (row_idx < nrowgrps));
	  assert((col_idx >= 0) and (col_idx < ncolgrps));
	  
	}
	
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
  }
  assert(offset == endpos);
  fin.close();
  
 
 
 /*
  int i = 0, j= 0;
  //for(int i = 0; i < nvertices; i++)	  
  for (auto& tile_r : tiles)
  {	  
      j = 0;
	  //for(int j = 0; j < mvertices; j++)
	  for (auto& tile_c: tile_r)
	  {
		  //std::vector<Triple> t = tiles[i][j].triples[k];
		  //for (int k = 0; k < tiles[i][j].triples->size(); k++)
		  for (auto& triple : *(tile_c.triples))
		  {
		  if(!rank);
			printf("%d:[%d][%d]=[%d %d]\n", rank, i, j, triple.row, triple.col);
		  }
		  j++;
	  }
	  i++;
  }
  */

  uint64_t sum1 = 0;
  //std::vector<std::vector<struct Triple>> outboxes(nranks);  
  
  for(uint32_t i = 0; i < local_tiles.size(); i++) 
  {
	uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
	uint32_t col = local_tiles[i] % ncolgrps;
	  sum1 += tiles[row][col].triples->size();
  }
	  
  
  /*
  for (uint32_t i = 0; i < nrowgrps; i++)
  {
	for (uint32_t j = 0; j < ncolgrps; j++)  
	{
	  if(tiles[i][j].rank == rank) {
        printf("tiles[%d][%d]->%lu \n", i, j, tiles[i][j].triples->size());
	  sum1 += tiles[i][j].triples->size();
	  }
	}
  }
  */
  printf("SUM=%lu %lu %d\n", sum, sum1, 4219314 + 3983833 + 4002485 + 4571584);	
  
 
   
 
 
  
  
  

  
  
  
  
  
  
  
//printf("yyyyyyyyyyyyyyyyyyyyyyyyy\n");
  
  

}





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
  	//std::cout << num_vertices << " "<< num_iterations <<  std::endl;
	
	load_binary(filepath, num_vertices, num_vertices);
	
	uint32_t i = 0;
	
	//uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
	//uint32_t col = local_tiles[i] % ncolgrps;
	//sum1 += tiles[row][col].triples->size();
	
	
	printf("rank=%d size=%lu, s=%lu\n", rank, local_tiles.size(), tiles[0][0].triples->size());
	
	
	for (auto& tile_r : tiles)
    {
	  for (auto& tile_c: tile_r)
	  {
		delete tile_c.triples;
		//tile_c.free_triples();
	  }
    }
	
	MPI_Finalize();

	return(0);

}

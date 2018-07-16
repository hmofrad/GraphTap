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
#include <sys/mman.h>
#include <type_traits>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <mpi.h>
#include <sched.h>

int sched_getcpu(void);

int nranks;
int rank;
int cpu_id;
bool is_master;


template <typename Weight>
struct Triple
{
    uint32_t row;
    uint32_t col;
    Weight weight;
    Triple(uint32_t row = 0, uint32_t col = 0, Weight weight = 0)
	    : row(row), col(col), weight(weight) {}  
	uint32_t get_weight() {return(this->weight);};	
    //static bool compare(const struct Triple<Weight>& a, const struct Triple<Weight>& b);
};

//template <typename Weight>
//uint32_t Triple<Weight>::has_weight()
//{
    //return(this.weight);
//}


//template <typename Weight>
//bool Triple<Weight>::compare(const struct Triple<Weight>& a, const struct Triple<Weight>& b)
//{
//	return((a.row == b.row) ? (a.col < b.col) : (a.row < b.row));
//}

struct Empty {};

template <>
struct Triple <Empty>
{
    uint32_t row;
    union {
        uint32_t col;
        Empty weight;
    };
	uint32_t get_weight() {return 1;};
    //static bool compare(const struct Triple<Empty>& a, const struct Triple<Empty>& b);
};

//template <>
//uint32_t Triple<Empty>::has_weight()
//{
    //return -1;
//}

//template <typename Weight>
//uint32_t has_weight(const struct Triple<Weight>& triple)
//{
//    return(triple.weight);
//}



	  


template <typename Weight>
struct Functor
{
	bool operator()(const struct Triple<Weight>& a, const struct Triple<Weight>& b)
	{
	    return((a.row == b.row) ? (a.col < b.col) : (a.row < b.row));
	}
};

//template <>
//bool Triple<Empty>::compare(const struct Triple<Empty>& a, const struct Triple<Empty>& b)
//{
//	return((a.row == b.row) ? (a.col < b.col) : (a.row < b.row));
//}

template<typename Weight>
struct CSR
{
	uint32_t nnz;
	uint32_t nrwos_plus_one;
	uint32_t* A;
	uint32_t* IA;
	uint32_t* JA;
	void free();
};


template<typename Weight>
void CSR<Weight>::free()
{
	munmap(CSR<Weight>::A, (this->nnz) * sizeof(uint32_t));
    munmap(CSR<Weight>::IA, (this->nrwos_plus_one) * sizeof(uint32_t));
    munmap(CSR<Weight>::JA, (this->nnz) * sizeof(uint32_t));	
}


template <typename Weight>
struct Tile2D
{ 
    template <typename Weight_>
    friend class Matrix;
    std::vector<struct Triple<Weight>>* triples;
    struct CSR<Weight>* csr;
    uint32_t rg, cg;
    uint32_t ith, jth, nth;
    int32_t rank;
};

template<typename Weight>
class Matrix
{
	template<typename Weight_>
	friend class Graph;
	
	public:	
        Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles);
        ~Matrix();

	private:
		const uint32_t nrows, ncols;
		const uint32_t ntiles, nrowgrps, ncolgrps;
	    const uint32_t tile_height, tile_width;	

	    std::vector<std::vector<struct Tile2D<Weight>>> tiles;
        std::vector<uint32_t> local_tiles;
		std::vector<struct Tile2D<Weight>> tiles_;
		
		uint32_t local_tile_of_tile(const struct Triple<Weight>& pair);
		struct Triple<Weight> tile_of_triple(const struct Triple<Weight>& triple);
		struct Triple<Weight> tile_of_local_tile(const uint32_t local_tile);
		
		
};

template<typename Weight>
Matrix<Weight>::Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles) 
    : nrows(nrows), ncols(ncols), ntiles(ntiles), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles / nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1) {};

template<typename Weight>
Matrix<Weight>::~Matrix() {};
	  
template <typename Weight>
uint32_t Matrix<Weight>::local_tile_of_tile(const struct Triple<Weight>& pair)
{
  return((pair.row * Matrix<Weight>::ncolgrps) + pair.col);
}	  
	  
template <typename Weight>
struct Triple<Weight> Matrix<Weight>::tile_of_triple(const struct Triple<Weight>& triple)
{
  return{triple.row / Matrix<Weight>::tile_height, triple.col / Matrix<Weight>::tile_width};
}

template <typename Weight>
struct Triple<Weight> Matrix<Weight>::tile_of_local_tile(const uint32_t local_tile)
{
  return{(local_tile - (local_tile % Matrix<Weight>::ncolgrps)) / Matrix<Weight>::ncolgrps, local_tile % Matrix<Weight>::ncolgrps};
}







template<typename Weight>
class Graph
{
	public:	
        Graph();
        ~Graph();
		
		void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_ = true);//uint32_t nvertices, uint32_t mvertices, uint32_t nedges);
		void free();
	private:
	    std::string filepath;
	    uint32_t nvertices;
		uint32_t mvertices;
		uint64_t nedges;
		bool directed;
		Matrix<Weight> *A;
};

template<typename Weight>
Graph<Weight>::Graph() : A(nullptr) {};

template<typename Weight>
Graph<Weight>::~Graph()
{
	delete Graph<Weight>::A;
};


template<typename Weight>
void Graph<Weight>::free()
{
	for (auto& tile_r : Graph<Weight>::A->tiles)
    {
	    for (auto& tile_c: tile_r)
	    {
			if(tile_c.rank == rank)
			{
				tile_c.csr->free();
				//munmap(tile_c.csr->A, (tile_c.csr->nnz) * sizeof(uint32_t));
				//munmap(tile_c.csr->IA, (Graph<Weight>::A->nrows + 1) * sizeof(uint32_t));
			    //munmap(tile_c.csr->JA, (tile_c.csr->nnz) * sizeof(uint32_t));	
			    //delete tile_c.triples;
			}
		}
	}
}






//template <typename Weight>
//struct Triple<Weight> tile_of_local_tile(const struct Triple<Weight> triple, const uint32_t local_tile, const uint32_t ncolgrps)
//{
//  return{(local_tile - (local_tile % ncolgrps)) / ncolgrps, local_tile % ncolgrps};
//}

//template <typename Weight>
//struct Triple<Weight> tile_of_triple(const struct Triple<Weight> triple, const uint32_t tile_height, const uint32_t tile_width)
//{
//  return{triple.row / tile_height, triple.col / tile_width};
//}
  



/*
void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols)
{

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
    struct Triple<uint64_t> header;

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
  
  uint64_t ntriples = (orig_filesize - offset) / sizeof(struct Triple<Weight>);
  share = ((filesize / nranks) / sizeof(struct Triple<Weight>)) * sizeof(struct Triple<Weight>);
  assert(share % sizeof(struct Triple<Weight>) == 0);

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
  uint32_t ntiles = nranks * nranks;
  uint32_t nrowgrps = sqrt(ntiles);
  uint32_t ncolgrps = ntiles / nrowgrps;
  uint32_t tile_height = (nrows / nrowgrps) + 1;
  uint32_t tile_width = (ncols / ncolgrps) + 1;
  
  if(!rank)
	  printf("nrows=%d ncols=%d tiles=%d nrowgrps=%d ncolgrps=%d tile_height=%d tile_width=%d endpos=%lu\n", nrows, ncols, ntiles, nrowgrps, ncolgrps, tile_height, tile_width, endpos);

  
  
  // Reserve the 2D vector of tiles. 
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
          //printf("%d: [%d][%d] %d\n", tiles[i][j].rank, i, j, (i * ncolgrps) + j);
		  }
	  }
	  //printf("\n");
    }
	
	
	//for(uint32_t i = 0; i < local_tiles.size(); i++)
	//{
		
		//uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
		//uint32_t col = local_tiles[i] % ncolgrps;
		//printf("[%d][%d] = %d ", row, col, local_tiles[i]);
	//}
    //printf("\n");  
	
 //}
 
  
  
  
  
  for (auto& tile_r : tiles)
  {
	  for (auto& tile_c: tile_r)
	  {
		tile_c.triples = new std::vector<struct Triple<Weight>>;
	  }
  }
  
  uint64_t sum = 0;
  struct Triple<Weight> triple;
int ii = 0;
  while (offset < endpos)
  {
    fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
	//uint32_t row_idx = triple.row / tile_height;
	//uint32_t col_idx = triple.col / tile_width;
	struct Triple<Weight> pair = tile_of_triple(triple, tile_height, tile_height);
	
	uint32_t local_idx = (pair.row * ncolgrps) + pair.col;
	//uint32_t local_idx = (row_idx * ncolgrps) + col_idx;

    //assert(((row_idx * ncolgrps) + col_idx) == 
	if (std::find(local_tiles.begin(), local_tiles.end(), local_idx) != local_tiles.end())
	{
		//tiles[row_idx][col_idx].triples->push_back(triple);
		tiles[pair.row][pair.col].triples->push_back(triple);
		//sum++;
	    //if(!rank)
		//{
		  //if(ii == 0)
	        //std::cout << tile_height << "," << tile_width << "," << "(" << triple.row << "," << triple.col << "):" << row_idx << " " << col_idx << " " << local_idx << std::endl;	
		//ii++;
		//}
	    
		//assert((row_idx >= 0) and (row_idx < nrowgrps));
	    //assert((col_idx >= 0) and (col_idx < ncolgrps));
	  
	}
	
    if(fin.gcount() != sizeof(Triple<Weight>))
    {
      std::cout << "read() failure" << std::endl;
      exit(1);
    }

    if ((offset & ((1L << 26) - 1L)) == 0)
    {
		;
      //std::cout << "| ";
      //fflush(stdout);
    }
    offset += sizeof(Triple<Weight>);
  }
  assert(offset == endpos);
  fin.close();
  
 
 

  //int i = 0, j= 0;
  //for (auto& tile_r : tiles)
  //{	  
      //j = 0;

	  //for (auto& tile_c: tile_r)
	  //{

		//  for (auto& triple : *(tile_c.triples))
		//  {
	//	  if(!rank);
//			printf("%d:[%d][%d]=[%d %d]\n", rank, i, j, triple.row, triple.col);
		  //}
		//  j++;
	  //}
	//  i++;
  //}
  

  //uint64_t sum1 = 0;
  //std::vector<std::vector<struct Triple>> outboxes(nranks);  
  
  //for(uint32_t i = 0; i < local_tiles.size(); i++) 
  //{
	//uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
	//uint32_t col = local_tiles[i] % ncolgrps;
	  //sum1 += tiles[row][col].triples->size();
  //}
  
	  
  
  
  //for (uint32_t i = 0; i < nrowgrps; i++)
  //{
	//for (uint32_t j = 0; j < ncolgrps; j++)  
	//{
	  //if(tiles[i][j].rank == rank) {
        //printf("tiles[%d][%d]->%lu \n", i, j, tiles[i][j].triples->size());
//	  sum1 += tiles[i][j].triples->size();
	//  }
	//}
  //}
  
  //printf("SUM=%lu %lu %d\n", sum, sum1, 4219314 + 3983833 + 4002485 + 4571584);	
  
 
 //int i = 0, j= 0;
//if(rank == 0)   
//{
   //for (uint32_t i = 0; i < nrowgrps; i++)
  //{
	//for (uint32_t j = 0; j < ncolgrps; j++)  
	//{
		//std::cout << tiles[i][j].rank << "|";
	//}
	//std::cout << "\n";
  //}
  //uint64_t s = 0;
  //for(uint32_t t: local_tiles)
  //{
	//uint32_t row = (t - (t % ncolgrps)) / ncolgrps;
    //uint32_t col = t % ncolgrps;
	//printf("%d %d %d %lu\n", tile, row, col, tiles[row][col].triples->size());
	//auto& tile = tiles[row][col];
	
	//for (auto& triple : *(tile.triples))
    //{
		//s++;
	    //printf("%d:[%d][%d]=[%d %d]\n", rank, row, col, triple.row, triple.col);
    //}
	
	
    //for (std::vector<Triple>::iterator it = tiles[row][col].triples.begin() ; it != tiles[row][col].triples.end(); ++it)
      //std::cout << ' ' << *it;
    //std::cout << '\n';
	
	//std::vector<Triple*> t = tiles[row][col].triples;//->data();
	//if(tiles[row][col].triples->size())
	//{
	  //printf("%d %d\n", tiles[row][col].triples->front().row, tiles[row][col].triples->front().col);
	//}
	//for(Triple& t: (tiles[row][col].triples))
	//{
		//;
		//printf("%d %d\n", t.row, t.col);
	//}
  //}
  //printf("degree=%lu\n", s);
  
  for(uint32_t t: local_tiles)
  {

	struct Triple<Weight> pair = tile_of_local_tile(triple, t, ncolgrps);
	auto& tile = tiles[pair.row][pair.col];
	tile.csr = new struct CSR;
	tile.csr->nnz = tile.triples->size();
	tile.csr->A = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	tile.csr->IA = (uint32_t*) mmap(nullptr, (nrows + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	tile.csr->JA = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	
	if(!rank)
	{
	for (auto& triple : *(tile.triples))
    {
	    printf("%d:[%d][%d]=[%d %d]\n", rank, pair.row, pair.col, triple.row, triple.col);
    }
	}
    uint32_t i = 0;
    uint32_t j = 1;
    uint32_t nextRow = 0;
    tile.csr->IA[0] = 0;
    for (auto& triple : *(tile.triples))
    {
      while(j < triple.row)
	  {
	    j++;
	    tile.csr->IA[j] = tile.csr->IA[j - 1];
	  }

      tile.csr->A[i] = 1;
	  tile.csr->IA[j]++;
	  tile.csr->JA[i] = triple.col;	
	  i++;
    }
  
    // Not necessary
    while(j < (nrows + 1))
    {
  	  j++;
      tile.csr->IA[j] = tile.csr->IA[j - 1];
    }

	if(!rank)
	{
	printf("csc:%d:[%d][%d]:\n", rank, pair.row, pair.col);
	for(i = 0; i < tile.csr->nnz; i++)
    {
	  printf("%d ", tile.csr->A[i]);
    }
    printf("\n");
    for(i = 0; i < (ncols + 1); i++)
    {
	  printf("%d ", tile.csr->IA[i]);
    }
    printf("\n");
	
    for(i = 0; i < tile.csr->nnz; i++)
    {
	  printf("%d ", tile.csr->JA[i]);
    }
    printf("\n");
	
  }

  
 }
  
}
*/
template<typename Weight>
void Graph<Weight>::load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_)
{
	// Initialize graph data
	// Note we keep using Graph<Weight> format to avoid confusion
	Graph<Weight>::filepath = filepath_;
	Graph<Weight>::nvertices = nrows;
	Graph<Weight>::mvertices = ncols;
	Graph<Weight>::nedges = 0;
	Graph<Weight>::directed = directed_;
	
    // Open matrix file.
    std::ifstream fin(Graph<Weight>::filepath.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        std::cout << "Unable to open input file" << std::endl;
        exit(1); 
    }


    // Obtain filesize
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
	fin.seekg(0, std::ios_base::beg);
	
	Graph<Weight>::A = new Matrix<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
	
	
	// Reserve the 2D vector of tiles. 
	Graph<Weight>::A->tiles.resize(Graph<Weight>::A->nrowgrps);
    for (uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
        Graph<Weight>::A->tiles[i].resize(Graph<Weight>::A->ncolgrps);

	struct Triple<Weight> pair;
    for (uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
    {
	    for (uint32_t j = 0; j < Graph<Weight>::A->ncolgrps; j++)  
	    {
			/*
			auto& tile = Graph<Weight>::A->tiles[i][j];
			tile.rg = i;
            tile.cg = j;
	        tile.rank = i;
            if(tile.rank == rank)
			{
		        pair.row = i;
		        pair.col = j;	
			    Graph<Weight>::A->local_tiles.push_back(Graph<Weight>::A->local_tile_of_tile(pair));
			}
			*/
	        Graph<Weight>::A->tiles[i][j].rg = i;
            Graph<Weight>::A->tiles[i][j].cg = j;
	        Graph<Weight>::A->tiles[i][j].rank = i;
		
			if(Graph<Weight>::A->tiles[i][j].rank == rank)
			{
		        pair.row = i;
		        pair.col = j;	
			    Graph<Weight>::A->local_tiles.push_back(Graph<Weight>::A->local_tile_of_tile(pair));
			}
	    }
    }
 
 /*
    for (uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
    {
	  for (uint32_t j = 0; j < Graph<Weight>::A->ncolgrps; j++)  
	  {
		  pair.row = i;
		  pair.col = j;
		  if(Graph<Weight>::A->tiles[i][j].rank == rank)
		  {
	        if(!rank)
		    {
		        printf(">>>[%d %d]\n", i, j);	
		    }
			//pair = Graph<Weight>::A->tile_of_local_tile(i);
			
		    //Graph<Weight>::A->local_tiles.push_back((i * Graph<Weight>::A->ncolgrps) + j);
			Graph<Weight>::A->local_tiles.push_back(Graph<Weight>::A->local_tile_of_tile(pair));
		  }
	  }
    }
	*/
	
	/*
	for(uint32_t i = 0; i < Graph<Weight>::A->local_tiles.size(); i++)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(i);
		if(!rank)
		{
		    printf("[%d %d]\n", pair.row, pair.col);	
		}
		Graph<Weight>::A->tiles[pair.row][pair.col].triples = new std::vector<struct Triple<Weight>>;
		
		///std::vector<struct Triple<Weight> triple> = Graph<Weight>::A->tiles[pair.row][pair.col].triples;
		//std::vector<struct Triple<Weight>>* triples;
		// = new std::vector<struct Triple<Weight>>;
	}
	*/
	
	
    /*
    for (auto& tile_r : Graph<Weight>::A->tiles)
    {
		
	    for (auto& tile_c: tile_r)
	    {
			if(tile_c.rank == rank)
			{
		    tile_c.triples = new std::vector<struct Triple<Weight>>;
			}
	    }
    }
	*/
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
	    pair = Graph<Weight>::A->tile_of_local_tile(t);
		auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		tile.triples = new std::vector<struct Triple<Weight>>;
	}
	
	
	struct Triple<Weight> triple;
	while (offset < filesize)
    {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
		/*
		if(!rank)
	    {
		    printf("%d %d %lu\n", triple.row, triple.col, sizeof(triple));
	    }
		*/
	    if(fin.gcount() != sizeof(Triple<Weight>))
        {
            std::cout << "read() failure" << std::endl;
            exit(1);
        }
		Graph<Weight>::nedges++;
		
	
	    pair = Graph<Weight>::A->tile_of_triple(triple);
		//uint32_t local_idx = Graph<Weight>::A->local_tile_of_tile(pair);
		//if(!rank)
		//{
		//    printf("%d %d %d %d\n", pair.row, pair.col,  local_idx, Graph<Weight>::A->tiles[pair.row][pair.col]);
		//}
	    //uint32_t local_idx = (pair.row * Graph<Weight>::A->nrowgrps) + pair.col;
		
		//if (std::find(Graph<Weight>::A->local_tiles.begin(), Graph<Weight>::A->local_tiles.end(), local_idx) != Graph<Weight>::A->local_tiles.end())
		if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)	
	    {
			/*
			if(!rank)
			{
			  printf("%d %d\n", pair.row, pair.col);
			}
			*/
		    Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
    	}
        offset += sizeof(Triple<Weight>);
    }
    assert(offset == filesize);
    fin.close();
	printf("done reading %lu\n", Graph<Weight>::nedges);	
	  
    // Creat the csr format
    // Allocate csr data structure
    // Sort triples
    // Populate csr 	
    for(uint32_t t: Graph<Weight>::A->local_tiles)
    {
	    pair = Graph<Weight>::A->tile_of_local_tile(t);
    	auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
	    tile.csr = new struct CSR<Weight>;
    	tile.csr->nnz = tile.triples->size();
		tile.csr->nrwos_plus_one = Graph<Weight>::A->nrows + 1;
	    tile.csr->A = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    	tile.csr->IA = (uint32_t*) mmap(nullptr, (tile.csr->nrwos_plus_one) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	    tile.csr->JA = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	/*
    	if(!rank)
	    {
	        for (auto& triple : *(Graph<Weight>::A->tiles[pair.row][pair.col].triples))
            {
	            printf("%d:[%d][%d]:  %d %d\n", rank, pair.row, pair.col, triple.row, triple.col);
    		}
	    }
		*/
	    Functor<Weight> f;
		std::sort(tile.triples->begin(), tile.triples->end(), f);
	    //std::sort(tile.triples->begin(), tile.triples->end(), struct Triple<Weight>::compare);
        uint32_t i = 0;
        uint32_t j = 1;
        tile.csr->IA[0] = 0;
        for (auto& triple : *(tile.triples))
        {
            while(j < triple.row)
	        {
	            j++;
	            tile.csr->IA[j] = tile.csr->IA[j - 1];
	        }			
			tile.csr->A[i] = triple.get_weight(); // In case weights are important
	        tile.csr->IA[j]++;
	        tile.csr->JA[i] = triple.col;	
	        i++;
        }
  
        // Not necessary
        while(j < (Graph<Weight>::A->nrows + 1))
        {
       	   j++;
           tile.csr->IA[j] = tile.csr->IA[j - 1];
        }

		/*
	if(!rank)
	{
	printf("csc:%d:[%d][%d]:\n", rank, pair.row, pair.col);
	for(i = 0; i < tile.csr->nnz; i++)
    {
	  printf("%d ", tile.csr->A[i]);
    }
    printf("\n");
    for(i = 0; i < (ncols + 1); i++)
    {
	  printf("%d ", tile.csr->IA[i]);
    }
    printf("\n");
	
    for(i = 0; i < tile.csr->nnz; i++)
    {
	  printf("%d ", tile.csr->JA[i]);
    }
    printf("\n");
	
	//struct Triple<uint32_t> triple1 = {10, 5, 4};
	//printf("%d %d %d\n", triple1.row, triple1.col, triple1.get_weight());
	//if (std::is_same<uint32_t, Empty>::value) 
	//{
	    //printf("%d %d %lu\n", triple1.row, triple1.col, sizeof(triple1));
	//}
	//else
	//{
		//printf(">> %d %d %d %lu\n", triple1.row, triple1.col, (uint32_t) triple1.weight, sizeof(triple1));
	//}
	//struct Triple<uint32_t> triple1 = {5, 4};
	//struct Triple<uint32_t> triple2 = {6 ,2};
	//printf("CMPPP = %d %lu\n",compare(triple1, triple2), sizeof(triple1) );
	
    }
	*/
	
    }
	
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
	    pair = Graph<Weight>::A->tile_of_local_tile(t);
    	auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		delete tile.triples;
		
	}
	
	
	
	
	
	
	/*
	for(uint32_t i = 0; i < Graph<Weight>::A->local_tiles.size(); i++)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(i);
		//if(!rank)
		  // printf("size=%lu %d [%d %d]\n", Graph<Weight>::A->tiles[pair.row][pair.col].triples->size(), rank, pair.row, pair.col);
		//if(Graph<Weight>::A->tiles[pair.row][pair.col].triples->size())
		delete Graph<Weight>::A->tiles[pair.row][pair.col].triples;
		//printf("size=%d\n", Graph<Weight>::A->tiles[pair.row][pair.col].triples->size());
	}
	*/
	/*
	for (auto& tile_r : Graph<Weight>::A->tiles)
    {
	    for (auto& tile_c: tile_r)
	    {
			if(tile_c.rank == rank)
			{
				munmap(tile_c.csr->A, (tile_c.csr->nnz) * sizeof(uint32_t));
				munmap(tile_c.csr->IA, (Graph<Weight>::A->nrows + 1) * sizeof(uint32_t));
			    munmap(tile_c.csr->JA, (tile_c.csr->nnz) * sizeof(uint32_t));	
			    delete tile_c.triples;
			}
		}
	}
	*/
	
	
	
	
	
	
}


using ew_t = Empty;
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

	std::string file_path = argv[1]; 
	uint32_t num_vertices = std::atoi(argv[2]);
  	uint32_t num_iterations = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
	bool directed = true;
	Graph<ew_t> G;
	G.load_binary(file_path, num_vertices, num_vertices, directed);
	printf("done load binary\n");
	G.free();
	
	

	
	
	
	//delete Graph<Weight>::A;
  	//std::cout << num_vertices << " "<< num_iterations <<  std::endl;
	
	//Matrix<Empty>* M = new Matrix<Empty>(num_vertices, num_vertices, nranks * nranks);
	//if(!rank)
		//M.gelrows()
	   
	
	
	
	
	//load_binary(filepath, num_vertices, num_vertices);
	
	//uint32_t i = 0;
	
	//uint32_t row = (local_tiles[i] - (local_tiles[i] % ncolgrps)) / ncolgrps;
	//uint32_t col = local_tiles[i] % ncolgrps;
	//sum1 += tiles[row][col].triples->size();
	
	
	//printf("rank=%d size=%lu, s=%lu\n", rank, local_tiles.size(), tiles[0][0].triples->size());
	
	
	
	
	

	
	MPI_Finalize();

	return(0);

}

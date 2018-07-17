/*
 * Test app 
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>

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
	void set_weight(Weight& w) {this->weight = w;};
	Weight get_weight() {return(this->weight);};	
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
	void set_weight(Empty& w) {};
	bool get_weight() {return 0;};
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
        void load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_ = true);
		
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
	fin.close();
    assert(offset == filesize);
    
	printf("done reading %lu\n", Graph<Weight>::nedges);	
	  
    // Creat the csr format
    // Allocate csr data structure
	// Sort triples and populate csr	
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




template<typename Weight>
void Graph<Weight>::load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_)
{
	// Initialize graph data
	// Note we keep using Graph<Weight> format to avoid confusion
	Graph<Weight>::filepath = filepath_;
	Graph<Weight>::nvertices = nrows;
	Graph<Weight>::mvertices = ncols;
	Graph<Weight>::nedges = 0;
	Graph<Weight>::directed = directed_;
	
	
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
 
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
	    pair = Graph<Weight>::A->tile_of_local_tile(t);
		auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		tile.triples = new std::vector<struct Triple<Weight>>;
	}
	

    // Open matrix file.
    std::ifstream fin(Graph<Weight>::filepath.c_str());
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
	
	// Skip comments
    std::string line;
    uint32_t position; // Fallback position
    do
	{
        position = fin.tellg();
        std::getline(fin, line);
    } while ((line[0] == '#') || (line[0] == '%'));
    fin.seekg(position, std::ios_base::beg);

	
	
	
	
	
	//printf("%d\n", position);
	//bool weighted = false;
	
	//std::getline(fin, line);
	//if((std::count(line.cbegin(), line.cend(), ' ') + 1) == 3) // src dst wd
	//{
	//	weighted = true;
	//}
	//fin.seekg(position, std::ios_base::beg);
	struct Triple<Weight> triple;
	std::istringstream iss;
	//struct Triple<Weight> triple1;
	//uint32_t i, j;
    //Weight wd;
	//Weight *p = &wd;
	//p = NULL;
    while (std::getline(fin, line) && !line.empty())
    {
		iss.clear();
        iss.str(line);
		
		if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
		{
            std::cout << "read() failure" << std::endl;
            exit(1);
        }
		
		iss >> triple.row >> triple.col;
		//if(weighted)
		//{
        //    iss >> triple1.row >> triple1.col  >> p; 
		//	triple1.set_weight(*p);
		//}
		//else
		//	iss >> triple1.row >> triple1.col;
		// Solely uncoment this for debug purpose 
		//if(!rank)
        //std::cout << "(i,j,w)=" << "(" << triple.row << "," << triple.col << "," << fin.tellg() << std::endl;// << "," << triple.get_weight() << ")" << line.empty() << std::endl;
	    
		
		Graph<Weight>::nedges++;
		
	
	    pair = Graph<Weight>::A->tile_of_triple(triple);
		//if(!rank)
			//std::cout << "(i,j)=" << "(" << pair.row << "," << pair.col << std::endl;
		if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)	
	    {
		    Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
    	}
		offset = fin.tellg();
	}
	
	//current = fin.tellg();
	//printf("%lu %lu\n", offset, filesize);
	fin.close();
	assert(offset == filesize);

	
    // Creat the csr format
    // Allocate csr data structure
    // Sort triples and populate csr
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
    }
	*/
	
    }
	
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
	    pair = Graph<Weight>::A->tile_of_local_tile(t);
    	auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		delete tile.triples;
		
	}
	
		
	
	
    
	
	//MPI_Finalize();
	//exit(0);
	

	
	
	
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
	//G.load_binary(file_path, num_vertices, num_vertices, directed);
	G.load_text(file_path, num_vertices, num_vertices, directed);
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

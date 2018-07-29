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

/*
struct delete_ptr { // Helper function to ease cleanup of container
    template <typename P>
    void operator () (P p) {
        delete p;
    }
};
*/ 

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
	bool get_weight() {return 1;};
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
struct Functor1
{
	bool operator()(const struct Triple<Weight>& a, const struct Triple<Weight>& b)
	{
	    return(((a.row == b.row) and (a.col == b.col)) ? true : false);
	}
};	  


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
	uint32_t nrows_plus_one;
	uint32_t* A;
	uint32_t* IA;
	uint32_t* JA;
	void allocate(uint32_t nnz, uint32_t nrows_plus_one);
	void free();
};

template<typename Weight>
void CSR<Weight>::allocate(uint32_t nnz, uint32_t nrows_plus_one)
{		
    CSR<Weight>::nnz = nnz;
	CSR<Weight>::nrows_plus_one = nrows_plus_one;
	CSR<Weight>::A = (uint32_t*) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	memset(CSR<Weight>::A, 0, CSR<Weight>::nnz);
    CSR<Weight>::IA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nrows_plus_one) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	memset(CSR<Weight>::IA, 0, CSR<Weight>::nrows_plus_one);
	CSR<Weight>::JA = (uint32_t*) mmap(nullptr, (CSR<Weight>::nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	memset(CSR<Weight>::JA, 0, CSR<Weight>::nnz);
}		

template<typename Weight>
void CSR<Weight>::free()
{
	munmap(CSR<Weight>::A, (this->nnz) * sizeof(uint32_t));
    munmap(CSR<Weight>::IA, (this->nrows_plus_one) * sizeof(uint32_t));
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

enum Tiling
{
  _1D_ROW,
  _1D_COL,
  _2D_
};


template<typename Weight>
class Partitioning
{	
    template<typename Weight_>
	friend class Matrix;

	template<typename Weight__>
	friend class Graph;
	
	public:	
		Partitioning(uint32_t nranks, uint32_t rank, uint32_t ntiles, uint32_t nrowgrps, uint32_t ncolgrps, Tiling tiling);
        ~Partitioning();
	
	private:
	    const uint32_t nranks, rank;
	    const uint32_t ntiles, nrowgrps, ncolgrps;
	    const Tiling tiling;
		
		uint32_t rank_ntiles, rank_nrowgrps, rank_ncolgrps;
		uint32_t rowgrp_nranks, colgrp_nranks;
		
		void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
		//void free();
		
};
//Graph<Weight>::A->partitioning = new Partitioning<Weight>(nranks, rank, nranks * nranks, Graph<Weight>::A->nrowgrps, Graph<Weight>::A->ncolgrps, tiling)
template<typename Weight>
Partitioning<Weight>::Partitioning(uint32_t nranks, uint32_t rank, uint32_t ntiles, uint32_t nrowgrps, uint32_t ncolgrps, Tiling tiling) 
    : nranks(nranks), rank(rank), ntiles(ntiles), nrowgrps(nrowgrps), ncolgrps(ncolgrps), rank_ntiles(ntiles / nranks), tiling(tiling)
{
	assert(Partitioning<Weight>::rank_ntiles * Partitioning<Weight>::nranks == ntiles);
    if (Partitioning<Weight>::tiling == Tiling::_1D_ROW)
    {
        Partitioning<Weight>::rowgrp_nranks = 1;
        Partitioning<Weight>::colgrp_nranks = nranks;
        assert(Partitioning<Weight>::rowgrp_nranks * Partitioning<Weight>::colgrp_nranks == Partitioning<Weight>::nranks);

        Partitioning<Weight>::rank_nrowgrps = 1;
        Partitioning<Weight>::rank_ncolgrps = Partitioning<Weight>::ncolgrps;
        assert(Partitioning<Weight>::rank_nrowgrps * Partitioning<Weight>::rank_ncolgrps == Partitioning<Weight>::rank_ntiles);
	}		
	else if (Partitioning<Weight>::tiling == Tiling::_2D_)
	{
		integer_factorize(Partitioning<Weight>::nranks, Partitioning<Weight>::rowgrp_nranks, Partitioning<Weight>::colgrp_nranks);
        assert(Partitioning<Weight>::rowgrp_nranks * Partitioning<Weight>::colgrp_nranks == Partitioning<Weight>::nranks);
    
        Partitioning<Weight>::rank_nrowgrps = Partitioning<Weight>::nrowgrps / Partitioning<Weight>::colgrp_nranks;
        Partitioning<Weight>::rank_ncolgrps = Partitioning<Weight>::ncolgrps / Partitioning<Weight>::rowgrp_nranks;
        assert(Partitioning<Weight>::rank_nrowgrps * Partitioning<Weight>::rank_ncolgrps == Partitioning<Weight>::rank_ntiles);
	}
};

template<typename Weight>
Partitioning<Weight>::~Partitioning()
{
	//Partitioning<Weight>::free();
};

//template<typename Weight>
//Partitioning<Weight>::free()
//{
	//delete ;
//};


template <class Weight>
void Partitioning<Weight>::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b)
{
  /* This approach is adapted from that of GraphPad. */
  a = b = sqrt(n);
  while (a * b != n)
  {
    b++;
    a = n / b;
  }
  assert(a * b == n);
}



template<typename Weight>
class Matrix
{
	template<typename Weight_>
	friend class Graph;
	
	public:	
        Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles, Tiling tiling);
        ~Matrix();

	private:
		const uint32_t nrows, ncols;
		const uint32_t ntiles, nrowgrps, ncolgrps;
	    const uint32_t tile_height, tile_width;	
		
		Partitioning<Weight>* partitioning;

	    std::vector<std::vector<struct Tile2D<Weight>>> tiles;
        std::vector<uint32_t> local_tiles;
		std::vector<uint32_t> local_segments;
		std::vector<uint32_t> local_col_segments;
		std::vector<uint32_t> local_row_segments;
		
		void init_mat();
		void del_triples();
		void init_csr();
		void del_csr();
		
		uint32_t local_tile_of_tile(const struct Triple<Weight>& pair);
		uint32_t segment_of_tile(const struct Triple<Weight>& pair);
		struct Triple<Weight> tile_of_triple(const struct Triple<Weight>& triple);
		struct Triple<Weight> tile_of_local_tile(const uint32_t local_tile);
		struct Triple<Weight>  rebase(const struct Triple<Weight>& pair);
		struct Triple<Weight>  base(const struct Triple<Weight>& pair);
		
		//std::vector<uint32_t> segments_of_local_tile();
};

template<typename Weight>
Matrix<Weight>::Matrix(uint32_t nrows, uint32_t ncols, uint32_t ntiles, Tiling tiling) 
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
  return{(triple.row / Matrix<Weight>::tile_height), (triple.col / Matrix<Weight>::tile_width)};
}

template <typename Weight>
struct Triple<Weight> Matrix<Weight>::tile_of_local_tile(const uint32_t local_tile)
{
  return{(local_tile - (local_tile % Matrix<Weight>::ncolgrps)) / Matrix<Weight>::ncolgrps, local_tile % Matrix<Weight>::ncolgrps};
}

template <typename Weight> 
uint32_t Matrix<Weight>::segment_of_tile(const struct Triple<Weight>& pair)
{
	return(pair.col);
}
template <typename Weight> 
struct Triple<Weight>  Matrix<Weight>::rebase(const struct Triple<Weight>& pair)
{
	return{(pair.row % Matrix<Weight>::tile_height), (pair.col % Matrix<Weight>::tile_width)};
}

/*
template <typename Weight> 
std::vector<uint32_t> Matrix<Weight>::segments_of_local_tile()
{
	//if(!rank)
		//printf(":: ");
	//int i = 0;
	std::vector<uint32_t> local_segments;
	for(uint32_t t: Matrix<Weight>::local_tiles)
	{
		Triple<Weight> pair = Matrix<Weight>::tile_of_local_tile(t);
		if (std::find(local_segments.begin(), local_segments.end(), pair.row) == local_segments.end())
		{
			local_segments.push_back(pair.row);
		}
		
	}
	return(local_segments);
	//if(!rank)
	//printf("\n");
    
}
*/
template<typename Weight>
void Matrix<Weight>::del_triples()
{
	// Delete triples
	Triple<Weight> pair;
	for(uint32_t t: Matrix<Weight>::local_tiles)
	{
	    pair = Matrix<Weight>::tile_of_local_tile(t);
    	auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		//std::for_each(tile.triples->begin(), tile.triples->end(), delete_ptr());
        //tile.triples->clear();
	    //for(auto& triple: *(tile.triples))	
		    //delete triple;		
		delete tile.triples;		
	}
	
}

template <typename Weight> 
void Matrix<Weight>::del_csr()
{
	Triple<Weight> pair;
	for(uint32_t t: Matrix<Weight>::local_tiles)
	{
		pair = Matrix<Weight>::tile_of_local_tile(t);
		auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		tile.csr->free();
		delete tile.csr;
	}
}

template<typename Weight>
struct Segment
{
	template<typename Weight_>
	friend class Vector;
	
	uint32_t* data;	
	uint32_t n;
    uint32_t nrows, ncols;
	uint32_t rg, cg;
	uint32_t rank;
	std::vector<std::vector<uint32_t>> other_ranks;
	void allocate();
	void free();
};

template<typename Weight>
void Segment<Weight>::allocate()
{
	Segment<Weight>::data = (uint32_t*) mmap(nullptr, (this->n) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	memset(Segment<Weight>::data, 0, this->n);
}

template<typename Weight>
void Segment<Weight>::free()
{
	munmap(Segment<Weight>::data, (this->n) * sizeof(uint32_t));
}





template<typename Weight>
class Vector
{
	template<typename Weight_>
	friend class Graph;
	
	public:
	    Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles);
	    ~Vector();
	
	private:
		const uint32_t nrows, ncols;
		const uint32_t nrowgrps, ncolgrps;
	    const uint32_t tile_height, tile_width;	// == segment_height
	
        std::vector<struct Segment<Weight>> segments;
	    std::vector<uint32_t> local_segments;
		
		void init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments);
};

template<typename Weight>
Vector<Weight>::Vector(uint32_t nrows, uint32_t ncols, uint32_t ntiles) 
    : nrows(nrows), ncols(ncols), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles/nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1) {};
	

template<typename Weight>
Vector<Weight>::~Vector() {};

template<typename Weight>
class Graph
{
	public:	
        Graph();
        ~Graph();
		
		void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_ = true);
        void load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_ = true);

	private:
	    std::string filepath;
	    uint32_t nvertices;
		uint32_t mvertices;
		uint64_t nedges;
		bool directed;
		Matrix<Weight>* A;
		Vector<Weight>* X;
		Vector<Weight>* Y;
		Vector<Weight>* V;
		
		void spmv();
		void free();
};

template<typename Weight>
Graph<Weight>::Graph() : A(nullptr) {};

template<typename Weight>
Graph<Weight>::~Graph()
{
	delete Graph<Weight>::A->partitioning;
	delete Graph<Weight>::A;
	
	delete Graph<Weight>::X;
	delete Graph<Weight>::Y;
	delete Graph<Weight>::V;
	
}


template<typename Weight>
void Graph<Weight>::free()
{
	Graph<Weight>::A->del_csr();
	
	for(uint32_t s: Graph<Weight>::X->local_segments)
	{
		auto& segment = Graph<Weight>::X->segments[s];
		segment.free();
	}
	
	for(uint32_t s: Graph<Weight>::Y->local_segments)
	{
		auto& segment = Graph<Weight>::Y->segments[s];
		segment.free();
	}
	
	for(uint32_t s: Graph<Weight>::V->local_segments)
	{
		auto& segment = Graph<Weight>::V->segments[s];
		segment.free();
	}
	
	
	
	/*
	for (auto& tile_r : Graph<Weight>::A->tiles)
    {
	    for (auto& tile_c: tile_r)
	    {
			if(tile_c.rank == rank)
			{
				tile_c.csr->free();
				delete tile_c.csr;
			}
		}
	}
	*/
	//for (auto& vec : Graph<Weight>::X->segments)
		//;
/*	
	for (auto& vec : Graph<Weight>::A->tiles)
    {
	    for (auto& tile_c: tile_r)
	    {
			if(tile_c.rank == rank)
			{
				auto& vec = Graph<Weight>::X->segments[
				vec.segments->free();
				break;
			}
		}
*/
		
	
	//for (auto& vec : Graph<Weight>::X->segments)
	//{
	    //vec.segments->free();
	//}
}


template<typename Weight>
void Graph<Weight>::load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_)
{
	// Initialize graph data
	// Note we keep using Graph<Weight> format to avoid confusion
	Graph<Weight>::filepath = filepath_;
	Graph<Weight>::nvertices = nrows;
	Graph<Weight>::mvertices = ncols;
	Graph<Weight>::nedges = 0;
	Graph<Weight>::directed = directed_;
	
	Graph<Weight>::A = new Matrix<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, tiling);
	Graph<Weight>::A->partitioning = new Partitioning<Weight>(nranks, rank, nranks * nranks, Graph<Weight>::A->nrowgrps, Graph<Weight>::A->ncolgrps, tiling);
	//if(!rank)
	//	printf("rank_ntiles = %d\n", Graph<Weight>::A->partitioning->rank_ntiles);
		
	Graph<Weight>::A->init_mat();
	
	std::vector<uint32_t> diag_ranks(Graph<Weight>::A->nrowgrps, -1);
	for(uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
	{
		diag_ranks[i] = Graph<Weight>::A->tiles[i][i].rank;
	}
	
	
	std::vector<std::vector<uint32_t>> other_ranks;
	other_ranks.resize(Graph<Weight>::A->partitioning->rank_nrowgrps);
	//rowgrp_nranks.resize(Graph<Weight>::A->partitioning->colgrp_nranks - 1);
	//for(uint32_t i = 0; i < Graph<Weight>::A->partitioning->rowgrp_nranks; i++)
		//rowgrp_nranks[i].resize(Graph<Weight>::A->partitioning->rowgrp_nranks - 1);
	
	
	struct Triple<Weight> pair;
	uint32_t i = 0;
	uint32_t prev_row = -1;
	if(!rank)
	{	
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(t);
		//if(prev_row != -1)
		//    prev_row = pair.row;
		
		//uint32_t i = pair.row;
		//uint32_t j = pair.col;
		
		//printf("[%d %d %d %d %d %d %d %d %d]\n", prev_row, pair.row, pair.col, i, prev_row, Graph<Weight>::A->partitioning->rank_nrowgrps, Graph<Weight>::A->partitioning->rank_ncolgrps, Graph<Weight>::A->partitioning->rowgrp_nranks, Graph<Weight>::A->partitioning->colgrp_nranks);
		if(prev_row != pair.row)
		{
            for(uint32_t j = 0; j < Graph<Weight>::A->ncolgrps; j++)
	    	{
		    	if((pair.row != j) and (Graph<Weight>::A->tiles[pair.row][j].rank != rank))
			    {
				    if (std::find(other_ranks[i].begin(), other_ranks[i].end(), Graph<Weight>::A->tiles[pair.row][j].rank) == other_ranks[i].end())
    				{
	    				other_ranks[i].push_back(Graph<Weight>::A->tiles[pair.row][j].rank);
						//printf("PUSH %d %d\n", i, Graph<Weight>::A->tiles[pair.row][j].rank);
		    		}
			    }
		    }
			i++;
			/*
			for(uint32_t ii; ii < Graph<Weight>::A->partitioning->rowgrp_nranks - 1; ii++)
		    {
			    printf("%d\n", rowgrp_nranks[i - 1][ii]);
		    }
			*/
		}
		prev_row = pair.row;
	    
	    
		
		
		/*
		for(uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
		{
			if (std::find(rowgrp_nranks[i].begin(), rowgrp_nranks[i].end(), Graph<Weight>::A->tiles[i][j].rank) == rowgrp_nranks[i].end())
			{
				
			}
		}
		*/
		
		
	}


    /*
	for(uint32_t i = 0; i < Graph<Weight>::A->nrowgrps; i++)
	{
		//rowgrp_nranks.push_back(Graph<Weight>::A->tiles[i][j].rank);
		uint32_t k = 0;
		for(uint32_t j = 0; j < Graph<Weight>::A->ncolgrps; j++)
		{
			if(i != j)
			{
				//rowgrp_nranks[i][k].push_back(Graph<Weight>::A->tiles[i][j].rank);
		        if (std::find(rowgrp_nranks[i].begin(), rowgrp_nranks[i].end(), Graph<Weight>::A->tiles[i][j].rank) == rowgrp_nranks[i].end())
		        {
					printf("%d %d %d\n", i,j,k);
				   // rowgrp_nranks[i][k] = Graph<Weight>::A->tiles[i][j].rank;
					k++;
		        }
			}
		}
	}
    
	*/
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
	    if(fin.gcount() != sizeof(Triple<Weight>))
        {
            std::cout << "read() failure" << std::endl;
            exit(1);
        }
		
	    pair = Graph<Weight>::A->tile_of_triple(triple);
		Graph<Weight>::nedges++;	

		if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)	
	    {
		    Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
    	}
        offset += sizeof(Triple<Weight>);
    }
	fin.close();
    assert(offset == filesize);
	
	/*
    if(!rank)
	{
		for(uint32_t t: Graph<Weight>::A->local_tiles)
		{
			pair = Graph<Weight>::A->tile_of_local_tile(t);
			auto& triples = *(Graph<Weight>::A->tiles[pair.row][pair.col].triples);
			for(auto& tt: triples)
				printf("%d[%d][%d]:%d %d\n", rank, pair.row, pair.col, tt.row, tt.col);
			
		}
	}
	*/
	
	if(!rank)
	{
		printf("[x]Reading\n");
	}
	
	
	Graph<Weight>::A->init_csr();
	
	
	
	
	
	Graph<Weight>::X = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
	Graph<Weight>::Y = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
	Graph<Weight>::V = new Vector<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks);
	
	
	//std::vector<uint32_t> local_segments = Graph<Weight>::A->segments_of_local_tile();
    Graph<Weight>::X->init_vec(diag_ranks, Graph<Weight>::A->local_col_segments);
	Graph<Weight>::Y->init_vec(diag_ranks, Graph<Weight>::A->local_row_segments);
	Graph<Weight>::V->init_vec(diag_ranks, Graph<Weight>::A->local_segments); // CHANGE THIS
	
	

	
	
	Graph<Weight>::spmv();
	
	Graph<Weight>::free();
	
	if(!rank)
	{
		printf("[x]SPMV\n");
	}
	
	
	//Graph<Weight>::X->local_segments = Graph<Weight>::A->segments_of_local_tile();
	//Graph<Weight>::Y->local_segments = (Graph<Weight>::A->segments_of_local_tile);
	//Graph<Weight>::V->local_segments = (Graph<Weight>::A->segments_of_local_tile);
	
	//std::copy(Graph<Weight>::A->segments_of_local_tile.begin(), Graph<Weight>::A->segments_of_local_tile.end(), std::back_inserter(startPop));

	//Graph<Weight>::X->local_segments = Graph<Weight>::A->segments_of_local_tile;
	//Graph<Weight>::Y->local_segments = Graph<Weight>::A->segments_of_local_tile;
	//Graph<Weight>::V->local_segments = Graph<Weight>::A->segments_of_local_tile;
	

	

/*	
	for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
    {
		Graph<Weight>::X->n ;
			Graph<Weight>::X->allocate();
			Matrix<Weight>::tiles[i][i].rank;
	}
	
		uint32_t* segment;	
    uint32_t n;
	uint32_t rg;
	uint32_t rank;
	void allocate();
	void free();
	*/
	
	
	//Graph<Weight>::A->spmv();
	//Graph<Weight>::free();
	
	/*
	if(!rank)
	{
		for(uint32_t t: Graph<Weight>::A->local_tiles)
		{
			pair = Graph<Weight>::A->tile_of_local_tile(t);
			auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
			
			printf("csc:%d:[%d][%d]\nA=", rank, pair.row, pair.col);
			for(uint32_t i = 0; i < tile.csr->nnz; i++)
			{
			  printf("%d ", tile.csr->A[i]);
			}
			printf("\nIA=");
			for(uint32_t i = 0; i < (tile.csr->nrows_plus_one); i++)
			{
			  printf("%d ", tile.csr->IA[i]);
			}
			printf("\nJA=");
			
			for(uint32_t i = 0; i < tile.csr->nnz; i++)
			{
			  printf("%d ", tile.csr->JA[i]);
			}
			printf("\n");
		}
	}
	*/
	
}


template<typename Weight>
void Vector<Weight>::init_vec(std::vector<uint32_t>& diag_ranks, std::vector<uint32_t>& local_segments)
{
	// Reserve the 1D vector of segments. 
	Vector<Weight>::segments.resize(Vector<Weight>::ncolgrps);
	//Vector<Weight>::local_segments;
	//local_segments

	for (uint32_t i = 0; i < Vector<Weight>::ncolgrps; i++)
    {
		Vector<Weight>::segments[i].n = Vector<Weight>::tile_height;
		Vector<Weight>::segments[i].nrows = Vector<Weight>::tile_height;
		Vector<Weight>::segments[i].ncols = Vector<Weight>::tile_width;
		Vector<Weight>::segments[i].rg = i;
		Vector<Weight>::segments[i].cg = i;
		Vector<Weight>::segments[i].rank = diag_ranks[i];
	}
	
	Vector<Weight>::local_segments = local_segments;
	
	for(uint32_t s: Vector<Weight>::local_segments)
	{
		Vector<Weight>::segments[s].allocate();
	}

	/*
	if(!rank)
	{
	for(uint32_t s: Vector<Weight>::local_segments)
	{
		printf("%d ", s);
	}
	printf("\n");
	}
	*/
	
}


template<typename Weight>
void Graph<Weight>::load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, Tiling tiling, bool directed_)
{
	
	// Initialize graph data
	// Note we keep using Graph<Weight> format to avoid confusion
	Graph<Weight>::filepath = filepath_;
	Graph<Weight>::nvertices = nrows;
	Graph<Weight>::mvertices = ncols;
	Graph<Weight>::nedges = 0;
	Graph<Weight>::directed = directed_;
	
	Graph<Weight>::A = new Matrix<Weight>(Graph<Weight>::nvertices, Graph<Weight>::mvertices, nranks * nranks, tiling);
	Graph<Weight>::A->partitioning = new Partitioning<Weight>(nranks, rank, nranks * nranks, Graph<Weight>::A->nrowgrps, Graph<Weight>::A->ncolgrps, tiling);
	Graph<Weight>::A->init_mat();
	
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

	struct Triple<Weight> triple;
	struct Triple<Weight> pair;
	std::istringstream iss;
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
		Graph<Weight>::nedges++;
	
	    pair = Graph<Weight>::A->tile_of_triple(triple);
		if(Graph<Weight>::A->tiles[pair.row][pair.col].rank == rank)	
	    {
		    Graph<Weight>::A->tiles[pair.row][pair.col].triples->push_back(triple);
    	}
		offset = fin.tellg();
	}
	fin.close();
	assert(offset == filesize);
	
	/*
	if(!rank)
	{
		for(uint32_t t: Graph<Weight>::A->local_tiles)
		{
			pair = Graph<Weight>::A->tile_of_local_tile(t);
			auto& triples = *(Graph<Weight>::A->tiles[pair.row][pair.col].triples);
			for(auto& tt: triples)
				printf("%d[%d][%d]:%d %d\n", rank, pair.row, pair.col, tt.row, tt.col);
			
		}
	}
	*/
	
	//Graph<Weight>::A->init_csr();
	//Graph<Weight>::A->spmv();
	//Graph<Weight>::free();
	/*
	if(!rank)
	{
		for(uint32_t t: Graph<Weight>::A->local_tiles)
		{
			pair = Graph<Weight>::A->tile_of_local_tile(t);
			auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
			
			printf("csc:%d:[%d][%d]:\n", rank, pair.row, pair.col);
			for(uint32_t i = 0; i < tile.csr->nnz; i++)
			{
			  printf("%d ", tile.csr->A[i]);
			}
			printf("\n");
			for(uint32_t i = 0; i < (tile.csr->nrows_plus_one); i++)
			{
			  printf("%d ", tile.csr->IA[i]);
			}
			printf("\n");
			
			for(uint32_t i = 0; i < tile.csr->nnz; i++)
			{
			  printf("%d ", tile.csr->JA[i]);
			}
			printf("\n");
		}
	}
	*/
}

template<typename Weight>
void Matrix<Weight>::init_mat()
{	

	// Reserve the 2D vector of tiles. 
	Matrix<Weight>::tiles.resize(Matrix<Weight>::nrowgrps);
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
        Matrix<Weight>::tiles[i].resize(Matrix<Weight>::ncolgrps);
    
	
    for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
    {
	    for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
	    {
	        Matrix<Weight>::tiles[i][j].rg = i;
            Matrix<Weight>::tiles[i][j].cg = j;
			if(Matrix<Weight>::partitioning->tiling == Tiling::_1D_ROW)
			{
	            Matrix<Weight>::tiles[i][j].rank = i;
				Matrix<Weight>::tiles[i][j].ith = Matrix<Weight>::tiles[i][j].rg / Matrix<Weight>::partitioning->colgrp_nranks;
                Matrix<Weight>::tiles[i][j].jth = Matrix<Weight>::tiles[i][j].cg / Matrix<Weight>::partitioning->rowgrp_nranks;
				Matrix<Weight>::tiles[i][j].nth = Matrix<Weight>::tiles[i][j].ith * Matrix<Weight>::partitioning->rank_ncolgrps + Matrix<Weight>::tiles[i][j].jth;
			}
			else if(Matrix<Weight>::partitioning->tiling == Tiling::_2D_)
			{
				Matrix<Weight>::tiles[i][j].rank = (j % Matrix<Weight>::partitioning->rowgrp_nranks) * Matrix<Weight>::partitioning->colgrp_nranks
            			                         + (i % Matrix<Weight>::partitioning->colgrp_nranks);
				Matrix<Weight>::tiles[i][j].ith = Matrix<Weight>::tiles[i][j].rg / Matrix<Weight>::partitioning->colgrp_nranks;
                Matrix<Weight>::tiles[i][j].jth = Matrix<Weight>::tiles[i][j].cg / Matrix<Weight>::partitioning->rowgrp_nranks;
				Matrix<Weight>::tiles[i][j].nth = Matrix<Weight>::tiles[i][j].ith * Matrix<Weight>::partitioning->rank_ncolgrps + Matrix<Weight>::tiles[i][j].jth;
			}
		
			//if(Matrix<Weight>::tiles[i][j].rank == rank)
			//{
		    //    pair.row = i;
		    //    pair.col = j;	
			//    Matrix<Weight>::local_tiles.push_back(Matrix<Weight>::local_tile_of_tile(pair));
			//}
	    }
    }

	std::vector<uint32_t> diag_ranks(Matrix<Weight>::nrowgrps, -1);
	for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
	{
		for (uint32_t j = i; j < Matrix<Weight>::ncolgrps; j++)  
		{
			if(not (std::find(diag_ranks.begin(), diag_ranks.end(), Matrix<Weight>::tiles[j][i].rank) != diag_ranks.end()))
			{
				std::swap(Matrix<Weight>::tiles[j], Matrix<Weight>::tiles[i]);
				diag_ranks[j] = Matrix<Weight>::tiles[j][i].rank;
				break;
			}
			
			
			//printf("%d ", Matrix<Weight>::tiles[i][j].rank);
		}
		//printf("\n");
	}
		
	struct Triple<Weight> pair;
	for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
	{
		for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
		{
			Matrix<Weight>::tiles[i][j].rg = i;
			Matrix<Weight>::tiles[i][j].cg = j;
			if(Matrix<Weight>::tiles[i][j].rank == rank)
			{
		        pair.row = i;
		        pair.col = j;	
			    Matrix<Weight>::local_tiles.push_back(Matrix<Weight>::local_tile_of_tile(pair));
				/*
				if (std::find(local_segments.begin(), local_segments.end(), pair.col) == local_segments.end())
		        {
			        local_segments.push_back(pair.col);
			    }	
				*/
				if (std::find(Matrix<Weight>::local_row_segments.begin(), Matrix<Weight>::local_row_segments.end(), pair.row) == local_row_segments.end())
		        {
			        Matrix<Weight>::local_row_segments.push_back(pair.row);
			    }	
				
				if (std::find(Matrix<Weight>::local_col_segments.begin(), Matrix<Weight>::local_col_segments.end(), pair.col) == local_col_segments.end())
		        {
			        Matrix<Weight>::local_col_segments.push_back(pair.col);
			    }	
				
			}
		}
	}
	
	// Initialize triples 
	for(uint32_t t: Matrix<Weight>::local_tiles)
	{
	    pair = Matrix<Weight>::tile_of_local_tile(t);
		auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		tile.triples = new std::vector<struct Triple<Weight>>;
	}
		
	if(!rank)
	{	
		for (uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
        {
	        for (uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)  
	        {
				printf("%02d ", Matrix<Weight>::tiles[i][j].rank);
		    }
			printf("\n");
	    }
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
}

template<typename Weight>
void Matrix<Weight>::init_csr()
{
	// Create the csr format
    // Allocate csr data structure
    // Sort triples and populate csr
	struct Triple<Weight> pair;
    for(uint32_t t: Matrix<Weight>::local_tiles)
    {

	    pair = Matrix<Weight>::tile_of_local_tile(t);
    	auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		tile.csr = new struct CSR<Weight>;
		//tile.csr->allocate(tile.triples->size(), Matrix<Weight>::nrows + 1);
		tile.csr->allocate(tile.triples->size(), Matrix<Weight>::tile_height + 1);
		
		//if(!rank)
		//{/
			//printf("local=%d %d %d\n", t, tile.triples->size(), (Matrix<Weight>::nrows + 1));
		//}		
		/*
	    tile.csr = new struct CSR<Weight>;
		
    	tile.csr->nnz = tile.triples->size();
		tile.csr->nrows_plus_one = Matrix<Weight>::nrows + 1;
		
	    tile.csr->A = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
		memset(tile.csr->A, 0, tile.csr->nnz);
    	tile.csr->IA = (uint32_t*) mmap(nullptr, (tile.csr->nrows_plus_one) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
		memset(tile.csr->IA, 0, tile.csr->nrows_plus_one);
	    tile.csr->JA = (uint32_t*) mmap(nullptr, (tile.csr->nnz) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	    memset(tile.csr->JA, 0, tile.csr->nnz);
		*/
	    Functor<Weight> f;
		std::sort(tile.triples->begin(), tile.triples->end(), f);
		
		// *** DO NOT DELETE
		//Functor1<Weight> f1;
		//tile.triples->erase(std::unique(tile.triples->begin(), tile.triples->end(), f1), tile.triples->end());

/*
	if(!rank)
	{
		for(auto& tt: *(tile.triples))
		    printf("%d[%d][%d]:%d %d\n", rank, pair.row, pair.col, tt.row, tt.col);			
		printf("\n\n");
	}
*/

MPI_Barrier(MPI_COMM_WORLD);
//if(!rank)
//{
	//printf(">>>Creating csr\n");


		
        uint32_t i = 0;
        uint32_t j = 1;
        tile.csr->IA[0] = 0;
        for (auto& triple : *(tile.triples))
        {
			pair = rebase(triple);
			/*
			if(!rank and t == 14 and triple.row > 1048508)
            {
            	printf("t=%d,j=%d,height=%d,row/h=(%d,%d),(%d %d)(%d %d)\n", t, j, Matrix<Weight>::tile_height, (triple.row % Matrix<Weight>::tile_height), (triple.col % Matrix<Weight>::tile_width)
				, triple.row, triple.col, pair.row, pair.col);
            }
			*/
            while((j - 1) != pair.row)
			//while((j - 1) != (triple.row % Matrix<Weight>::tile_height))
	        {
	            j++;
	            tile.csr->IA[j] = tile.csr->IA[j - 1];
	        }			
			tile.csr->A[i] = triple.get_weight(); // In case weights are implemented
	        tile.csr->IA[j]++;
	        tile.csr->JA[i] = pair.col;	
	        i++;
        }
		
        //if(!rank)
		//{
//			printf("after=%d\n", j);
	//	}
		
        // Not necessary
        while(j < (Matrix<Weight>::tile_height))
        {
       	   j++;
           tile.csr->IA[j] = tile.csr->IA[j - 1];
        }
}
   // }	
	
	Matrix<Weight>::del_triples();
	/*
	// Delete triples
	for(uint32_t t: Matrix<Weight>::local_tiles)
	{
	    pair = Matrix<Weight>::tile_of_local_tile(t);
    	auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		//std::for_each(tile.triples->begin(), tile.triples->end(), delete_ptr());
        //tile.triples->clear();
	    //for(auto& triple: *(tile.triples))	
		    //delete triple;		
		delete tile.triples;		
	}
	*/

}

template<typename Weight>
void Graph<Weight>::spmv()
{
	Triple<Weight> pair;
	if(!rank)
	{
		/*
		for(uint32_t t: Graph<Weight>::A->local_tiles)
		{
			pair = Graph<Weight>::A->tile_of_local_tile(t);
		    auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
			
			uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
			auto& x_seg = Graph<Weight>::X->segments[s];
			uint32_t n = x_seg.n;
			
			printf("t=%d(nnz=%d,nrow+1=%d,s=%d,n=%d)\n", t, tile.csr->nnz, tile.csr->nrows_plus_one, s, n);
		}
		printf("\n");
		*/
		/*
		for(uint32_t s: Graph<Weight>::A->local_segments)
		{
			auto& x_seg = Graph<Weight>::X->segments[s];
			auto& y_seg = Graph<Weight>::Y->segments[s];
			uint32_t n = x_seg.n;
			printf("(%d %d)", s, n);
		}
		printf("\n");
		*/
	}
//if(!rank)
//{	
    //uint32_t i = 0, j = 0, k = 0 ,l = 0;
	
	uint32_t x_c = 0, y_r = 0, y_r_old = 0;
	// Data initialization
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(t);
		auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
		x_c = pair.col;
        y_r = pair.row;
		
		auto& x_seg = Graph<Weight>::X->segments[x_c];
		auto& y_seg = Graph<Weight>::Y->segments[y_r];
		
		uint32_t nitems = x_seg.n;
		for(uint32_t i = 0; i < nitems; i++)
		{
			x_seg.data[i] = 1;
			y_seg.data[i] = 0;
		}
		
	}	
	
	
	// Data processing 
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(t);
		auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
		x_c = pair.col;
		y_r = pair.row;
		
		/*
		if(y_r != y_r_old)
		{
		    y_r_old = y_r;
			if(!rank)
			{
				printf("Communication + Computation %d %d\n", tile.nth, tile.ith);
			}
		}
		else
		{
			if(!rank)
			{
				printf("Computation %d %d\n", tile.nth, tile.ith);
			}
		}
		*/
       //if(!rank)
		//printf("(%d %d)\n", Graph<Weight>::A->tiles[pair.row][pair.col].rg, Graph<Weight>::A->tiles[pair.row][pair.col].cg);
		
		auto& x_seg = Graph<Weight>::X->segments[x_c];
		auto& y_seg = Graph<Weight>::Y->segments[y_r];
		
		//uint32_t nrows = x_seg.ncols;
		/*
		for(uint32_t i = 0; i < x_seg.ncols; i++)
		{
			x_seg.data[i] = 1;
			//y_seg.data[i] = 0;
		}
		
		for(uint32_t i = 0; i < y_seg.nrows; i++)
		{
			y_seg.data[i] = 0;
		}
		*/
		
		
		//if(!rank)
		//{
		//printf("T[%d][%d] S[%d]\n", pair.row, pair.col, s);
		
		//while(i < tile.csr->nnz)
		//for(i = 0; i < tile.csr->nnz; i++)
		uint32_t k = 0;
		for(uint32_t i = 0; i < tile.csr->nrows_plus_one - 1; i++)
		{
			uint32_t nnz_per_row = tile.csr->IA[i + 1] - tile.csr->IA[i];
			for(uint32_t j = 0; j < nnz_per_row; j++)
			{
				//if(!rank)
				//printf("TILE[%d][%d]:%d: A[%d]=%d, JA[%d]=%d, X[%d]=%d, Y[%d]=%d \n", pair.row, pair.col, i, k, tile.csr->A[k], k, tile.csr->JA[k], i, x_seg.data[i], i, y_seg.data[i]);
			
				y_seg.data[i] += tile.csr->A[k] * x_seg.data[i];
				k++;
			}
		}
		
		bool communication = ((tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks == 0);
		
		if((rank == 0 or rank == 4 or  rank == 8 or  rank == 12) and pair.row == 0)
		{
		if(communication)
		{
			if(y_seg.rank == rank)
			{
				// MPI_Recv
				printf("receive(%d)\n", rank);
			}
			else
			{
				// MPI_Send
				// int MPI_Send(y_seg.data, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
				
				printf("send(%d) to %d\n", rank, y_seg.rank);
			}
		}
		}
		
		
		if(!rank)
		{
		    printf("[%d %d] %d %d %d %d %d %d\n",x_c, y_r, tile.ith, tile.jth,  tile.nth, Graph<Weight>::A->partitioning->colgrp_nranks, (tile.nth + 1) % Graph<Weight>::A->partitioning->colgrp_nranks, communication);	
		}
		
		
	}

	
	
	
	/*
	uint32_t v = 0;
	for(uint32_t t: Graph<Weight>::A->local_tiles)
	{
		pair = Graph<Weight>::A->tile_of_local_tile(t);
		auto& tile = Graph<Weight>::A->tiles[pair.row][pair.col];
		uint32_t s = Graph<Weight>::A->segment_of_tile(pair);
		auto& y_seg = Graph<Weight>::Y->segments[s];
		if(y_seg.rank == rank)
		{
			for(uint32_t i = 0; i < y_seg.ncols; i++)
		    {
				v += y_seg.data[i];
			}
		}			
		else
		{
           ;
			
		}
		
				
	}
	
	if(!rank)
		printf("SUM=%d\n", v);
	*/
	//Vector<Weight>::segments[i].rank
	
//}
	
	/*
	for(uint32_t s: Graph<Weight>::A->local_segments)
	{
		
		auto& x_seg = Graph<Weight>::X->segments[s];
		auto& y_seg = Graph<Weight>::Y->segments[s];
		uint32_t n = x_seg.n;
		for(uint32_t i = 0; i < n; i++)
		{
			x_seg.data[i] = 1;
			y_seg.data[i] = 0;
		}
				
		
	}
	*/
	//printf("\n");
	
	
	//Vector<Weight>::segments[i].n
	
	
	//for(uint32_t t: Graph<Weight>::X->local_segments)
	//{
	//	Graph<Weight>::X
		
	//}
	

	
	
	/*
	
	uint32_t X[Matrix<Weight>::nrows]; 
	for(uint32_t i = 0; i < Matrix<Weight>::nrows; i++)
	{
		X[i] = i;
	}
	uint32_t Y[Matrix<Weight>::nrows] = {0};  
	
	struct Triple<Weight> pair;
    for(uint32_t t: Matrix<Weight>::local_tiles)
	{
	    pair = Matrix<Weight>::tile_of_local_tile(t);
		auto& tile = Matrix<Weight>::tiles[pair.row][pair.col];
		if(!rank)
		{
			printf("[%d][%d]\n", pair.row, pair.col);
			uint32_t i = 0, j = 0, k = 0 ,l = 0;
			for(i = 0; i < tile.csr->nrows_plus_one - 1; i++)
			{
			   // printf("%d[ ", tile.csr->IA[i + 1] - tile.csr->IA[i]);
				if(tile.csr->IA[i + 1] - tile.csr->IA[i])
					printf("%d ", i);				
				for(j = 0; j < tile.csr->IA[i + 1] - tile.csr->IA[i]; j++)
				{
					printf("(%d %d %d) ", tile.csr->A[k], tile.csr->JA[k], tile.csr->A[k] * X[tile.csr->JA[k]]);
					Y[tile.csr->JA[k]] = tile.csr->A[k] * X[tile.csr->JA[k]];
					k++;
					if(j + 1 == tile.csr->IA[i + 1] - tile.csr->IA[i])
					    printf(" \n");
				}

				
			}
			//printf("\n", );
		}
	}
	if(!rank)
	{
		for(uint32_t i = 0; i < Matrix<Weight>::nrows; i++)
		    printf("%d ", Y[i]);
		printf("\n");
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
	G.load_binary(file_path, num_vertices, num_vertices, _2D_, directed);
	//G.load_text(file_path, num_vertices, num_vertices, _1D_ROW, directed);
	//G.A->spmv();
	
	
	if(!rank)
	{
		printf("[x]MAIN\n");
	}
	
	//G.free();
	
	
	

	
	
	
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


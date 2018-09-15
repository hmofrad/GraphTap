/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP
 
#include <cmath>
#include <algorithm>
#include <vector>
#include "tiling.hpp" 
#include "vector.hpp" 
#include "types.hpp" 

enum Filtering_type
{
  _SRCS_, // Rows
  _SNKS_, // Columns
  _BOTH_,
  _NONE_,
  _SOME_
}; 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D
{ 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight, Integer_Type>> *triples;
    struct CSR<Weight, Integer_Type> *csr;
    struct CSC<Weight, Integer_Type> *csc;
    struct CSR<Weight, Integer_Type> *tcsr;
    struct CSC<Weight, Integer_Type> *tcsc;
    uint32_t rg, cg; // Row group, Column group
    uint32_t ith, jth, nth, mth, kth; // ith row, jth column, nth local row order tile, mth local column order tile, and kth global tile
    uint32_t rank;
    uint32_t leader_rank_rg, leader_rank_cg;
    uint32_t rank_rg, rank_cg;
    uint32_t leader_rank_rg_rg, leader_rank_cg_cg;
    uint64_t nedges;
    Integer_Type nsources, nsinks, nisolated, nregular;
    Integer_Type nnz_col;
    bool allocated;
    
    void allocate_triples();
    void free_triples();
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Tile2D<Weight, Integer_Type, Fractional_Type>::allocate_triples()
{
    if (!triples)
        triples = new std::vector<Triple<Weight, Integer_Type>>;
}
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Tile2D<Weight, Integer_Type, Fractional_Type>::free_triples()
{
    triples->clear();
    triples->shrink_to_fit();
    delete triples;
    triples = nullptr;
}
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Matrix
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Vertex_Program;
    
    public:    
        Matrix(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, 
               Tiling_type tiling_type_, Compression_type compression_type_, Filtering_type filtering_type_, bool parread_);
        ~Matrix();

    private:
        Integer_Type nrows, ncols;
        uint32_t ntiles, nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width;    
        
        Tiling *tiling;
        Compression_type compression_type;
        bool parread;
        //Vector<Weight, Integer_Type, Integer_Type> *E; //empty columns
        //Vector<Weight, Integer_Type, Integer_Type> *I; // Indices
        Filtering_type filtering_type;
        Vector<Weight, Integer_Type, Integer_Type> *R = nullptr;  // Non-empty rows
        Vector<Weight, Integer_Type, char> *I = nullptr; // Row indices
        Vector<Weight, Integer_Type, Integer_Type> *IV = nullptr; // Row indices values
        
        Vector<Weight, Integer_Type, Integer_Type> *C = nullptr;  // Non-empty columns
        Vector<Weight, Integer_Type, char> *J = nullptr; // Column indices
        Vector<Weight, Integer_Type, Integer_Type> *JV = nullptr; // Column indices values


        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles;
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles_rg;
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles_cg;
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;
        std::vector<int32_t> local_row_segments;
        std::vector<int32_t> local_col_segments;
        
        
        std::vector<uint32_t> leader_ranks;
        
        std::vector<uint32_t> leader_ranks_rg;
        std::vector<uint32_t> leader_ranks_cg;
        
        std::vector<int32_t> all_rowgrp_ranks;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        std::vector<int32_t> all_colgrp_ranks; 
        std::vector<int32_t> all_colgrp_ranks_accu_seg;
        
        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        std::vector<int32_t> follower_colgrp_ranks; 
        std::vector<int32_t> follower_colgrp_ranks_accu_seg;
        
        
        std::vector<int32_t> all_rowgrp_ranks_rg;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> all_colgrp_ranks_cg;
        std::vector<int32_t> all_colgrp_ranks_accu_seg_cg;
        
        std::vector<int32_t> follower_rowgrp_ranks_rg;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> follower_colgrp_ranks_cg;
        std::vector<int32_t> follower_colgrp_ranks_accu_seg_cg;
        
        int32_t owned_segment, accu_segment_rg, accu_segment_cg, accu_segment_row, accu_segment_col;
        // In case of owning multiple segments
        std::vector<int32_t> owned_segment_vec;
        std::vector<int32_t> accu_segment_rg_vec;
        std::vector<int32_t> accu_segment_cg_vec;
        std::vector<int32_t> accu_segment_row_vec;
        std::vector<int32_t> accu_segment_col_vec;
        
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        
        void free_tiling();
        void init_matrix();
        void del_triples();
        void init_tiles();
        void init_compression();
        void init_csr();
        void init_csc();
        void init_tcsr();
        void init_tcsc();
        void init_bv();
        void del_csr();
        void del_csc();
        void del_dcsr();
        void del_compression();
        void print(std::string element);
        void distribute();
        void filter(Filtering_type filtering_type_);
        void debug(int what_rank);
        void init_filtering();
        
        
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type> &triple);
        uint32_t local_tile_of_triple(const struct Triple<Weight, Integer_Type> &triple);
        
        uint32_t segment_of_tile(const struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type> &pair);
        void insert(const struct Triple<Weight, Integer_Type> &triple);
        void test(const struct Triple<Weight, Integer_Type> &triple);
        
        std::vector<int32_t> sort_indices(const std::vector<int32_t> &v);
        void indexed_sort(std::vector<int32_t> &v1, std::vector<int32_t> &v2);        
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, Tiling_type tiling_type_, 
    Compression_type compression_type_, Filtering_type filtering_type_, bool parread_)
{
    nrows = nrows_;
    ncols = ncols_;
    ntiles = ntiles_;
    nrowgrps = sqrt(ntiles_);
    ncolgrps = ntiles_ / nrowgrps;
    tile_height = (nrows_ / nrowgrps) + 1;
    tile_width = (ncols_ / ncolgrps) + 1;
    parread = parread_;
    
    // Initialize tiling 
    tiling = new Tiling(Env::nranks, ntiles, nrowgrps, ncolgrps, tiling_type_);
    compression_type = compression_type_;
    filtering_type = filtering_type_;
    init_matrix();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::free_tiling()
{
    delete tiling;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::~Matrix()
{
    //delete tiling;
    //if(!Env::rank)
      //  printf("Delete matrix\n");
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::tile_of_local_tile(const uint32_t local_tile)
{
    return{(local_tile - (local_tile % ncolgrps)) / ncolgrps, local_tile % ncolgrps};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::tile_of_triple(const struct Triple<Weight, Integer_Type> &triple)
{
    return{(triple.row / tile_height), (triple.col / tile_width)};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::segment_of_tile(const struct Triple<Weight, Integer_Type> &pair)
{
    return(pair.col);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::base(const struct Triple<Weight, Integer_Type> &pair, 
                      Integer_Type rowgrp, Integer_Type colgrp)
{
   return{(pair.row + (rowgrp * tile_height)), (pair.col + (colgrp * tile_width))};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::rebase(const struct Triple<Weight, Integer_Type> &pair)
{
    return{(pair.row % tile_height), (pair.col % tile_width)};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::test(const struct Triple<Weight, Integer_Type> &triple)
{        
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    uint32_t t = (pair.row * tiling->ncolgrps) + pair.col;
    if(not (std::find(local_tiles.begin(), local_tiles.end(), t) != local_tiles.end()))
    {
        //fprintf(stderr, "Local tiles[r=%d]: ", Env::rank);
        //for(uint32_t i : local_tiles)
        //    fprintf(stderr, " %d ", i);
        //fprintf(stderr, "\n");
        printf("Invalid[r=%d,t=%d]: Tile[%d][%d] [%d %d]\n", Env::rank, t, pair.row, pair.col, triple.row, triple.col);
        fprintf(stderr, "Invalid[r=%d,t=%d]: Tile[%d][%d] [%d %d]\n", Env::rank, t, pair.row, pair.col, triple.row, triple.col);
        Env::exit(1);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::local_tile_of_triple(const struct Triple<Weight, Integer_Type> &triple)
{
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    uint32_t t = (pair.row * tiling->ncolgrps) + pair.col;
    return(t);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::insert(const struct Triple<Weight, Integer_Type> &triple)
{
    /*
    if(triple.row >= nrows)
        printf("%d (triple.row %d < tile_height %d)\n", Env::rank, triple.row, nrows);
    if(triple.col >= ncols)
        printf("%d (triple.col %d < tile_width %d)\n", Env::rank, triple.col, ncols);
    
    assert(triple.row < nrows);
    assert(triple.col < ncols);
    */
    
    /*
    if(pair.row >= nrowgrps)
        printf("%d (pair.row %d< nrowgrps %d)\n", Env::rank, pair.row, nrowgrps);
    
    if(pair.col >= ncolgrps)
        printf("%d (pair.col %d < ncolgrps %d)\n", Env::rank, pair.col, ncolgrps);
    
    assert(pair.row < nrowgrps);
    assert(pair.col < ncolgrps);
    
    if(!Env::rank)
    {
        Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
        
        for(uint32_t t : local_tiles)
            printf("%d %d\n", t, tile);
    }
    */
    
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    /*
    if(!parread)
    {
        uint32_t t = (pair.row * tiling->ncolgrps) + pair.col;
        if(not (std::find(local_tiles.begin(), local_tiles.end(), t) != local_tiles.end()))
        {
            fprintf(stderr, "R[%d]-Invalid[t=%d]: Tile[%d][%d] [%d %d]\n", Env::rank, t, pair.row, pair.col, triple.row, triple.col);
            Env::exit(1);
        }
    }
    */
    tiles[pair.row][pair.col].triples->push_back(triple);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
std::vector<int32_t> Matrix<Weight, Integer_Type, Fractional_Type>::sort_indices(const std::vector<int32_t> &v) 
{
    std::vector<int32_t> idx(v.size());
    for( int i = 0; i < v.size(); i++ )
        idx[i] = i;
    //std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&v](int32_t i1, int32_t i2) {return v[i1] < v[i2];});
  return idx;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::indexed_sort(std::vector<int32_t> &v1, std::vector<int32_t> &v2) 
{
    // Sort v2 based on v1 order
    std::vector<int32_t> idx = sort_indices(v1);
    std::sort(v1.begin(), v1.end());
    int32_t max = *std::max_element(v2.begin(), v2.end());
    std::vector<int32_t> temp(max + 1);
    int32_t i = 0;
    for(int32_t j: idx)
    {
        temp[v2[j]] = i;
        i++;
    }
    std::sort(v2.begin(), v2.end(),[&temp](int32_t i1, int32_t i2) {return temp[i1] < temp[i2];});           
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_matrix()
{
    // Reserve the 2D vector of tiles. 
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++)
        tiles[i].resize(ncolgrps);
    
    // Initialize tiles 
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            tile.rg = i;
            tile.cg = j;
            if(tiling->tiling_type == Tiling_type::_1D_ROW)
            {
                tile.rank = i;
                tile.ith  = tile.rg / tiling->colgrp_nranks;
                tile.jth  = tile.cg / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = i;
                tile.leader_rank_cg = j;
                
                tile.leader_rank_rg_rg = j;
                tile.leader_rank_cg_cg = i;
            }
            else if(tiling->tiling_type == Tiling_type::_1D_COL)
            {
                tile.rank = j;
                tile.ith  = tile.rg / tiling->colgrp_nranks;
                tile.jth  = tile.cg / tiling->rowgrp_nranks;

                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = j;
                tile.leader_rank_cg = i;
                
                tile.leader_rank_rg_rg = j;
                tile.leader_rank_cg_cg = i;
            }
            else if(tiling->tiling_type == Tiling_type::_2D_)
            {
                tile.rank = ((j % tiling->rowgrp_nranks) * tiling->colgrp_nranks) +
                                   (i % tiling->colgrp_nranks);
                tile.ith = tile.rg   / tiling->colgrp_nranks;
                tile.jth = tile.cg   / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = i;
                tile.leader_rank_cg = j;
                
                tile.leader_rank_rg_rg = i;
                tile.leader_rank_cg_cg = j;
            }
            else if(tiling->tiling_type == Tiling_type::_NUMA_)
            {
                
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks
                                                        + (j % tiling->rowgrp_nranks);
                
                tile.ith = tile.rg / tiling->colgrp_nranks; 
                tile.jth = tile.cg / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = i;
                tile.leader_rank_cg = j;
                
                tile.leader_rank_rg_rg = i;
                tile.leader_rank_cg_cg = j;
            }
            tile.nth   = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.mth   = (tile.jth * tiling->rank_nrowgrps) + tile.ith;
            tile.allocate_triples();
            //tile.triples = new std::vector<Triple<Weight, Integer_Type>>;
            tile.allocated = false;
        }
    }
    //print("rank");
    
    //print("rank_cg");
    
    //Env::barrier();
    //Env::exit(1);
    /*
    * Reorganize the tiles so that each rank is placed in
    * at least one diagonal tile then calculate 
    * the leader ranks per row group.
    */
    leader_ranks.resize(nrowgrps, -1);
    leader_ranks_rg.resize(nrowgrps);
    leader_ranks_cg.resize(ncolgrps);
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = i; j < ncolgrps; j++)  
        {
            if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[j][i].rank)
                 != leader_ranks.end()))
            {
                std::swap(tiles[j], tiles[i]);
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
        leader_ranks_rg[i] = tiles[i][i].rank_rg;
        leader_ranks_cg[i] = tiles[i][i].rank_cg;
    }
    
    //Calculate local tiles and local column segments
    struct Triple<Weight, Integer_Type> pair;
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            tile.rg = i;
            tile.cg = j;
            tile.kth   = (tile.rg * tiling->ncolgrps) + tile.cg;
            if(tile.rank == Env::rank)
            {
                pair.row = i;
                pair.col = j;    
                local_tiles.push_back(tile.kth);
                local_tiles_row_order.push_back(tile.kth);
                if (std::find(local_col_segments.begin(), local_col_segments.end(), pair.col) == local_col_segments.end())
                {
                    local_col_segments.push_back(pair.col);
                }
                
                if (std::find(local_row_segments.begin(), local_row_segments.end(), pair.row) == local_row_segments.end())
                {
                    local_row_segments.push_back(pair.row);
                }
            }
            tile.leader_rank_rg = tiles[i][i].rank;
            tile.leader_rank_cg = tiles[j][j].rank;
                
            tile.leader_rank_rg_rg = tiles[i][i].rank_rg;
            tile.leader_rank_cg_cg = tiles[j][j].rank_cg;
            if((tile.rank == Env::rank) and (i == j))
            {
                owned_segment = i;
                owned_segment_vec.push_back(owned_segment);
            }
        }
    }
    
    for (uint32_t j = 0; j < ncolgrps; j++)
    {
        for (uint32_t i = 0; i < nrowgrps; i++)  
        {
            auto &tile = tiles[i][j];
            if(tile.rank == Env::rank)
            {
                local_tiles_col_order.push_back(tile.kth);
            }
        }
    }
    
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        if(pair.row == pair.col)
        {            
            for(uint32_t j = 0; j < ncolgrps; j++)
            {
                if(tiles[pair.row][j].rank == Env::rank) 
                {
                    if(std::find(all_rowgrp_ranks.begin(), all_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                              == all_rowgrp_ranks.end())
                    {
                        all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                        all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                
                        all_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                        all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                    }
                }
                else
                {
                    if(std::find(follower_rowgrp_ranks.begin(), follower_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                            == follower_rowgrp_ranks.end())
                    {
                        all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                        all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                        
                        all_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                        all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                        
                        follower_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                        follower_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                        
                        follower_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                        follower_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                    }
                }
            }
            
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                if(tiles[i][pair.col].rank == Env::rank) 
                {
                    if(std::find(all_colgrp_ranks.begin(), all_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                              == all_colgrp_ranks.end())
                    {
                        all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                        all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
            
                        all_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                        all_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                    }
                }
                else
                {
                    if(std::find(follower_colgrp_ranks.begin(), follower_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                              == follower_colgrp_ranks.end())
                    {
                        all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                        all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                        
                        all_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                        all_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                        
                        follower_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                        follower_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                        
                        follower_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                        follower_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                    }
                }
            }
            break;
            /* We do not keep iterating as the ranks in row/col groups are the same */
        }
    }
    

    // Optimization: Spilitting communicator among row/col groups       
    indexed_sort(all_rowgrp_ranks, all_rowgrp_ranks_accu_seg);
    indexed_sort(all_rowgrp_ranks_rg, all_rowgrp_ranks_accu_seg_rg);
    // Make sure there is at least one follower
    if(follower_rowgrp_ranks.size() > 1)
    {
        indexed_sort(follower_rowgrp_ranks, follower_rowgrp_ranks_accu_seg);
        indexed_sort(follower_rowgrp_ranks_rg, follower_rowgrp_ranks_accu_seg_rg);
    }
    indexed_sort(all_colgrp_ranks, all_colgrp_ranks_accu_seg);
    indexed_sort(all_colgrp_ranks_cg, all_colgrp_ranks_accu_seg_cg);
    // Make sure there is at least one follower
    if(follower_colgrp_ranks.size() > 1)
    {
        indexed_sort(follower_colgrp_ranks, follower_colgrp_ranks_accu_seg);
        indexed_sort(follower_colgrp_ranks_cg, follower_colgrp_ranks_accu_seg_cg);
    }

    if(Env::comm_split and not Env::get_comm_split())
    {
        //if(!Env::rank)
            //printf("+++++++++++++++++++++++==comm_split+++++++++++++++++++++++++++\n");
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        Env::set_comm_split();
    }
 
    // Calculate accumulator segments for X and Y
    
    // Which column index in my rowgrps is mine when I'm the accumulator
    for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++)
    {
        if(all_rowgrp_ranks[j] == Env::rank)
        {
            accu_segment_rg = j;
            accu_segment_rg_vec.push_back(accu_segment_rg);
        }
    }
    
    // Which row index in my colgrps is mine when I'm the accumulator
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        if(all_colgrp_ranks[j] == Env::rank)
        //if(local_col_segments[j] == owned_segment)
        {
            accu_segment_cg = j;
            accu_segment_cg_vec.push_back(accu_segment_cg);
        }
    } 
    
    // Which rowgrps/colgrps segment is mine
    owned_segment_vec.push_back(owned_segment);   
    
    // Which rowgrp is mine
    for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++)
    {
        if(leader_ranks[local_row_segments[j]] == Env::rank)
        {
        //if(leader_ranks[local_row_segments[j]] == Env::rank)
            accu_segment_row = j;
            accu_segment_row_vec.push_back(accu_segment_row);
        }
    }
    
    // Which colgrp is mine
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        if(leader_ranks[local_col_segments[j]] == Env::rank)
        //if(local_col_segments[j] == owned_segment)
        {
            accu_segment_col = j;
            accu_segment_col_vec.push_back(accu_segment_col);
        }
    } 
    
    /*
    if(Env::rank == 2)
    {
        printf("accu_segment_rg=%d\n", accu_segment_rg);
        printf("accu_segment_cg=%d\n", accu_segment_cg);
        printf("accu_segment_row=%d\n", accu_segment_row);
        printf("accu_segment_col=%d\n", accu_segment_col);
        
    }
    */
    // Print tiling assignment
    if(Env::is_master)
    {
        printf("Tiling Info: %d x %d [rowgrps x colgrps] with height of %d\n", nrowgrps, ncolgrps, tile_height);
        printf("Tiling Info: %d x %d [rowgrp_nranks x colgrp_nranks]\n", tiling->rowgrp_nranks, tiling->colgrp_nranks);
        printf("Tiling Info: %d x %d [rank_nrowgrps x rank_ncolgrps]\n", tiling->rank_nrowgrps, tiling->rank_ncolgrps);
    }
    print("rank");
    Env::barrier();
    //print("kth");
    // Want some debug info?
    //Env::barrier();
    //debug(-1);
    //Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::debug(int what_rank)
{
    if(Env::rank == what_rank)
    {
        uint32_t other, accu;
        printf("all_rowgrp_ranks\n");
        for(uint32_t j = 0; j < tiling->rowgrp_nranks; j++)
        {
            other = all_rowgrp_ranks[j];
            accu = all_rowgrp_ranks_accu_seg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        printf("all_colgrp_ranks\n");
        for(uint32_t j = 0; j < tiling->colgrp_nranks; j++)
        {
            other = all_colgrp_ranks[j];
            accu = all_colgrp_ranks_accu_seg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        printf("follower_rowgrp_ranks\n");
        for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++)
        {
            other = follower_rowgrp_ranks[j];
            accu = follower_rowgrp_ranks_accu_seg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        printf("follower_colgrp_ranks\n");
        for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
        {
            other = follower_colgrp_ranks[j];
            accu = follower_colgrp_ranks_accu_seg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        printf("all_rowgrp_ranks_rg\n");
        for(uint32_t j = 0; j < tiling->rowgrp_nranks; j++)
        {
            other = all_rowgrp_ranks_rg[j];
            accu = all_rowgrp_ranks_accu_seg_rg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        printf("all_colgrp_ranks_cg\n");
        for(uint32_t j = 0; j < tiling->colgrp_nranks; j++)
        {
            other = all_colgrp_ranks_cg[j];
            accu = all_colgrp_ranks_accu_seg_cg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");
        
        
        printf("follower_rowgrp_ranks_rg\n");
        for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++)
        {
            other = follower_rowgrp_ranks_rg[j];
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");   
        
        printf("follower_colgrp_ranks_cg\n");
        for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
        {
            other = follower_colgrp_ranks_cg[j];
            accu = follower_colgrp_ranks_accu_seg_cg[j];
            printf("[%d %d] ", other, accu);
        }
        printf("\n");   
        
        printf("local_col_segments\n");
        for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
        {
            other = local_col_segments[j];
            printf("[%d] ", other);
        }
        printf("\n");   
        
        printf("local_row_segments\n");
        for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++)
        {
            other = local_row_segments[j];
            printf("[%d] ", other);
        }
        printf("\n"); 
        
        printf("leader_ranks\n");
        for(uint32_t j = 0; j < Env::nranks; j++)
        {
            other = leader_ranks[j];
            printf("[%d] ", other);
        }
        printf("\n"); 
        
        printf("leader_ranks_rg\n");
        for(uint32_t j = 0; j < tiling->nrowgrps; j++)
        {
            other = leader_ranks_rg[j];
            printf("[%d] ", other);
        }
        printf("\n"); 
        
        printf("leader_ranks_cg\n");
        for(uint32_t j = 0; j < tiling->ncolgrps; j++)
        {
            other = leader_ranks_cg[j];
            printf("[%d] ", other);
        }
        printf("\n"); 
        
        printf("os=%d asr=%d asc=%d\n", owned_segment, accu_segment_rg, accu_segment_cg);   
    }    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::print(std::string element)
{
    if(Env::is_master)
    {    
        uint32_t skip = 15;
        for (uint32_t i = 0; i < nrowgrps; i++)
        {
            for (uint32_t j = 0; j < ncolgrps; j++)  
            {
                auto& tile = tiles[i][j];
                if(element.compare("rank") == 0) 
                    printf("%02d ", tile.rank);
                else if(element.compare("kth") == 0) 
                    printf("%3d ", tile.kth);
                else if(element.compare("ith") == 0) 
                    printf("%2d ", tile.ith);
                else if(element.compare("jth") == 0) 
                    printf("%2d ", tile.jth);
                else if(element.compare("rank_rg") == 0) 
                    printf("%2d ", tile.rank_rg);
                else if(element.compare("rank_cg") == 0) 
                    printf("%2d ", tile.rank_cg);
                else if(element.compare("leader_rank_rg") == 0) 
                    printf("%2d ", tile.leader_rank_rg);
                else if(element.compare("leader_rank_cg") == 0) 
                    printf("%2d ", tile.leader_rank_cg);
                else if(element.compare("nedges") == 0) 
                    printf("%lu ", tile.nedges);
                if(j > skip)
                {
                    printf("...");
                    break;
                }
            }
            printf("\n");
            if(i > skip)
            {
                printf(".\n.\n.\n");
                break;
            }
        }
        printf("\n");
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tiles()
{
    if(parread)
    {
        if(Env::is_master)
            printf("Edge distribution among %d ranks\n", Env::nranks);     
        distribute();
    }
    
    Triple<Weight, Integer_Type> pair;
    RowSort<Weight, Integer_Type> f_row;
    ColSort<Weight, Integer_Type> f_col;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.nedges = tile.triples->size();
        if(tile.nedges)
        {
            tile.allocated = true;
            if(compression_type == Compression_type::_CSR_)
                std::sort(tile.triples->begin(), tile.triples->end(), f_row);
            if(compression_type == Compression_type::_CSC_)
                std::sort(tile.triples->begin(), tile.triples->end(), f_col);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_filtering()
{
    if(filtering_type == _NONE_)
    {
       if(Env::is_master)
            printf("Vertex filtering: Skipped filtering\n");
    }
    else if(filtering_type == _SRCS_)
    {
        if(Env::is_master)
            printf("Vertex filtering: Removing empty sources\n");     
        
        filter(_SRCS_);
        filter(_SNKS_);
    }
    else if(filtering_type == _SNKS_)
    {
       if(Env::is_master)
            printf("Vertex filtering: Removing empty sinks\n");     
        
        filter(_SNKS_);
    }
    else if(filtering_type == _BOTH_)
    {
        if(Env::is_master)
            printf("Vertex filtering: Removing both empty sources and sinks\n");     
        
        filter(_SRCS_);
        filter(_SNKS_);
    }
    else if(filtering_type == _SOME_)
    {
        if(Env::is_master)
            printf("Vertex filtering: Removing isolated rows and columns\n");     
        
        filter(_SRCS_);
        filter(_SNKS_);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter(Filtering_type filtering_type_)
{
    uint32_t rank_nrowgrps_, rank_ncolgrps_;
    uint32_t rowgrp_nranks_, colgrp_nranks_;
    Integer_Type tile_length;
    std::vector<int32_t> local_row_segments_;
    std::vector<int32_t> all_rowgrp_ranks_accu_seg_;
    std::vector<int32_t> accu_segment_row_vec_;
    std::vector<uint32_t> local_tiles_row_order_;
    int32_t accu_segment_rg_, accu_segment_row_;
    std::vector<int32_t> follower_rowgrp_ranks_; 
    std::vector<int32_t> follower_rowgrp_ranks_accu_seg_;
    std::vector<Integer_Type> nnz_sizes_all, nnz_sizes_loc;
    uint32_t nrowgrps_;
    
    
    if(filtering_type_ == _SRCS_)
    {
        rank_nrowgrps_ = tiling->rank_nrowgrps;
        rank_ncolgrps_ = tiling->rank_ncolgrps;
        rowgrp_nranks_ = tiling->rowgrp_nranks;
        colgrp_nranks_ = tiling->colgrp_nranks;
        nrowgrps_ = tiling->nrowgrps;
        tile_length = tile_height;
        local_row_segments_ = local_row_segments;
        all_rowgrp_ranks_accu_seg_ = all_rowgrp_ranks_accu_seg;
        follower_rowgrp_ranks_ = follower_rowgrp_ranks;
        follower_rowgrp_ranks_accu_seg_ = follower_rowgrp_ranks_accu_seg;
        
        local_tiles_row_order_ = local_tiles_row_order;  
        accu_segment_rg_ = accu_segment_rg;
        accu_segment_row_ = accu_segment_row;
        accu_segment_row_vec_ = accu_segment_row_vec;
    }
    else if(filtering_type_ == _SNKS_)
    {
        rank_nrowgrps_ = tiling->rank_ncolgrps;
        rank_ncolgrps_ = tiling->rank_nrowgrps;
        rowgrp_nranks_ = tiling->colgrp_nranks;
        colgrp_nranks_ = tiling->rowgrp_nranks;
        nrowgrps_ = tiling->ncolgrps;
        tile_length = tile_width;
        local_row_segments_ = local_col_segments;
        all_rowgrp_ranks_accu_seg_ = all_colgrp_ranks_accu_seg;
        follower_rowgrp_ranks_ = follower_colgrp_ranks;
        follower_rowgrp_ranks_accu_seg_ = follower_colgrp_ranks_accu_seg;
        
        local_tiles_row_order_ = local_tiles_col_order;  
        accu_segment_rg_ = accu_segment_cg;
        accu_segment_row_ = accu_segment_col;
        accu_segment_row_vec_ = accu_segment_col_vec;
    }
    
    
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;
    uint32_t leader, follower, my_rank, accu, this_segment;
    uint32_t tile_th, pair_idx;
    bool vec_owner, communication;
    uint32_t fi = 0, fo = 0;
    std::vector<Vector<Weight, Integer_Type, Integer_Type> *> F;
    Vector<Weight, Integer_Type, Integer_Type> *F_;
    MPI_Datatype type = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        if(local_row_segments_[j] == owned_segment)
        {
            std::vector<Integer_Type> tile_length_sizes(rowgrp_nranks_, tile_length);
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes, all_rowgrp_ranks_accu_seg_);
        }
        else
        {
            std::vector<Integer_Type> tile_length_sizes(1, tile_length);
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes, accu_segment_row_vec_);
        }
        F.push_back(F_);
    }
    
    for(uint32_t t: local_tiles_row_order_)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        
        if(filtering_type_ == _SRCS_)
        {
            tile_th = tile.nth;
            pair_idx = pair.row;
        }
        else if(filtering_type_ == _SNKS_)
        {
            tile_th = tile.mth;
            pair_idx = pair.col;
        }
        
        vec_owner = (leader_ranks[pair_idx] == Env::rank);
        if(vec_owner)
            fo = accu_segment_rg_;
        else
            fo = 0;
        
        auto *Fp = F[fi];
        auto &f_seg = Fp->segments[fo];
        auto *f_data = (Integer_Type *) f_seg.D->data;
        Integer_Type f_nitems = f_seg.D->n;
        Integer_Type f_nbytes = f_seg.D->nbytes;        
        
        if(tile.allocated)
        {
            if(filtering_type_ == _SRCS_)
            {
                for (auto& triple : *(tile.triples))
                {
                    auto pair1 = rebase(triple);
                    f_data[pair1.row]++;
                    //printf("%d %d\n", Env::rank, f_data[pair.row] );
                }
            }
            else if(filtering_type_ == _SNKS_)
            {
                for (auto& triple : *(tile.triples))
                {
                    auto pair1 = rebase(triple);
                    f_data[pair1.col]++;
                    //printf("%d %d\n", Env::rank, f_data[pair.row] );
                }
            }
            
            /*
            if(compression_type == Compression_type::_CSR_)
            {
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                if(filtering_type_ == _SRCS_)
                {
                    for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                    {
                        for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                        {       
                            f_data[i]++;
                        }
                    }
                }
                else if(filtering_type_ == _SNKS_)
                {
                    for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                    {
                        for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                        {       
                            f_data[JA[j]]++;
                        }
                    }
                }
            }
            else if(compression_type == Compression_type::_CSC_)
            {
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                if(filtering_type_ == _SRCS_)
                {
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                        {
                            f_data[ROW_INDEX[i]]++;
                        }
                    }
                    
                }
                else if(filtering_type_ == _SNKS_)
                {
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                        {
                            f_data[j]++;
                        }
                    }
                }
            }
            */
        }
        
        communication = (((tile_th + 1) % rank_ncolgrps_) == 0);
        if(communication)
        {
            if(filtering_type_ == _SRCS_)
                leader = tile.leader_rank_rg;
            if(filtering_type_ == _SNKS_)
                leader = tile.leader_rank_cg;
            my_rank = Env::rank;
            if(leader == my_rank)
            {
                for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++)               
                {
                    follower = follower_rowgrp_ranks_[j];
                    accu = follower_rowgrp_ranks_accu_seg_[j];
                    auto &fj_seg = Fp->segments[accu];
                    auto *fj_data = (Integer_Type *) fj_seg.D->data;
                    Integer_Type fj_nitems = fj_seg.D->n;
                    Integer_Type fj_nbytes = fj_seg.D->nbytes;
                    MPI_Irecv(fj_data, fj_nitems, type, follower, pair_idx, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }   
            else
            {
                MPI_Isend(f_data, f_nitems, type, leader, pair_idx, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            fi++;
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();    
    
    fi = accu_segment_row_;
    auto *Fp = F[fi];
    fo = accu_segment_rg_;
    auto &f_seg = Fp->segments[fo];
    auto *f_data = (Integer_Type *) f_seg.D->data;
    Integer_Type f_nitems = f_seg.D->n;
    Integer_Type f_nbytes = f_seg.D->nbytes;

    for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++)
    {
        accu = follower_rowgrp_ranks_accu_seg_[j];
        auto &fj_seg = Fp->segments[accu];
        auto *fj_data = (Integer_Type *) fj_seg.D->data;
        Integer_Type fj_nitems = fj_seg.D->n;
        Integer_Type fj_nbytes = fj_seg.D->nbytes;                        
        for(uint32_t i = 0; i < fj_nitems; i++)
        {
            if(fj_data[i])
                f_data[i] += fj_data[i];
        }
    }
    
    Integer_Type nnz_local = 0;
    for(uint32_t i = 0; i < f_nitems; i++)
    {
        if(f_data[i])
            nnz_local++;
    }
    
    //printf("%d %d\n", Env::rank, nnz_local);
    
    
    nnz_sizes_all.resize(nrowgrps_);
    nnz_sizes_all[owned_segment] = nnz_local;
    
    //Env::barrier();
    for (uint32_t j = 0; j < nrowgrps_; j++)
    {
        uint32_t r = leader_ranks[j];
        if (j != owned_segment)
        {
            MPI_Sendrecv(&nnz_sizes_all[owned_segment], 1, MPI_UNSIGNED, r, Env::rank, &nnz_sizes_all[j], 1, MPI_UNSIGNED, 
                                                                         r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
        
    }
    Env::barrier();     
    
    /*
   // MPI_Status status;
    for (uint32_t j = 0; j < nrowgrps_; j++)
    {
        uint32_t r = leader_ranks[j];
        //if(!Env::rank)
        //printf("%d %d %d %d\n", Env::rank, j, r, owned_segment);
        //if (j == owned_segment)
          //  MPI_Bcast(&nnz_sizes_all[j], 1, MPI_UNSIGNED, r, Env::MPI_WORLD);
        //
        if (r == Env::rank)
        {
            for (uint32_t i = 0; i < nrowgrps_; i++)
            {
                uint32_t k = leader_ranks[i];
                if (k != Env::rank)
                {
                    MPI_Isend(&nnz_sizes_all[owned_segment], 1, MPI_UNSIGNED, k, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            //  MPI_recv(&nnz_sizes_all[j], 1, MPI_UNSIGNED, r, j, Env::MPI_WORLD, &status);
        }
        else
        {
            
            MPI_Irecv(&nnz_sizes_all[j], 1, MPI_UNSIGNED, r, j, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
       
    }
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    */
    
    //printf("%d %d \n", Env::rank, nnz_sizes_all[owned_segment]);
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        this_segment = local_row_segments_[j];
        nnz_sizes_loc.push_back(nnz_sizes_all[this_segment]);
    }
    
    
    
    Vector<Weight, Integer_Type, Integer_Type> *T = new Vector<Weight, Integer_Type, Integer_Type>(nnz_sizes_loc,  local_row_segments_);
    
    std::vector<Integer_Type> tile_length_sizes(rank_nrowgrps_, tile_length);
    Vector<Weight, Integer_Type, char> *K = new Vector<Weight, Integer_Type, char>(tile_length_sizes,  local_row_segments_);
    Vector<Weight, Integer_Type, Integer_Type> *KV = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes,  local_row_segments_);
    
    MPI_Datatype type1 = Types<Weight, Integer_Type, char>::get_data_type();
    if(nnz_sizes_all[owned_segment])
    {
        auto &tj_seg = T->segments[accu_segment_row_];
        auto *tj_data = (Integer_Type *) tj_seg.D->data;
        
        auto &kj_seg = K->segments[accu_segment_row_];
        auto *kj_data = (char *) kj_seg.D->data;
        
        auto &kvj_seg = KV->segments[accu_segment_row_];
        auto *kvj_data = (Integer_Type *) kvj_seg.D->data;
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < f_nitems; i++)
        {
            if(f_data[i])
            {
                tj_data[j] = i;
                kj_data[i] = 1;
                kvj_data[i] = j; 
                j++;
            }
            else
            {
                kj_data[i] = 0;
                kvj_data[i] = 0; // Don't put -1 because Integer_Type might be unsigned int
            }
        }
        assert(j == nnz_sizes_all[owned_segment]);
    }
    
  
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        
        auto &tj_seg = T->segments[j];
        auto *tj_data = (Integer_Type *) tj_seg.D->data;
        Integer_Type tj_nitems = tj_seg.D->n;
        Integer_Type tj_nbytes = tj_seg.D->nbytes;
        
        if(tj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++)
                {
                    follower = follower_rowgrp_ranks_[i];
                    //printf("send %d --> %d %d %d %d\n", leader, follower, this_segment, j, ej_seg.D->n); 

                    MPI_Isend(tj_data, tj_nitems, type, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                //printf("recv %d <-- %d %d %d %d\n", Env::rank, leader, this_segment, j, ej_seg.D->n); 
                                        
                MPI_Irecv(tj_data, tj_nitems, type, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        
        auto &kj_seg = K->segments[j];
        auto *kj_data = (char *) kj_seg.D->data;
        Integer_Type kj_nitems = kj_seg.D->n;
        Integer_Type kj_nbytes = kj_seg.D->nbytes;
        
        if(kj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++)
                {
                    follower = follower_rowgrp_ranks_[i];
                    MPI_Isend(kj_data, kj_nitems, type1, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                MPI_Irecv(kj_data, kj_nitems, type1, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        
        auto &kvj_seg = KV->segments[j];
        auto *kvj_data = (Integer_Type *) kvj_seg.D->data;
        Integer_Type kvj_nitems = kvj_seg.D->n;
        Integer_Type kvj_nbytes = kvj_seg.D->nbytes;
        
        if(kvj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++)
                {
                    follower = follower_rowgrp_ranks_[i];
                    MPI_Isend(kvj_data, kvj_nitems, type, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                MPI_Irecv(kvj_data, kvj_nitems, type, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    
    /*
    if(!Env::rank)
    {
        Integer_Type sum = 0;
        for(uint32_t j = 0; j < nrowgrps_; j++)
        {
            sum += nnz_sizes_all[j];
            printf(" r=%d %d ", Env::rank, nnz_sizes_all[j]);
        }
        printf("/%d\n", sum);
    }
     */  
    /*
    if(1)
    {
        //for(uint32_t j = 0; j < nrowgrps_; j++)
        //    printf("%d ", nnz_sizes_all[j]);
        //printf("\n");
        
        for(uint32_t j = 0; j < rank_nrowgrps_; j++)
        {
            auto &tj_seg = T->segments[j];
            //if(cj_seg.allocated)
            //{
            auto *tj_data = (Integer_Type *) tj_seg.D->data;
            
            Integer_Type tj_nitems = tj_seg.D->n;
            Integer_Type tj_nbytes = tj_seg.D->nbytes;
            printf(">>> %d %d %d\n", Env::rank, tj_data == nullptr, tj_nitems);
            auto &kj_seg = K->segments[j];
            auto *kj_data = (Integer_Type *) kj_seg.D->data;
            Integer_Type kj_nitems = kj_seg.D->n;
            Integer_Type kj_nbytes = kj_seg.D->nbytes;  
            
            printf(">>> %d %d %d %d %d\n", Env::rank, tj_data == nullptr, tj_nitems, kj_data == nullptr, kj_nitems);
            
            for(uint32_t i = 0; i < tj_nitems; i++)
            {
               printf("a[%d]=%d %d\n", i, kj_data[i], tj_data[kj_data[i]]);
            }
            printf("\n");
            //}
        }
        
    }
    */
    
    
    if(!Env::rank)
    {
        printf("nnz_sizes_all[%d]: ", filtering_type_);
        for(uint32_t i = 0 ; i < nrowgrps_; i++)
            printf("%d ", nnz_sizes_all[i]);
        printf("\n");
        
        printf("nnz_sizes_loc[%d]: ", filtering_type_);
        for(uint32_t i = 0 ; i < rank_nrowgrps_; i++)
            printf("%d ", nnz_sizes_loc[i]);
        printf("\n");
        
        /*
        for(uint32_t j = 0; j < rank_nrowgrps_; j++)
        {
            auto &k_seg = K->segments[j];
            char *k_data = (char *) k_seg.D->data;
            Integer_Type k_nitems = k_seg.D->n;
            
            for(uint32_t i = 0; i < k_nitems; i++)
                printf("%d %d\n", i, k_data[i]);
        }
        */    
        
    }
    
    if(filtering_type_ == _SRCS_)
    {
        nnz_row_sizes_all = nnz_sizes_all;
        nnz_row_sizes_loc = nnz_sizes_loc;
        R = T;
        I = K;
        IV = KV;
    }
    else if(filtering_type_ == _SNKS_)
    {
        nnz_col_sizes_all = nnz_sizes_all;
        nnz_col_sizes_loc = nnz_sizes_loc;
        C = T;
        J = K;
        JV = KV;
    }
    
    for (uint32_t i = 0; i < rank_nrowgrps_; i++)
    {
        F_ = F[i];
        F_->del_vec();
        delete F_;
    } 
    
    //int retval = 0;
    //retval = MPI_Type_commit(&type);
    //assert(retval == MPI_SUCCESS);
    //retval = MPI_Type_free(&type);
    //assert(retval == MPI_SUCCESS);
    //assert(retval == MPI_SUCCESS);   
    //retval = MPI_Type_free(&type1);
    //assert(retval == MPI_SUCCESS);   
    

    Env::barrier();
}

/* Borrowed from LA3 code @
   https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/dist_matrix2d.hpp
*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::distribute()
{
    /* Sanity check on # of edges */
    uint64_t nedges_start_local = 0, nedges_end_local = 0,
             nedges_start_global = 0, nedges_end_global = 0;
             
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            if(tile.triples->size() > 0)
            {
                nedges_start_local += tile.triples->size();
            }
        }
    }

    MPI_Datatype MANY_TRIPLES;
    const uint32_t many_triples_size = 1;
    
    MPI_Type_contiguous(many_triples_size * sizeof(Triple<Weight, Integer_Type>), MPI_BYTE, &MANY_TRIPLES);
    MPI_Type_commit(&MANY_TRIPLES);
    
    std::vector<std::vector<Triple<Weight, Integer_Type>>> outboxes(Env::nranks);
    std::vector<std::vector<Triple<Weight, Integer_Type>>> inboxes(Env::nranks);
    std::vector<uint32_t> inbox_sizes(Env::nranks);
    
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            if(tile.rank != Env::rank)
            {
                //if(!Env::rank)
                //    printf("r=%d s=%lu k=%d\n", tile.rank, tile.triples->size(), tile.kth);
                auto &outbox = outboxes[tile.rank];
                outbox.insert(outbox.end(), tile.triples->begin(), tile.triples->end());
                tile.free_triples();
            }
        }
    }
    
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;
    MPI_Status status;
    
    //Env::barrier();
    for (uint32_t r = 0; r < Env::nranks; r++)
    {
        if (r != Env::rank)
        {
            auto &outbox = outboxes[r];
            uint32_t outbox_size = outbox.size();
            MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, Env::rank, &inbox_sizes[r], 1, MPI_UNSIGNED, 
                                                        r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    Env::barrier();
    
    /*
    for (uint32_t r = 0; r < Env::nranks; r++)
    {
        //uint32_t r = leader_ranks[j];
        //if(!Env::rank)
        //printf("%d %d %d %d\n", Env::rank, j, r, owned_segment);
        //if (j == owned_segment)
          //  MPI_Bcast(&nnz_sizes_all[j], 1, MPI_UNSIGNED, r, Env::MPI_WORLD);
        //
        if (r == Env::rank)
        {
            for (uint32_t i = 0; i < Env::nranks; i++)
            {
                //uint32_t k = leader_ranks[i];
                if (i != Env::rank)
                {
                    auto &outbox = outboxes[i];
                    uint32_t outbox_size = outbox.size();   
                    MPI_Isend(&outbox_size, 1, MPI_UNSIGNED, i, Env::rank, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
        }
        else
        {
            
            MPI_Irecv(&inbox_sizes[r], 1, MPI_UNSIGNED, r, r, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
       
    }
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    */
    
    

     

    for (uint32_t i = 0; i < Env::nranks; i++)
    {
        uint32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank)
        {
            //if(!Env::rank)
            //    printf("-->%d %lu\n",r, outboxes[r].size());
            auto &outbox = outboxes[r];
            uint32_t outbox_bound = outbox.size() + many_triples_size;
            outbox.resize(outbox_bound);
            /* Send the triples with many_triples_size padding. */
            //MPI_Send(outbox.data(), outbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD);
            MPI_Isend(outbox.data(), outbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);
            //MPI_Wait(&request, &status);
            out_requests.push_back(request);
        }
    }     
     
     
    for (uint32_t i = 0; i < Env::nranks; i++)
    {
        uint32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank)
        {
            //if(!Env::rank)
            //    printf("<<--%d %lu\n",r, outboxes[r].size());
            auto &inbox = inboxes[r];
            uint32_t inbox_bound = inbox_sizes[r] + many_triples_size;
            inbox.resize(inbox_bound);
            /* Recv the triples with many_triples_size padding. */
//            MPI_Recv(inbox.data(), inbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD, &status);
            MPI_Irecv(inbox.data(), inbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);
            //MPI_Wait(&request, &status);

            in_requests.push_back(request);
        }
    }



     


    
    
    //MPI_Request request;



    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();


    //Env::barrier();
    
    for (uint32_t r = 0; r < Env::nranks; r++)
    {
        if (r != Env::rank)
        {
            auto &inbox = inboxes[r];
            for (uint32_t i = 0; i < inbox_sizes[r]; i++)
            {
                test(inbox[i]);
                insert(inbox[i]);
            }

            inbox.clear();
            inbox.shrink_to_fit();
        }
    }
    
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
    out_requests.clear();
    Env::barrier();    
    

    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        nedges_end_local += tile.triples->size();
        //tile.nedges = tile.triples->size();
        //if(tile.nedges)
        //    tile.allocated = true;
    //if(!Env::rank)
      //  printf("%lu\n", tile.triples->size());
    }
    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    MPI_Allreduce(&nedges_end_local, &nedges_end_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges_start_global == nedges_end_global);
    if(Env::is_master)
        printf("Sanity check for exchanging %lu edges is done\n", nedges_end_global);
    
    auto retval = MPI_Type_free(&MANY_TRIPLES);
    assert(retval == MPI_SUCCESS);   
    //Env::barrier();
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression()
{
    /*
    
    if(parread)
    {
       if(Env::is_master)
            printf("Edge distribution among %d ranks\n", Env::nranks);     
        distribute();
    }
    
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.nedges = tile.triples->size();
        if(tile.nedges)
            tile.allocated = true;
        //if(!Env::rank)
        //    printf("t=%d n=%lu\n", t, tile.triples->size());
    }
    */

    
    //if(filtering_type == _NONE_)
    //{
    if(compression_type == Compression_type::_CSR_)
    {
        if(Env::is_master)
            printf("Edge compression: CSR\n");
        
        if(filtering_type == _NONE_)
            init_csr();
        else if(filtering_type == _SOME_)
            init_tcsr();
    }
    else if(compression_type == Compression_type::_CSC_)
    {
        if(Env::is_master)
            printf("Edge compression: CSC\n");
        if(filtering_type == _NONE_)
            init_csc();
        else if(filtering_type == _SOME_)
            init_tcsc();
    }
    else
    {
        fprintf(stderr, "Invalid compression type\n");
        Env::exit(1);
    }    
        /*
    }
    else if(filtering_type == _SOME_)
    {
        if(compression_type == Compression_type::_CSR_)
        {
            if(Env::is_master)
                printf("Edge compression: _CSR_\n");
            init_tcsr();
        }
        else if(compression_type == Compression_type::_CSC_)
        {
            if(Env::is_master)
                printf("Edge compression: _CSC_\n");
            init_tcsc();
        }
        else
        {
            fprintf(stderr, "Invalid compression type\n");
            Env::exit(1);
        }
        
    }
    */
    //filter();
    //printf("Outside filter %d\n", Env::rank);
    //filter(_ROW_);
    //filter(_COL_);
    
    //filter1();
    //Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsr()
{
    //if(!Env::rank)
        //printf("We are testing this\n");
    uint32_t yi = 0, xi = 0, next_row = 0;
    
    struct Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {        
        auto &r_seg = R->segments[yi];
        auto *r_data = (Integer_Type *) r_seg.D->data;
        Integer_Type r_nitems = r_seg.D->n;
        
        auto &i_seg = I->segments[yi];
        char *i_data = (char *) i_seg.D->data;
        Integer_Type i_nitems = i_seg.D->n;

        auto &iv_seg = IV->segments[yi];
        auto *iv_data = (Integer_Type *) iv_seg.D->data;
        Integer_Type iv_nitems = iv_seg.D->n;
        
        auto &c_seg = C->segments[xi];
        auto *c_data = (Integer_Type *) c_seg.D->data;
        Integer_Type c_nitems = c_seg.D->n;
        
        auto &j_seg = J->segments[xi];
        char *j_data = (char *) j_seg.D->data;
        Integer_Type j_nitems = j_seg.D->n;

        auto &jv_seg = JV->segments[xi];
        auto *jv_data = (Integer_Type *) jv_seg.D->data;
        Integer_Type jv_nitems = jv_seg.D->n;
        
        
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        uint32_t n = 0;
        if(tile.allocated)
        {

            
            tile.csr = new struct CSR<Weight, Integer_Type>(tile.triples->size(), r_nitems + 1);
            uint32_t i = 0; // CSR Index
            uint32_t j = 1; // Row index
            uint32_t k = 0; // Row indirected index
            /* A hack over partial specialization because 
               we didn't want to duplicate the code for 
               Empty weights though! */
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A->data;
            #endif
            
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data; // Rows
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data; // Cols
            IA[0] = 0;
            for (auto& triple : *(tile.triples))
            {
                pair = rebase(triple);
                //printf("r=%d c=%d r_i=%d c_i=%d\n", pair.row, pair.col, iv_data[pair.row], jv_data[pair.col]);
                /*
                if((j - 1) != pair.row)
                {
                    j++;
                    IA[j] = IA[j - 1];
                }
                */
                /*
                while((j - 1) != pair.row)
                {
                    //printf("j-1=%d i[j-1]=%d\n", j - 1, i_data[j - 1]);
                    if(i_data[j - 1])
                    {
                        k++;
                        IA[k] = IA[k - 1];
                    }
                    j++;
                }
                */
                
                if(iv_data[pair.row] > k)
                {
                    IA[iv_data[pair.row] + 1] = IA[iv_data[pair.row]];
                    k = iv_data[pair.row];
                    
                    //if(iv_data[pair.row] != k)
                    //{
                    //    printf("Increase %d %d\n", iv_data[pair.row], k);
                    //    assert(iv_data[pair.row] == k);
                    //}
                }
                
                // In case weights are there
                #ifdef HAS_WEIGHT
                A[i] = triple.weight;
                #endif
                
                //IA[k]++;
                //k = iv_data[pair.row];
                IA[iv_data[pair.row] + 1]++;
                JA[i] = jv_data[pair.col];    
                i++;
                n++;
            }
            
            //if(!Env::rank)
            //{
                
                //Env::barrier();
                //if(!Env::rank)
                   // printf("Edge compression: TCSR\n");
                //if(Env::rank == 0)
                //{
                /*    
                printf("rank=%d tile=%d\n", Env::rank, t);
                for(uint32_t i = 0; i < r_nitems; i++)
                {
                    printf("%d: ", i);
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {
                        printf("%d ", JA[j]);
                    }
                    printf("\n");
                }
                */
                //}
                //Env::barrier();
                
                /*
                if(Env::rank == 1)
                {
                printf("rank=%d tile=%d\n", Env::rank, t);
                for(uint32_t i = 0; i < r_nitems; i++)
                {
                    printf("%d: ", i);
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {
                        printf("%d ", JA[j]);
                    }
                    printf("\n");
                }
                }
                Env::barrier();
                */
            //}
        }    
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row)
        {
            xi = 0;
            yi++;
        }  
//printf("%d %d %d\n", Env::rank, t, n);        
    }
    
    //del_dcsr();
    //R, I, IV
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc()
{
    struct Triple<Weight, Integer_Type> pair;
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        auto &i_seg = I->segments[yi];
        char *i_data = (char *) i_seg.D->data;
        Integer_Type i_nitems = i_seg.D->n;
        
        auto &r_seg = R->segments[yi];
        auto *r_data = (Integer_Type *) r_seg.D->data;
        Integer_Type r_nitems = r_seg.D->n;

        auto &iv_seg = IV->segments[yi];
        auto *iv_data = (Integer_Type *) iv_seg.D->data;
        Integer_Type iv_nitems = iv_seg.D->n;
        
        auto &j_seg = J->segments[xi];
        char *j_data = (char *) j_seg.D->data;
        Integer_Type j_nitems = j_seg.D->n;
        
        auto &c_seg = C->segments[xi];
        auto *c_data = (Integer_Type *) c_seg.D->data;
        Integer_Type c_nitems = c_seg.D->n;

        auto &jv_seg = JV->segments[xi];
        auto *jv_data = (Integer_Type *) jv_seg.D->data;
        Integer_Type jv_nitems = jv_seg.D->n;
        
        
        //if(tile.allocated)
        //{
            //tile.nedges = tile.triples->size();
         //   tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            //tile.allocated = true;
        //}        
        
        //std::sort(tile.triples->begin(), tile.triples->end(), f);
        
        //uint32_t i = 0; // CSR Index
        //uint32_t j = 1; // Row index
        //if(!Env::rank)
        //{
        //    printf("t=%d a=%d n=%lu\n", t, tile.allocated, tile.nedges);
        //}
        
        if(tile.allocated)
        {
            tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), c_nitems + 1);
            uint32_t i = 0; // CSC Index
            uint32_t j = 1; // Col index
            uint32_t k = 0; // Col indirected index
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data; // ROW_INDEX
            Integer_Type *COL_PTR = (Integer_Type *) tile.csc->COL_PTR->data; // COL_PTR
            COL_PTR[0] = 0;
            for (auto& triple : *(tile.triples))
            {
                pair = rebase(triple);
                /*
                while((j - 1) != pair.col)
                {
                    j++;
                    COL_PTR[j] = COL_PTR[j - 1];
                }  
                */     
                if(jv_data[pair.col] > k)                
                //if((k - 1) != jv_data[pair.col])
                {
                    COL_PTR[jv_data[pair.col] + 1] = COL_PTR[jv_data[pair.col]];
                    k = jv_data[pair.col];
                    //k++;
                }
                
                // In case weights are there
                #ifdef HAS_WEIGHT
                VAL[i] = triple.weight;
                #endif
                
                COL_PTR[jv_data[pair.col] + 1]++;
                ROW_INDEX[i] = iv_data[pair.row];
                i++;
            }
            /*
            printf("rank=%d tile=%d\n", Env::rank, t);
            for(uint32_t i = 0; i < c_nitems; i++)
            {
                printf("%d: ", i);
                for(uint32_t j = COL_PTR[i]; j < COL_PTR[i + 1]; j++)
                {
                    printf("%d ", ROW_INDEX[j]);
                }
                printf("\n");
            }
            */
            
        }
        
        
        
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row)
        {
            xi = 0;
            yi++;
        }  
    }    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csr()
{
    /* Create the the csr format by allocating the csr data structure
       and then Sorting triples and populating the csr */
    struct Triple<Weight, Integer_Type> pair;
    //RowSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        //if(tile.allocated)
        //{
        //    tile.csr = new struct CSR<Weight, Integer_Type>(tile.triples->size(), tile_height + 1);
        //}        
        
        //std::sort(tile.triples->begin(), tile.triples->end(), f);
        

        if(tile.allocated)
        {
            tile.csr = new struct CSR<Weight, Integer_Type>(tile.triples->size(), tile_height + 1);
            uint32_t i = 0; // CSR Index
            uint32_t j = 1; // Row index
            /* A hack over partial specialization because 
               we didn't want to duplicate the code for 
               Empty weights though! */
            #ifdef HAS_WEIGHT
            Weight *A = (Weight *) tile.csr->A->data;
            #endif
            
            Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
            Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
            IA[0] = 0;
            for (auto& triple : *(tile.triples))
            {
                //nedges_local++;
                pair = rebase(triple);
                while((j - 1) != pair.row)
                {
                    j++;
                    IA[j] = IA[j - 1];
                }
                
                // In case weights are there
                #ifdef HAS_WEIGHT
                A[i] = triple.weight;
                #endif
                
                IA[j]++;
                JA[i] = pair.col;    
                i++;
            }
            
            while(j < tile_height)
            {
                j++;
                IA[j] = IA[j - 1];
            }   
            /*
            Env::barrier();

            if(Env::rank == 0)
            {
            printf("rank=%d tile=%d\n", Env::rank, t);
            for(uint32_t i = 0; i < tile_height - 1; i++)
            {
                printf("%d: ", i);
                for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                {
                    printf("%d ", JA[j]);
                }
                printf("\n");
            }
            }
            Env::barrier();
            if(Env::rank == 1)
            {
            printf("rank=%d tile=%d\n", Env::rank, t);
            for(uint32_t i = 0; i < tile_height - 1; i++)
            {
                printf("%d: ", i);
                for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                {
                    printf("%d ", JA[j]);
                }
                printf("\n");
            }
            }
            Env::barrier();
            */
            
        }
    }
    
    //del_triples();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csc()
{
    struct Triple<Weight, Integer_Type> pair;
    //ColSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        //if(tile.allocated)
        //{
            //tile.nedges = tile.triples->size();
         //   tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            //tile.allocated = true;
        //}        
        
        //std::sort(tile.triples->begin(), tile.triples->end(), f);
        
        ///uint32_t i = 0; // CSR Index
        //uint32_t j = 1; // Row index
        //if(!Env::rank)
        //{
        //    printf("t=%d a=%d n=%lu\n", t, tile.allocated, tile.nedges);
        //}
        
        if(tile.allocated)
        {
            tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            uint32_t i = 0; // CSC Index
            uint32_t j = 1; // Col index
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data; // ROW_INDEX
            Integer_Type *COL_PTR = (Integer_Type *) tile.csc->COL_PTR->data; // COL_PTR
            COL_PTR[0] = 0;
            for (auto& triple : *(tile.triples))
            {
                pair = rebase(triple);
                while((j - 1) != pair.col)
                {
                    j++;
                    COL_PTR[j] = COL_PTR[j - 1];
                }            
                // In case weights are there
                #ifdef HAS_WEIGHT
                VAL[i] = triple.weight;
                #endif
                
                COL_PTR[j]++;
                ROW_INDEX[i] = pair.row;
                i++;
            }
            
            while(j < tile_width)
            {
                j++;
                COL_PTR[j] = COL_PTR[j - 1];
            }
        }
    }

    //del_triples();
}
/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_compression()
{
    if(compression_type == Compression_type::_CSR_)
    {
        del_csr();
    }
    else if(compression_type == Compression_type::_CSC_)
    {
        del_csc();
    }
    else
    {
        fprintf(stderr, "Invalid compression type\n");
        Env::exit(1);
    }        
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_dcsr()
{
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.allocated)
        {
            delete tile.tcsr;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_csr()
{
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.allocated)
        {
            delete tile.csr;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_csc()
{
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.allocated)
        {
            delete tile.csc;
        }
    }
}
*/


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_compression()
{
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.allocated)
        {
            if(compression_type == Compression_type::_CSR_)
                delete tile.csr;
            else if(compression_type == Compression_type::_CSC_)
                delete tile.csc;
            else if(compression_type == Compression_type::_TCSR_)
                delete tile.tcsr;
            else if(compression_type == Compression_type::_TCSC_)
                delete tile.tcsc;
        }
    }
}




template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_triples()
{
    // Delete triples
    if(parread)
    {
        Triple<Weight, Integer_Type> pair;
        for(uint32_t t: local_tiles_row_order)
        {
            pair = tile_of_local_tile(t);
            auto& tile = tiles[pair.row][pair.col];
            tile.free_triples();
        }
        
    }
    else
    {
        for (uint32_t i = 0; i < nrowgrps; i++)
        {
            for (uint32_t j = 0; j < ncolgrps; j++)  
            {
                auto &tile = tiles[i][j];
                tile.free_triples();
            }
        }
    } 
}

#endif

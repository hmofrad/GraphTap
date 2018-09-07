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

enum Order_type
{
  _ROW_,
  _COL_
}; 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D
{ 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight, Integer_Type>> *triples;
    struct CSR<Weight, Integer_Type> *csr;
    struct CSC<Weight, Integer_Type> *csc;
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
               Tiling_type tiling_type, Compression_type compression_type);
        ~Matrix();

    private:
        Integer_Type nrows, ncols;
        uint32_t ntiles, nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width;    
        
        Tiling *tiling;
        Compression_type compression;
        //Vector<Weight, Integer_Type, Integer_Type> *E; //empty columns
        //Vector<Weight, Integer_Type, Integer_Type> *I; // Indices
        
        Vector<Weight, Integer_Type, Integer_Type> *R = nullptr;  // Non-empty rows
        Vector<Weight, Integer_Type, Integer_Type> *I = nullptr; // Row indices
        
        Vector<Weight, Integer_Type, Integer_Type> *C = nullptr;  // Non-empty columns
        Vector<Weight, Integer_Type, Integer_Type> *J = nullptr; // Column indices
        


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
        void init_compression(bool parread);
        void init_csr();
        void init_csc();
        void init_bv();
        void del_csr();
        void del_csc();
        void del_compression();
        void print(std::string element);
        void distribute();
        void filter1();
        void filter(Order_type order_type);
        void debug(int what_rank);
        
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type> &triple);
        uint32_t segment_of_tile(const struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type> &pair);
        void insert(const struct Triple<Weight, Integer_Type> &triple);
        
        std::vector<int32_t> sort_indices(const std::vector<int32_t> &v);
        void indexed_sort(std::vector<int32_t> &v1, std::vector<int32_t> &v2);        
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, Tiling_type tiling_type, Compression_type compression_type)
{
    nrows = nrows_;
    ncols = ncols_;
    ntiles = ntiles_;
    nrowgrps = sqrt(ntiles_);
    ncolgrps = ntiles_ / nrowgrps;
    tile_height = (nrows_ / nrowgrps) + 1;
    tile_width = (ncols_ / ncolgrps) + 1;
    
    // Initialize tiling 
    tiling = new Tiling(Env::nranks, ntiles, nrowgrps, ncolgrps, tiling_type);
    compression = compression_type;
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
void Matrix<Weight, Integer_Type, Fractional_Type>::insert(const struct Triple<Weight, Integer_Type> &triple)
{
    if(triple.row >= nrows)
        printf("%d (triple.row %d < tile_height %d)\n", Env::rank, triple.row, nrows);
    if(triple.col >= ncols)
        printf("%d (triple.col %d < tile_width %d)\n", Env::rank, triple.col, ncols);
    
    assert(triple.row < nrows);
    assert(triple.col < ncols);
    
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    
    if(pair.row >= nrowgrps)
        printf("%d (pair.row %d< nrowgrps %d)\n", Env::rank, pair.row, nrowgrps);
    
    if(pair.col >= ncolgrps)
        printf("%d (pair.col %d < ncolgrps %d)\n", Env::rank, pair.col, ncolgrps);
    
    assert(pair.row < nrowgrps);
    assert(pair.col < ncolgrps);
    
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
            tile.nth   = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.mth   = (tile.jth * tiling->rank_nrowgrps) + tile.ith;
            tile.allocate_triples();
            tile.allocated = false;
        }
    }
    
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
    if(Env::comm_split)
    {
        indexed_sort(all_rowgrp_ranks, all_rowgrp_ranks_accu_seg);
        indexed_sort(all_rowgrp_ranks_rg, all_rowgrp_ranks_accu_seg_rg);
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
        // Make sure there is at least one follower
        if(follower_rowgrp_ranks.size() > 1)
        {
            indexed_sort(follower_rowgrp_ranks, follower_rowgrp_ranks_accu_seg);
            indexed_sort(follower_rowgrp_ranks_rg, follower_rowgrp_ranks_accu_seg_rg);
        }
        
        indexed_sort(all_colgrp_ranks, all_colgrp_ranks_accu_seg);
        indexed_sort(all_colgrp_ranks_cg, all_colgrp_ranks_accu_seg_cg);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        if(follower_colgrp_ranks.size() > 1)
        {
            indexed_sort(follower_colgrp_ranks, follower_colgrp_ranks_accu_seg);
            indexed_sort(follower_colgrp_ranks_cg, follower_colgrp_ranks_accu_seg_cg);
        }
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
    print("rank");
    // Want some debug info?
    //Env::barrier();
    debug(-1);
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
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression(bool parread)
{
    
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
    

    if(Env::is_master)
        printf("Starting edge compression ...\n");
    
    if(compression == Compression_type::_CSR_)
    {
        init_csr();
    }
    else if(compression == Compression_type::_CSC_)
    {
        init_csc();
    }
    else
    {
        fprintf(stderr, "Invalid compression type\n");
        Env::exit(1);
    }    
    //filter();
    //printf("Outside filter %d\n", Env::rank);
    filter(_ROW_);
    filter(_COL_);
    
    //filter1();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter(Order_type order_type)
{
    uint32_t rank_nrowgrps_, rank_ncolgrps_;
    uint32_t rowgrp_nranks_, colgrp_nranks_;
    //uint32_t rank_ngrps;
    Integer_Type tile_length;
    std::vector<int32_t> local_row_segments_;
    std::vector<int32_t> all_rowgrp_ranks_accu_seg_;
    std::vector<int32_t> accu_segment_row_vec_;//, accu_segment_col_vec_;
    std::vector<uint32_t> local_tiles_row_order_;
    int32_t accu_segment_rg_, accu_segment_row_;//, accu_segment_cg_, accu_segment_col_;
    std::vector<int32_t> follower_rowgrp_ranks_; 
    std::vector<int32_t> follower_rowgrp_ranks_accu_seg_;
    std::vector<Integer_Type> nnz_sizes_all, nnz_sizes_loc;
    uint32_t nrowgrps_;
    
    
    if(order_type == _ROW_)
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
        //nnz_sizes_all = nnz_row_sizes_all;
        //nnz_sizes_loc = nnz_row_sizes_loc;
    }
    else if(order_type == _COL_)
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
        //nnz_sizes_all = nnz_col_sizes_all;
        //nnz_sizes_loc = nnz_col_sizes_loc;
    }
    
    
    Env::barrier();
    //printf("Here %d\n", Env::rank);
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    //MPI_Comm communicator;
    MPI_Request request;
    //MPI_Status status;
    uint32_t leader, follower, my_rank, accu, this_segment;
    uint32_t tile_th, pair_idx;
    bool vec_owner, communication;
    uint32_t fi = 0, fo = 0;
    std::vector<Vector<Weight, Integer_Type, Integer_Type> *> F;
    Vector<Weight, Integer_Type, Integer_Type> *F_;
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        if(local_row_segments_[j] == owned_segment)
        {
            std::vector<Integer_Type> tile_length_sizes(rowgrp_nranks_, tile_length);
            //printf("11111.rank=%d sz1=%lu sz2=%lu tl%d\n", Env::rank, tile_length_sizes.size(), tile_length_sizes.size(), tile_length);
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes, all_rowgrp_ranks_accu_seg_);
            //if(Env::rank == 0)
                
        }
        else
        {
            std::vector<Integer_Type> tile_length_sizes(1, tile_length);
            //printf("2222.rank=%d sz1=%lu sz2=%lu tl=%d\n", Env::rank, tile_length_sizes.size(), tile_length_sizes.size(), tile_length);
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes, accu_segment_row_vec_);
            //if(Env::rank == 0)
                //printf("2.rank=%d j=%d lrs=%d og=%d %lu %lu\n", Env::rank, j, local_row_segments_[j], owned_segment, tile_length_sizes.size(), all_rowgrp_ranks_accu_seg_.size());
        }
        
        //if(Env::rank == 0)
          ///  printf("rank=%d j=%d lrs=%d og=%d\n", Env::rank, j, local_row_segments_[j], owned_segment);
        F.push_back(F_);
    }
    //printf("Inside filter ... %d\n", Env::rank);
    //Env::barrier();
    
    for(uint32_t t: local_tiles_row_order_)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        //if(Env::rank == 11)
        //    printf("rank=%d tile=%d\n", Env::rank, t);
        
        if(order_type == _ROW_)
        {
            tile_th = tile.nth;
            pair_idx = pair.row;
        }
        else if(order_type == _COL_)
        {
            tile_th = tile.mth;
            pair_idx = pair.col;
        }
        
        vec_owner = (leader_ranks[pair_idx] == Env::rank);
        if(vec_owner)
            fo = accu_segment_rg_;
        else
            fo = 0;
        
        Vector<Weight, Integer_Type, Integer_Type> *Fp = F[fi];
        Segment<Weight, Integer_Type, Integer_Type> &f_seg = Fp->segments[fo];
        Integer_Type *f_data = (Integer_Type *) f_seg.D->data;
        Integer_Type f_nitems = f_seg.D->n;
        Integer_Type f_nbytes = f_seg.D->nbytes;        
        

        if(tile.allocated)
        {

            if(compression == Compression_type::_CSR_)
            {
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                if(order_type == _ROW_)
                {
                    for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                    {
                        for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                        {       
                            f_data[i]++;
                        }
                    }
                }
                else if(order_type == _COL_)
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
            else if(compression == Compression_type::_CSC_)
            {
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                if(order_type == _ROW_)
                {
                    for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                    {
                        for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                        {
                            f_data[ROW_INDEX[i]]++;
                        }
                    }
                    
                }
                else if(order_type == _COL_)
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
        }
        
        communication = (((tile_th + 1) % rank_ncolgrps_) == 0);
        if(communication)
        {
            if(order_type == _ROW_)
                leader = tile.leader_rank_rg;
            if(order_type == _COL_)
                leader = tile.leader_rank_cg;
            my_rank = Env::rank;
            
            //printf("%d %d %d\n", my_rank, leader, pair_idx);
            
            if(leader == my_rank)
            {
                for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++)               
                {
                    follower = follower_rowgrp_ranks_[j];
                    accu = follower_rowgrp_ranks_accu_seg_[j];
                    //printf("RECV: rank=%d: leader=%d <-- follower=%d row=%d\n", Env::rank, leader, follower, pair_idx);
                    Segment<Weight, Integer_Type, Integer_Type> &fj_seg = Fp->segments[accu];
                    Integer_Type *fj_data = (Integer_Type *) fj_seg.D->data;
                    Integer_Type fj_nitems = fj_seg.D->n;
                    Integer_Type fj_nbytes = fj_seg.D->nbytes;
                    MPI_Irecv(fj_data, fj_nbytes, MPI_BYTE, follower, pair_idx, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }   
            else
            {
                MPI_Isend(f_data, f_nbytes, MPI_BYTE, leader, pair_idx, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
                //printf("SEND: rank=%d: me=%d --> leader=%d row=%d\n", Env::rank, my_rank, leader, pair_idx);
            }
            fi++;
        }
    }
    //printf("0.####################### %d\n", Env::rank);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    //printf("1.####################### %d\n", Env::rank);
    //Env::barrier();
    
    
    
    fi = accu_segment_row_;
    Vector<Weight, Integer_Type, Integer_Type> *Fp = F[fi];
    fo = accu_segment_rg_;
    Segment<Weight, Integer_Type, Integer_Type> &f_seg = Fp->segments[fo];
    Integer_Type *f_data = (Integer_Type *) f_seg.D->data;
    Integer_Type f_nitems = f_seg.D->n;
    Integer_Type f_nbytes = f_seg.D->nbytes;

    for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++)
    {

        accu = follower_rowgrp_ranks_accu_seg_[j];
        Segment<Weight, Integer_Type, Integer_Type> &fj_seg = Fp->segments[accu];
        Integer_Type *fj_data = (Integer_Type *) fj_seg.D->data;
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
            MPI_Sendrecv(&nnz_sizes_all[owned_segment], 1, MPI_UNSIGNED, r, 0, &nnz_sizes_all[j], 1, MPI_UNSIGNED, 
                                                        r, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    //Env::barrier();  
    assert(nnz_local == nnz_sizes_all[owned_segment]);
    //printf("%d %d \n", Env::rank, nnz_sizes_all[owned_segment]);
    
    //printf("2.####################### %d\n", Env::rank);
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        this_segment = local_row_segments_[j];
        nnz_sizes_loc.push_back(nnz_sizes_all[this_segment]);
    }
    
    
    
    Vector<Weight, Integer_Type, Integer_Type> *T = new Vector<Weight, Integer_Type, Integer_Type>(nnz_sizes_loc,  local_row_segments_);
    
    std::vector<Integer_Type> tile_length_sizes(rank_nrowgrps_, tile_length);
    Vector<Weight, Integer_Type, Integer_Type> *K = new Vector<Weight, Integer_Type, Integer_Type>(tile_length_sizes,  local_row_segments_);
    
    //printf("3.####################### %d\n", Env::rank);
    if(nnz_sizes_all[owned_segment])
    {
        Segment<Weight, Integer_Type, Integer_Type> &tj_seg = T->segments[accu_segment_row_];
        Integer_Type *tj_data = (Integer_Type *) tj_seg.D->data;
        
        Segment<Weight, Integer_Type, Integer_Type> &kj_seg = K->segments[accu_segment_row_];
        Integer_Type *kj_data = (Integer_Type *) kj_seg.D->data;
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < f_nitems; i++)
        {
            if(f_data[i])
            {
                tj_data[j] = i;
                kj_data[i] = j;
                j++;
            }
        }
        assert(j == nnz_sizes_all[owned_segment]);
    }
    
    
    //printf("4.####################### %d\n", Env::rank);
    
    
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        
        Segment<Weight, Integer_Type, Integer_Type> &tj_seg = T->segments[j];
        Integer_Type *tj_data = (Integer_Type *) tj_seg.D->data;
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

                    MPI_Isend(tj_data, tj_nbytes, MPI_BYTE, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                //printf("recv %d <-- %d %d %d %d\n", Env::rank, leader, this_segment, j, ej_seg.D->n); 
                                        
                MPI_Irecv(tj_data, tj_nbytes, MPI_BYTE, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    //Env::barrier();
    // printf("5.####################### %d\n", Env::rank);
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        
        Segment<Weight, Integer_Type, Integer_Type> &kj_seg = K->segments[j];
        Integer_Type *kj_data = (Integer_Type *) kj_seg.D->data;
        Integer_Type kj_nitems = kj_seg.D->n;
        Integer_Type kj_nbytes = kj_seg.D->nbytes;
        
        if(kj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++)
                {
                    follower = follower_rowgrp_ranks_[i];
                    //printf("send %d --> %d %d %d %d\n", leader, follower, this_segment, j, ej_seg.D->n); 

                    MPI_Isend(kj_data, kj_nbytes, MPI_BYTE, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                //printf("recv %d <-- %d %d %d %d\n", Env::rank, leader, this_segment, j, ej_seg.D->n); 
                                        
                MPI_Irecv(kj_data, kj_nbytes, MPI_BYTE, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier(); 
    
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
    
    
    if(order_type == _ROW_)
    {
        nnz_row_sizes_all = nnz_sizes_all;
        nnz_row_sizes_loc = nnz_sizes_loc;
        R = T;
        I = K;
    }
    else if(order_type == _COL_)
    {
        nnz_col_sizes_all = nnz_sizes_all;
        nnz_col_sizes_loc = nnz_sizes_loc;
        C = T;
        J = K;
    }
    
    for (uint32_t i = 0; i < rank_nrowgrps_; i++)
    {
        F_ = F[i];
        F_->del_vec();
        delete F_;
    } 
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter1()
{
    Env::barrier();
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Comm communicator;
    MPI_Request request;
    MPI_Status status;
    uint32_t leader, follower, my_rank, accu, this_segment;
    uint32_t tile_th, pair_idx;
    bool vec_owner, communication;
    uint32_t fi = 0, fo = 0;
    std::vector<Vector<Weight, Integer_Type, Integer_Type> *> F;
    Vector<Weight, Integer_Type, Integer_Type> *F_;
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        if(local_col_segments[j] == owned_segment)
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_height, all_colgrp_ranks_accu_seg);
        else
            F_ = new Vector<Weight, Integer_Type, Integer_Type>(tile_height, accu_segment_col_vec);
        F.push_back(F_);
    }
    
    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        
        tile_th = tile.mth;
        pair_idx = pair.col;
        
        vec_owner = (leader_ranks[pair_idx] == Env::rank);
        if(vec_owner)
            fo = accu_segment_cg;
        else
            fo = 0;
        
        auto *Fp = F[fi];
        auto &f_seg = Fp->segments[fo];
        auto *f_data = (Integer_Type *) f_seg.D->data;
        Integer_Type f_nitems = f_seg.D->n;
        Integer_Type f_nbytes = f_seg.D->nbytes;        
        

        if(tile.allocated)
        {

            if(compression == Compression_type::_CSR_)
            {
                Integer_Type *IA = (Integer_Type *) tile.csr->IA->data;
                Integer_Type *JA = (Integer_Type *) tile.csr->JA->data;
                Integer_Type nrows_plus_one_minus_one = tile.csr->nrows_plus_one - 1;
                Integer_Type nnz_per_row;
                for(uint32_t i = 0; i < nrows_plus_one_minus_one; i++)
                {
                    for(uint32_t j = IA[i]; j < IA[i + 1]; j++)
                    {       
                        f_data[JA[j]]++;
                    }
                }
            }
            else if(compression == Compression_type::_CSC_)
            {
                Integer_Type *COL_PTR   = (Integer_Type *) tile.csc->COL_PTR->data;
                Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data;
                Integer_Type ncols_plus_one_minus_one = tile.csc->ncols_plus_one - 1;
                Integer_Type nnz_per_col;
                
                for(uint32_t j = 0; j < ncols_plus_one_minus_one; j++)
                {
                    for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                    {
                        f_data[j]++;
                    }
                }
            }
        }
        
        communication = (((tile_th + 1) % tiling->rank_nrowgrps) == 0);
        if(communication)
        {
            if(Env::comm_split)
            {
                leader = tile.leader_rank_cg_cg;
                my_rank = Env::rank_cg;
                communicator = Env::colgrps_comm;
            }
            else
            {
                leader = tile.leader_rank_cg;
                my_rank = Env::rank;
                communicator = Env::MPI_WORLD;
            }
            
            if(leader == my_rank)
            {

                for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)               
                {

                    if(Env::comm_split)
                    {
                        follower = follower_colgrp_ranks_cg[j];
                        accu = follower_colgrp_ranks_accu_seg_cg[j];
                    }
                    else
                    {
                        follower = follower_colgrp_ranks[j];
                        accu = follower_colgrp_ranks_accu_seg[j];
                    }
                    auto &fj_seg = Fp->segments[accu];
                    auto *fj_data = (Integer_Type *) fj_seg.D->data;
                    Integer_Type fj_nitems = fj_seg.D->n;
                    Integer_Type fj_nbytes = fj_seg.D->nbytes;
                    MPI_Irecv(fj_data, fj_nbytes, MPI_BYTE, follower, pair_idx, communicator, &request);
                    in_requests.push_back(request);
                }
            }   
            else
            {
                MPI_Isend(f_data, f_nbytes, MPI_BYTE, leader, pair_idx, communicator, &request);
                out_requests.push_back(request);
            }
            fi++;
        }
    }
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    //Env::barrier();
    
    
    fi = accu_segment_col;
    auto *Fp = F[fi];
    fo = accu_segment_cg;
    auto &f_seg = Fp->segments[fo];
    auto *f_data = (Integer_Type *) f_seg.D->data;
    Integer_Type f_nitems = f_seg.D->n;
    Integer_Type f_nbytes = f_seg.D->nbytes;

    for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
    {
        if(Env::comm_split)
            accu = follower_colgrp_ranks_accu_seg_cg[j];
        else
            accu = follower_colgrp_ranks_accu_seg[j];
        
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
    
    Integer_Type nnz_col_local = 0;
    for(uint32_t i = 0; i < f_nitems; i++)
    {
        if(f_data[i])
            nnz_col_local++;
    }
    
    
    //std::vector<Integer_Type> 
    nnz_col_sizes_all.resize(tiling->ncolgrps);
    nnz_col_sizes_all[owned_segment] = nnz_col_local;
    //Env::barrier();
    for (uint32_t j = 0; j < tiling->ncolgrps; j++)
    {
        uint32_t r = leader_ranks[j];
        if (j != owned_segment)
        {
            MPI_Sendrecv(&nnz_col_sizes_all[owned_segment], 1, MPI_UNSIGNED, r, 0, &nnz_col_sizes_all[j], 1, MPI_UNSIGNED, 
                                                        r, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    //Env::barrier();  
    assert(nnz_col_local == nnz_col_sizes_all[owned_segment]);
    printf("r=%d nnz=%d\n", Env::rank, nnz_col_sizes_all[owned_segment]);
    //struct Basic_Storage<Integer_Type, Integer_Type> *t = new struct Basic_Storage<Integer_Type, Integer_Type>(0);
//F_ = new Vector<Weight, Integer_Type, Fractional_Type>(tile_height, all_colgrp_ranks_accu_seg);

    //nnz_col_sizes_loc(tiling->rank_ncolgrps);
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        this_segment = local_col_segments[j];
        nnz_col_sizes_loc.push_back(nnz_col_sizes_all[this_segment]);
    }
    
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
            printf("%d ", nnz_col_sizes_loc[j]);
    printf(" %d\n", Env::rank);
    
    /*
    if(!Env::rank)
    {
        for(uint32_t j = 0; j < tiling->rowgrp_nranks; j++)
        {
            uint32_t index = find(leader_ranks.begin(), leader_ranks.end(), all_rowgrp_ranks[j]) - leader_ranks.begin();
            //;
            printf("[[[[%d %d %d %d] ", j, all_rowgrp_ranks[j], all_rowgrp_ranks_accu_seg[j], nnz_col_sizes_all[index]);
        }
        

        //leader_ranks[/
        printf("\n");
    }
    */
    
    
    

    C = new Vector<Weight, Integer_Type, Integer_Type>(nnz_col_sizes_loc,  local_col_segments);
    
    std::vector<Integer_Type> tile_height_sizes(tiling->rank_ncolgrps, tile_height);
    J = new Vector<Weight, Integer_Type, Integer_Type>(tile_height_sizes,  local_col_segments);
    /*
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        auto &ej_seg = E->segments[j];
        auto *ej_data = (Integer_Type *) ej_seg.D->data;
        Integer_Type ej_nitems = ej_seg.D->n;
        Integer_Type ej_nbytes = ej_seg.D->nbytes;                        
         printf("[%d %d %d %d]", ej_nitems, ej_nbytes, ej_seg.allocated, tile_height);
    }
    printf(" %d\n", Env::rank);
    */
    
    if(nnz_col_sizes_all[owned_segment])
    {
        auto &cj_seg = C->segments[accu_segment_col];
        auto *cj_data = (Integer_Type *) cj_seg.D->data;
        
        auto &jj_seg = J->segments[accu_segment_col];
        auto *jj_data = (Integer_Type *) jj_seg.D->data;
        
        Integer_Type j = 0;
        for(uint32_t i = 0; i < f_nitems; i++)
        {
            if(f_data[i])
            {
                cj_data[j] = i;
                jj_data[i] = j;
                j++;
                //printf("j=%d ej=%d\n", j-1, ej_data[j-1]);
            }
        }
        assert(j == nnz_col_sizes_all[owned_segment]);
    }
    
    
    
    printf(">>%d %d %d\n", Env::rank, owned_segment, accu_segment_col);    
    
    
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        
        this_segment = local_col_segments[j];
        leader = leader_ranks[this_segment];
        
        auto &cj_seg = C->segments[j];
        auto *cj_data = (Integer_Type *) cj_seg.D->data;
        Integer_Type cj_nitems = cj_seg.D->n;
        Integer_Type cj_nbytes = cj_seg.D->nbytes;
        
        if(cj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++)
                {
                    follower = follower_colgrp_ranks[i];
                    //printf("send %d --> %d %d %d %d\n", leader, follower, this_segment, j, ej_seg.D->n); 

                    MPI_Isend(cj_data, cj_nbytes, MPI_BYTE, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                //printf("recv %d <-- %d %d %d %d\n", Env::rank, leader, this_segment, j, ej_seg.D->n); 
                                        
                MPI_Irecv(cj_data, cj_nbytes, MPI_BYTE, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();
    
    
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        
        this_segment = local_col_segments[j];
        leader = leader_ranks[this_segment];
        
        auto &jj_seg = J->segments[j];
        auto *jj_data = (Integer_Type *) jj_seg.D->data;
        Integer_Type jj_nitems = jj_seg.D->n;
        Integer_Type jj_nbytes = jj_seg.D->nbytes;
        
        if(jj_seg.allocated)
        {
            if(this_segment == owned_segment)
            {
                for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++)
                {
                    follower = follower_colgrp_ranks[i];
                    //printf("send %d --> %d %d %d %d\n", leader, follower, this_segment, j, ej_seg.D->n); 

                    MPI_Isend(jj_data, jj_nbytes, MPI_BYTE, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else
            {
                //printf("recv %d <-- %d %d %d %d\n", Env::rank, leader, this_segment, j, ej_seg.D->n); 
                                        
                MPI_Irecv(jj_data, jj_nbytes, MPI_BYTE, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();
    
    
    /*
    if(Env::rank == 0)
    {
        for(uint32_t j = 0; j < tiling->ncolgrps; j++)
            printf("%d ", nnz_col_sizes_all[j]);
        printf("\n");
        
        for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
        {
            auto &ej_seg = E->segments[j];
            auto *ej_data = (Integer_Type *) ej_seg.D->data;
            Integer_Type ej_nitems = ej_seg.D->n;
            Integer_Type ej_nbytes = ej_seg.D->nbytes;
            
            auto &ij_seg = I->segments[j];
            auto *ij_data = (Integer_Type *) ij_seg.D->data;
            Integer_Type ij_nitems = ij_seg.D->n;
            Integer_Type ij_nbytes = ij_seg.D->nbytes;  
            for(uint32_t i = 0; i < ij_nitems; i++)
            {
               printf("a[%d]=%d %d\n", i, ij_data[i], ej_data[ij_data[i]]);
               
            }
            printf("\n");
        }
        
    }
    */
    
    
    
    
    //struct Vector<Integer_Type, Integer_Type> *isolated_col_;
    /*
    //struct Basic_Storage<Integer_Type, Integer_Type> *isolated_col_;
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        uint32_t this_segment = local_col_segments[j];
        if(this_segment == owned_segment)
        {
            isolated_col_ = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz_col_sizes[owned_segment]);
            if(nnz_col_sizes[owned_segment])
            {
                
                auto *icol_data_ = (Integer_Type *) isolated_col_->data;
                Integer_Type j = 0;
                for(uint32_t i = 0; i < f_nitems; i++)
                {
                    if(f_data[i])
                    {
                        icol_data_[j] = i;
                        j++;
                    }
                }
            }
            //else
                //isolated_col_ = nullptr;
        }
        else
        {
            isolated_col_ = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz_col_sizes[this_segment]);
            //if(nnz_col_sizes[this_segment])
            //    isolated_col_ = new struct Basic_Storage<Integer_Type, Integer_Type>(nnz_col_sizes[this_segment]);
            //else
            //    isolated_col_ = nullptr;
        }
        isolated_col.push_back(isolated_col_);
    }
    
    
    printf("r=%d sz=%lu\n", Env::rank, isolated_col.size());
    
    
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        
        this_segment = local_col_segments[j];
        leader = leader_ranks[this_segment];
        if(this_segment == owned_segment)
        {
            if(nnz_col_sizes[owned_segment])
            {
                for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++)
                {
                    follower = follower_colgrp_ranks[i];
                    printf("send %d --> %d %d %d\n", leader, follower, this_segment, j); 
                    auto *icol_data = (Integer_Type *) isolated_col[j]->data;
                    Integer_Type icol_nitems = isolated_col[j]->n;
                            
                    MPI_Isend(icol_data, icol_nitems, MPI_UNSIGNED, follower, owned_segment, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            
        }
        else
        {
            if(nnz_col_sizes[this_segment])
            {
                //leader = leader_ranks[this_segment];
                printf("recv %d <-- %d %d %d\n", Env::rank, leader, this_segment, j); 

                auto *icol_data = (Integer_Type *)  isolated_col[j]->data;
                Integer_Type icol_nitems = isolated_col[j]->n;
                MPI_Irecv(icol_data, icol_nitems, MPI_UNSIGNED, leader, this_segment, Env::MPI_WORLD, &request);
                in_requests.push_back(request);   
            }
            
        }
    } 
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();
    */
    
    
    /*
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++)
    {
        this_segment = local_col_segments[j];
        if(nnz_col_sizes[this_segment])
        {
            //printf("r=%d s=%d ni=%d sz=%d\n", Env::rank, this_segment, isolated_col[j]->n, tile_height);
            auto *icol_data = (Integer_Type *) isolated_col[j]->data;
            Integer_Type icol_nitems = isolated_col[j]->n;
            for(uint32_t i = 0; i < icol_nitems; i++)
            {
                printf("r=%d s=%d i=%d d=%d\n", Env::rank, j, i, icol_data[i]);
            }
        }
    }
    */
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
    {
        F[i]->del_vec();
        delete F[i];
    }
    //delete E;
   
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
    //Env::barrier();
    for (uint32_t r = 0; r < Env::nranks; r++)
    {
        if (r != Env::rank)
        {
            auto &outbox = outboxes[r];
            uint32_t outbox_size = outbox.size();
            MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, 0, &inbox_sizes[r], 1, MPI_UNSIGNED, 
                                                        r, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    //Env::barrier();
    std::vector<MPI_Request> outreqs;
    std::vector<MPI_Request> inreqs;
    MPI_Request request;
    MPI_Status status;
     
     
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
            outreqs.push_back(request);
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

            inreqs.push_back(request);
        }
    }

     

    
    
    //MPI_Request request;



    
    MPI_Waitall(inreqs.size(), inreqs.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(outreqs.size(), outreqs.data(), MPI_STATUSES_IGNORE);
    //Env::barrier();
    
    for (uint32_t r = 0; r < Env::nranks; r++)
    {
        if (r != Env::rank)
        {
            auto &inbox = inboxes[r];
            for (uint32_t i = 0; i < inbox_sizes[r]; i++)
                insert(inbox[i]);

            inbox.clear();
            inbox.shrink_to_fit();
        }
    }
    
    
    

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
    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1,MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    MPI_Allreduce(&nedges_end_local, &nedges_end_global, 1,MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges_start_global == nedges_end_global);
    if(Env::is_master)
        printf("Sanity check for exchanging %lu edges is done\n", nedges_end_global);
    
    auto retval = MPI_Type_free(&MANY_TRIPLES);
    assert(retval == MPI_SUCCESS);   
    Env::barrier();
    
        /*
        1121410
        1130954
        975723
        977375
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csr()
{
        uint64_t nedges_local = 0;
    uint64_t nedges_global = 0;
    /* Create the the csr format by allocating the csr data structure
       and then Sorting triples and populating the csr */
    struct Triple<Weight, Integer_Type> pair;
    RowSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        if(tile.allocated)
        {
            tile.csr = new struct CSR<Weight, Integer_Type>(tile.triples->size(), tile_height + 1);
            //tile.allocated = true;
        }        
        
        std::sort(tile.triples->begin(), tile.triples->end(), f);
        
        uint32_t i = 0; // CSR Index
        uint32_t j = 1; // Row index
        if(tile.allocated)
        {
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
                nedges_local++;
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
        }
    }    
    del_triples();
    
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(!Env::rank)
            printf("1. %lu\n", nedges_global);
        
                  
        
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csc()
{
    struct Triple<Weight, Integer_Type> pair;
    ColSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        if(tile.allocated)
        {
            //tile.nedges = tile.triples->size();
            tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            //tile.allocated = true;
        }        
        
        std::sort(tile.triples->begin(), tile.triples->end(), f);
        
        uint32_t i = 0; // CSR Index
        uint32_t j = 1; // Row index
        if(tile.allocated)
        {
            #ifdef HAS_WEIGHT
            Weight *VAL = (Weight *) tile.csc->VAL->data;
            #endif
            
            Integer_Type *ROW_INDEX = (Integer_Type *) tile.csc->ROW_INDEX->data; // JA
            Integer_Type *COL_PTR = (Integer_Type *) tile.csc->COL_PTR->data; // IA
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
    //printf("DONE compression %d\n",Env::rank);
    del_triples();
    //printf("DONE deletion %d\n",Env::rank);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_compression()
{
    if(compression == Compression_type::_CSR_)
    {
        del_csr();
    }
    else if(compression == Compression_type::_CSC_)
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


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_triples()
{
    // Delete triples
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.free_triples();
    }
}

#endif
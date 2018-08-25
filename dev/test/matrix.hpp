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


enum Vertex_type
{
  _SRC,
  _SNK,
  _ISO,
  _REG
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
        
        int32_t owned_segment, accu_segment_rg, accu_segment_cg;
        // In case of owning multiple segments
        std::vector<int32_t> owned_segment_vec;
        std::vector<int32_t> accu_segment_rg_vec;
        std::vector<int32_t> accu_segment_cg_vec;
        
        void init_matrix();
        void del_triples();
        void init_compression();
        void init_csr();
        void init_csc();
        void init_bv();
        void del_csr();
        void del_csc();
        void del_compression();
        void print(std::string element);
        void distribute();
        void filter();
        
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
Matrix<Weight, Integer_Type, Fractional_Type>::~Matrix()
{
    delete tiling;
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
    
    // Calculate row/col leader ranks and accumulator segments
/*    
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        if(pair.row == pair.col)
        {
            all_rowgrp_ranks.push_back(tiles[pair.row][pair.col].rank);
            all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][pair.col].cg);
            
            all_rowgrp_ranks_rg.push_back(tiles[pair.row][pair.col].rank_rg);
            all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][pair.col].cg);
            for(uint32_t j = 0; j < ncolgrps; j++)
            {
                if((tiles[pair.row][j].rank != Env::rank) 
                    and (std::find(follower_rowgrp_ranks.begin(), follower_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                    == follower_rowgrp_ranks.end()))
                {
                    all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                    all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                    
                    follower_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                    follower_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                    
                    all_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                    all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                    
                    follower_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                    follower_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                    
                }
            }
            
            all_colgrp_ranks.push_back(tiles[pair.row][pair.col].rank);
            all_colgrp_ranks_accu_seg.push_back(tiles[pair.row][pair.col].rg);
            
            all_colgrp_ranks_cg.push_back(tiles[pair.row][pair.col].rank_cg);
            all_colgrp_ranks_accu_seg_cg.push_back(tiles[pair.row][pair.col].rg);
            
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                if((tiles[i][pair.col].rank != Env::rank) 
                    and (std::find(follower_colgrp_ranks.begin(), follower_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                    == follower_colgrp_ranks.end()))
                {
                    all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                    all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                    
                    follower_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                    follower_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                    
                    all_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                    all_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                    
                    follower_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                    follower_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                }
            }
        }
    }
*/    
    
    
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
                        
                        accu_segment_rg = tiles[pair.row][j].cg;
                        accu_segment_rg_vec.push_back(accu_segment_rg);
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
                        
                        accu_segment_cg = tiles[i][pair.col].rg;
                        accu_segment_cg_vec.push_back(accu_segment_cg);
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

    
 
    // Print tiling assignment
    print("rank");
    Env::barrier();
    
        
    
    


    
    //uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
      //        std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
    
    // Optimization: Spilitting communicator among row/col groups   
    if(Env::comm_split)
    {
        indexed_sort(all_rowgrp_ranks, all_rowgrp_ranks_accu_seg);
        indexed_sort(all_rowgrp_ranks_rg, all_rowgrp_ranks_accu_seg_rg);
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
        // Make sure there is at least one follower
        if(follower_rowgrp_ranks.size())
        {
            indexed_sort(follower_rowgrp_ranks, follower_rowgrp_ranks_accu_seg);
            indexed_sort(follower_rowgrp_ranks_rg, follower_rowgrp_ranks_accu_seg_rg);
        }
        
        indexed_sort(all_colgrp_ranks, all_colgrp_ranks_accu_seg);
        indexed_sort(all_colgrp_ranks_cg, all_colgrp_ranks_accu_seg_cg);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        if(follower_colgrp_ranks.size())
        {
            indexed_sort(follower_colgrp_ranks, follower_colgrp_ranks_accu_seg);
            indexed_sort(follower_colgrp_ranks_cg, follower_colgrp_ranks_accu_seg_cg);
        }
    }
    
    
    
    uint32_t other, accu;
    if(!Env::rank)
    {
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
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression()
{
    if(Env::is_master)
        printf("Edge distribution among %d ranks\n", Env::nranks);
    
    distribute();
    filter();
    
    
    
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
}

#include "vector.hpp"

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter()
{
    //print("nedges");
    uint32_t leader;
    uint32_t other;
    uint32_t accu;
    MPI_Request request;
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    
    //std::vector<uint64_t> inboxes(tiling->rowgrp_nranks);
    //memset(inboxes, 0, tiling->rowgrp_nranks * sizeof(uint64_t));
    //std::vector<std::vector<uint64_t>> inbox_sizes(tiling->rowgrp_nranks, std::vector<uint64_t>(tiling->colgrp_nranks));
    
    //= new Vector<Weight, Integer_Type, Fractional_Type>(nrows, local_col_segments);

    //XX->del_vec_1();
    
    //printf("r=%d own=%d accu=%d\n", Env::rank, Env::owned_segment, Env::accu_segment);   
 
 /*
    uint32_t owned_diag_segment = std::distance(A->leader_ranks.begin(), 
          std::find(A->leader_ranks.begin(), A->leader_ranks.end(), Env::rank));
    X = new Vector<Weight, Integer_Type, Fractional_Type>(A->nrows, A->ncols, A->nrowgrps, A->ncolgrps, 
                A->tile_height, A->tile_width, owned_diag_segment, A->leader_ranks,
                A->leader_ranks_rg, A->leader_ranks_cg, A->local_col_segments);
    for(uint32_t xi : X->local_segments)
    {
        auto &x_seg = X->segments[xi];
        populate(x, x_seg);
    }
*/ 

//Vector<Empty, Integer_Type, Fractional_Type> *S;
  //      std::vector<Vector<Empty, Integer_Type, Fractional_Type> *> Y;
 //Vector<Weight, Integer_Type, Fractional_Type> *X;
    //Vector<Weight, Integer_Type, Fractional_Type> *X = new Vector<Weight, Integer_Type, Fractional_Type>(nrows, local_col_segments);
    std::vector<Vector<Weight, Integer_Type, uint64_t> *> X;
    Vector<Weight, Integer_Type, uint64_t> *XX;

    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];

        if(((tile.mth + 1) % tiling->rank_nrowgrps) == 0)
        {
            if(pair.col == owned_segment)
            {
                //printf("ALL %d\n" ,t);
                XX = new Vector<Weight, Integer_Type, uint64_t>(1, all_colgrp_ranks_accu_seg);
                //if(Env::rank == 1)
                    //printf("ALL %d %lu %d %d\n" ,t, all_colgrp_ranks_accu_seg.size(), all_colgrp_ranks_accu_seg[0], all_colgrp_ranks_accu_seg[1]);
            }
            else
            {
                XX = new Vector<Weight, Integer_Type, uint64_t>(1, accu_segment_cg_vec);
                //printf("ACCU %d\n", t);
            }
            X.push_back(XX);
        }
    }
    //printf("SZ=%lu\n",X.size());
    uint32_t x = 0, xi = 0, xj = 0, xk = 0, xo = 0; 
    Vector<Weight, Integer_Type, uint64_t> *Xp = nullptr;    
    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        
        if(pair.col == owned_segment)
            xo = accu_segment_cg;
        else
            xo = 0;
        
        Xp = X[xi];
        auto &x_seg = Xp->segments[xo];
        auto *x_data = (uint64_t *) x_seg.D->data;
        Integer_Type x_nitems = x_seg.D->n;
        Integer_Type x_nbytes = x_seg.D->nbytes;
        if(tile.allocated)
        {
            for(uint32_t i = 0; i < x_nitems; i++)
                x_data[0] += tile.nedges;             
        }
        
        bool communication = (((tile.mth + 1) % tiling->rank_nrowgrps) == 0);
        if(communication)
        {
            if(Env::comm_split)
                leader = tile.leader_rank_cg_cg;
            else
                leader = tile.leader_rank_cg;
            
            if(leader == Env::rank)
            {
                for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
                {
                    if(Env::comm_split)
                    {
                        
                        other = follower_colgrp_ranks_cg[j];
                        accu = follower_colgrp_ranks_accu_seg_cg[j];
                        auto &xj_seg = Xp->segments[accu];
                        auto *xj_data = (uint64_t *) xj_seg.D->data;
                        Integer_Type xj_nitems = xj_seg.D->n;
                        Integer_Type xj_nbytes = xj_seg.D->nbytes;
                        
                        MPI_Irecv(xj_data, xj_nbytes, MPI::UNSIGNED_LONG, other, pair.col, Env::colgrps_comm, &request);
                        if(!Env::rank)
                            printf("<< <accu=%d xk=%d %d %p d=%lu>>>\n", accu, xk, xi, xj_seg, xj_data[0]);
                    }
                    else
                    {
                        other = follower_colgrp_ranks[j];
                        accu = follower_colgrp_ranks_accu_seg[j];
                        auto &xj_seg = Xp->segments[accu];
                        auto *xj_data = (uint64_t *) xj_seg.D->data;
                        Integer_Type xj_nitems = xj_seg.D->n;
                        Integer_Type xj_nbytes = xj_seg.D->nbytes;
                        MPI_Irecv(xj_data, xj_nbytes, MPI::UNSIGNED_LONG, other, pair.col, Env::MPI_WORLD, &request);
                    }
                    in_requests.push_back(request);
                }
                xk = xi;
            }
            else 
            {
                if(Env::comm_split)
                    MPI_Isend(x_data, x_nbytes, MPI::UNSIGNED_LONG, leader, pair.col, Env::colgrps_comm, &request);
                else
                    MPI_Isend(x_data, x_nbytes, MPI::UNSIGNED_LONG, leader, pair.col, Env::MPI_WORLD,    &request);
                out_requests.push_back(request);
                if(Env::rank == 1)
                    printf("<<<send xk=%d xi=%d>>>\n", xk, xi);
            }                
            xi++;
            //if(!Env::rank)
              //  printf("%d %d %d\n" , t, xk, xi);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    
    
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    Env::barrier();
    if(!Env::rank)
        printf("xk=%d\n", xk);
    Xp = X[xk];
    xo = accu_segment_cg;
    auto &x_seg = Xp->segments[xo];
    auto *x_data = (uint64_t *) x_seg.D->data;
    Integer_Type x_nitems = x_seg.D->n;
    Integer_Type x_nbytes = x_seg.D->nbytes;
    
    for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
    {
        accu = follower_colgrp_ranks_accu_seg_cg[j];
        auto &xj_seg = Xp->segments[accu];
        //auto *xj_data = (uint64_t *) xj_seg.D->data;
        //Integer_Type xj_nitems = xj_seg.D->n;
        //if(Env::rank == 1)
           //printf("vl=%d\n", Xp->local_segments);
        //
        //Integer_Type xj_nitems = xj_seg.D->n;
        //Integer_Type xj_nbytes = xj_seg.D->nbytes;
        //for(uint32_t i = 0; i < x_nitems; i++)
        //    x_data[i] += xj_data[i];        
    }
    
    

    
    //printf("r=%d e=%lu\n", Env::rank, x_data[0]);
        
        
            //printf("mine %d %d %d %d %d\n", t, x_nitems, x_nbytes, x_seg.g, accu_segment_cg);
            /*
                for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
                {
                    xj = follower_colgrp_ranks_accu_seg[j];
                    auto &xj_seg = Xp->segments[xj];
                    auto *xj_data = (uint64_t *) xj_seg.D->data;
                    Integer_Type xj_nitems = xj_seg.D->n;
                    Integer_Type xj_nbytes = xj_seg.D->nbytes;
                    //printf("mine %d %d %d %d %d %d %d\n", t, j, follower_colgrp_ranks_accu_seg[j], follower_colgrp_ranks_cg[j], xj_nitems, xj_nbytes, xj_seg.g);
                }
                */
                /*
            }
            else
            {
                auto &x_seg = Xp->segments[0];
                auto *x_data = (uint64_t *) x_seg.D->data;
                Integer_Type x_nitems = x_seg.D->n;
                Integer_Type x_nbytes = x_seg.D->nbytes;  
               // printf("Not mine %d %d %d %d\n", t, x_nitems, x_nbytes, x_seg.g);
            }
       // }
        */
        //if(tile.allocated)
        //{
            //inbox_sizes += tile.nedges;
            
        //}   
        
        

        
        //printf("t=%d c=%d rg=%d\n", t, tile.mth, ((tile.mth + 1) % tiling->rank_nrowgrps) == 0);
    //}
         
    
    
    
               
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
    {
        X[i]->del_vec_1();
        delete X[i];
    }
    
    /*
    for(uint32_t t: local_tiles_col_order)
    {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        
        if(tile.allocated)
        {
            if(Env::comm_split)
                leader = tile.leader_rank_cg_cg;
            else
                leader = tile.leader_rank_cg;
            
            if(leader == Env::rank)
            {
                for(uint32_t j = 0; j < tiling->colgrp_nranks - 1; j++)
                {
                    if(Env::comm_split)
                    {
                        other = follower_colgrp_ranks_cg[j];
                        accu = follower_colgrp_ranks_accu_seg_cg[j];
                        if(!Env::rank)
                            printf("t=%d o=%d acc=%d\n", t, other, accu);
                        MPI_Irecv(&inbox_sizes[accu], 1, MPI::UNSIGNED_LONG, other, pair.row, Env::rowgrps_comm, &request);
                        //MPI_Wait(&request, MPI_STATUS_IGNORE);
                        
                    }
                    else
                    {
                        other = follower_colgrp_ranks[j];
                        accu = follower_colgrp_ranks_accu_seg[j];
                        MPI_Irecv(&inbox_sizes[accu], 1, MPI::UNSIGNED_LONG, other, pair.row, Env::MPI_WORLD, &request);
                        //MPI_Wait(&request, MPI_STATUS_IGNORE);
                    }
                    //inbox_sizes[accu]
                    in_requests.push_back(request);
                }
            }
            else
            {
                if(Env::comm_split)
                    MPI_Isend(&tile.nedges, 1, MPI::UNSIGNED_LONG, leader, pair.row, Env::colgrps_comm, &request);
                else
                    MPI_Isend(&tile.nedges, 1, MPI::UNSIGNED_LONG, leader, pair.row, Env::MPI_WORLD,    &request);
                out_requests.push_back(request);
            }   
        }
        
    }
    
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    
    
    
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    */
    
    
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
    std::vector<std::vector<Triple<Weight>>> inboxes(Env::nranks);
    std::vector<uint32_t> inbox_sizes(Env::nranks);
    
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            if(tile.rank != Env::rank)
            {
                auto &outbox = outboxes[tile.rank];
                outbox.insert(outbox.end(), tile.triples->begin(), tile.triples->end());
                tile.free_triples();
            }
        }
    }
    
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
    
    std::vector<MPI_Request> outreqs;
    std::vector<MPI_Request> inreqs;
    MPI_Request request;

    for (uint32_t i = 0; i < Env::nranks; i++)
    {
        uint32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank)
        {
            auto &inbox = inboxes[r];
            uint32_t inbox_bound = inbox_sizes[r] + many_triples_size;
            inbox.resize(inbox_bound);
            /* Recv the triples with many_triples_size padding. */
            MPI_Irecv(inbox.data(), inbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);
            inreqs.push_back(request);
        }
    }
    
    for (uint32_t i = 0; i < Env::nranks; i++)
    {
        uint32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank)
        {
            auto &outbox = outboxes[r];
            uint32_t outbox_bound = outbox.size() + many_triples_size;
            outbox.resize(outbox_bound);
            /* Send the triples with many_triples_size padding. */
            MPI_Isend(outbox.data(), outbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);
            outreqs.push_back(request);
        }
    }
    
    MPI_Waitall(inreqs.size(), inreqs.data(), MPI_STATUSES_IGNORE);

    
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
    
    MPI_Waitall(outreqs.size(), outreqs.data(), MPI_STATUSES_IGNORE);
    Env::barrier();

    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        nedges_end_local += tile.triples->size();
        tile.nedges = tile.triples->size();
        if(tile.nedges)
            tile.allocated = true;
    }
    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1,MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    MPI_Allreduce(&nedges_end_local, &nedges_end_global, 1,MPI::UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges_start_global == nedges_end_global);
    if(Env::is_master)
        printf("Sanity check for exchanging %lu edges is done\n", nedges_end_global);
    
    auto retval = MPI_Type_free(&MANY_TRIPLES);
    assert(retval == MPI_SUCCESS);   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csr()
{
    /* Create the the csr format by allocating the csr data structure
       and then Sorting triples and populating the csr */
    struct Triple<Weight, Integer_Type> pair;
    RowSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles_row_order)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        if(tile.triples->size())
        {
            tile.csr = new struct CSR<Weight, Integer_Type>(tile.triples->size(), tile_height + 1);
            tile.allocated = true;
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
        
        if(tile.triples->size())
        {
            tile.nedges = tile.triples->size();
            tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            tile.allocated = true;
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
    
    del_triples();
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
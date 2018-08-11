/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <cmath>
#include "tiling.hpp" 
#include <algorithm>

 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D
{ 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight>>* triples;
    //struct CSR<Weight>* csr;
    //struct BV<char> *bv;
    uint32_t rg, cg;
    uint32_t ith, jth, nth;
    uint32_t rank;
}; 
 
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Matrix
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    public:    
        Matrix(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, Tiling_type tiling_type);
        ~Matrix();

    private:
        Integer_Type nrows, ncols;
        uint32_t tile_height, tile_width;    
        uint32_t ntiles, nrowgrps, ncolgrps;
        
        Tiling* tiling;

        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles;
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_segments;
        std::vector<uint32_t> local_col_segments;
        
        std::vector<uint32_t> leader_ranks;
        std::vector<uint32_t> other_row_ranks_accu_seg;
        std::vector<uint32_t> other_col_ranks_accu_seg;

        std::vector<uint32_t> other_rowgrp_ranks;
        std::vector<uint32_t> rowgrp_ranks_accu_seg;
        std::vector<uint32_t> other_colgrp_ranks; 
        std::vector<uint32_t> colgrp_ranks_accu_seg;
        
        void init_matrix();
        void del_triples();
        void init_csr();
        void init_bv();
        void del_csr();
        
        uint32_t local_tile_of_tile(const struct Triple<Weight>& pair);
        uint32_t segment_of_tile(const struct Triple<Weight>& pair);
        struct Triple<Weight> tile_of_triple(const struct Triple<Weight>& triple);
        struct Triple<Weight> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight> rebase(const struct Triple<Weight>& pair);
        struct Triple<Weight> base(const struct Triple<Weight>& pair, Integer_Type rowgrp, Integer_Type colgrp);
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, Tiling_type tiling_type)
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
    init_matrix();
    

    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::~Matrix()
{
    delete tiling;
};
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
                
            }
            if(tiling->tiling_type == Tiling_type::_1D_COL)
            {
                tile.rank = j;
                tile.ith  = tile.rg / tiling->colgrp_nranks;
                tile.jth  = tile.cg / tiling->rowgrp_nranks;   
            }
            else if(tiling->tiling_type == Tiling_type::_2D_)
            {
                tile.rank = ((j % tiling->rowgrp_nranks) * tiling->colgrp_nranks) +
                                   (i % tiling->colgrp_nranks);
                tile.ith = tile.rg   / tiling->colgrp_nranks;
                tile.jth = tile.cg   / tiling->rowgrp_nranks;
            }
            tile.nth  = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
        }
    }
    
    /*
    * Reorganize the tiles so that each rank is placed in
    * at least one diagonal tile then calculate 
    * the leader ranks per row group.
    */
    leader_ranks.resize(nrowgrps, -1);
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = i; j < ncolgrps; j++)  
        {
            if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[i][j].rank)
                 != leader_ranks.end()))
            {
                std::swap(tiles[j], tiles[i]);
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
    }
    
    
    struct Triple<Weight> pair;
    printf("%lu\n", sizeof(pair));
    for (uint32_t i = 0; i < nrowgrps; i++)
    {
        for (uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            tile.rg = i;
            tile.cg = j;
            if(tile.rank == Env::rank)
            {
                pair.row = i;
                pair.col = j;    
                local_tiles.push_back(local_tile_of_tile(pair));
                printf("%d %d %d %d\n", local_tile_of_tile(pair), tile.ith, tile.jth, tile.nth);

                if (std::find(local_col_segments.begin(), local_col_segments.end(), pair.col) == local_col_segments.end())
                {
                    local_col_segments.push_back(pair.col);
                }
            }
        }
    }
    
    
    
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::local_tile_of_tile(const struct Triple<Weight>& pair)
{
  return((pair.row * ncolgrps) + pair.col);
}


/*
template<typename Weight>
void Matrix<Weight>::init_mat()
{    


    
    
    
    

    
    


    for(uint32_t t: Matrix<Weight>::local_tiles)
    {
        pair = Matrix<Weight>::tile_of_local_tile(t);
        if(pair.row == pair.col)
        {
            for(uint32_t j = 0; j < Matrix<Weight>::ncolgrps; j++)
            {
                if((Matrix<Weight>::tiles[pair.row][j].rank != rank) 
                    and (std::find(other_rowgrp_ranks.begin(), other_rowgrp_ranks.end(), Matrix<Weight>::tiles[pair.row][j].rank) == other_rowgrp_ranks.end()))
                {
                    Matrix<Weight>::other_rowgrp_ranks.push_back(Matrix<Weight>::tiles[pair.row][j].rank);
                    Matrix<Weight>::rowgrp_ranks_accu_seg.push_back(Matrix<Weight>::tiles[pair.row][j].cg);
                }
            }
            
            for(uint32_t i = 0; i < Matrix<Weight>::nrowgrps; i++)
            {
                if((Matrix<Weight>::tiles[i][pair.col].rank != rank) 
                    and (std::find(other_colgrp_ranks.begin(), other_colgrp_ranks.end(), Matrix<Weight>::tiles[i][pair.col].rank) == other_colgrp_ranks.end()))
                {
                    Matrix<Weight>::other_colgrp_ranks.push_back(Matrix<Weight>::tiles[i][pair.col].rank);
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
                if(j > 15)
                {
                    printf("...");
                    break;
                }
            }
            printf("\n");
            if(i > 15)
            {
                printf(".\n.\n.\n");
                break;
            }
        }
        printf("\n");
    }
}
*/



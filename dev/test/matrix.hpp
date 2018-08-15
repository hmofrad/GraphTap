/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <cmath>
#include <algorithm>
#include <vector>
#include "tiling.hpp" 




//#include <type_traits>


 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D
{ 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight, Integer_Type>> *triples;
    struct CSR<Weight, Integer_Type> *csr;
    struct CSC<Weight, Integer_Type> *csc;
    //struct BV<char> *bv;
    uint32_t rg, cg; // Row group, Column group
    uint32_t ith, jth, nth; // ith row, jth column and nth local tile
    uint32_t kth; // kth global tile
    uint32_t rank;
    bool allocated;
}; 
 
 
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
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_segments;
        std::vector<uint32_t> local_col_segments;
        
        std::vector<uint32_t> leader_ranks;

        std::vector<uint32_t> follower_rowgrp_ranks;
        std::vector<uint32_t> rowgrp_ranks_accu_seg;
        std::vector<uint32_t> follower_colgrp_ranks; 
        std::vector<uint32_t> colgrp_ranks_accu_seg;
        
        std::vector<std::vector<uint32_t>> rowgrp_ranks_accu_seg_dup;
        
        void init_matrix();
        void del_triples();
        void init_compression();
        void init_csr();
        void init_csc();
        void init_bv();
        void del_csr();
        void del_csc();
        
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type> &triple);
        uint32_t segment_of_tile(const struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type> &pair);
        void insert(const struct Triple<Weight, Integer_Type> &triple);
        
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, Tiling_type tiling_type, Compression_type compression_type)
      //: nrows(nrows_), ncols(ncols_), ntiles(ntiles_), nrowgrps(sqrt(ntiles_)), ncolgrps(ntiles_ / nrowgrps),
      //  tile_height((nrows_ / nrowgrps) + 1), tile_width((ncols_ / ncolgrps) + 1)
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
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp)
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
    tiles[triple.row][triple.col].triples->push_back(triple);
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
            tile.nth   = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.allocated = false;
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
            if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[j][i].rank)
                 != leader_ranks.end()))
            {
                std::swap(tiles[j], tiles[i]);
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
        //if(!Env::rank)
          //  printf("%d ", leader_ranks[i]);
    }
    //if(!Env::rank)
      //  printf("\n");
    
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
                if (std::find(local_col_segments.begin(), local_col_segments.end(), pair.col) == local_col_segments.end())
                {
                    local_col_segments.push_back(pair.col);
                }
            }
        }
    }
    
    // Calculate leader ranks and accumulator segments
    for(uint32_t t: local_tiles)
    {
        pair = tile_of_local_tile(t);
        if(pair.row == pair.col)
        {
            for(uint32_t j = 0; j < ncolgrps; j++)
            {
                if((tiles[pair.row][j].rank != Env::rank) 
                    and (std::find(follower_rowgrp_ranks.begin(), follower_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                    == follower_rowgrp_ranks.end()))
                {
                    follower_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                    rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                }
            }
            
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                if((tiles[i][pair.col].rank != Env::rank) 
                    and (std::find(follower_colgrp_ranks.begin(), follower_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                    == follower_colgrp_ranks.end()))
                {
                    follower_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                    colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                }
            }
        }
    }
    rowgrp_ranks_accu_seg_dup.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        rowgrp_ranks_accu_seg_dup[i].resize(tiling->rowgrp_nranks - 1);
    
    if(!Env::rank)
    {
        /*
        for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        {
            for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++)    
            {
                rowgrp_ranks_accu_seg_dup[i][j] = rowgrp_ranks_accu_seg[j];
                printf("%d ", rowgrp_ranks_accu_seg_dup[i][j]);
            }
            printf("\n");
        }
        */
        /*
        for(uint32_t s: follower_rowgrp_ranks)
            printf("%d ", s);
        printf("\n");
        
        for(uint32_t s: rowgrp_ranks_accu_seg)
            printf("%d %d ", s, tiling->rank_nrowgrps);
        printf("\n");
        */
        
    }
    
    
    // Initialize triples 
    for(uint32_t t: local_tiles)
    {
        
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.triples = new std::vector<struct Triple<Weight, Integer_Type>>;
    }
    
    // Print tiling assignment
    if(Env::is_master)
    {    
        uint32_t skip = 16;
        for (uint32_t i = 0; i < nrowgrps; i++)
        {
            for (uint32_t j = 0; j < ncolgrps; j++)  
            {
                auto& tile = tiles[i][j];
                printf("%02d ", tile.rank);
                
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


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csr()
{
    /* Check if weights are empty or not */
    bool has_weight = (std::is_same<Weight, Empty>::value) ? false : true;
    //printf("type=%d\n", has_weight);
    
    /* Create the the csr format by allocating the csr data structure
       and then Sorting triples and populating the csr */
    struct Triple<Weight, Integer_Type> pair;
    RowSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles)
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
            Weight *A = nullptr;
            if(has_weight)
            {
                A = (Weight *) tile.csr->A->data;
            }
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
                if(has_weight)
                {
                    A[i] = triple.weight;
                }
                IA[j]++;
                JA[i] = pair.col;    
                i++;
                //printf("%d %d %d\n", triple.row, triple.col, triple.weight);
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
    bool has_weight = (std::is_same<Weight, Empty>::value) ? false : true;
    
    struct Triple<Weight, Integer_Type> pair;
    ColSort<Weight, Integer_Type> f;
    for(uint32_t t: local_tiles)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        
        if(tile.triples->size())
        {
            tile.csc = new struct CSC<Weight, Integer_Type>(tile.triples->size(), tile_width + 1);
            tile.allocated = true;
        }        
        
        std::sort(tile.triples->begin(), tile.triples->end(), f);
        
        uint32_t i = 0; // CSR Index
        uint32_t j = 1; // Row index
        if(tile.allocated)
        {
            Weight *VAL = nullptr;
            if(has_weight)
            {
                VAL = (Weight *) tile.csc->VAL->data;
            }
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
                if(has_weight)
                {
                    VAL[i] = triple.weight;
                }
                COL_PTR[j]++;
                ROW_INDEX[i] = pair.row;    
                i++;

                //printf("%d %d %d %d %d\n", t, triple.row, triple.col, pair.row, pair.col);
                //i++;
            }
            
            while(j < tile_width)
            {
                j++;
                COL_PTR[j] = COL_PTR[j - 1];
            }  
            
            /*
        uint32_t k = 0;
        for(i = 0; i < tile.csc->ncols_plus_one - 1; i++)
        {
            uint32_t nnz_per_col = COL_PTR[i + 1] - COL_PTR[i];
                for(j = 0; j < nnz_per_col; j++)
                {
                    printf("%d %d\n", i, ROW_INDEX[k]);
                    k++;
                }
        }
            printf("%d\n", tile.csc->ncols_plus_one );
*/            
        }
    }    
    del_triples();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_csr()
{
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles)
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
    for(uint32_t t: local_tiles)
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
    for(uint32_t t: local_tiles)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        delete tile.triples;        
    }
    
}

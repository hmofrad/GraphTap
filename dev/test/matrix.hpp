/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

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
    uint32_t ith, jth, nth, kth; // ith row, jth column, nth local tile and kth global tile
    uint32_t rank;
    uint32_t rank_rg, rank_cg;
    Integer_Type nedges_tile;
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
        std::vector<int32_t> local_col_segments;
        
        std::vector<uint32_t> leader_ranks;

        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        std::vector<int32_t> follower_colgrp_ranks; 
        std::vector<int32_t> follower_colgrp_ranks_accu_seg;
        
        std::vector<uint32_t> leader_ranks_rg;
        std::vector<uint32_t> leader_ranks_cg;
        
        std::vector<int32_t> all_rowgrp_ranks;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        std::vector<int32_t> all_colgrp_ranks; 
        std::vector<int32_t> all_colgrp_ranks_accu_seg;
        
        
        std::vector<int32_t> all_rowgrp_ranks_rg;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg_rg;
        
        std::vector<int32_t> follower_rowgrp_ranks_rg;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg_rg;
        
        void init_matrix();
        void del_triples();
        void init_compression();
        void init_csr();
        void init_csc();
        void init_bv();
        void del_csr();
        void del_csc();
        void del_compression();
        void print();
        
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type> &triple);
        uint32_t segment_of_tile(const struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type> &pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type> &pair);
        void insert(const struct Triple<Weight, Integer_Type> &triple);
        
        std::vector<int32_t> sort_indices(const std::vector<int32_t> &v);
        void indexed_sort(std::vector<int32_t> &v1, std::vector<int32_t> &v2);
        
        void distribute();
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
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
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
            }
            if(tiling->tiling_type == Tiling_type::_1D_COL)
            {
                tile.rank = j;
                tile.ith  = tile.rg / tiling->colgrp_nranks;
                tile.jth  = tile.cg / tiling->rowgrp_nranks;

                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
            }
            else if(tiling->tiling_type == Tiling_type::_2D_)
            {
                tile.rank = ((j % tiling->rowgrp_nranks) * tiling->colgrp_nranks) +
                                   (i % tiling->colgrp_nranks);
                tile.ith = tile.rg   / tiling->colgrp_nranks;
                tile.jth = tile.cg   / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
            }
            tile.nth   = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
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
            all_rowgrp_ranks.push_back(tiles[pair.row][pair.col].rank);
            all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][pair.col].cg);
            for(uint32_t j = 0; j < ncolgrps; j++)
            {
                if((tiles[pair.row][j].rank != Env::rank) 
                    and (std::find(follower_rowgrp_ranks.begin(), follower_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                    == follower_rowgrp_ranks.end()))
                {
                    follower_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                    follower_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                    
                    all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                    all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                    
                    follower_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                    follower_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                    
                }
            }
            
            all_colgrp_ranks.push_back(tiles[pair.row][pair.col].rank);
            all_colgrp_ranks_accu_seg.push_back(tiles[pair.row][pair.col].rg);
            for(uint32_t i = 0; i < nrowgrps; i++)
            {
                if((tiles[i][pair.col].rank != Env::rank) 
                    and (std::find(follower_colgrp_ranks.begin(), follower_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                    == follower_colgrp_ranks.end()))
                {
                    follower_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                    follower_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                    
                    all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                    all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                    
                }
            }
        }
    }

    // Print tiling assignment
    print();
    
    // Optimization: Spilitting communicator among row/col groups   
    if(Env::comm_split)
    {
        indexed_sort(all_rowgrp_ranks, all_rowgrp_ranks_accu_seg);
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
        
        indexed_sort(all_colgrp_ranks, all_colgrp_ranks_accu_seg);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        
        if(follower_rowgrp_ranks.size() > 0)
        {
            indexed_sort(follower_rowgrp_ranks, follower_rowgrp_ranks_accu_seg);
        
            indexed_sort(follower_rowgrp_ranks_rg, follower_rowgrp_ranks_accu_seg_rg);
            follower_rowgrp_ranks_accu_seg_rg = follower_rowgrp_ranks_accu_seg;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::print()
{
    if(Env::is_master)
    {    
        uint32_t skip = 15;
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
    for(uint32_t t: local_tiles)
    {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        nedges_end_local += tile.triples->size();
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
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression()
{
    distribute();
    
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
            
           
        uint32_t k = 0;
        for(j = 0;j < tile.csc->ncols_plus_one - 1; j++)
        {
            uint32_t nnz_per_col = COL_PTR[j + 1] - COL_PTR[j];
                //for(i = 0; i < nnz_per_col; i++)
                for(uint32_t i = COL_PTR[j]; i < COL_PTR[j + 1]; i++)
                {
                    //printf("%d %d %d\n", i, ROW_INDEX[k], VAL[k]);
                    printf("r=%d t=%d ci=%d =ri%d w=%d \n", Env::rank, t, j, ROW_INDEX[j], 1);
                    k++;
                }
        }
            //printf("%d\n", tile.csc->ncols_plus_one );
        
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
        tile.free_triples();
    }
}

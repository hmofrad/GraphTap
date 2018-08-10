/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <cmath>
 
enum Tiling
{
  _1D_ROW,
  _1D_COL,
  _2D_
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Matrix
{
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    public:    
        Matrix(Integer_Type nrows, Integer_Type ncols, Integer_Type ntiles, Tiling tiling);
        ~Matrix();

    private:
        const Integer_Type nrows, ncols;
        const Integer_Type ntiles, nrowgrps, ncolgrps;
        const Integer_Type tile_height, tile_width;    
        
        //Partitioning<Weight>* partitioning;

        //std::vector<std::vector<struct Tile2D<Weight>>> tiles;
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_segments;
        std::vector<uint32_t> local_col_segments;
        
        std::vector<uint32_t> diag_ranks;
        std::vector<uint32_t> other_row_ranks_accu_seg;
        std::vector<uint32_t> other_col_ranks_accu_seg;

        std::vector<uint32_t> other_rowgrp_ranks;
        std::vector<uint32_t> rowgrp_ranks_accu_seg;
        std::vector<uint32_t> other_colgrp_ranks; 
        std::vector<uint32_t> colgrp_ranks_accu_seg;
        
        void init_mat();
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
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows, Integer_Type ncols, Integer_Type ntiles, Tiling tiling) 
    : nrows(nrows), ncols(ncols), ntiles(ntiles), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles / nrowgrps),
        tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1) {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::~Matrix() {};
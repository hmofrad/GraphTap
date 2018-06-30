#include <algorithm>
#include <numeric>
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>
#include "utils/env.h"
#include "structures/bitvector.h"
#include "matrix/graph.h"

const bool TRANSPOSE = false;

const bool TWOD_STAGGERED = true; // Nested
const bool _TWOD_STAGGERED = true;

const bool NUMA = true; //Nested
const bool _NUMA = true;


template <class Weight, class Tile>
DistMatrix2D<Weight, Tile>::DistMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles,
                                         Partitioning partitioning)
    : Base(nrows, ncols, ntiles, partitioning),
      nranks(Env::nranks), rank(Env::rank), rank_ntiles(ntiles / nranks)
{
  /* Number of tiles must be a multiple of the number of ranks. */
  assert(rank_ntiles * nranks == ntiles);

  if (partitioning == Partitioning::_1D_COL)
  {
    /* Number of ranks sharing each rowgroup and colgroup. */
    rowgrp_nranks = nranks;
    colgrp_nranks = 1;
    assert(rowgrp_nranks * colgrp_nranks == nranks);
    /* Number of rowgroups and colgroups per rank. */
    rank_nrowgrps = nrowgrps;
    rank_ncolgrps = 1;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
	assign_tiles();
  }
  else if (partitioning == Partitioning::_1D_ROW)
  {
    rowgrp_nranks = 1;
    colgrp_nranks = nranks;
    assert(rowgrp_nranks * colgrp_nranks == nranks);

    rank_nrowgrps = 1;
    rank_ncolgrps = ncolgrps;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
	assign_tiles();

   // LOG.fatal("1D_ROW partitioning Not implemented! \n");
  }
  else if (partitioning == Partitioning::_2D)
  {
	  
	Env::nmachines = 4;
	/* Machine configuration */
	uint32_t rowgrp_nmachines = Env::nmachines;
	uint32_t colgrp_nmachines = Env::nmachines;
    uint32_t nranks_ = Env::nranks / Env::nmachines;
    uint32_t rowgrp_nranks_;
	uint32_t colgrp_nranks_;
    integer_factorize(nranks_, rowgrp_nranks_, colgrp_nranks_);
    assert(rowgrp_nranks_ * colgrp_nranks_ == nranks_);
    //if(!Env::rank)
	//  printf("-1.%d %d %d \n", rowgrp_nranks_, colgrp_nranks_, nranks_);
  
  
    
	uint32_t ntiles_ = nranks_ * nranks_;
	uint32_t rank_ntiles_ = ntiles_ / nranks_;
	uint32_t nrowgrps_ = sqrt(ntiles_);
    uint32_t ncolgrps_ = ntiles_ / nrowgrps_;
    uint32_t rank_nrowgrps_ = nrowgrps_ / colgrp_nranks_;
    uint32_t rank_ncolgrps_ = ncolgrps_ / rowgrp_nranks_;
	assert(rank_nrowgrps_ * rank_ncolgrps_ == rank_ntiles_);
    //if(!Env::rank)
	//  printf("0.%d rank_ncolgrps_=%d %d \n", rank_nrowgrps_, rank_ncolgrps_, nranks_);
	

	
  	std::vector<std::vector<Tile>> tiles_(nranks_);
    for (uint32_t x = 0; x < nranks_; x++)
      tiles_[x].resize(nranks_);
  
    integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
	//rowgrp_nranks = rowgrp_nranks * Env::nmachines;
    assert(rowgrp_nranks * colgrp_nranks == Env::nranks);
	//if(!Env::rank)
	//  printf("1.%d %d %d \n", rowgrp_nranks, colgrp_nranks, Env::nranks);
	
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
	
	//if(!Env::rank)
	//  printf("2.%d %d %d \n", rank_nrowgrps, rank_ncolgrps, rank_ntiles);
	
    tiles.resize(nranks);
    for (uint32_t x = 0; x < nranks; x++)
      tiles[x].resize(nranks);

  
    // Rank per machine configuration
    uint32_t nranks_s = Env::nmachines;
    uint32_t ntiles_s = nranks_s * nranks_s;
    uint32_t rank_ntiles_s = ntiles_s / nranks_s;
  
    uint32_t rowgrp_nranks_s;
    uint32_t colgrp_nranks_s;  
    integer_factorize(nranks_s, rowgrp_nranks_s, colgrp_nranks_s);
    assert(rowgrp_nranks_s * colgrp_nranks_s == nranks_s);

    uint32_t nrowgrps_s = sqrt(ntiles_s);
    uint32_t ncolgrps_s = ntiles_s / nrowgrps_s;
    uint32_t rank_nrowgrps_s = nrowgrps_s / colgrp_nranks_s;
    uint32_t rank_ncolgrps_s = ncolgrps_s / rowgrp_nranks_s;
    assert(rank_nrowgrps_s * rank_ncolgrps_s == rank_ntiles_s);

    std::vector<std::vector<Tile>> super_tiles(nranks_s);
    for (uint32_t x = 0; x < nranks_s; x++)
      super_tiles[x].resize(nranks_s);	

    // Machine tiles 
    map(super_tiles, nranks_s, TRANSPOSE);

	if(!Env::rank)
	{
		
      for (uint32_t rg = 0; rg < nranks_s; rg++)
      {
        for (uint32_t cg = 0; cg < nranks_s; cg++)
        {
          LOG.info<true, false>("%02d ", super_tiles[rg][cg].rank);
        }
        LOG.info<true, false>("\n");
      }	
      LOG.info<true, false>("\n");
	  
	  
	  for (uint32_t rg = 0; rg < nranks_s; rg++)
      {
        for (uint32_t cg = 0; cg < nranks_s; cg++)
        {
		  if(!super_tiles[rg][cg].rank)
            LOG.info<true, false>("[%02d %02d %02d %02d %02d]", super_tiles[rg][cg].rg, super_tiles[rg][cg].cg, super_tiles[rg][cg].ith, super_tiles[rg][cg].jth, super_tiles[rg][cg].nth);
        }
        LOG.info<true, false>("\n");
      }	
	}
	
		if(!Env::rank)
	printf("%d\n", nranks_s);
	
	 
 // Env::finalize();		
  //exit(0);	
   
	
	
	for (uint32_t rm = 0; rm < rowgrp_nmachines; rm++)
	{
      for (uint32_t cm = 0; cm < colgrp_nmachines; cm++)
      {
		uint32_t rank_offset = super_tiles[rm][cm].rank;
		uint32_t ith_offset = super_tiles[rm][cm].ith;
		uint32_t jth_offset = super_tiles[rm][cm].jth;
		uint32_t nth_offset = super_tiles[rm][cm].nth;
		// Ranks per machine tiles
		map(tiles_, nranks_, TRANSPOSE);
		for (uint32_t rg = 0; rg < nranks_; rg++)
		{
	      for (uint32_t cg = 0; cg < nranks_; cg++)
	      {
			uint32_t rg_idx = (rm * nranks_) + rg;
			uint32_t cg_idx = (cm * nranks_) + cg;
			tiles[rg_idx][cg_idx].rank = tiles_[rg][cg].rank + (rank_offset * nranks_);
	        tiles[rg_idx][cg_idx].ith = (ith_offset * rank_nrowgrps_) + tiles_[rg][cg].ith;
			tiles[rg_idx][cg_idx].jth = (jth_offset * rank_ncolgrps_) + tiles_[rg][cg].jth;
			tiles[rg_idx][cg_idx].nth = (tiles[rg_idx][cg_idx].ith * rank_ncolgrps_ * rank_ncolgrps_s) + tiles[rg_idx][cg_idx].jth;
			tiles[rg_idx][cg_idx].rg = rg_idx;
			tiles[rg_idx][cg_idx].cg = cg_idx;
		  }
		}
      }
    }
	

	

	/*
   if(!Env::rank)
   {
    for (uint32_t rg = 0; rg < Env::nranks; rg++)
    {
      for (uint32_t cg = 0; cg < Env::nranks; cg++)
      {
        LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      }
      LOG.info<true, false>("\n");
    }		   
	LOG.info<true, false>("\n");   

	
    for (uint32_t rg = 0; rg < Env::nranks; rg++)
    {
      for (uint32_t cg = 0; cg < Env::nranks; cg++)
      {
		  if(!tiles[rg][cg].rank)
            LOG.info<true, false>("[%02d %02d %02d %02d %02d]", tiles[rg][cg].rg, tiles[rg][cg].cg, tiles[rg][cg].ith, tiles[rg][cg].jth, tiles[rg][cg].nth);
      }
      LOG.info<true, false>("\n");
    }	
	
  }	
  
  Env::finalize();		
  exit(0);	
  */

  }
  else if (partitioning == Partitioning::_TEST)
  {
    LOG.fatal("TEST partitioning Not implemented! \n");
  }
  /* A large MPI type to support buffers larger than INT_MAX. */
  MPI_Type_contiguous(many_triples_size * sizeof(Triple<Weight>), MPI_BYTE, &MANY_TRIPLES);
  MPI_Type_commit(&MANY_TRIPLES);

  //assign_tiles();
  // print_info();

  /**** TODO: Place this in a function; this makes sure leader _always_ exists in row+col group.
        NOTE: It doesn't at all guarantee that the rowgrps == colgrps of a process!
              This cannot be the case in non-square nranks at least. *****/
/*
  BitVector bv(nranks);

  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    if (bv.count() == bv.size())
      bv.clear();

    for (uint32_t rg_ = rg; rg_ < nrowgrps; rg_++)
    {
      if (not bv.touch(tiles[rg_][rg].rank))
      {
        using std::swap;
        swap(tiles[rg_], tiles[rg]);
        break;
      }
    }
  }

  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      auto& tile = tiles[rg][cg];
      tile.rg = rg;
      tile.cg = cg;
    }
  }
*/


  if(_TWOD_STAGGERED)
  {
    integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
    assert(rowgrp_nranks * colgrp_nranks == nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    map(tiles, nranks, TRANSPOSE);
  }

  if(_NUMA)
  {
	uint32_t nranks_machine = Env::nranks / Env::nmachines; ///Env::machines_nranks[0];
    rowgrp_nranks = Env::nmachines;
    colgrp_nranks = nranks_machine;
    assert(rowgrp_nranks * colgrp_nranks == Env::nranks);  
    rank_nrowgrps = nranks_machine;
    rank_ncolgrps = Env::nmachines;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    map(tiles, nranks,  TRANSPOSE);
  }

  print_info();
  
  //Env::finalize();		
  //exit(0);	
  
  
  //Env::finalize();		
  //exit(0);	
}

template <class Weight, class Tile>
DistMatrix2D<Weight, Tile>::~DistMatrix2D()
{
  /* Cleanup the MPI type. */
  auto retval = MPI_Type_free(&MANY_TRIPLES);
  assert(retval == MPI_SUCCESS);
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::assign_tiles()
{
  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      auto& tile = tiles[rg][cg];
      tile.rg = rg;
      tile.cg = cg;

      if (this->partitioning == Partitioning::_1D_COL)
      {
        tile.rank    = cg; //(cg % rowgrp_nranks) * colgrp_nranks + (rg % colgrp_nranks);
        tile.ith     = rg / colgrp_nranks;
        tile.jth     = cg / rowgrp_nranks;
      }
      else if (this->partitioning == Partitioning::_1D_ROW)
      {
        tile.rank    = rg;
        tile.ith     = rg / colgrp_nranks;
        tile.jth     = cg / rowgrp_nranks;
        //LOG.fatal("1D_ROW partitioning Not implemented! \n");
      }
      else if (this->partitioning == Partitioning::_2D)
      {
        tile.rank = (cg % rowgrp_nranks) * colgrp_nranks + (rg % colgrp_nranks);
        tile.ith = rg / colgrp_nranks;
        tile.jth = cg / rowgrp_nranks;
		
		
        
        // NUMA_2D
        tile.rank = (cg % rowgrp_nranks) * colgrp_nranks + (rg / rowgrp_nranks);
        tile.ith = rg % rowgrp_nranks;
        tile.jth = cg / rowgrp_nranks;
		
        
      }

      tile.nth = tile.ith * rank_ncolgrps + tile.jth;
    }
  }
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b)
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

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::print_info()
{
  /* Print assignment information. */
  LOG.info("#> Assigned the tiles to the %u ranks.\n"
           "#> Each rank has been assigned %u local tiles across %u rowgroups and %u colgroups.\n"
           "#> Each rowgroup is divided among %u ranks.\n"
           "#> Each colgroup is divided among %u ranks.\n", nranks, rank_ntiles, rank_nrowgrps,
           rank_ncolgrps, rowgrp_nranks, colgrp_nranks);

  /* Print a 2D grid of tiles, each annotated with the owner's rank. */
  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  //for (uint32_t rg = 0; rg < std::min(nrowgrps, 10u); rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    //for (uint32_t cg = 0; cg < std::min(ncolgrps, 10u); cg++)
    {
      LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      // LOG.info("[%02d] ", tiles[x][y].nth);
    }

    //if (ncolgrps > 10u)
      //LOG.info<true, false>(" ...");
    LOG.info<true, false>("\n");
  }
  
  
  for (uint32_t rg = 0; rg < Env::nranks; rg++)
  {
    for (uint32_t cg = 0; cg < Env::nranks; cg++)
    {
		if(!tiles[rg][cg].rank)
          LOG.info<true, false>("[%02d %02d %02d %02d %02d]", tiles[rg][cg].rg, tiles[rg][cg].cg, tiles[rg][cg].ith, tiles[rg][cg].jth, tiles[rg][cg].nth);
    }
    LOG.info<true, false>("\n");
  }

  //if (nrowgrps > 10u)
    //LOG.info<true, false>(" ...\n");
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::distribute()
{
  std::vector<std::vector<Triple<Weight>>> outboxes(nranks);
  std::vector<std::vector<Triple<Weight>>> inboxes(nranks);
  std::vector<uint32_t> inbox_sizes(nranks);

  /* Copy the triples of each (non-self) rank to its outbox. */
  for (auto& tilegrp : tiles)
  {
    for (auto& tile : tilegrp)
    {
      if (tile.rank == rank)
        continue;

      auto& outbox = outboxes[tile.rank];
      outbox.insert(outbox.end(), tile.triples->begin(), tile.triples->end());
      tile.free_triples();
      tile.allocate_triples();
    }
  }

  for (uint32_t r = 0; r < nranks; r++)
  {
    if (r == rank)
      continue;

    auto& outbox = outboxes[r];
    uint32_t outbox_size = outbox.size();
    MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, 0, &inbox_sizes[r], 1, MPI_UNSIGNED, r, 0,
                 Env::MPI_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<MPI_Request> outreqs;
  std::vector<MPI_Request> inreqs;
  MPI_Request request;

  for (uint32_t i = 0; i < nranks; i++)
  {
    uint32_t r = (rank + i) % nranks;
    if (r == rank)
      continue;

    auto& inbox = inboxes[r];
    uint32_t inbox_bound = inbox_sizes[r] + many_triples_size;
    inbox.resize(inbox_bound);

    /* Send/Recv the triples with many_triples_size padding. */
    MPI_Irecv(inbox.data(), inbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD,
              &request);
    inreqs.push_back(request);
  }

  for (uint32_t i = 0; i < nranks; i++)
  {
    uint32_t r = (rank + i) % nranks;
    if (r == rank)
      continue;

    auto& outbox = outboxes[r];
    uint32_t outbox_bound = outbox.size() + many_triples_size;
    outbox.resize(outbox_bound);

    /* Send the triples with many_triples_size padding. */
    MPI_Isend(outbox.data(), outbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD,
              &request);

    outreqs.push_back(request);
  }

  MPI_Waitall(inreqs.size(), inreqs.data(), MPI_STATUSES_IGNORE);

  for (uint32_t r = 0; r < nranks; r++)
  {
    if (r == rank)
      continue;

    auto& inbox = inboxes[r];
    for (uint32_t i = 0; i < inbox_sizes[r]; i++)
      Base::insert(inbox[i]);

    inbox.clear();
    inbox.shrink_to_fit();
  }

  LOG.info<false, false>("|");

  MPI_Waitall(outreqs.size(), outreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Barrier(Env::MPI_WORLD);
  LOG.info<true, false>("\n");
  // LOG.info("Done blocking on recvs!\n");
}


template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::map(std::vector<std::vector<Tile>> &tiles_, uint32_t nranks_, bool transpose)
{	
  uint32_t ntiles_ = nranks_ * nranks_;
  uint32_t rank_ntiles_ = ntiles_ / nranks_;
  
  uint32_t rowgrp_nranks_;
  uint32_t colgrp_nranks_;  
  integer_factorize(nranks_, rowgrp_nranks_, colgrp_nranks_);
  assert(rowgrp_nranks_ * colgrp_nranks_ == nranks_);
  //if(!Env::rank)
  //printf("nranks_=%d, %d %d\n", nranks_, rowgrp_nranks_, colgrp_nranks_);
  
  uint32_t nrowgrps_ = sqrt(ntiles_);
  uint32_t ncolgrps_ = ntiles_ / nrowgrps_;
  uint32_t rank_nrowgrps_ = nrowgrps_ / colgrp_nranks_;
  uint32_t rank_ncolgrps_ = ncolgrps_ / rowgrp_nranks_;
  //if(!Env::rank)
  //printf("1.ntiles_=%d, %d %d\n", ntiles_, nrowgrps_, ncolgrps_);
  assert(rank_nrowgrps_ * rank_ncolgrps_ == rank_ntiles_);
  //printf("2.rank_ntiles_=%d, %d %d\n", rank_ntiles_, rank_nrowgrps_, rank_ncolgrps_);
  
  for (uint32_t rg = 0; rg < nrowgrps_; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps_; cg++)
    {
	  if(TWOD_STAGGERED)
	  {		  
	    if(transpose)
	    {
		  auto& tile = tiles_[cg][rg];
          tile.rg = cg;
          tile.cg = rg;
		
		  tile.rank = ((cg % rowgrp_nranks_) * colgrp_nranks_) + (rg % colgrp_nranks_); // + (cg % colgrp_nranks_);
		  tile.ith =  (cg / rowgrp_nranks_);
          tile.jth =  (rg / colgrp_nranks_);
		
          tile.nth = (tile.ith * rank_nrowgrps_) + tile.jth;
	    }
        else
        {	
          auto& tile = tiles_[rg][cg];
          tile.rg = rg;
          tile.cg = cg;

	      tile.rank = ((cg % rowgrp_nranks_) * colgrp_nranks_) + (rg % colgrp_nranks_);
          tile.ith =  (rg / colgrp_nranks_);
          tile.jth =  (cg / rowgrp_nranks_);
   	  	  
          tile.nth = (tile.ith * rank_ncolgrps_) + tile.jth;
        }
	  }
	  
	  if(NUMA) 
      {
		if(transpose)
		{
          auto& tile = tiles_[cg][rg];
          tile.rg = cg;
          tile.cg = rg;

	      tile.rank = (cg % rowgrp_nranks) * colgrp_nranks + (rg / rowgrp_nranks);
          tile.ith = cg / rowgrp_nranks;
          tile.jth = rg % rowgrp_nranks;
		   
		  tile.nth = (tile.ith * rank_ncolgrps_) + tile.jth;
		}
		else
		{
		  auto& tile = tiles_[rg][cg];
          tile.rg = rg;
          tile.cg = cg;

	      tile.rank = (cg % rowgrp_nranks) * colgrp_nranks + (rg / rowgrp_nranks);
          tile.ith = rg % rowgrp_nranks;
          tile.jth = cg / rowgrp_nranks;
		   
		  tile.nth = (tile.ith * rank_nrowgrps_) + tile.jth;
		}
	  }
    }
  }
  
  
  BitVector bv(ntiles_);
  for (uint32_t rg = 0; rg < nrowgrps_; rg++)
  {
    if (bv.count() == bv.size())
      bv.clear();

    for (uint32_t rg_ = rg; rg_ < nrowgrps_; rg_++)
    {
      if (not bv.touch(tiles_[rg_][rg].rank))
      {
        using std::swap;
        swap(tiles_[rg_], tiles_[rg]);
        break;
      }
    }
  }
  
  for (uint32_t rg = 0; rg < nrowgrps_; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps_; cg++)
    {
      auto& tile = tiles_[rg][cg];
      tile.rg = rg;
      tile.cg = cg;
    }
  }
  
  
  
  /*
   if(!Env::rank)
   {
    LOG.info<true, false>("\n");
    for (uint32_t rg = 0; rg < nrowgrps_; rg++)
    {
      for (uint32_t cg = 0; cg < ncolgrps_; cg++)
      {
        LOG.info<true, false>("%02d ", tiles_[rg][cg].rank);
      }
      LOG.info<true, false>("\n");
    }	
   }
  */
  
}
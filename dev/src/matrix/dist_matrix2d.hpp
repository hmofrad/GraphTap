#include <algorithm>
#include <numeric>
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>
#include "utils/env.h"
#include "structures/bitvector.h"
#include "matrix/graph.h"


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
  }
  else if (partitioning == Partitioning::_1D_ROW)
  {
    rowgrp_nranks = 1;
    colgrp_nranks = nranks;
    assert(rowgrp_nranks * colgrp_nranks == nranks);

    rank_nrowgrps = 1;
    rank_ncolgrps = ncolgrps;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);

   // LOG.fatal("1D_ROW partitioning Not implemented! \n");
  }
  else if (partitioning == Partitioning::_2D)
  {
	/*
    // Number of ranks sharing each rowgroup and colgroup.
    integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
    // _1D using _2D
    //rowgrp_nranks = 1;
    //colgrp_nranks = nranks;

    assert(rowgrp_nranks * colgrp_nranks == nranks);
    // Number of rowgroups and colgroups per rank
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);

	std::vector<int> machines_ntiles(Env::nmachines);	
	for(int i = 0; i < Env::nmachines; i++) 
	{
		machines_ntiles[i] = Env::machines_nranks[i] * Env::nranks;
	}
	
	if(!Env::rank){
		//printf("%d %d %d %d\n", rank, nranks, Env::rank, Env::nmachines); //, 
		//printf("%s %d %d\n", Env::machines[0].c_str(), Env::machines_ranks[0][0], Env::machines_cores[0][0]);
		printf("machines_nranks=%d machines_ncores=%d machines_nsockets=%d\n", Env::machines_nranks[0], Env::machines_ncores[0], Env::machines_nsockets[0]);
		
		//printf("%d %d %d\n", Env::nranks, Env::machines_nranks[0], Env::machines_nranks[0]/Env::machines_ncores[0]);
		
		printf("machines_nranks=%d machines_ncores=%d machines_nsockets=%d\n", Env::machines_nranks[0], Env::machines_ncores[0], Env::machines_nsockets[0]);
		printf("%d\n", machines_ntiles[0]);
		
	}
	*/
	

	
	Env::nmachines = 4;
	uint32_t super_nranks = Env::nmachines;
	uint32_t super_ntiles = Env::nmachines * Env::nmachines;
	uint32_t super_rank_ntiles = super_ntiles / super_nranks;
	//assert(super_ntiles = Env::nranks / Env::nmachines);
	
	if(!rank)
	  printf("super_nranks=%d super_ntiles=%d super_rank_ntiles=%d\n", super_nranks, super_ntiles, super_rank_ntiles);
	
	
	std::vector<std::vector<Tile>> super_tiles(super_ntiles);
    for (uint32_t x = 0; x < super_ntiles; x++)
      super_tiles[x].resize(super_ntiles);
  
    uint32_t super_rowgrp_nranks;
	uint32_t super_colgrp_nranks;
    
    integer_factorize(super_nranks, super_rowgrp_nranks, super_colgrp_nranks);
    assert(super_rowgrp_nranks * super_colgrp_nranks == super_nranks);
	if(!rank)
		printf("super_rowgrp_nranks=%d super_colgrp_nranks=%d super_ntiles=%d\n", super_rowgrp_nranks, super_colgrp_nranks, super_ntiles);
	
    /* Number of rowgroups and colgroups per rank. */
	uint32_t super_nrowgrps = sqrt(super_ntiles);
	uint32_t super_ncolgrps = super_ntiles / super_nrowgrps;
	uint32_t super_rank_nrowgrps = super_nrowgrps / super_colgrp_nranks;
	uint32_t super_rank_ncolgrps = super_ncolgrps / super_rowgrp_nranks;
	

	
	
	if(!rank)
	{
		printf("super_nrowgrps=%d super_ncolgrps=%d\n",super_nrowgrps, super_ncolgrps);
		printf("super_rank_nrowgrps=%d super_rank_nrowgrps=%d super_rank_ntiles=%d\n", super_rank_nrowgrps, super_rank_ncolgrps, super_rank_ntiles);
	}
    //assert(super_rank_nrowgrps * super_rank_ncolgrps == super_rank_ntiles);
	
	
	
	
	
	for (uint32_t rg = 0; rg < super_nrowgrps; rg++)
    {
      for (uint32_t cg = 0; cg < super_ncolgrps; cg++)
      {
        auto& tile = super_tiles[rg][cg];
        tile.rg = rg;
        tile.cg = cg;

        tile.rank = (cg % super_rowgrp_nranks) * super_colgrp_nranks + (rg % super_colgrp_nranks);
        tile.ith = rg / super_colgrp_nranks;
        tile.jth = cg / super_rowgrp_nranks;
		
        tile.nth = tile.ith * rank_ncolgrps + tile.jth;
      }
    }
	
	
    BitVector bv(super_ntiles);

    for (uint32_t rg = 0; rg < super_nrowgrps; rg++)
    {
      if (bv.count() == bv.size())
        bv.clear();

      for (uint32_t rg_ = rg; rg_ < super_nrowgrps; rg_++)
      {
        if (not bv.touch(super_tiles[rg_][rg].rank))
        {
          using std::swap;
          swap(super_tiles[rg_], super_tiles[rg]);
          break;
        }
      }
    }

    for (uint32_t rg = 0; rg < super_nrowgrps; rg++)
    {
      for (uint32_t cg = 0; cg < super_ncolgrps; cg++)
      {
        auto& tile = super_tiles[rg][cg];
        tile.rg = rg;
        tile.cg = cg;
      }
    }
  
	for (uint32_t rg = 0; rg < super_nrowgrps; rg++)
    {
      for (uint32_t cg = 0; cg < super_ncolgrps; cg++)
      {
        LOG.info<true, false>("%02d ", super_tiles[rg][cg].rank);
      }
      LOG.info<true, false>("\n");
    }	
  
  
  
  
	Env::finalize();
	exit(0);			
	
	
	//int colgrp_nranks_ = 16;
	//int rowgrp_nranks_ = 16;
	int bigtile_nranks = Env::nranks / Env::nmachines;
	integer_factorize(bigtile_nranks, rowgrp_nranks, colgrp_nranks);
	//rowgrp_nranks = sqrt(Env::nmachines);
	//colgrp_nranks = bigtile_nranks;
	
	if(!Env::rank)
  	  printf("bigtile_nranks=%d, rowgrp_nranks=%d, colgrp_nranks=%d\n", bigtile_nranks, rowgrp_nranks, colgrp_nranks);
	

    
	int rowgrp_nmachines = sqrt(Env::nmachines);
	int colgrp_nmachines = sqrt(Env::nmachines);
	assert(rowgrp_nmachines * colgrp_nmachines == Env::nmachines);
	
	if(!Env::rank)
	  printf("colgrp_nranks=%d, rowgrp_nmachines=%d\n", rowgrp_nmachines, colgrp_nmachines);

	rank_nrowgrps = (Env::nranks / rowgrp_nmachines) / colgrp_nranks;
    rank_ncolgrps = (Env::nranks / rowgrp_nmachines) / rowgrp_nranks;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);

	Env::finalize();
	exit(0);		
	
	
	int bigtile_width = Env::nranks / rowgrp_nmachines;
	int bigtile_height = Env::nranks / colgrp_nmachines;
	assert(bigtile_height * colgrp_nmachines == Env::nranks);
	if(!Env::rank)
	printf("nmachines=%d, nranks=%d, rowgrp_nranks=%d, colgrp_nranks=%d, rowgrp_nmachines=%d, colgrp_nmachines=%d, rank_nrowgrps=%d, rank_ncolgrps=%d, bigtile_width=%d, bigtile_height=%d\n", 
	   Env::nmachines, Env::nranks, rowgrp_nranks, colgrp_nranks, rowgrp_nmachines, colgrp_nmachines, rank_nrowgrps, rank_ncolgrps, bigtile_width, bigtile_height);
	

	Env::finalize();
	exit(0);		
	
	
    tiles.resize(Env::nranks);
    for (uint32_t x = 0; x < Env::nranks; x++)
      tiles[x].resize(Env::nranks);	
	int t = 0;
	for (uint32_t rm = 0; rm < rowgrp_nmachines; rm++)
	{
      for (uint32_t cm = 0; cm < rowgrp_nmachines; cm++)
	  {
        for (uint32_t rg = rm * bigtile_width; rg < (rm * bigtile_width) + bigtile_width; rg++)
        {
          for (uint32_t cg = cm * bigtile_height; cg < (cm * bigtile_height) + bigtile_height; cg++)
          {
            auto& tile = tiles[rg][cg];
            tile.rg = rg;
            tile.cg = cg;	
		
            tile.rank = ((cg % rowgrp_nranks) * colgrp_nranks + (rg % colgrp_nranks)) + ((rm * (rowgrp_nmachines * bigtile_nranks)) + (cm * bigtile_nranks)); 
            tile.ith = rg / colgrp_nranks;
            tile.jth = cg / rowgrp_nranks;

            tile.nth = tile.ith * rank_ncolgrps + tile.jth;	 
			if(!rank)
			  printf("[%d %d %d %d %d] ", rm, cm, rg, cg, tile.rank);
	      }
		  if(!rank)		
		    printf("\n");
        }
		if(!rank)		
		  printf("\n");
	    t++;
	  }
	}
	
	
	for (uint32_t rg = 0; rg < Env::nranks; rg++)
    {
      for (uint32_t cg = 0; cg < Env::nranks; cg++)
      {
        LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      }
      LOG.info<true, false>("\n");
    }

	


	if(!rank)
	{
  for (uint32_t rm = 0; rm < rowgrp_nmachines; rm++)
  {
	
    for (uint32_t cm = 0; cm < rowgrp_nmachines; cm++)
    {
      BitVector bv(nranks);
	  printf("(%d %d)\n", rm, cm);  
	  for (uint32_t rg = rm * bigtile_width; rg < (rm * bigtile_width) + bigtile_width; rg++)
      {
        if (bv.count() == bv.size())
          bv.clear();

	    for (uint32_t rg_ = rg; rg_ < (rm * bigtile_width) + bigtile_width; rg_++)
        //for (uint32_t rg_ = cm * bigtile_height; rg_ < (cm * bigtile_height) + bigtile_height; rg_++)
        {
          if (not bv.touch(tiles[rg_][rg].rank))
          {
            //using std::iter_swap;
			//swap(tiles[rg_], tiles[rg]);
			
		     printf(">>> %d %d [%d %d] \n", rg, rg_, (cm * bigtile_height), ((cm * bigtile_height) + bigtile_height));
			 if((rg > ncolgrps) or (rg_ >ncolgrps ))
				 exit(0);
			
			std::swap_ranges(tiles[rg_].begin() + (cm * bigtile_height), tiles[rg_].begin() + ((cm * bigtile_height) + bigtile_height), tiles[rg].begin());
            
			//if(!rank)
		      //printf("[%d %d %d %d %d] ", rm, cm, rg, cg, tile.rank);
            break;
          }
        }
      }
    }
  }
  printf("done!");
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

	
	LOG.info<true, false>("\n");
	LOG.info<true, false>("\n");
			  if(!rank)		
		    printf("\n");
	
	
    for (uint32_t rg = 0; rg < Env::nranks; rg++)
    {
      for (uint32_t cg = 0; cg < Env::nranks; cg++)
      {
        LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      }
      LOG.info<true, false>("\n");
    }
	
	
	
    //print_info();
	//printf("%d\n", Env::nmachines);
	Env::finalize();
	exit(0);

	int num_sockets = Env::machines_nsockets[0]; //NUM_SOCKETS
	rowgrp_nranks = Env::nmachines;
	colgrp_nranks = Env::machines_nranks[0];
	assert(rowgrp_nranks * colgrp_nranks == nranks);
	
	rank_nrowgrps = nranks / Env::machines_nranks[0];
    rank_ncolgrps = Env::nmachines;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);

    /*
    // Numa-aware _2D
    int nmachines = 4; // 4;
    int nsockets_machine = 2; // 2;
    int ncores_socket = 1;
    rowgrp_nranks = ncores_socket * nmachines;
    colgrp_nranks = nsockets_machine;
    assert(rowgrp_nranks * colgrp_nranks == nranks);
    
    rank_nrowgrps = nrowgrps / nsockets_machine;
    rank_ncolgrps = nsockets_machine;
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    */

  }
  else if (partitioning == Partitioning::_TEST)
  {
    LOG.fatal("TEST partitioning Not implemented! \n");
  }
  /* A large MPI type to support buffers larger than INT_MAX. */
  MPI_Type_contiguous(many_triples_size * sizeof(Triple<Weight>), MPI_BYTE, &MANY_TRIPLES);
  MPI_Type_commit(&MANY_TRIPLES);

  assign_tiles();
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
  print_info();
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
  //for (uint32_t rg = 0; rg < nrowgrps; rg++)
  for (uint32_t rg = 0; rg < std::min(nrowgrps, 10u); rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    for (uint32_t cg = 0; cg < std::min(ncolgrps, 10u); cg++)
    {
      LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      // LOG.info("[%02d] ", tiles[x][y].nth);
    }

    if (ncolgrps > 10u)
      LOG.info<true, false>(" ...");
    LOG.info<true, false>("\n");
  }

  if (nrowgrps > 10u)
    LOG.info<true, false>(" ...\n");
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

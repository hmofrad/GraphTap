#include <cstdio>
#include <cstdlib>
#include <functional>

#include "utils/dist_timer.h"
#include "pr.h"

#include <iostream>
#include <set>
#include <algorithm>
/* Calculate Pagerank for a directed input graph. */


void run(std::string filepath, vid_t nvertices, uint32_t niters)
{
  /* Calculate out-degrees */
  Graph<ew_t> GR; // reverse graph for out-degree
  GR.load_directed(true, filepath, nvertices, true);  // reverse
  //GR.load_directed(true, filepath, nvertices, true, false, Hashing::BUCKET, Partitioning::_1D_ROW);  // reverse

  DegVertex<ew_t> vp_degree(&GR, true);  // stationary

  DistTimer degree_timer("Degree Execution");
  vp_degree.execute(1);
  degree_timer.stop();

  //vp_degree.display();
  GR.free();  // free degree graph

  /* Calculate Pagerank */
  Graph<ew_t> G;
  G.load_directed(true, filepath, nvertices);
  //G.load_directed(true, filepath, nvertices, false, false, Hashing::BUCKET, Partitioning::_1D_ROW);

  /* Pagerank initialization using out-degrees */
  PrVertex vp(&G, true);  // stationary

  vp.initialize(vp_degree);
  //vp.display();
  vp_degree.free();  // free degree states

  Env::barrier();
  DistTimer pr_timer("Pagerank Execution");
  vp.execute(niters);
  pr_timer.stop();

  vp.display();
  degree_timer.report();
  pr_timer.report();

  /* For correctness checking */
  long deg_checksum = vp.reduce<long>(
      [&](uint32_t idx, const PrState& s) -> long { return s.degree; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Degree Checksum = %lu \n", deg_checksum);

  /* For correctness checking */
  fp_t pr_checksum = vp.reduce<fp_t>(
      [&](uint32_t idx, const PrState& s) -> fp_t { return s.rank; },  // mapper
      [&](fp_t& a, const fp_t& b) { a += b; });  // reducer
  LOG.info("Pagerank Checksum = %lf \n", pr_checksum);
}


int main(int argc, char* argv[])
{
  //Env::init();
  Env::init(RankOrder::KEEP_ORIGINAL);

  /* Print usage. */
  if (argc < 3)
  {
    LOG.info("Usage: %s <filepath> <num_vertices: 0 if header present> "
                 "[<iterations> (default: until convergence)] \n", argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  std::string filepath = argv[1];
  vid_t nvertices = (vid_t) std::atol(argv[2]);
  uint32_t niters = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;
  std::cout << Env::rank << "," << Env::core_name << "," << Env::core_id << std::endl;
  /*
  std::cout << Env::rank << "," << Env::core_name << "," << Env::core_id << std::endl;
  std::vector<int> core_ids(Env::nranks);
  
  MPI_Gather(&Env::core_id, 1, MPI_INT, core_ids.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int core_name_len = strlen(Env::core_name);
  std::vector<int> recvcounts(Env::nranks);
  
  int max_length = 0;
  MPI_Allreduce(&core_name_len, &max_length, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  char *str_padded[max_length + 1]; // + 1 for '\n'
  memset(str_padded, '\0', max_length + 1);
  memcpy(str_padded, &Env::core_name, max_length + 1);
  
  int total_length = (max_length + 1) * Env::nranks; 
  std::string total_string (total_length, '\0');
  MPI_Allgather(str_padded, max_length + 1, MPI_CHAR, (void *) total_string.data(), max_length + 1, MPI_CHAR, MPI_COMM_WORLD);

   

  int offset = 0;
  std::vector<std::string> machines_all;
  machines_all.clear();
  for(int i = 0; i < Env::nranks; i++) 
  {
    machines_all.push_back(total_string.substr(offset, max_length + 1));
    offset += max_length + 1;
  }
	   
	   
  if (!Env::rank) {  
    for(int i = 0; i < Env::nranks; i++)
      printf("rank=%d, core_id=%d, cup_name=%s\n", i, core_ids[i] , machines_all[i].c_str());
  
  
  
  int nmachines = std::set<std::string>(machines_all.begin(), machines_all.end()).size();
  
  std::vector<std::string> machines = machines_all; 
  std::vector<std::string>::iterator it;
  it = std::unique(machines.begin(), machines.end());
  machines.resize(std::distance(machines.begin(),it));

  std::vector<int> machines_nranks(nmachines, 0); 
  //std::vector<int> machines_nsockets(nmachines, 0); 
  std::vector<std::vector<int>> machines_ranks(nmachines);
  std::vector<std::vector<int>> machines_cores(nmachines);

  //vector<int> myRow(1,5);
//myVector.push_back(myRow);
// add element to row
//myVector[0].push_back(1);

   // print out content:
  //std::cout << "myvector contains:";
  //for (it=machines.begin(); it!=machines.end(); ++it)
  //  std::cout << ' ' << *it;
  //std::cout << '\n';

  //std::vector<std::string>::iterator it1;
  for (it=machines_all.begin(); it!=machines_all.end(); it++) {
	  //ptrdiff_t pos = find(machines.begin(), machines.end(), *it) - machines.begin();
	  int idx = distance(machines.begin(), find(machines.begin(), machines.end(), *it));
	  assert((idx > 0) && (idx < machines.size()));
	  
	  machines_nranks[idx]++;
	  int idx1 = it - machines_all.begin();

	  machines_ranks[idx].push_back(idx);
	  machines_cores[idx].push_back(core_ids[idx1]);
	  
	  
	std::cout << " " << *it << "," << idx << "," <<  machines[idx].c_str()  << "," << idx1 << ".." << core_ids[idx1] << std::endl;	  

	  //if(std::find(machines.begin(), machines.end(), *it) != machines.end())
	//for (it1=machines.begin(); it1!=machines.end(); ++it1) {
		//std::cout << " " << *it << "," << *it1 << ( *it == *it1) << std::endl;
	  //it.compare(
	  
	//}
	//break;
  }

  std::vector<int>::iterator it1;
  std::vector<int>::iterator it2;
  for(int i = 0; i < nmachines; i++) {
    for(int j= 0; j < machines_ranks[i].size(); j++) {
      std::cout << machines[i] << " " << machines_ranks[i][j] <<  " " << machines_cores[i][j] << std::endl;
	}

  }
  //for (it1=machines.begin(); it1!=machines.end(); ++it1) {
	//  for (it2=machines_ranks.begin(); it2!=machines_ranks.end(); ++it2) {
		//  int idx = it - machines.begin();
		  //int idx1 = it2 - machines_ranks[idx].begin();
		  //std::cout << *it1 << " " << *it2 <<  " " << machines_cores[idx][idx1] << std::endl;
		  
//	  }
	  
 // }
  
  

  //for(int i = 0; i < nmachines; i++)
	//  printf("%d\n", machines_nranks[i]);
      //printf("rank=%d, core_id=%d, cup_name=%s\n", i, core_ids[i] , machines_all[i].c_str());
  
  	  
    
    //}
	  

  
  }
  */
  
  
  
  //run(filepath, nvertices, niters);

  Env::finalize();
  return 0;
}

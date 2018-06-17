#include <cstdio>
#include <cstdlib>
#include <functional>

#include "utils/dist_timer.h"
#include "pr.h"

#include <iostream>
#include <set>
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
  std::vector<std::string> str;
  str.clear();
  for(int i = 0; i < Env::nranks; i++) 
  {
    str.push_back(total_string.substr(offset, max_length + 1));
    offset += max_length + 1;
  }
	   
	   
  if (!Env::rank) {  
    for(int i = 0; i < Env::nranks; i++)
      printf("rank=%d, core_id=%d, cup_name=%s\n", i, core_ids[i] , str[i].c_str());
  }
  
  
  int nmachines = std::set<std::string>(str.begin(), str.end()).size();
  
  
  //printf("%d\n", nmachines);
  //printf("%d %d %d\n" ,NUM_SOCKETS, NUM_CORES_PER_SOCKET, NUM_CORES_PER_MACHINE);
	
	
	
	

 
  
  //MPI_Allgather(&Env::core_id, 1, MPI_INT, core_ids.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  //MPI_Allgather(&Env::core_id, 1, MPI_INT, core_ids.data(), 1, MPI_INT, MPI_COMM_WORLD);
  

  /*
  if(!Env::rank) 
  {
	  
  }
  else
  {
     MPI_Send(&Env::cpuid, 1, MPI_INT, Env::rank, 0, MPI_COMM_WORLD);	  
  }
*/	  
  
  
  
  //run(filepath, nvertices, niters);

  Env::finalize();
  return 0;
}

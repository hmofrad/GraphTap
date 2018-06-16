#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "pr.h"

#include <iostream>

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
  
  //int root = -1;
  //if(!Env::rank)
   // root = Env::rank;
  std::cout << Env::rank << "," << Env::cpu_name << "," << Env::cpu_id << std::endl;
  std::vector<int> cpu_ids(Env::nranks);
  
  
   
  MPI_Gather(&Env::cpu_id, 1, MPI_INT, cpu_ids.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  //MPI_MAX_PROCESSOR_NAME
  
  //if(!Env::rank)
  //{
//	  for(int i = 0; i < Env::nranks; i++) 
	//	  std::cout << i << ":" << cpu_ids[i] << std::endl;
  //}
  
 

  int cpu_name_len = strlen(Env::cpu_name);
  std::vector<int> recvcounts(Env::nranks);

  //MPI_Gather(&cpu_name_len, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  //MPI_Allgather(&cpu_name_len, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  //int total_length = 0;
  //MPI_Allreduce(&cpu_name_len, &total_length, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  int max_length = 0;
  MPI_Allreduce(&cpu_name_len, &max_length, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  max_length = max_length + 1; // '\n'
  
  int total_length = max_length * Env::nranks; 
  std::string total_string (total_length, '\0');
  //char *total_string_ = (char *) malloc(total_length_ * sizeof(char));
  //char *temp_string = (char *) malloc(max_length * sizeof(char));
  //memset(temp_string, 0, max_length);
  //strcpy(temp_string, Env::cpu_name);
  //printf("%s\n", temp_string);
  MPI_Allgather(&Env::cpu_name, max_length, MPI_CHAR, (void *) total_string.data(), max_length, MPI_CHAR, MPI_COMM_WORLD);
  //printf(">>>>%s\n", total_string_);
  //std::vector<int> recvcounts_(Env::nranks);
  //recvcounts_
  //MPI_ALLGather(&Env::cpu_name, cpu_name_len, MPI_CHAR, total_string_, recvcounts.data(), displacement.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
  
  
  /*
  if(!Env::rank)
  {
	  for(int i = 0; i < Env::nranks; i++) 
		  std::cout << i << ":" << recvcounts[i] << std::endl;
  }

  std::vector<int> displacement(Env::nranks);
  int total_length = 0;
  char *total_string = NULL;
  if(!Env::rank)
  {
    displacement[0] = 0 ;
    total_length += recvcounts[0] + 1;
	for (int i=1; i < Env::nranks; i++) 
	{
	  total_length += recvcounts[i] + 1;
	  displacement[i] = displacement[i-1] + recvcounts[i-1] + 1;
	}
    total_string = (char *) malloc(total_length * sizeof(char));
	memset(total_string, 0, total_length);
  }
*/
  //char *mystring = (char *)strings[Env::nranks];
  //int mylen = strlen(mystring);


  //MPI_Gatherv(&Env::cpu_name, cpu_name_len, MPI_CHAR, total_string, recvcounts.data(), displacement.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
  
  
  //MPI_Gather(&Env::cpu_name, cpu_name_len, MPI_CHAR, total_string, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
   if (!Env::rank) {
	   int k = 0;
	   //std::string j = NULL;
	   char *j = (char *) total_string.data();
		std::vector<std::string> str;
		str.clear();
	   for(int i = 0; i < Env::nranks; i++) 
	   {
		   //char *j = total_string + displacement[i];
           //printf("%d: <%s> %d %d %d\n", Env::rank, j, strlen(j), recvcounts[i], displacement[i]);
		   
		   //j = &total_string + k;
		   
		   printf("+++%s %d\n", j, cpu_ids[i]);
		   str.push_back(j);
		   k += max_length;
		   j = (char *) total_string.data() + k;
		   
	   }
	   
	   for(int i = 0; i < Env::nranks; i++) 
		   printf(">>%s\n", str[i]);
	   
	   //total_string[0] = X;
	     // printf(">>%s\n", str[0]);
	   //printf("%d %d %d\n", total_length, total_length_, max_length);
		   
		
        //free(totalstring);
        //free(displs);
        //free(recvcounts);
		
		
		std::string s = "What is the right way to split a string into a vector of strings";
std::stringstream ss(s);
std::istream_iterator<std::string> begin(ss);
std::istream_iterator<std::string> end;
//printf(">>%x\n", ss);
std::vector<std::string> vstrings(begin, end);
std::copy(vstrings.begin(), vstrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
		
		
    }

 
  
  //MPI_Allgather(&Env::cpu_id, 1, MPI_INT, cpu_ids.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  //MPI_Allgather(&Env::cpu_id, 1, MPI_INT, cpu_ids.data(), 1, MPI_INT, MPI_COMM_WORLD);
  

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

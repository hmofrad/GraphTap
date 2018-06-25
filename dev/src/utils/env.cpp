#include <sys/time.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include "utils/env.h"

#include <iostream>


int Env::rank  ;  // my rank

int Env::nranks;  // num of ranks

bool Env::is_master;  // rank == 0?

MPI_Comm Env::MPI_WORLD;

char Env::core_name[MPI_MAX_PROCESSOR_NAME];
int Env::core_id;
int Env::nmachines;
std::vector<std::string> Env::machines;
std::vector<int> Env::machines_nranks;
std::vector<std::vector<int>> Env::machines_ranks;
std::vector<std::vector<int>> Env::machines_cores;
std::vector<std::unordered_set<int>> Env::machines_cores_uniq;
std::vector<int> Env::machines_ncores;
std::vector<int> Env::machines_nsockets;

//std:vector<int> test;


void Env::init(RankOrder order)
{
  int mpi_threading;
  MPI_Init_thread(0, nullptr, MPI_THREAD_MULTIPLE, &mpi_threading);

  /* Set Environment Variables. */
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  is_master = rank == 0;

  MPI_WORLD = MPI_COMM_WORLD;
  if (order != RankOrder::KEEP_ORIGINAL)
    shuffle_ranks(order);

  // Affinity 
  affinity();
}

void Env::finalize()
{ MPI_Finalize(); }

void Env::exit(int code)
{
  finalize();
  std::exit(code);
}

void Env::barrier()
{ MPI_Barrier(MPI_WORLD); }


void Env::shuffle_ranks(RankOrder order)
{
  std::vector<int> ranks(nranks);

  if (is_master)
  {
    int seed = order == RankOrder::FIXED_SHUFFLE ? 0 : now();
    srand(seed);
    std::iota(ranks.begin(), ranks.end(), 0);  // ranks = range(len(ranks))
    std::random_shuffle(ranks.begin() + 1, ranks.end());

    assert(ranks[0] == 0);
  }

  MPI_Bcast(ranks.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Group world_group, reordered_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, nranks, ranks.data(), &reordered_group);
  MPI_Comm_create(MPI_COMM_WORLD, reordered_group, &MPI_WORLD);

  MPI_Comm_rank(MPI_WORLD, &rank);
}


double Env::now()
{
  struct timeval tv;
  auto retval = gettimeofday(&tv, nullptr);
  assert(retval == 0);
  return (double) tv.tv_sec + (double) tv.tv_usec / 1e6;
}

void Env::affinity()
{
  int cpu_name_len;
  MPI_Get_processor_name(core_name, &cpu_name_len);
  core_id = sched_getcpu();

  std::vector<int> core_ids = std::vector<int>(nranks);
  MPI_Gather(&core_id, 1, MPI_INT, core_ids.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
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
  
  // Tokenizing the string!
  int offset = 0;
  std::vector<std::string> machines_all;
  machines_all.clear();
  for(int i = 0; i < Env::nranks; i++) 
  {
    machines_all.push_back(total_string.substr(offset, max_length + 1));
    offset += max_length + 1;
  }
  /*
  if(is_master)
  {
	  for(int i = 0; i < Env::nranks; i++)
	    printf("rank=%d, core_id=%d, cup_name=%s\n", i, core_ids[i] , machines_all[i].c_str());
  }
  */
  nmachines = std::set<std::string>(machines_all.begin(), machines_all.end()).size();
  
  //std::vector<std::string> 
  machines = machines_all; 
  std::vector<std::string>::iterator it;
  it = std::unique(machines.begin(), machines.end());
  machines.resize(std::distance(machines.begin(),it));

  //std::vector<int> machines_nranks(&nmachines, 0);
  machines_nranks.resize(nmachines, 0);
  //std::vector<std::vector<int>> 
  machines_ranks.resize(nmachines);
  //std::vector<std::vector<int>> 
  machines_cores.resize(nmachines);
  std::vector<std::unordered_set<int>> machines_cores_uniq(nmachines);
  

  
  
  for (it=machines_all.begin(); it!=machines_all.end(); it++)
  {
    int idx = distance(machines.begin(), find(machines.begin(), machines.end(), *it));
    assert((idx >= 0) && (idx < machines.size()));
	  
    machines_nranks[idx]++;
    int idx1 = it - machines_all.begin();

    machines_ranks[idx].push_back(idx1);
	assert((core_ids[idx1] >= 0) && (core_ids[idx1] < NUM_CORES_PER_MACHINE));
	machines_cores[idx].push_back(core_ids[idx1]);
	
	//int core_uniq_sz = machines_cores_uniq[idx].size();
    machines_cores_uniq[idx].insert(core_ids[idx1]);
	//if(machines_cores_uniq[idx].size() == (core_uniq_sz + 1))
	//{
		//if(NUM_CORES_PER_SOCKET
		//machines_nsockets[idx]++;
	//}
	  	  
	//std::cout << " " << *it << "," << idx << "," <<  machines[idx].c_str()  << "," << idx1 << ".." << core_ids[idx1] << std::endl;
  }
  
  //std::vector<int> 
  machines_ncores.resize(nmachines, 0);
  //std::vector<int>
  machines_nsockets.resize(nmachines, 0);
  std::vector<int> sockets_per_machine(NUM_SOCKETS, 0);
 // if(is_master) {
	  for(int i = 0; i < nmachines; i++)
	  {
	    std::unordered_set<int>::iterator it1;
		for(it1=machines_cores_uniq[i].begin(); it1!=machines_cores_uniq[i].end();it1++)
		{
			int socket_id = *it1 / NUM_CORES_PER_SOCKET;
			sockets_per_machine[socket_id] = 1;
			if(!rank)
			    std::cout << i << " " << *it1 << " " << socket_id << ", ";
		}
		if(!rank)
		 std::cout << "\n";
		machines_ncores[i] = machines_cores_uniq[i].size();
		machines_nsockets[i] = std::accumulate(sockets_per_machine.begin(), sockets_per_machine.end(), 0);
	  }
  //}
  
      

  //std::vector<std::set<int> > my_sets[1];
	//int number = 10;
	//auto it3 = my_sets[0].begin();
	//my_sets[0].insert(it3,number);
	
	//std::vector<std::unordered_set<int>> machines_cores(nmachines);
	
	
  //machines_cores[0].insert(1);                        // copy insertion

  
  if(is_master) 
  {
	  std::vector<int>::iterator it1;
	  std::vector<int>::iterator it2;
	  for(int i = 0; i < nmachines; i++)
	  {
		std::cout << "Machine=" << machines[i] << "(rank,core): ";
		for(int j= 0; j < machines_ranks[i].size(); j++) {
		  std::cout << "(" << machines_ranks[i][j] <<  "," << machines_cores[i][j] << ")";
		}
		std::cout << " unique_core(core):";
		std::unordered_set<int>::iterator iter;
		for(iter=machines_cores_uniq[i].begin(); iter!=machines_cores_uniq[i].end();++iter)
		{
			std::cout << "(" << *iter << ")";
		}
		
		std::cout << "| machine_nranks=" << machines_nranks[i];
		std::cout << "| machine_ncores=" << machines_ncores[i];
		std::cout << "| machine_nsockets=" << machines_nsockets[i] << "\n";
      }
	  
  }
  
  
 // test

  
  
  

}



#ifndef ENV_H
#define ENV_H

#include <sched.h>
#include <vector>
#include <set>
#include <unordered_set>

#include <mpi.h>
#include "utils/enum.h"



#define NUM_SOCKETS 2
#define NUM_CORES_PER_SOCKET 14
#define NUM_CORES_PER_MACHINE (NUM_SOCKETS * NUM_CORES_PER_SOCKET)

int sched_getcpu(void);

class RankOrder : public Enum {
public:
  using Enum::Enum;
  static constexpr int KEEP_ORIGINAL  = 0;
  static constexpr int FIXED_SHUFFLE  = 1;  // Default
  static constexpr int RANDOM_SHUFFLE = 2;
};


class Env
{
public:

  Env();

  static int rank;    // my rank

  static int nranks;  // num of ranks

  static bool is_master;  // rank == 0?

  static MPI_Comm MPI_WORLD;

  static void init(RankOrder order = RankOrder::FIXED_SHUFFLE);

  static void finalize();

  static void exit(int code);

  static void barrier();  // global barrier

  static double now();  // timestamp

  static char core_name[]; // Core name = hostname
  static int core_id; // Core id
  static int nmachines; // Number of allocated machines
  static std::vector<std::string> machines; // Number of machines
  static std::vector<int> machines_nranks; // Number of ranks per machine
  static std::vector<std::vector<int>> machines_ranks;
  static std::vector<std::vector<int>> machines_cores;
  static std::vector<std::unordered_set<int>> machines_cores_uniq;
  static std::vector<int> machines_ncores; // Number of cores per machine
  static std::vector<int> machines_nsockets; // Number of sockets available per machine

  
  
private:
  static void shuffle_ranks(RankOrder order);
  
  static void affinity(); // Affinity
};


#endif

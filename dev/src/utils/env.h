#ifndef ENV_H
#define ENV_H

#include <mpi.h>
#include "utils/enum.h"

#include <sched.h>

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

  //static int getcpuid(); // get CPU id

private:
  static void shuffle_ranks(RankOrder order);
};


#endif

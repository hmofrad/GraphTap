/*
 * env.hpp: MPI runtime utilities
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <mpi.h>
#include <cassert>
#include <iostream>

class Env
{
    public:
    Env();
    
    static MPI_Comm MPI_WORLD;
    static int rank;
    static int nranks;
    static bool is_master;
    static void init();
    
    static double start;
    static double finish;    
    static double clock();
    static void   tick();
    static void   tock(std::string preamble);
    
    static void barrier();
    static void finalize();
    static void exit(int code);
};

int  Env::rank = -1;
int  Env::nranks = -1;
bool Env::is_master = false;
MPI_Comm Env::MPI_WORLD;

double Env::start = 0;
double Env::finish = 0;
 
void Env::init()
{
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;
    MPI_Init_thread(nullptr, nullptr, required, &provided);
    assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks >= 0);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);
    
    is_master = rank == 0;
    
    MPI_WORLD = MPI_COMM_WORLD;
}

double Env::clock()
{
    return(MPI_Wtime());
}

void Env::tick()
{
    start = Env::clock();
}

void Env::tock(std::string preamble)
{
    finish = Env::clock();
    double elapsed_time = finish - start;
    printf("%s time: %f seconds\n", preamble.c_str(), elapsed_time);
}

void Env::finalize()
{
    MPI_Finalize();
}

void Env::exit(int code)
{
  Env::finalize();
  std::exit(code);
}

void Env::barrier()
{
    MPI_Barrier(MPI_WORLD); 
}
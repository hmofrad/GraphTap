/*
 * env.hpp: MPI runtime utilities
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <mpi.h>
#include <cassert>
#include <iostream>
#include <vector>

class Env
{
    public:
    Env();
    
    static MPI_Comm MPI_WORLD;
    static int rank;
    static int nranks;
    static bool is_master;
    static void init();
    static void barrier();
    static void finalize();
    static void exit(int code);
    
     
    static MPI_Group rowgrps_group_, rowgrps_group;
    static MPI_Comm rowgrps_comm;         
    static int rank_rg;
    static int nranks_rg;
    static MPI_Group colgrps_group_, colgrps_group;
    static MPI_Comm colgrps_comm;         
    static int rank_cg;
    static int nranks_cg;
    static void grps_init(std::vector<int32_t> &grps_ranks, int32_t grps_nranks, 
               int &grps_rank_, int &grps_nranks_,
               MPI_Group &grps_group_, MPI_Group &grps_group, MPI_Comm &grps_comm);
    static void rowgrps_init(std::vector<int32_t> &rowgrps_ranks, int32_t rowgrps_nranks);
    static void colgrps_init(std::vector<int32_t> &colgrps_ranks, int32_t colgrps_nranks);               

    static double start;
    static double finish;    
    static double clock();
    static void   tick();
    static void   tock(std::string preamble);
};

MPI_Comm Env::MPI_WORLD;
int  Env::rank = -1;
int  Env::nranks = -1;
bool Env::is_master = false;

MPI_Group Env::rowgrps_group_;
MPI_Group Env::rowgrps_group;
MPI_Comm Env::rowgrps_comm;
int  Env::rank_rg = -1;
int  Env::nranks_rg = -1;

MPI_Group Env::colgrps_group_;
MPI_Group Env::colgrps_group;
MPI_Comm Env::colgrps_comm;
int  Env::rank_cg = -1;
int  Env::nranks_cg = -1;



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

void Env::grps_init(std::vector<int32_t> &grps_ranks, int grps_nranks, int &grps_rank_, int &grps_nranks_,
                    MPI_Group &grps_group_, MPI_Group &grps_group, MPI_Comm &grps_comm)
{
    
    MPI_Comm_group(MPI_COMM_WORLD, &grps_group_);
    MPI_Group_incl(grps_group_, grps_nranks, grps_ranks.data(), &grps_group);
    MPI_Comm_create(MPI_COMM_WORLD, grps_group, &grps_comm);
    
    if (MPI_COMM_NULL != grps_comm) 
    {
        MPI_Comm_rank(grps_comm, &grps_rank_);
        MPI_Comm_size(grps_comm, &grps_nranks_);
    }
}

void Env::rowgrps_init(std::vector<int32_t> &rowgrps_ranks, int32_t rowgrps_nranks)
{
    grps_init(rowgrps_ranks, rowgrps_nranks, rank_rg, nranks_rg, rowgrps_group_, rowgrps_group, rowgrps_comm);
}

void Env::colgrps_init(std::vector<int32_t> &colgrps_ranks, int32_t colgrps_nranks)
{
    grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_group_, colgrps_group, colgrps_comm);   
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
    MPI_Group_free(&rowgrps_group_);
    MPI_Group_free(&rowgrps_group);
    MPI_Comm_free(&rowgrps_comm);
    
    MPI_Group_free(&colgrps_group_);
    MPI_Group_free(&colgrps_group);
    MPI_Comm_free(&colgrps_comm);
     
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

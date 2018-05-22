#include <iostream>
#include <mpi.h>

using namespace std;


int getNodeCount(void)
{
   int rank, is_rank0, nodes;
   MPI_Comm shmcomm;

   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
   MPI_Comm_rank(shmcomm, &rank);
   is_rank0 = (rank == 0) ? 1 : 0;
   MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Comm_free(&shmcomm);
   return nodes;
}



int main(int argc, char **argv) {

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[world_size];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    //printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
    cout << "World size = " << world_size << endl;
	cout << "Node size = " << getNodeCount() << endl;

    MPI_Finalize();

    return 0;
}

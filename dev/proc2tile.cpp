#include <iostream>
#include <mpi.h>
#include <string.h>

using namespace std;


int getCount(const char *description, int split_type) {
  int rank, is_rank0, nodes;
  MPI_Comm shmcomm;

  MPI_Comm_split_type(MPI_COMM_WORLD, split_type, 0, MPI_INFO_NULL, &shmcomm);
  //MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &rank);
  cout << "Rank=" << rank << "," << description <<  endl;
  if(!strcmp(description, "OMPI_COMM_TYPE_CORE")) {
    is_rank0 = 1;
  } else {
    is_rank0 = (rank == 0) ? 1 : 0;
  }
   
   MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Comm_free(&shmcomm);
   if(rank == 0) {
    cout << description << ": " << nodes << endl;
  }
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
  MPI_Barrier(MPI_COMM_WORLD);
    //printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
    //cout << "World size = " << world_size << endl;

    //int count = 0;
  int node_count = getCount("OMPI_COMM_TYPE_NODE", OMPI_COMM_TYPE_NODE);     // 0

    //count = getCount(OMPI_COMM_TYPE_HWTHREAD); // 1
  int core_count = getCount("OMPI_COMM_TYPE_CORE", OMPI_COMM_TYPE_CORE);     // 2
    //count = getCount(OMPI_COMM_TYPE_L1CACHE);  // 3 
    //count = getCount(OMPI_COMM_TYPE_L2CACHE);  // 4
    //count = getCount(OMPI_COMM_TYPE_L3CACHE);  // 5
  int socket_count = getCount("OMPI_COMM_TYPE_SOCKET", OMPI_COMM_TYPE_SOCKET);   // 6
    //count = getCount(OMPI_COMM_TYPE_NUMA);     // 7
    //count = getCount(OMPI_COMM_TYPE_BOARD);    // 8
    //count = getCount(OMPI_COMM_TYPE_HOST);     // 9
    //count = getCount(OMPI_COMM_TYPE_CU);       // 10
    //count = getCount(OMPI_COMM_TYPE_CLUSTER);  // 11

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int proc_count = world_size;
  int sockets_per_node;
  int cores_per_socket;
  if(rank == 0) {
    sockets_per_node = socket_count / node_count;
    cores_per_socket = (core_count / node_count) / sockets_per_node;
    cout << "node_count=" << node_count << ",core_count=" << core_count << ",socket_count=" << socket_count << endl;
    cout << "node_count=" << node_count << ",sockets_per_node=" << sockets_per_node << ",cores_per_socket=" << cores_per_socket << endl;
    
    int grid = proc_count * proc_count;
    int tiles_per_node = grid / node_count;
    int tiles_per_socket = tiles_per_node / sockets_per_node;
    int tiles_per_core = tiles_per_socket / cores_per_socket;


    cout << "proc_count=" << proc_count << endl;
    cout << "grid=" << grid << ",tiles_per_node=" << tiles_per_node << ",tiles_per_socket=" << tiles_per_socket << ",tiles_per_core=" << tiles_per_core << endl;
    
    int i, j, k, l = 0, tile;
    for(i = 0; i < proc_count; i++) {
      cout << "|";
      for(j = 0; j < proc_count; j++) {
          int tile = (i * proc_count) + j;
          if(tile <= 9) {
            cout << "0" << tile << "|";
          } else {
            cout << tile << "|";
          }
      }
      cout << endl;
    }

    for(i = 0; i < node_count; i++) {
      cout << "node=" << i << endl;
      for(j = 0; j < sockets_per_node; j++) {
        cout << "  socket=" << j << endl;
        for(k = 0; k < cores_per_socket; k++) {
          cout << "    core=" << k << endl;
          

        }

      }
    }

    cout << proc_count / node_count << endl;
    




  }

/*
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == -1) {
      cout << "OMPI_COMM_TYPE_NODE    : " << getCount(OMPI_COMM_TYPE_NODE) << endl;
      cout << "OMPI_COMM_TYPE_HWTHREAD: " << getCount(OMPI_COMM_TYPE_HWTHREAD) << endl;
      cout << "OMPI_COMM_TYPE_CORE    : " << getCount(OMPI_COMM_TYPE_CORE) << endl;
      cout << "OMPI_COMM_TYPE_L1CACHE : " << getCount(OMPI_COMM_TYPE_L1CACHE) << endl;
      cout << "OMPI_COMM_TYPE_L2CACHE : " << getCount(OMPI_COMM_TYPE_L2CACHE) << endl;
      cout << "OMPI_COMM_TYPE_L3CACHE : " << getCount(OMPI_COMM_TYPE_L3CACHE) << endl;
      cout << "OMPI_COMM_TYPE_L3CACHE : " << getCount(OMPI_COMM_TYPE_L3CACHE) << endl;
      cout << "OMPI_COMM_TYPE_SOCKET  : " << getCount(OMPI_COMM_TYPE_SOCKET) << endl;
      cout << "OMPI_COMM_TYPE_NUMA    : " << getCount(OMPI_COMM_TYPE_NUMA) << endl;
      cout << "OMPI_COMM_TYPE_BOARD   : " << getCount(OMPI_COMM_TYPE_BOARD) << endl;
      cout << "OMPI_COMM_TYPE_HOST    : " << getCount(OMPI_COMM_TYPE_HOST) << endl;
      cout << "OMPI_COMM_TYPE_CU      : " << getCount(OMPI_COMM_TYPE_CU) << endl;
      cout << "OMPI_COMM_TYPE_CLUSTER : " << getCount(OMPI_COMM_TYPE_CLUSTER) << endl;
    }
*/  

    MPI_Finalize();

    return 0;
}

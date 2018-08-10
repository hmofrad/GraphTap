#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <mpi.h>
#include <sys/time.h> 
//#include <iostream>

//using namespace std;


#define nstrings 1
const char *const strings[nstrings] = {
 "Hello world! "
 };

 int main(int argc, char** argv) {
	
	struct timeval t1, t2;
    double elapsedTime;
	
	gettimeofday(&t1, NULL);

	MPI_Init(NULL, NULL);
	int rank;
	int nranks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	
	//srand(time(NULL) + rank);
	//int my_len =  (rand() % 10) + 1; // str_len   \in [1, 9]
	//int my_char = (rand() % 26) + 65; // str_char \in [65, 90] = [A, Z]
	
	int myStringNum = rank % nstrings;
	char *my_str = (char *)strings[myStringNum];
	int my_len = strlen(my_str);
	//char my_str[my_len + 1];
	//memset(my_str, my_char, my_len);
	///my_str[my_len] = '\0';
	//printf("rank %d of %d has string=%s with size=%zu\n",
   	//        rank, nranks, my_str, strlen(my_str));
	
	int max_len = 0;
	MPI_Allreduce(&my_len, &max_len, 1, 
	              MPI_INT, MPI_MAX, MPI_COMM_WORLD); 
				  
	// + 1 for taking account of null pointer at the end ['\n']
	char *my_str_padded[max_len + 1]; 
	memset(my_str_padded, '\0', max_len + 1);
	memcpy(my_str_padded, my_str, my_len);
	
	char *all_str = NULL;
	if(!rank) {
		int all_len = (max_len + 1) * nranks;
		all_str = malloc(all_len * sizeof(char));	
		memset(all_str, '\0', all_len);
	}
	
	MPI_Gather(my_str_padded, max_len + 1, MPI_CHAR, 
	                 all_str, max_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
					 
	gettimeofday(&t2, NULL);
	
	// compute and print the elapsed time in millisec
	if(!rank) {
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("%f ms.\n", elapsedTime);
	}
		 
	/*
	if(!rank) {
		char *str_idx = all_str;
		int rank_idx = 0;
		while(*str_idx) {
			printf("rank %d sent string=%s with size=%zu\n", 
			        rank_idx, str_idx, strlen(str_idx));
			str_idx = str_idx + max_len + 1;
			rank_idx++;
		}
		
	}
	*/
	
	MPI_Finalize();
	return(0);	
}
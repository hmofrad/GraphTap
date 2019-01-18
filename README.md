# GraphTap: Efiicient Distributed Graph Analytics using Triply Compressed Sparse Matrix Format

## Build
    make clean && make
## Run
    mpirun -np 4 bin/./pr   rmat10.bin   1024 20
    mpirun -np 4 bin/./sssp rmat10_w.bin 1024 1
    mpirun -np 4 bin/./bfs  rmat10.bin   1024 1
    mpirun -np 4 bin/./cc   rmat10.bin   1024
##Contact
    Mohammad Hasanzadeh Mofrad
    moh18@pitt.edu

# GraphTap Distributed Graph Analytics System

## Build
    make clean && make

## Run
    mpirun -np 4 bin/./pr   data/rmat10_1024.bin   1024 20
    mpirun -np 4 bin/./sssp data/rmat10_1024_w.bin 1024 0
    mpirun -np 4 bin/./bfs  data/rmat10_1024.bin   1024 0
    mpirun -np 4 bin/./cc   data/rmat10_1024.bin   1024

##
Mohammad Hasanzadeh Mofrad, Rami Melhem, Yousuf Ahmad and Mohammad Hammoud. [“Efficient Distributed Graph Analytics using Triply Compressed Sparse Format.”](http://people.cs.pitt.edu/~hasanzadeh/files/papers/PID6084671.pdf) In proceedings of IEEE Cluster, Albuquerque, NM USA, 2019

## Contact
    Mohammad Hasanzadeh Mofrad
    m.hasanzadeh.mofrad@gmail.com

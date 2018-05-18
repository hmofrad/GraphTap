# C++ and MPI Compilers (minimum -std=c++14)
CXX = g++
MPI_CXX = mpicxx


# Usage:
#  make
#  make <tk> run app=<app_name> [np=<num_procs> (default=1)]
#                               [tpp=<threads_per_proc> (default=2)]
#                               [mf=<machine_file> (default=./machinefile)]
#                               args="<input_arguments>"
#     where <tk> = { ga (graph_analytics) | ir (information_retrieval) | gs (graph_simulation) }


# make run args
#app ?=
np ?= 1
tpp ?= 2
mf ?= ./machinefile
#args ?=


# Toolkits <tk> (default=ga)
ga = graph_analytics
ir = information_retrieval
gs = graph_simulation


# Compiler Flags:
# For debugging, enable DEBUG and disable OPTIMIZE
DNWARN = -Wno-comment -Wno-literal-suffix -Wno-sign-compare \
         -Wno-unused-function -Wno-unused-variable -Wno-reorder
THREADED = -fopenmp -D_GLIBCXX_PARALLEL
#DEBUG = -g -fsanitize=undefined,address -lasan -lubsan
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native


SRC_UTILS = $(wildcard src/utils/*.cpp)


.PHONY: all tools apps ga ga_all ir ir_all gs gs_all

all: clean tools apps

clean:
	rm -rf bin

ga: $(eval tk=$(ga))
	$(eval tk=$(ga))
ga_all: pr
#ga_all: ga degree pr sssp bfs cc tc

ir:
	$(eval tk=$(ir))
ir_all: ir tfidf tfidf_batch

gs:
	$(eval tk=$(gs))
gs_all: gs graphsim

apps: ga_all
#apps: ga_all ir_all gs_all

tools:
	@mkdir -p bin/tools
	$(CXX) -std=c++14 $(DNWARN) $(OPTIMIZE) $(DEBUG) src/tools/mtx2bin.cpp -o bin/tools/mtx2bin
	$(MPI_CXX) -std=c++14 $(DNWARN) $(OPTIMIZE) $(DEBUG) src/tools/cw2bin.cpp \
			-lboost_serialization -lboost_mpi -o bin/tools/cw2bin

run: #$(app)
	export OMP_NUM_THREADS=$(tpp); \
	mpirun -np $(np) -machinefile $(mf) bin/$(tk)/$(app) $(args)

.DEFAULT:
	@mkdir -p bin/$(tk)
	$(MPI_CXX) -std=c++14 $(DNWARN) $(THREADED) $(OPTIMIZE) $(DEBUG) \
		$(SRC_UTILS) src/apps/$(tk)/$@.cpp -lboost_serialization -I src -o bin/$(tk)/$@
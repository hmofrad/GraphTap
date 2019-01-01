/*
 * compressed_column.hpp: Column compressed storage implementaion
 * Compressed Sparse Column (CSC)
 * Double Compressed Sparse Column (DCSC)
 * Triple Compressed Sparse Column (TCSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP

#include <sys/mman.h>
#include <cstring> 
#include <algorithm>
 
enum Compression_type
{
   _CSC_, // Compressed Sparse Col
  _DCSC_, // Compressed Sparse Col
  _TCSC_, // Compressed Sparse Col
 _TCSC1_, // Compressed Sparse Col
};

template<typename Weight, typename Integer_Type>
struct Compressed_column {
    public:
        virtual ~Compressed_column() {}
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                              const Integer_Type tile_height,
                              const Integer_Type tile_width){};
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,
                              const std::vector<char>&         nnzcols_bitvector,
                              const std::vector<Integer_Type>& nnzcols_indices){};
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,                
                              const std::vector<char>&         nnzrows_bitvector,
                              const std::vector<Integer_Type>& nnzrows_indices, 
                              const std::vector<char>&         nnzcols_bitvector,
                              const std::vector<Integer_Type>& nnzcols_indices, 
                              const std::vector<Integer_Type>& regular_rows_indices,
                              const std::vector<char>&         regular_rows_bitvector,
                              const std::vector<Integer_Type>& source_rows_indices,
                              const std::vector<char>&         source_rows_bitvector,
                              const std::vector<Integer_Type>& regular_columns_indices,
                              const std::vector<char>&         regular_columns_bitvector,
                              const std::vector<Integer_Type>& sink_columns_indices,
                              const std::vector<char>&         sink_columns_bitvector){};
};


template<typename Weight, typename Integer_Type>
struct CSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        CSC_BASE(uint64_t nnz_, Integer_Type ncols_);
        ~CSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width);
        uint64_t nnz;
        Integer_Type ncols;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
};

template<typename Weight, typename Integer_Type>
CSC_BASE<Weight, Integer_Type>::CSC_BASE(uint64_t nnz_, Integer_Type ncols_) {
    nnz = nnz_;
    ncols = ncols_;
    #ifdef HAS_WEIGHT
    if((A = (Weight*) mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(Weight));
    #endif
    
    if((IA = (Integer_Type*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(Integer_Type));
    
    if((JA = (Integer_Type*) mmap(nullptr, (ncols + 1) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, (ncols + 1) * sizeof(Integer_Type));
}

template<typename Weight, typename Integer_Type>
CSC_BASE<Weight, Integer_Type>::~CSC_BASE() {
    #ifdef HAS_WEIGHT
    if(munmap(A, nnz * sizeof(Weight)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    #endif
    
    if(munmap(IA, nnz * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, (ncols + 1) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}

template<typename Weight, typename Integer_Type>
void CSC_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                                              const Integer_Type tile_height, 
                                              const Integer_Type tile_width) {
    struct Triple<Weight, Integer_Type> pair;
    uint32_t i = 0; // Row Index
    uint32_t j = 1; // Col index
    JA[0] = 0;
    for (auto& triple : *triples) {
        pair  = {(triple.row % tile_height), (triple.col % tile_width)};
        while((j - 1) != pair.col) {
            j++;
            JA[j] = JA[j - 1];
        }            
        #ifdef HAS_WEIGHT
        A[i] = triple.weight;
        #endif
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (ncols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
}

template<typename Weight, typename Integer_Type>
struct DCSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        DCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_);
        ~DCSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,
                              const std::vector<char>&         nnzcols_bitvector,
                              const std::vector<Integer_Type>& nnzcols_indices);
        uint64_t nnz;
        Integer_Type nnzcols;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
        Integer_Type* JC; // COL_IDX
};

template<typename Weight, typename Integer_Type>
DCSC_BASE<Weight, Integer_Type>::DCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    #ifdef HAS_WEIGHT
    if((A = (Weight*) mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(Weight));
    #endif
    
    if((IA = (Integer_Type*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(Integer_Type));
    
    if((JA = (Integer_Type*) mmap(nullptr, (nnzcols + 1) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, (nnzcols + 1) * sizeof(Integer_Type));
    
    if((JC = (Integer_Type*) mmap(nullptr, nnzcols * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC, 0, nnzcols * sizeof(Integer_Type));
}

template<typename Weight, typename Integer_Type>
DCSC_BASE<Weight, Integer_Type>::~DCSC_BASE() {
    #ifdef HAS_WEIGHT
    if(munmap(A, nnz * sizeof(Weight)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    #endif
    
    if(munmap(IA, nnz * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, (nnzcols + 1) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JC, nnzcols * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}

template<typename Weight, typename Integer_Type>
void DCSC_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                                               const Integer_Type tile_height, 
                                               const Integer_Type tile_width,
                                               const std::vector<char>&         nnzcols_bitvector,
                                               const std::vector<Integer_Type>& nnzcols_indices) {
    struct Triple<Weight, Integer_Type> pair;
    uint32_t i = 0; // Row Index
    uint32_t j = 1; // Col index
    JA[0] = 0;
    for (auto& triple : *triples) {
        pair  = {(triple.row % tile_height), (triple.col % tile_width)};
        while((j - 1) != nnzcols_indices[pair.col]) {
            j++;
            JA[j] = JA[j - 1];
        }            
        #ifdef HAS_WEIGHT
        A[i] = triple.weight;
        #endif
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (nnzcols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
    // Column indices
    Integer_Type k = 0;
    for(j = 0; j < tile_width; j++) {
        if(nnzcols_bitvector[j]) {
            JC[k] = j;
            k++;
        }
    }
}


template<typename Weight, typename Integer_Type>
struct TCSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        TCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_, Integer_Type nnzrows_);
        ~TCSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,                
                              const std::vector<char>&         nnzrows_bitvector,
                              const std::vector<Integer_Type>& nnzrows_indices, 
                              const std::vector<char>&         nnzcols_bitvector,
                              const std::vector<Integer_Type>& nnzcols_indices, 
                              const std::vector<Integer_Type>& regular_rows_indices,
                              const std::vector<char>&         regular_rows_bitvector,
                              const std::vector<Integer_Type>& source_rows_indices,
                              const std::vector<char>&         source_rows_bitvector,
                              const std::vector<Integer_Type>& regular_columns_indices,
                              const std::vector<char>&         regular_columns_bitvector,
                              const std::vector<Integer_Type>& sink_columns_indices,
                              const std::vector<char>&         sink_columns_bitvector);
                              
        void allocate_local_reg(Integer_Type nnzcols_regulars_local_);
        void allocate_local_src(Integer_Type nnzcols_sources_local_);
        void allocate_local_src_reg(Integer_Type nnzcols_sources_regulars_local_);
        void allocate_local_reg_snk(Integer_Type nnzcols_regulars_sinks_local_);
        void allocate_local_src_snk(Integer_Type nnzcols_sources_sinks_local_);
        uint64_t nnz;
        Integer_Type nnzcols;
        Integer_Type nnzcols_regulars;
        Integer_Type nnzcols_regulars_local;
        Integer_Type nnzcols_sources_local;
        Integer_Type nnzcols_sources_regulars_local;
        Integer_Type nnzcols_regulars_sinks_local;
        Integer_Type nnzcols_sources_sinks_local;
        Integer_Type nnzrows;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
        Integer_Type* JC; // COL_IDX
        Integer_Type* IR; // ROW_PTR
        Integer_Type  NC_REG_R_REG_C;
        Integer_Type* JC_REG_R_REG_C;
        Integer_Type* JA_REG_R_REG_C;
        Integer_Type  NC_REG_R_SNK_C;
        Integer_Type* JC_REG_R_SNK_C;
        Integer_Type* JA_REG_R_SNK_C;
        
        Integer_Type* JA_REG_C;  // COL_PTR_REG_COL
        Integer_Type* JC_REG_C;  // COL_IDX_REG_COL
        Integer_Type* JA_REG_R;  // COL_PTR_REG_ROW
        Integer_Type* JA_SRC_R;  // COL_PTR_SRC_ROW
        Integer_Type* J_SRC_C;  // COL_IDX_SRC_COL
        Integer_Type* JA_REG_RC; // COL_PTR_REG_COL_REG_ROW
        Integer_Type* JA_SRC_RC; // COL_PTR_REG_COL_SRC_ROW
        Integer_Type* J_SRC_RC;  // COL_IDX_REG_COL_SRC_ROW
        Integer_Type* JC_NNZ_REG_C;  // COL_IDX_NNZ_REG_COL
        Integer_Type* JA_REG_R_SNK_C;
        Integer_Type* J_REG_SNK_C;
        Integer_Type* JA_SRC_R_SNK_C;
        Integer_Type* J_SRC_SNK_C;
};

template<typename Weight, typename Integer_Type>
TCSC_BASE<Weight, Integer_Type>::TCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_, Integer_Type nnzrows_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    nnzrows = nnzrows_;
    //nnzcols_regulars = nnzcols_regulars_;
    if(nnz and nnzcols and nnzrows) {
        #ifdef HAS_WEIGHT
        if((A = (Weight*) mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(A, 0, nnz * sizeof(Weight));
        #endif
        if((IA = (Integer_Type*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(IA, 0, nnz * sizeof(Integer_Type));
        
        if((JA = (Integer_Type*) mmap(nullptr, (nnzcols + 1) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA, 0, (nnzcols + 1) * sizeof(Integer_Type));
        
        if((JC = (Integer_Type*) mmap(nullptr, nnzcols * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JC, 0, nnzcols * sizeof(Integer_Type));
        
        if((IR = (Integer_Type*) mmap(nullptr, nnzrows * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(IR, 0, nnzrows * sizeof(Integer_Type));
    }
    
    /*
    if((JA_REG_C = (Integer_Type*) mmap(nullptr, (nnzcols_regulars * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_C, 0, (nnzcols_regulars * 2) * sizeof(Integer_Type));
    
    if((JC_REG_C = (Integer_Type*) mmap(nullptr, nnzcols_regulars * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC_REG_C, 0, nnzcols_regulars * sizeof(Integer_Type));
    */
    /*
    if((JA_REG_R = (Integer_Type*) mmap(nullptr, (nnzcols * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_R, 0, (nnzcols * 2) * sizeof(Integer_Type));
    */

    
    /*
    if((JA_REG_RC = (Integer_Type*) mmap(nullptr, (nnzcols_regulars * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_RC, 0, (nnzcols_regulars * 2) * sizeof(Integer_Type));            
    
    if((JC_NNZ_REG_C = (Integer_Type*) mmap(nullptr, nnzcols_regulars * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC_NNZ_REG_C, 0, nnzcols_regulars * sizeof(Integer_Type));
    */
}

template<typename Weight, typename Integer_Type>
TCSC_BASE<Weight, Integer_Type>::~TCSC_BASE() {
    if(nnz and nnzcols and nnzrows) {
        #ifdef HAS_WEIGHT
        if(munmap(A, nnz * sizeof(Weight)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        #endif
        if(munmap(IA, nnz * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(JA, (nnzcols + 1) * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(JC, nnzcols * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(IR, nnzrows * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
        }
    }
   
    /*
    if(munmap(JA_REG_C, (nnzcols_regulars_local * 2) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    */
    
    if(munmap(JC_REG_C, nnzcols_regulars_local * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA_REG_R, (nnzcols * 2) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(nnzcols_sources_local) {
        if(munmap(JA_SRC_R, (nnzcols_sources_local * 2) * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(J_SRC_C, nnzcols_sources_local * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
    }
    
    if(munmap(JA_REG_RC, (nnzcols_regulars_local * 2) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    /*
    if(munmap(JA_SRC_RC, (nnzcols_regulars_local * 2) * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(J_SRC_RC, nnzcols_regulars_local * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    */
    
    if(nnzcols_sources_regulars_local > 0) {
        if(munmap(JA_SRC_RC, (nnzcols_sources_regulars_local * 2) * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(J_SRC_RC, nnzcols_sources_regulars_local * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
    }
    if(nnzcols_regulars_sinks_local > 0) {
        if(munmap(JA_REG_R_SNK_C, (nnzcols_regulars_sinks_local * 2) * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(J_REG_SNK_C, nnzcols_regulars_sinks_local * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
    }
    
    if(nnzcols_sources_sinks_local > 0) {
        if(munmap(JA_SRC_R_SNK_C, (nnzcols_sources_sinks_local * 2) * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(J_SRC_SNK_C, nnzcols_sources_sinks_local * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
    }
    
   
    if(munmap(JC_NNZ_REG_C, nnzcols_regulars_local * sizeof(Integer_Type)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::allocate_local_reg(Integer_Type nnzcols_regulars_local_) {
    nnzcols_regulars_local = nnzcols_regulars_local_;
    if((JA_REG_C = (Integer_Type*) mmap(nullptr, (nnzcols_regulars_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_C, 0, (nnzcols_regulars_local * 2) * sizeof(Integer_Type));
    
    if((JC_REG_C = (Integer_Type*) mmap(nullptr, nnzcols_regulars_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC_REG_C, 0, nnzcols_regulars_local * sizeof(Integer_Type));

    if((JA_REG_RC = (Integer_Type*) mmap(nullptr, (nnzcols_regulars_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_RC, 0, (nnzcols_regulars_local * 2) * sizeof(Integer_Type));            
        
    /*    
    if((JA_SRC_RC = (Integer_Type*) mmap(nullptr, (nnzcols_regulars_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_SRC_RC, 0, (nnzcols_regulars_local * 2) * sizeof(Integer_Type));
    
    if((J_SRC_RC = (Integer_Type*) mmap(nullptr, nnzcols_regulars_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(J_SRC_RC, 0, nnzcols_regulars_local * sizeof(Integer_Type));
    */
    if((JC_NNZ_REG_C = (Integer_Type*) mmap(nullptr, nnzcols_regulars_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC_NNZ_REG_C, 0, nnzcols_regulars_local * sizeof(Integer_Type));
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::allocate_local_src(Integer_Type nnzcols_sources_local_) {
    nnzcols_sources_local = nnzcols_sources_local_;
    if(nnzcols_sources_local > 0) {
        if((JA_SRC_R = (Integer_Type*) mmap(nullptr, (nnzcols_sources_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA_SRC_R, 0, (nnzcols_sources_local * 2) * sizeof(Integer_Type));
        
        if((J_SRC_C = (Integer_Type*) mmap(nullptr, nnzcols_sources_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(J_SRC_C, 0, nnzcols_sources_local * sizeof(Integer_Type));
    }
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::allocate_local_src_reg(Integer_Type nnzcols_sources_regulars_local_) {
    nnzcols_sources_regulars_local = nnzcols_sources_regulars_local_;
    if(nnzcols_sources_regulars_local > 0) {
        if((JA_SRC_RC = (Integer_Type*) mmap(nullptr, (nnzcols_sources_regulars_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA_SRC_RC, 0, (nnzcols_sources_regulars_local * 2) * sizeof(Integer_Type));
        
        if((J_SRC_RC = (Integer_Type*) mmap(nullptr, nnzcols_sources_regulars_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(J_SRC_RC, 0, nnzcols_sources_regulars_local * sizeof(Integer_Type));
    }
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::allocate_local_reg_snk(Integer_Type nnzcols_regulars_sinks_local_) {
    nnzcols_regulars_sinks_local = nnzcols_regulars_sinks_local_;
    if(nnzcols_regulars_sinks_local > 0) {
        if((JA_REG_R_SNK_C = (Integer_Type*) mmap(nullptr, (nnzcols_regulars_sinks_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA_REG_R_SNK_C, 0, (nnzcols_regulars_sinks_local * 2) * sizeof(Integer_Type));
        
        if((J_REG_SNK_C = (Integer_Type*) mmap(nullptr, nnzcols_regulars_sinks_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(J_REG_SNK_C, 0, nnzcols_regulars_sinks_local * sizeof(Integer_Type));
    }
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::allocate_local_src_snk(Integer_Type nnzcols_sources_sinks_local_) {
    nnzcols_sources_sinks_local = nnzcols_sources_sinks_local_;
    if(nnzcols_sources_sinks_local > 0) {
        if((JA_SRC_R_SNK_C = (Integer_Type*) mmap(nullptr, (nnzcols_sources_sinks_local * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA_SRC_R_SNK_C, 0, (nnzcols_sources_sinks_local * 2) * sizeof(Integer_Type));
        
        if((J_SRC_SNK_C = (Integer_Type*) mmap(nullptr, nnzcols_sources_sinks_local * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(J_SRC_SNK_C, 0, nnzcols_sources_sinks_local * sizeof(Integer_Type));
    }
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                                               const Integer_Type tile_height, 
                                               const Integer_Type tile_width,                
                                               const std::vector<char>&         nnzrows_bitvector,
                                               const std::vector<Integer_Type>& nnzrows_indices, 
                                               const std::vector<char>&         nnzcols_bitvector,
                                               const std::vector<Integer_Type>& nnzcols_indices, 
                                               const std::vector<Integer_Type>& regular_rows_indices,
                                               const std::vector<char>&         regular_rows_bitvector,
                                               const std::vector<Integer_Type>& source_rows_indices,
                                               const std::vector<char>&         source_rows_bitvector,
                                               const std::vector<Integer_Type>& regular_columns_indices,
                                               const std::vector<char>&         regular_columns_bitvector,
                                               const std::vector<Integer_Type>& sink_columns_indices,
                                               const std::vector<char>&         sink_columns_bitvector) {

/*
populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples, 
                                               const std::vector<char>& nnzcols_bitvector,
                                               const std::vector<Integer_Type>& nnzcols_indices, 
                                               const std::vector<Integer_Type>& nnzcols_regulars_indices,
                                               const std::vector<char>& nnzcols_regulars_bitvector,
                                               const std::vector<char>& nnzcols_sinks_bitvector,
                                               const std::vector<char>& nnzrows_bitvector,
                                               const std::vector<Integer_Type>& nnzrows_indices, 
                                               const std::vector<char>& nnzrows_sources_bitvector,
                                               const Integer_Type tile_height, 
                                               const Integer_Type tile_width) {
*/                                                   
    struct Triple<Weight, Integer_Type> pair;
    Integer_Type i = 0; // Row Index
    Integer_Type j = 1; // Col index
    JA[0] = 0;
    for (auto& triple : *triples) {
        pair  = {(triple.row % tile_height), (triple.col % tile_width)};
        while((j - 1) != nnzcols_indices[pair.col]) {
            j++;
            JA[j] = JA[j - 1];
        }            
        #ifdef HAS_WEIGHT
        A[i] = triple.weight;
        #endif
        JA[j]++;
        IA[i] = nnzrows_indices[pair.row];
        i++;
    }
    while((j + 1) < (nnzcols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
    // Column indices
    Integer_Type k = 0;
    for(j = 0; j < tile_height; j++) {
        if(nnzcols_bitvector[j]) {
            JC[k] = j;
            k++;
        }
    }
    /*
    // Refine JC array indices
    for(j = 1; j < nnzcols + 1; j++) {
        if(JA[j - 1] == JA[j]) {
            JC[j] = JC[j-1];
        }
    }
    */
    
    // Rows indices
    k = 0;
    for(i = 0; i < tile_height; i++) {
        if(nnzrows_bitvector[i]) {
            //IR[k] = nnzrows_indices[i];
            IR[k] = i;
            k++;
        }
    }
    
    
        /*
        Integer_Type  NC_REG_R_REG_C;
        Integer_Type* JC_REG_R_REG_C;
        Integer_Type* JA_REG_R_REG_C;
        Integer_Type  NC_REG_R_SNK_C;
        Integer_Type* JC_REG_R_SNK_C;
        Integer_Type* JA_REG_R_SNK_C;
        */
    
    
    if((JA_REG_R = (Integer_Type*) mmap(nullptr, (nnzcols * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_R, 0, (nnzcols * 2) * sizeof(Integer_Type));
    
    
    
    
    /*
    if(!Env::rank) {
        for(int i = 0; i < nnzrows; i++) {
            printf("%d ", IR[i]);
        }
        printf(" %d, %d\n", k, nnzrows );
        
    }
    */
    
    /*
    if(!Env::rank) {
        for(j = 0; j < nnzcols; j++) {
            printf("%d:%d\n", j, JC[j]);
            for(i = JA[j]; i < JA[j+1]; i++) {
                printf("   %d %d\n", i, IA[i]);
            }
        }
    }
    */ 
    /*
    // Regular columns pointers/indices
    Integer_Type j1 = 0;
    Integer_Type j2 = 0;
    k = 0;
    while((j1 < nnzcols) and (j2 < nnzcols_regulars)) {
        if(JC[j1] == nnzcols_regulars_indices[j2]) {
                k++;
                j1++;
                j2++;
        }
        else if (JC[j1] < nnzcols_regulars_indices[j2])
            j1++;
        else
            j2++;
    }
    allocate_local_reg(k);

    if(Env::rank == 2 or Env::rank == 3)
        printf("rank=%d, nnzcols=%d nnzcols_regulars=%d nnzcols_regulars_local=%d\n", Env::rank, nnzcols, nnzcols_regulars, nnzcols_regulars_local);
    
    j1 = 0;
    j2 = 0;        
    k = 0;
    Integer_Type l = 0;
    while((j1 < nnzcols) and (j2 < nnzcols_regulars)) {
        if(JC[j1] == nnzcols_regulars_indices[j2]) {
            JC_REG_C[k] = JC[j1];
            JC_NNZ_REG_C[k] = j1;
            k++;
            JA_REG_C[l] = JA[j1];
            JA_REG_C[l + 1] = JA[j1 + 1];
            l += 2;
            //if(!Env::rank)
            //    printf("JC[%d]=%d, REG[%d]=%d JA_REG_C=%d,%d\n", j1, JC[j1], j2, nnzcols_regulars_indices[j2], JA_REG_C[l-2], JA_REG_C[l-1]);
            j1++;
            j2++;


        }
        else if (JC[j1] < nnzcols_regulars_indices[j2])
            j1++;
        else
            j2++;
    }    
    */
    // Moving source rows to the end
    Integer_Type l = 0;
    uint32_t s = 0;
    Integer_Type m = 0;
    Integer_Type n = 0;
    std::vector<Integer_Type> r;
    for(j = 0; j < nnzcols; j++) {
        for(i = JA[j]; i < JA[j + 1]; i++) {
            if(source_rows_bitvector[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            s += n;
            if(m > n) {
                for(Integer_Type p = 0; p < n; p++) {
                    for(Integer_Type q = JA[j+1] - 1; q >= JA[j]; q--) {
                        if(source_rows_bitvector[IR[IA[q]]] != 1) {
                            #ifdef HAS_WEIGHT
                            std::swap(A[r[p]], A[q]);
                            #endif
                            std::swap(IA[r[p]], IA[q]);
                            break;
                        }
                        else {
                            if(r[p] == q)
                                break;
                        }
                    }
                }
            }
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
    }
    //printf("%d %d\n", Env::rank, s);
    /*
    if(Env::rank == 1) {
        for(uint32_t j = 0; j < nnzcols; j++) {
            printf("j=%d/%d\n", j, JC[j]);
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("   IA[%d]=%d/%d\n", i, IA[i], nnzrows_sources_bitvector[IR[IA[i]]]);
            }
        }
    }
    */
    //Env::barrier();
    //Env::exit(0);
    // NNZ columns pointers without source rows (1st iteration)
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    for(j = 0; j < nnzcols; j++) {
        //if(!Env::rank)
          //  printf("%d/%d:\n", j, JC[j]);
        for(i = JA[j]; i < JA[j + 1]; i++) {
            //if(!Env::rank)
              //  printf("   %d %d %d\n", i, IA[i], nnzrows_sources_bitvector[IR[IA[i]]]);
            if(source_rows_bitvector[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }   
        }
        if(m > 0) {
            n = r.size();
            JA_REG_R[l] = JA[j];
            JA_REG_R[l + 1] = JA[j + 1] - n;            
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_R[l] = JA[j];
            JA_REG_R[l + 1] = JA[j + 1];
            l += 2;  
        }
    }
    /*
    // NNZ columns pointers for source rows (1st iteration)
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    for(j = 0; j < nnzcols; j++) {
        for(i = JA[j]; i < JA[j + 1]; i++) {
            if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                k++;
                break;
            }   
        }
    }
    allocate_local_src(k);
    k = 0;
    for(j = 0; j < nnzcols; j++) {
        for(i = JA[j]; i < JA[j + 1]; i++) {
            if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }   
        }
        if(m > 0) {
            n = r.size();
            JA_SRC_R[l] = JA[j + 1] - n;
            JA_SRC_R[l + 1] = JA[j + 1]; 
            l += 2; 
            J_SRC_C[k] = j;
            k++;
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
    }
    //if(!Env::rank) {
    //for(j = 0; j < nnzcols_sources_local * 2; j = j + 2)
    //    printf("%d %d %d %d\n", j, J_SRC_C[j/2], JA_SRC_R[j], JA_SRC_R[j+1]);
   // }
    
    
    //if(!Env::rank) {
       // printf("k=%d\n", k);
    //}
    
    //Env::barrier();
    //Env::exit(0);
    
    
    // Regular columns pointers without source rows (2nd to last iteration)
    l = 0;       
    m = 0;
    n = 0;
    Integer_Type o = 0;
    r.clear();
    r.shrink_to_fit(); 
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars_local; j++, k = k + 2) {
        for(uint32_t i = JA_REG_C[k]; i < JA_REG_C[k + 1]; i++) {
            if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                m = (JA_REG_C[k+1] - JA_REG_C[k]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            JA_REG_RC[l] = JA_REG_C[k];
            JA_REG_RC[l + 1] = JA_REG_C[k + 1] - n;   
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_RC[l] = JA_REG_C[k];
            JA_REG_RC[l + 1] = JA_REG_C[k + 1];
            l += 2;  
        }
    }
    // Regular columns pointers for source rows (Last iteration)
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    //if(!Env::rank) {
    for(j = 0; j < nnzcols; j++) {
        if(nnzcols_regulars_bitvector[JC[j]]) {            
            for(i = JA[j]; i < JA[j + 1]; i++) {
                if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                    k++;
                    break;
                }   
            }
        }
    }
    allocate_local_src_reg(k);
    k = 0;
    for(j = 0; j < nnzcols; j++) {
        if(nnzcols_regulars_bitvector[JC[j]]) {
            for(i = JA[j]; i < JA[j + 1]; i++) {
                if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                    m = (JA[j+1] - JA[j]);
                    r.push_back(i);
                }   
            }
            if(m > 0) {
            n = r.size();
            JA_SRC_RC[l] = JA[j + 1] - n;
            JA_SRC_RC[l + 1] = JA[j + 1]; 
            l += 2; 
            J_SRC_RC[k] = j;
            k++;
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
            }
        }
    }
    
    // Sink columns pointers for regular rows (Last iteration)
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    
    //for(j = 0; j < nnzcols; j++) {
    //    if(nnzcols_sinks_bitvector[JC[j]]) {
    //        k++;
            
            //printf("[%d %d %d %d]\n", j, JC[j], nnzcols_sinks_bitvector[JC[j]], JA[j + 1] - JA[j]);
    //        for(i = JA[j]; i < JA[j + 1]; i++) {
    //            if(not (nnzrows_sources_bitvector[IR[IA[i]]] == 1)) {
    //                k++;    
    //                break;
    //            }   
    //        }
    //    }
    //}
    allocate_local_reg_snk(k);
   // printf("Rank=%d/%d\n", Env::rank, k);
    k = 0;
    for(j = 0; j < nnzcols; j++) {
        if(nnzcols_sinks_bitvector[JC[j]]) {
            for(i = JA[j]; i < JA[j + 1]; i++) {
                if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                    m = (JA[j+1] - JA[j]);
                    r.push_back(i);
                }   
            }
            if(m > 0) {
            n = r.size();
            JA_REG_R_SNK_C[l] = JA[j];
            JA_REG_R_SNK_C[l + 1] = JA[j + 1] - n;            
            l += 2; 
            J_REG_SNK_C[k] = j;
            k++;
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
            }
            else {
                JA_REG_R_SNK_C[l] = JA[j];
                JA_REG_R_SNK_C[l + 1] = JA[j + 1]; 
                l += 2;    
                J_REG_SNK_C[k] = j;
                k++;                
            }
        }
    }
    
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    
    for(j = 0; j < nnzcols; j++) {
        if(nnzcols_sinks_bitvector[JC[j]]) {
            for(i = JA[j]; i < JA[j + 1]; i++) {
                if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                    k++;    
                    break;
                }   
            }
        }
    }
    allocate_local_src_snk(k);
    //printf("Rank=%d/%d\n", Env::rank, k);
    k = 0;
    for(j = 0; j < nnzcols; j++) {
        if(nnzcols_sinks_bitvector[JC[j]]) {
            for(i = JA[j]; i < JA[j + 1]; i++) {
                if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                    m = (JA[j+1] - JA[j]);
                    r.push_back(i);
                }   
            }
            if(m > 0) {
                n = r.size();
                JA_SRC_R_SNK_C[l] = JA[j + 1] - n;
                JA_SRC_R_SNK_C[l + 1] = JA[j + 1]; 
                l += 2; 
                J_SRC_SNK_C[k] = j;
                k++;
                m = 0;
                n = 0;
                r.clear();
                r.shrink_to_fit();
            }
        }
    }
    */    

        
    
    
    // TRASHHHHH
    //Integer_Type* JA_SRC_R_SNK_C;
    //Integer_Type* J_SRC_SNK_C;
   // Env::barrier();
   // Env::exit(0);
    
    
    
    //if(!Env::rank) {
      //  for(uint32_t j = 0, k = 0; j < nnzcols_sources_regulars_local; j++, k = k + 2) {
            //printf("j=%d/%d:\n", j, J_SRC_RC[j]);
            //for(uint32_t i = JA_SRC_RC[k]; i < JA_SRC_RC[k + 1]; i++) {
            //    printf("    %d %d %d\n", i, IA[i], nnzrows_sources_bitvector[IR[IA[i]]]);
            //}
        //}
    //}
    //Env::barrier();
    //Env::exit(0);
    
    //for(i = 0; i < nnzcols_sinks_bitvector.size(); i++)
        //printf("%d ", nnzcols_sinks_bitvector[i]);
    //printf("\n\n");
    //for(i = 0; i < nnzcols_regulars_local; i++)
    //    printf("%d ", JC_REG_C[i]);
    //printf("\n\n");
    //for(i = 0; i < nnzcols; i++)
    //    printf("%d ", JC[i]);
    //printf("\n\n");
    
    
    //}
    /*
    //allocate_local_src(k);
    l = 0;       
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars_local; j++, k = k + 2) {
        for(uint32_t i = JA_REG_C[k]; i < JA_REG_C[k + 1]; i++) {
            if(nnzrows_sources_bitvector[IR[IA[i]]] == 1) {
                m = (JA_REG_C[k+1] - JA_REG_C[k]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            JA_SRC_RC[l] = JA_REG_C[k] - n;
            JA_SRC_RC[l + 1] = JA_REG_C[k + 1];
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_RC[l] = JA_REG_C[k];
            JA_REG_RC[l + 1] = JA_REG_C[k + 1];
            l += 2;  
        }
    }
    */
    
    
    /*
    if(!Env::rank) {
        for(uint32_t j = 0, k = 0; j < nnzcols_regulars_local; j++, k = k + 2) {
            printf("j=%d/%d\n", j, JC_NNZ_REG_C[j]);
            for(uint32_t i = JA_REG_C[k]; i < JA_REG_C[k + 1]; i++) {
                printf("   %d %d %d\n", i, IA[i], nnzrows_sources_bitvector[IR[IA[i]]]);
            }
        }
        printf("\n\n");
        for(uint32_t j = 0, k = 0; j < nnzcols_regulars_local; j++, k = k + 2) {
            printf("j=%d/%d\n", j, JC_NNZ_REG_C[j]);
            for(uint32_t i = JA_REG_RC[k]; i < JA_REG_RC[k + 1]; i++) {
                printf("   %d %d %d\n", i, IA[i], nnzrows_sources_bitvector[IR[IA[i]]]);
            }
        }
        
    }
    */
   
    //Env::barrier();
    //Env::exit(0);

//printf("%d %d %d\n", Env::rank, l/2, nnzcols_regulars_local);    
    
    
    
    
    //Env::barrier();
    //Env::exit(0);
       
}
#endif
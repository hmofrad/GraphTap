/*
 * csc.cpp: CSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include "base_csc.hpp" 
#include <chrono>

void populate_csc(struct CSC *csc) {
    uint32_t *A  = (uint32_t *) csc->A;  // Weight      
    uint32_t *IA = (uint32_t *) csc->IA; // ROW_INDEX
    uint32_t *JA = (uint32_t *) csc->JA; // COL_PTR
    uint32_t ncols = csc->  ncols_plus_one - 1;
    
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &pair: *pairs) {
        while((j - 1) != pair.col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (ncols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
}

void walk_csc(struct CSC *csc) {
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t ncols = csc->ncols_plus_one - 1;
    
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], j, A[i]);
        }
    }
}

uint64_t spmv_csc(struct CSC *csc, std::vector<double>& x, std::vector<double>& y) {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t ncols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += x[j];
            num_operations++;
        }
    }
    return(num_operations);
}

void copy_csc_v_x(const std::vector<double> v, const std::vector<double> d, std::vector<double>& x) {
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
        x[i] = d[i] ? (v[i]/d[i]) : 0;   
}

void copy_csc_y_v(const std::vector<double> y, std::vector<double>& v) {
    uint32_t nrows = y.size();
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = y[i];
}

void values_csc(const std::vector<double> v, int num_vals = 10)
{
    for(uint32_t i = 0; i < num_vals; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

double checksum_csc(const std::vector<double> v) {
    double value = 0.0;
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void update_csc(std::vector<double>& v, std::vector<double> y, double alpha) {
    uint32_t nrows = v.size();
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void stats_csc(std::vector<double>& v, double time, uint64_t total_size, uint64_t num_operations)
{
    std::cout << "CSC SpMV kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time:    " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Final value:     " << checksum_csc(v) <<std::endl;
    std::cout << "Num SpMV Ops:    " << num_operations <<std::endl;
    //values_csc(v);
}


void run_pagerank_csc(std::string file_path, uint32_t num_vertices, int num_iters) {
    // Degree program
    pairs = new std::vector<struct Pair>;
    read_binary(file_path);
    column_sort();
    struct CSC *csc = new struct CSC(pairs->size(), num_vertices);
    populate_csc(csc);
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk_csc(csc);
    std::vector<double> v(num_vertices);
    std::vector<double> x(num_vertices, 1);
    std::vector<double> y(num_vertices);
    (void)spmv_csc(csc, x, y);
    copy_csc_y_v(y, v);
    //values_csc(v);
    //(void)checksum_csc(v);
    delete csc;
    csc = nullptr;
    
    // PageRank program
    double alpha = 0.15;
    uint64_t num_operations = 0;
    uint64_t total_size = 0;
    pairs = new std::vector<struct Pair>;
    read_binary(file_path, true);
    column_sort();
    csc = new struct CSC(pairs->size(), num_vertices);
    populate_csc(csc);
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    std::vector<double> d(num_vertices);
    d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    for(int i = 0; i < 20; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        copy_csc_v_x(v, d, x);
        num_operations += spmv_csc(csc, x, y);
        update_csc(v, y, alpha);
    }
    t2 = std::chrono::steady_clock::now();
    total_size += csc->size;
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats_csc(v, t, total_size, num_operations);
    delete csc;
    csc = nullptr;
}
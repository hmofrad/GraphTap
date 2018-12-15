/*
 * csc_.hpp: CSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CSC__HPP
#define CSC__HPP 

#include <chrono>

#include "pair.hpp" 
#include "io.cpp" 
#include "base_csc.hpp" 
#include "csc.hpp"

class CSC_ : protected CSC {
    using CSC::CSC;
    public:
        virtual void run_pagerank();
    protected:
        virtual void filter();
        virtual void destroy_filter();
        void destroy_vectors();
        
        virtual void message();
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        std::vector<char> rows;
        std::vector<uint32_t> rows_vals_nnz;
        std::vector<uint32_t> rows_vals_all;
        uint32_t nnz_rows;
        std::vector<char> cols;
        std::vector<uint32_t> cols_vals_nnz;
        std::vector<uint32_t> cols_vals_all;        
        uint32_t nnz_cols;
};

/*
class CSC_ {
    public:
        CSC_() {};
        CSC_(const std::string file_path_, const uint32_t num_vertices_, const uint32_t num_iterations_) 
            : file_path(file_path_), num_vertices(num_vertices_), num_iterations(num_iterations_) {}
        ~CSC_() {};
        void run_pagerank();
    private:
        std::string file_path = "\0";
        uint32_t num_vertices = 0;
        uint32_t num_iterations = 0;
        uint64_t num_edges = 0;
        uint32_t num_rows = 0;
        std::vector<struct Pair> *pairs = nullptr;
        struct Base *csc = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x;
        std::vector<double> y;
        double alpha = 0.15;
        uint64_t num_operations = 0;
        uint64_t total_size = 0;

        void populate();
        void walk();
        void message();        
        uint64_t spmv();
        void update();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time);
        void space();
        
        void filter();
        void destroy_filter();
        std::vector<char> rows;
        std::vector<uint32_t> rows_vals_nnz;
        std::vector<uint32_t> rows_vals_all;
        uint32_t nnz_rows;
        
        std::vector<char> cols;
        std::vector<uint32_t> cols_vals_nnz;
        std::vector<uint32_t> cols_vals_all;        
        uint32_t nnz_cols;
};
*/
void CSC_::filter() {
    rows.resize(num_vertices);
    cols.resize(num_vertices);
    for(auto &pair: *pairs)
    {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_vals_all.resize(num_vertices);
    cols_vals_all.resize(num_vertices);
    for(uint32_t i = 0; i < num_vertices; i++)
    {
        if(rows[i] == 1)
        {
            rows_vals_nnz.push_back(i);
            rows_vals_all[i] = rows_vals_nnz.size() - 1;
        }
        if(cols[i] == 1)
        {
            cols_vals_nnz.push_back(i);
            cols_vals_all[i] = cols_vals_nnz.size() - 1;
        }
    }
    nnz_rows = rows_vals_nnz.size();
    nnz_cols = cols_vals_nnz.size();
}

void CSC_::destroy_filter() {
    rows.clear();
    rows.shrink_to_fit();
    cols.clear();
    cols.shrink_to_fit();
    rows_vals_nnz.clear();
    rows_vals_nnz.shrink_to_fit();
    rows_vals_all.clear();
    rows_vals_all.shrink_to_fit();
    cols_vals_nnz.clear();
    cols_vals_nnz.shrink_to_fit();
    cols_vals_all.clear();
    cols_vals_all.shrink_to_fit();
    nnz_rows = 0;
    nnz_cols = 0;
}

void CSC_::destroy_vectors() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void CSC_::run_pagerank() {    
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    csc = new struct Base(num_edges, num_vertices);
    populate();
    filter();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    v.resize(num_rows);
    x.resize(nnz_cols, 1);
    y.resize(nnz_rows);
    (void)spmv();
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] =  y[i];
    //(void)checksum();
    //display();
    delete csc;
    csc = nullptr;
    destroy_filter();
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    csc = new struct Base(num_edges, num_vertices);
    populate();
    filter();
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    destroy_vectors();
    x.resize(nnz_cols);
    y.resize(nnz_rows);
    d.resize(num_vertices);
    d = v;
    std::fill(v.begin(), v.end(), alpha);
    
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    for(uint32_t i = 0; i < num_iterations; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        num_operations += spmv();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "CSC SpMSpV");
    display();
    delete csc;
    csc = nullptr;
}

/*
void CSC_::populate() {
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

void CSC_::walk() {
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t num_cols = csc->ncols_plus_one - 1;
    
    for(uint32_t j = 0; j < num_cols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], j, A[i]);
        }
    }
}
*/
void CSC_::message() {
    for(uint32_t i = 0; i < nnz_cols; i++)
        x[i] = d[cols_vals_nnz[i]] ? (v[cols_vals_nnz[i]]/d[cols_vals_nnz[i]]) : 0;   
}


uint64_t CSC_::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t num_cols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < num_cols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[rows_vals_all[IA[i]]] += (A[i] * x[cols_vals_all[j]]);
            num_operations++;
        }
    }
    return(num_operations);
}


void CSC_::update() {
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void CSC_::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * rows_vals_all.size()) + (sizeof(uint32_t) * rows_vals_nnz.size());
    total_size += (sizeof(uint32_t) * cols_vals_all.size()) + (sizeof(uint32_t) * cols_vals_nnz.size());
}
/*

double CSC_::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < num_rows; i++)
        value += v[i];
    return(value);
}

void CSC_::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void CSC_::stats(double time)
{
    std::cout << "CSC SpMSpV kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time:    " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num SpMV Ops:    " << num_operations <<std::endl;
    std::cout << "Final value:     " << checksum() <<std::endl;
}

*/
#endif
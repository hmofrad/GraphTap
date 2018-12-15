/*
 * csc.hpp: CSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CSC_HPP
#define CSC_HPP

#include <chrono>

#include "pair.hpp" 
#include "io.cpp" 
#include "base_csc.hpp" 

class CSC {
    public:
        CSC() {};
        CSC(const std::string file_path_, const uint32_t num_vertices_, const uint32_t num_iterations_) 
            : file_path(file_path_), num_vertices(num_vertices_), num_iterations(num_iterations_) {}
        ~CSC() {};
        virtual void run_pagerank();
    protected:
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

        virtual void populate();
        virtual void message();        
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        void walk();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};

void CSC::run_pagerank() {
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    csc = new struct Base(num_edges, num_vertices);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    v.resize(num_rows);
    x.resize(num_rows, 1);
    y.resize(num_rows);
    (void)spmv();
    v = y;
    //(void)checksum();
    //display();
    delete csc;
    csc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    csc = new struct Base(num_edges, num_vertices);
    populate();
    total_size += csc->size;
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    d.resize(num_rows);
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
    stats(t, "CSC SpMV");
    display();
    delete csc;
    csc = nullptr;
}

void CSC::populate() {
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

void CSC::walk() {
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

void CSC::message() {
    for(uint32_t i = 0; i < num_rows; i++)
        x[i] = d[i] ? (v[i]/d[i]) : 0;   
}

uint64_t CSC::spmv() {
    uint64_t num_operations = 0;
    uint32_t *A  = (uint32_t *) csc->A;
    uint32_t *IA = (uint32_t *) csc->IA;
    uint32_t *JA = (uint32_t *) csc->JA;
    uint32_t num_cols = csc->ncols_plus_one - 1;
    for(uint32_t j = 0; j < num_cols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            num_operations++;
        }
    }
    return(num_operations);
}

void CSC::update() {
    for(uint32_t i = 0; i < num_rows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void CSC::space() {
    total_size += csc->size;
}

double CSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < num_rows; i++)
        value += v[i];
    return(value);
}

void CSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void CSC::stats(double time, std::string type)
{
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << num_operations <<std::endl;
    std::cout << "Final value    : " << checksum() <<std::endl;
}

#endif
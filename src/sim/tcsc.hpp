/*
 * tcsc.cpp: TCSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TCSC_HPP
#define TCSC_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "base_tcsc.hpp" 

 
class TCSC {  
    public:
        TCSC() {};
        TCSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~TCSC() {};
        virtual void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair> *pairs = nullptr;
        struct Base_tcsc *tcsc = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x;
        std::vector<double> x_r;
        std::vector<double> y;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        
        uint32_t nnzrows_ = 0;
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzrows_regulars_ = 0;
        std::vector<uint32_t> rows_regulars_all;
        std::vector<uint32_t> rows_regulars_nnz;
        uint32_t nnzrows_sources_ = 0;
        std::vector<uint32_t> rows_sources_all;
        std::vector<uint32_t> rows_sources_nnz;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        uint32_t nnzcols_regulars_ = 0;
        std::vector<uint32_t> cols_regulars_all;
        std::vector<uint32_t> cols_regulars_nnz;
        uint32_t nnzcols_sinks_ = 0;
        std::vector<uint32_t> cols_sinks_all;
        std::vector<uint32_t> cols_sinks_nnz;
        void construct_filter();
        void destruct_filter();
        
        void populate();
        void walk();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
        virtual void message();        
        virtual uint64_t spmv();
        virtual void message_regular();        
        virtual uint64_t spmv_regular();
        virtual void update();
        virtual void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};

void TCSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    tcsc = new struct Base_tcsc(nedges, nnzcols_, nnzrows_, nnzcols_regulars_);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors_degree();
    (void)spmv();
    uint32_t *IR = (uint32_t *) tcsc->IR;
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++)
        v[IR[i]] =  y[i];
    //(void)checksum();
    printf("checksum=s%f\n", checksum());
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete tcsc;
    tcsc = nullptr;

    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    tcsc = new struct Base_tcsc(nedges, nnzcols_, nnzrows_, nnzcols_regulars_);
    populate();        
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    construct_vectors_pagerank();
    
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(rows[i] == 1)
            d[i] = v[i];
    }   
    
    
    //d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    if(niters == 1)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        noperations += spmv();
        update();
    }
    else
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        noperations += spmv();
        update();
        
        for(uint32_t i = 1; i < niters; i++)
        {
            std::fill(x.begin(), x.end(), 0);
            std::fill(x_r.begin(), x_r.end(), 0);
            std::fill(y.begin(), y.end(), 0);
            //message_regular();
            //noperations += spmv_regular();
            message_regular();
            noperations += spmv_regular();
            update();
            //printf("%d\n", i);
        }
        
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "TCSC SpMV");
    display();
    delete tcsc;
    tcsc = nullptr;
    destruct_vectors_pagerank();
}

void TCSC::construct_filter() {
    nnzrows_ = 0;
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;        
    }
    rows_all.resize(nvertices);
    nnzcols_regulars_ = 0;
    cols_regulars_all.resize(nvertices);
    nnzcols_sinks_ = 0;
    cols_sinks_all.resize(nvertices);
    
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = nnzrows_;
            nnzrows_++;
        }
        if(cols[i] == 1) {
            nnzcols_++;
            if(rows[i] == 0) {
                cols_sinks_nnz.push_back(i);
                cols_sinks_all[i] = nnzcols_sinks_;
                nnzcols_sinks_++;
            }
            if(rows[i] == 1) {
                cols_regulars_nnz.push_back(i);
                cols_regulars_all[i] = nnzcols_regulars_;
                nnzcols_regulars_++;
            }
        }
    }
    
  //  for(uint32_t i = 0; i < 10; i++)
  //      printf("i=%d all=%d nnz=%d\n", i, cols_regulars_all[i], cols_regulars_nnz[i]);
    
}

void TCSC::destruct_filter() {
    rows.clear();
    rows.shrink_to_fit();
    nnzrows_ = 0;
    rows_all.clear();
    rows_all.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit(); 
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_ = 0;
    cols_sinks_nnz.clear();
    cols_sinks_all.shrink_to_fit();
    nnzcols_sinks_ = 0;
    cols_regulars_nnz.clear();
    cols_regulars_all.shrink_to_fit();
    nnzcols_regulars_ = 0;
}

void TCSC::populate() {
    uint32_t *A  = (uint32_t *) tcsc->A;  // WEIGHT      
    uint32_t *IA = (uint32_t *) tcsc->IA; // ROW_IDX
    uint32_t *JA = (uint32_t *) tcsc->JA; // COL_PTR
    uint32_t *JC = (uint32_t *) tcsc->JC; // COL_IDX
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    auto &p = pairs->front();
    JC[0] = p.col;
    for(auto &pair: *pairs) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_all[pair.row];
        i++;
    }
    uint32_t *IR = (uint32_t *) tcsc->IR; // ROW_PTR
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++)
        IR[i] = rows_nnz[i];
    uint32_t *JA_R = (uint32_t *) tcsc->JA_R; // COL_PTR_REG
    uint32_t *JC_R = (uint32_t *) tcsc->JC_R; // COL_IDX_REG
    uint32_t nnzcols = tcsc->nnzcols;
    uint32_t k = 0;
    uint32_t l = 0;
    int s = 0;
    for(uint32_t j = 0; j < nnzcols; j++) {
        if(JC[j] ==  cols_regulars_nnz[k]) {
            JC_R[k] = JC[j];
            k++;
            JA_R[l] = JA[j];
            JA_R[l + 1] = JA[j + 1];
            l += 2;
        }
      //  else
        //{
          // s += (JA[j+1] - JA[j]);
            //printf("Skipping JC=%d JA=%d JA+1=%d\n", JC[j], JA[j], JA[j+1]);
        //}
    }
    //printf("Skipping %d ops\n", s);
    
    /*
    for(auto &pair: *pairs) {
        if((k - 1) != cols_regulars_all[pair.col]) {
            JC_R[k - 1] = pair.col;
            k++;
            JA_R[k] = JA_R[k - 1];
        }
    }
    */
    /*
    
    //printf("%d %d %d %d s=%d\n", tcsc->nnzcols, tcsc->nnzcols_regulars, nnzcols_, nnzcols_sinks_, s);
    for(uint32_t i = 0; i < 10; i++)
    {
        //if(JC[i] == JC_R[i]
            printf("i=%d JC=%d JA=%d JA+1=%d\n", i, JC[i], JA[i], JA[i+1]);
    }
    printf("\n");
    k = 0;
    for(uint32_t i = 0; i < 10; i = i + 2)
    {
        
        //if(JC[i] == JC_R[i]
            printf("k=%d JC_R=%d JA_R=%d, JA_R+1=%d\n", k, JC_R[k], JA_R[i], JA_R[i+1]);
            k++;
    }
    */
    
    //for(uint32_t j = 0, k = 0; j < nnzcols_regulars_; j++, k = k + 2) {
       // if(j > 544)
         //  printf("j=%d k=%d JC=%d JA_R=%d JA_R+1=%d\n", j, k, JC[j], JA_R[k], JA_R[k+1]);
    //}
    
    
}

void TCSC::walk() {
    uint32_t *A  = (uint32_t *) tcsc->A;
    uint32_t *IA = (uint32_t *) tcsc->IA;
    uint32_t *JA = (uint32_t *) tcsc->JA;
    uint32_t *JC = (uint32_t *) tcsc->JC;
    uint32_t nnzcols = tcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], JC[j], A[i]);
        }
    }
}

void TCSC::construct_vectors_degree() {
    v.resize(nrows);
    x.resize(nnzcols_, 1);
    y.resize(nnzrows_);
}

void TCSC::destruct_vectors_degree() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void TCSC::construct_vectors_pagerank() {
    x.resize(nnzcols_);
    x_r.resize(nnzcols_regulars_);
    y.resize(nnzrows_);
    d.resize(nrows);
}

void TCSC::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    x_r.clear();
    x_r.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void TCSC::message() {
    uint32_t *JC = (uint32_t *) tcsc->JC;
    uint32_t nnzcols = tcsc->nnzcols;
    //printf("nnzcols=%d\n", nnzcols);
    for(uint32_t i = 0; i < nnzcols; i++)
    {
        //if(JC[i] == 268)
        //printf("i=%d jc=%d row=%d d=%f v=%f x=%f\n", i, JC[i], rows[JC[i]], d[JC[i]], v[JC[i]], x[i]);    
        //if(i < 100)
        //{
        x[i] = d[JC[i]] ? (v[JC[i]]/d[JC[i]]) : 0;   
          //  if(JC[i] == 268)
        //printf("2=%d jc=%d row=%d d=%f v=%f x=%f\n", i, JC[i], rows[JC[i]], d[JC[i]], v[JC[i]], x[i]);   
        
    //}
    }
}

uint64_t TCSC::spmv() {
    uint64_t noperations = 0;
    uint32_t *A  = (uint32_t *) tcsc->A;
    uint32_t *IA = (uint32_t *) tcsc->IA;
    uint32_t *JA = (uint32_t *) tcsc->JA;
    uint32_t *JC = (uint32_t *) tcsc->JC;
    uint32_t nnzcols = tcsc->nnzcols;
    //printf("nnzcols=%d\n", nnzcols);
    int n = 0;
    for(uint32_t j = 0; j < nnzcols; j++) {
       // if(j > 544)
         //   printf("spmv j=%d JC=%d JA=%d JA+1=%d nnzcols_=%d\n", j, JC[j], JA[j], JA[j+1], nnzcols_);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            //if(IA[i] == 32){
              //  printf("1.IA[i]=%d, y[IA[i]]=%f JC=%d x=%f\n", IA[i], y[IA[i]], JC[j], x[j]);
                //n++;
            //}
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
            //if(IA[i] == 32){
              //  printf("2.IA[i]=%d, y[IA[i]]=%f JC=%d x=%f\n", IA[i], y[IA[i]], JC[j], x[j]);
                //n++;
            //}
        }
    }
    //printf("SPMV noperations=%d\n", n);
    return(noperations);
}

void TCSC::message_regular() {
    uint32_t *JC_R = (uint32_t *) tcsc->JC_R;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    //printf("nnzcols_regulars=%d, %d/%d\n", nnzcols_regulars, nnzcols_regulars_, nnzcols_);
    for(uint32_t i = 0; i < nnzcols_regulars; i++)
    {
        //if(i < 100)
        //{
        x_r[i] = d[JC_R[i]] ? (v[JC_R[i]]/d[JC_R[i]]) : 0;   
        //x_r[i] = d[JC[i]] ? (v[JC[i]]/d[JC[i]]) : 0;          
        //printf("i=%d jc_r=%d rows=%d d=%f v=%f x=%f\n", i, JC_R[i], rows[JC_R[i]], d[JC_R[i]], v[JC_R[i]], x_r[i]);
        //}
    }
}

uint64_t TCSC::spmv_regular() {
    uint64_t noperations = 0;
    uint32_t *A  = (uint32_t *) tcsc->A;
    uint32_t *IA = (uint32_t *) tcsc->IA;
    uint32_t *JA_R = (uint32_t *) tcsc->JA_R;
    uint32_t *JC_R = (uint32_t *) tcsc->JC_R;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    //printf("nnzcols_regulars=%d\n", nnzcols_regulars);
    int n = 0;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
      //  if(j > 544)
        //    printf("spmv_regular: j=%d k=%d JC_R=%d JA_R=%d JA_R+1=%d nnzcols_regulars=%d\n", j, k, JC_R[j], JA_R[k], JA_R[k+1], nnzcols_regulars);
    
        for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
            //if(IA[i] == 32) {
              //  printf("1.IA[i]=%d, y[IA[i]]=%f JC=%d x=%f\n", IA[i], y[IA[i]], JC_R[j], x_r[j]);
                //n++;
            //}
            y[IA[i]] += (A[i] * x_r[j]);
            noperations++;
            //if(IA[i] == 32) {
              //  printf("2.IA[i]=%d, y[IA[i]]=%f JC=%d x=%f\n", IA[i], y[IA[i]], JC_R[j], x_r[j]);
                //n++;
            //}
        }
    }
    //printf("SPMSPV noperations=%d\n", n);
    return(noperations);
}

void TCSC::update() {
    uint32_t *IR = (uint32_t *) tcsc->IR;
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++){
      //  if(IR[i] == 41)
        //    printf("1.i=%d IR[i]=%d v[IR[i]]=%f\n", i, IR[i], v[IR[i]]);
        v[IR[i]] = alpha + (1.0 - alpha) * y[i];
        //if(IR[i] == 41)
          //  printf("2.i=%d IR[i]=%d v[IR[i]]=%f\n", i, IR[i], v[IR[i]]);
    }
}

void TCSC::space() {
    total_size += tcsc->size;
    //total_size += (sizeof(uint32_t) * rows_all.size()) + (sizeof(uint32_t) * rows_nnz.size());
}

double TCSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void TCSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void TCSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations <<std::endl;
    std::cout << "Final value    : " << checksum() <<std::endl;
}



/*
class TCSC : protected CSC {
    using CSC::CSC;    
    public:
        virtual void run_pagerank();
    protected:
        virtual void populate();
        virtual void message();
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        void message_reg();
        void spmv_reg();

        void filter();
        void destroy_filter();
        void destroy_vectors();
        
        std::vector<char> rows;
        std::vector<uint32_t> rows_vals_nnz;
        std::vector<uint32_t> rows_vals_all;
        uint32_t nnz_rows;
        std::vector<char> cols;
        std::vector<uint32_t> cols_vals_nnz;
        std::vector<uint32_t> cols_vals_all;        
        uint32_t nnz_cols;
        
        std::vector<char> srcs;
        std::vector<char>     regs_cols_vals_all;
        std::vector<uint32_t> regs_cols_vals_nnz;
        std::vector<char>     snks_cols_vals_all;
        std::vector<uint32_t> snks_cols_vals_nnz;
        std::vector<uint32_t> regs_2_nnz;
        std::vector<uint32_t> snks_2_nnz;
        uint32_t nnz_regs_cols;
        uint32_t nnz_snks_cols;
};

void TCSC::run_pagerank() {
    // Degree program
    num_rows = num_vertices;
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs);
    column_sort(pairs);
    filter();
    csc = new struct Base_csc(num_edges, nnz_cols);
    populate();
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
    printf("%f\n", checksum());
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    num_edges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    filter();
    csc = new struct Base_csc(num_edges, nnz_cols);
    populate();
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    destroy_vectors();
    x.resize(nnz_cols);
    y.resize(nnz_rows);
    d.resize(num_rows);
    d = v;
    std::fill(v.begin(), v.end(), alpha);
    
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    
    if(num_iterations == 1) {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        num_operations += spmv();
        update();
    }
    else if(num_iterations == 2) {
        ;
    }
    else if(num_iterations > 2) {
        // Sources
        
        // Regulars
        for(uint32_t i = 1; i < num_iterations - 1; i++) {
            std::fill(x.begin(), x.end(), 0);
            std::fill(y.begin(), y.end(), 0);
            message_reg();
            num_operations += spmv();
            update();
        }
        
        // Sinks
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "TCSC SpMSpV");
    display();
    delete csc;
    csc = nullptr;
}


void TCSC::filter() {
    rows.resize(num_vertices);
    cols.resize(num_vertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_vals_all.resize(num_vertices);
    cols_vals_all.resize(num_vertices);
    //srcs.resize(num_vertices);
    regs_cols_vals_all.resize(num_vertices);
    snks_cols_vals_all.resize(num_vertices);
    for(uint32_t i = 0; i < num_vertices; i++) {
        if(rows[i] == 1) {
            rows_vals_nnz.push_back(i);
            rows_vals_all[i] = rows_vals_nnz.size() - 1;
        }
        
        if(cols[i] == 1) {
            cols_vals_nnz.push_back(i);
            cols_vals_all[i] = cols_vals_nnz.size() - 1;
            if (rows[i] == 1) { // Regular columns
                regs_cols_vals_nnz.push_back(i);
                //regs_cols_vals_all[i] = 1;
                regs_cols_vals_all[i] = regs_cols_vals_nnz.size() - 1;
                regs_2_nnz.push_back(cols_vals_nnz.size() - 1);
            }
            else { // Sink columns
                snks_cols_vals_nnz.push_back(i);
                snks_cols_vals_all[i] = snks_cols_vals_nnz.size() - 1;
                snks_2_nnz.push_back(cols_vals_nnz.size() - 1);
            }
        }
        //if((rows[i] == 1) and (cols[i] == 0))
        //    srcs[i] = 1;
        
        
       // if(cols[i] == 1) {
            // Regular columns
         //   if (rows[i] == 1)
           //     regs_cols_vals_all[i] = 1;
            // Sink columns
           // else // if(rows[i] == 0)
             //   snks_cols_vals_all[i] = 1;
        //}

        
    }
    nnz_rows = rows_vals_nnz.size();
    nnz_cols = cols_vals_nnz.size();
    nnz_regs_cols = regs_cols_vals_nnz.size();
    nnz_snks_cols = snks_cols_vals_nnz.size();
    printf("%d %d %d %d\n", nnz_cols , nnz_regs_cols, nnz_snks_cols, nnz_regs_cols + nnz_snks_cols);
}

void TCSC::destroy_filter() {
    rows.clear();
    rows.shrink_to_fit();
    rows_vals_nnz.clear();
    rows_vals_nnz.shrink_to_fit();
    rows_vals_all.clear();
    rows_vals_all.shrink_to_fit();
    nnz_rows = 0;
    
    cols.clear();
    cols.shrink_to_fit();
    cols_vals_nnz.clear();
    cols_vals_nnz.shrink_to_fit();
    cols_vals_all.clear();
    cols_vals_all.shrink_to_fit();
    nnz_cols = 0;
    
    regs_cols_vals_all.clear();
    regs_cols_vals_all.shrink_to_fit();
    regs_cols_vals_nnz.clear();
    regs_cols_vals_nnz.shrink_to_fit();
    nnz_regs_cols = 0;
    snks_cols_vals_all.clear();
    snks_cols_vals_all.shrink_to_fit();
    snks_cols_vals_nnz.clear();
    snks_cols_vals_nnz.shrink_to_fit();
    nnz_snks_cols = 0;
    regs_2_nnz.clear();
    regs_2_nnz.shrink_to_fit();
    snks_2_nnz.shrink_to_fit();
}

void TCSC::destroy_vectors() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void TCSC::populate() {
    uint32_t *A  = (uint32_t *) csc->A;  // Weight      
    uint32_t *IA = (uint32_t *) csc->IA; // ROW_INDEX
    uint32_t *JA = (uint32_t *) csc->JA; // COL_PTR
    uint32_t ncols = csc->  ncols_plus_one - 1;
    
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto &pair: *pairs) {
        if((j - 1) != cols_vals_all[pair.col]) {
            j++;
            JA[j] = JA[j - 1];
            //snks_vals_nnz
            
            //if(regs_cols_vals_all[pair.col]) {
              //  regs_cols_vals_nnz.push_back(j);
                //printf("SINKs %d %d\n", pair.col, j);
            //}
            //else if(snks_cols_vals_all[pair.col]) {
            //    snks_cols_vals_nnz.push_back(j);
                //printf("SINKs %d %d\n", pair.col, j);
            //}
           // else
             //   exit(0);
            
            
        
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_vals_all[pair.row];
        i++;
        
        //else if(snks[pair.col])
            //printf("Regular%d\n", pair.col);
        
    }
    //for(int i = 0; i < snks_cols_vals_nnz.size(); i++)
        //printf("%d %d\n", i, snks_cols_vals_nnz[i]);
    //printf("nnz_cols=%d nnz_cols=%d regs_cols_vals_nnz=%lu snks_cols_vals_nnz=%lu\n", num_rows, nnz_cols, regs_cols_vals_nnz.size(), snks_cols_vals_nnz.size() );
}

void TCSC::message() {
    for(uint32_t i = 0; i < nnz_cols; i++)
        x[i] = d[cols_vals_nnz[i]] ? (v[cols_vals_nnz[i]]/d[cols_vals_nnz[i]]) : 0;   
}

void TCSC::message_reg() {
    for(uint32_t i = 0; i < nnz_regs_cols; i++)
        x[regs_2_nnz[i]] = d[regs_cols_vals_nnz[i]] ? (v[regs_cols_vals_nnz[i]]/d[regs_cols_vals_nnz[i]]) : 0;   
}

uint64_t TCSC::spmv() {
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




void TCSC::update() {
    for(uint32_t i = 0; i < nnz_rows; i++)
        v[rows_vals_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void TCSC::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * rows_vals_all.size()) + (sizeof(uint32_t) * rows_vals_nnz.size());
    total_size += (sizeof(uint32_t) * cols_vals_all.size()) + (sizeof(uint32_t) * cols_vals_nnz.size());
}
*/
#endif
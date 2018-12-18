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
        std::vector<double> y_r;
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
        std::vector<char> rows_sources;
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
        virtual uint64_t spmv_regular_();
        virtual uint64_t spmv_regular_regular();
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
    
    std::cout << "checksum=" <<  checksum() << std::endl;
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
        
        for(uint32_t i = 1; i < niters - 1; i++)
        {
            //std::fill(x.begin(), x.end(), 0);
            std::fill(x_r.begin(), x_r.end(), 0);
            std::fill(y.begin(), y.end(), 0);
            message_regular();
            noperations += spmv_regular_();
            update();
        }
        //std::fill(x.begin(), x.end(), 0);
        std::fill(x_r.begin(), x_r.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_regular();
        noperations += spmv_regular();
        update();
        
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
    
    nnzrows_regulars_ = 0;
    rows_regulars_all.resize(nvertices);
    
    rows_sources.resize(nvertices);
    nnzrows_sources_ = 0;
    rows_sources_all.resize(nvertices);    
    
    nnzcols_regulars_ = 0;
    cols_regulars_all.resize(nvertices);
    
    nnzcols_sinks_ = 0;
    cols_sinks_all.resize(nvertices);
    
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = nnzrows_;
            nnzrows_++;
            if(cols[i] == 0) {
                rows_sources[i] = 1;
                rows_sources_nnz.push_back(i);
                rows_sources_all[i] = nnzrows_sources_;
                nnzrows_sources_++;
            }
            if(cols[i] == 1) {
                rows_regulars_nnz.push_back(i);
                rows_regulars_all[i] = nnzrows_regulars_;
                nnzrows_regulars_++;   
            }
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
    printf("nnzrows_sources_=%d nnzrows_regulars=%d, nrows=%d\n", nnzrows_sources_, nnzrows_regulars_, nnzrows_);
    printf("nnzcols_sinks=%d nnzcols_regulars=%d, ncols=%d\n", nnzcols_sinks_, nnzcols_regulars_, nnzcols_);
    //for(uint32_t i = 0; i < nrows; i++)
        //printf("rows_sources_all[%d]=%d\n", i, rows_sources_all[i]);
    
    //printf("nrows=%d nnzrows=%d nnzrows_regulars=%d nnzrows_sources=%d\n", nrows, nnzrows_, nnzrows_regulars_, nnzrows_sources_);
    //for(uint32_t i = 0; i < nnzrows_sources_; i++)
      //  printf("rows_sources_nnz[%d]=%d rows_sources_all[rows_sources_nnz[i]]=%d rows_sources[rows_sources_nnz[i]]=%d\n", i, rows_sources_nnz[i], rows_sources_all[rows_sources_nnz[i]], rows_sources[rows_sources_nnz[i]]);
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
    
    rows_sources_nnz.clear();
    rows_sources_nnz.shrink_to_fit();
    rows_sources_all.clear();
    rows_sources_all.shrink_to_fit();
    nnzrows_sources_ = 0;
    rows_sources.clear();
    rows_sources.shrink_to_fit();
            
    rows_regulars_nnz.clear();
    rows_regulars_nnz.shrink_to_fit();
    rows_regulars_all.clear();
    rows_regulars_all.shrink_to_fit();
    nnzrows_regulars_ = 0;
    
    cols_sinks_nnz.clear();
    cols_sinks_nnz.shrink_to_fit();
    cols_sinks_all.clear();
    cols_sinks_all.shrink_to_fit();
    nnzcols_sinks_ = 0;
    
    cols_regulars_nnz.clear();
    cols_regulars_nnz.shrink_to_fit();
    cols_regulars_all.clear();
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
        
        //if(rows_regulars_all[pair.row])
        //    printf("regular row=%d\n", pair.row);
        //if(rows_sources[pair.row])
          //  printf("source row=%d, col=%d IA[i-1]=%d rows_nnz[IA[i-1]]=%d rows_sources_all[IA[i-1]]=%d\n", pair.row, pair.col, IA[i-1], rows_nnz[IA[i-1]], rows_sources_all[rows_nnz[IA[i-1]]]);
        //if(pair.row == 5)
        //    break;

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
    for(uint32_t j = 0; j < nnzcols; j++) {
        if(JC[j] ==  cols_regulars_nnz[k]) {
            JC_R[k] = JC[j];
            k++;
            JA_R[l] = JA[j];
            JA_R[l + 1] = JA[j + 1];
            l += 2;
        }
    }
    int s = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    std::vector<uint32_t> r;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            //printf("i=%d IA[i]=%d IR[IA[i]]=%d rows_sources[IR[IA[i]]]=%d\n", i, IA[i], IR[IA[i]], rows_sources[IR[IA[i]]]);
            //if(rows_sources[IR[IA[i]]] == 1)
                //printf("source\n");
             //   printf(">>IA=%d i=%d j=%d jc=%d nnz_row=%d all_row=%d %d %d\n", IA[i], IR[IA[i]], j, JC[j], rows_sources_nnz[IA[i]], rows_regulars_all[IR[IA[i]]],    rows_regulars_nnz[IA[i]], rows_sources_all[IR[IA[i]]]);
             if(rows_sources[IR[IA[i]]] == 1)
             {
                 m = (JA[j+1] - JA[j]);
                 r.push_back(i);
                 //r.push_back(i - JA[j]);
                // printf("begin=%d end=%d len=%d idx=%d row=%d\n", JA[j], JA[j+1], m, r.back(), IR[IA[r.back()]]);
                 //printf("IR[IA[i]]=%d rows_sources_all[IR[IA[i]]]=%d i=%d imx=%d\n", IR[IA[i]], rows_sources_all[IR[IA[i]]], (i - JA[j]), (JA[j+1] - JA[j]));

                     
                 //std::swap(IR[IA[i]], IR[IA[i+1]]);
            //if(rows_sources[IR[IA[i]]] == 1)
                //break;
             }
        }
        
        if(m > 0) {
            n = r.size();
            s += n;
            //assert(m >= n);
          //  printf("Swapping j=%d m=%d n=%d\n", j, m, n);
           // for(uint32_t p = 0; p < n; p++) {
           //     printf("[%d %d]", r[p], IA[r[p]]);
            //}
           // printf("\n");
            //while(true) {
                if(m > n) {
                    //break;
                //else {
                    for(uint32_t p = 0; p < n; p++) {
                        for(uint32_t q = JA[j+1] - 1; q >= JA[j]; q--) {    
                            //printf("p %d q %d\n", p, q);
                            if(rows_sources[IR[IA[q]]] != 1) {
                                
                              //  printf("swap %d %d\n",r[p], q);
                                //printf("1.swap %d %d %d %d\n", IA[r[p]], IR[IA[r[p]]], IA[q], IR[IA[q]]);
                                
                                
                                std::swap(IA[r[p]], IA[q]);
                                std::swap(A[r[p]], A[q]);
                                //printf("2.swap %d %d %d %d\n", IA[r[p]], IR[IA[r[p]]], IA[q], IR[IA[q]]);
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
        
        //if(j == 2)
          //  break;
        
    }
    
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 

    uint32_t *JA_C = (uint32_t *) tcsc->JA_C; // COL_PTR_REG_ALL
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            if(rows_sources[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }
            
        }
    
        if(m > 0) {
            //printf("j=%d m=%d n=%lu\n", j, m, r.size());
            n = r.size();
            JA_C[l] = JA[j];
            JA_C[l + 1] = JA[j + 1] - n;            
            l += 2; 
            
            //if(m == n)
                //x++;
            
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_C[l] = JA[j];
            JA_C[l + 1] = JA[j + 1];
            l += 2;  
        }
    }
    
    /*
    printf("1111111111111\n");
    
    for(uint32_t j = 0, k = 0; j < nnzcols; j++, k = k + 2) {
        if(j > 560)
        {
        printf("j=%d\n", j);
        for(uint32_t i = JA_C[k]; i < JA_C[k + 1]; i++) {
            //if(rows_sources[IR[IA[i]]])
                printf("    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
        }
        }
    }
    
    printf("22222222222222\n");
    
    for(uint32_t j = 0; j < nnzcols; j++) {
        if(j > 560)
        {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            //if(rows_sources[IR[IA[i]]])
                printf("    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
        }
        }
    }
    */
    
       
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    l = 0;
    //int x = 0;
    uint32_t *JA_RC = (uint32_t *) tcsc->JA_RC; // COL_PTR_REG_REG
    //for(uint32_t j = 0; j < nnzcols; j++) {
      //  for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
  uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
            if(rows_sources[IR[IA[i]]] == 1) {
                m = (JA_R[k+1] - JA_R[k]);
                r.push_back(i);
            }
            
            //if(j == 549)
                
           
            /*
        if(JC[j] ==  cols_regulars_nnz[k]) {
            JC_R[k] = JC[j];
            k++;
            JA_R[l] = JA[j];
            JA_R[l + 1] = JA[j + 1];
            l += 2;
            */
        }
        if(m > 0) {
            //printf("j=%d m=%d n=%lu\n", j, m, r.size());
            n = r.size();
            JA_RC[l] = JA_R[k];
            JA_RC[l + 1] = JA_R[k + 1] - n;            
            l += 2; 
            
            //if(m == n)
                //x++;
            
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_RC[l] = JA_R[k];
            JA_RC[l + 1] = JA_R[k + 1];
            l += 2;  
        }
    }
    

    
    
    //printf("%d %d %d %d %d skip=%d\n", l, l / 2, nnzcols, nnzcols_regulars_, nnzcols_sinks_, x);
    
    
    //printf("l/2=%d %d\n", l/2, nnzcols_regulars);
    /*
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        //for(uint32_t j = 0; j < nnzcols; j++) {
        //j = 570;    
        printf("j=%d\n", j);
        for(uint32_t i = JA_RC[k]; i < JA_RC[k + 1]; i++) {
    //    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            if(rows_sources[IR[IA[i]]])
                printf("    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
        }
    }
    */
/*    
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        //printf("j=%d\n", j);
        //bool tf = false;
        
        if(j == 549) {
            printf("j=%d\n", j);
        for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
        //for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
            //if(rows_sources[IR[IA[i]]]) {
            //    tf = true;
          //  }
        //}
        //if(tf) {
            
            
            //for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
                printf("1.    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
            //}
            //printf("\n");
            }
            //for(uint32_t i = JA_RC[k]; i < JA_RC[k + 1]; i++) {
            //    printf("    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
            //}
        }
    }     
*/    

    /*
    printf("\n");
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        
        if(j == 549) {
            printf("2.j=%d\n", j);
        for(uint32_t i = JA_RC[k]; i < JA_RC[k + 1]; i++) {
            
            //
            printf("2.    i=%d, %d, j=%d, value=%d %d\n", i, IA[i], JC[j], A[i], rows_sources[IR[IA[i]]]);
            }
        }
    }
    */
    //prin
    
    
    
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
    y_r.resize(nnzrows_regulars_);
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
    y_r.clear();
    y_r.shrink_to_fit();
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
    //printf("1 SPMV noperations=%lu\n", noperations);
    return(noperations);
}

uint64_t TCSC::spmv_regular_regular() {
    uint64_t noperations = 0;
    uint32_t *A  = (uint32_t *) tcsc->A;
    uint32_t *IA = (uint32_t *) tcsc->IA;
    uint32_t *JA_C = (uint32_t *) tcsc->JA_C;
    uint32_t *JC = (uint32_t *) tcsc->JC;
    uint32_t nnzcols = tcsc->nnzcols;
    //printf("nnzcols=%d\n", nnzcols);
    //int n = 0;
    for(uint32_t j = 0, k = 0; j < nnzcols; j++, k = k + 2) {
        for(uint32_t i = JA_C[k]; i < JA_C[k + 1]; i++) {
    
    
    //for(uint32_t j = 0; j < nnzcols; j++) {
       // if(j > 544)
         //   printf("spmv j=%d JC=%d JA=%d JA+1=%d nnzcols_=%d\n", j, JC[j], JA[j], JA[j+1], nnzcols_);
      //  for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
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
    //printf("1 SPMV noperations=%lu\n", noperations);
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
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
            y[IA[i]] += (A[i] * x_r[j]);
            noperations++;
        }
    }
    //printf("2 SPMV noperations=%lu\n", noperations);
    return(noperations);
}

uint64_t TCSC::spmv_regular_() {
    uint64_t noperations = 0;
    uint32_t *A  = (uint32_t *) tcsc->A;
    uint32_t *IA = (uint32_t *) tcsc->IA;
    uint32_t *IR = (uint32_t *) tcsc->IR;
    uint32_t *JA_RC = (uint32_t *) tcsc->JA_RC;
    uint32_t *JA_R = (uint32_t *) tcsc->JA_R;
    uint32_t *JC_R = (uint32_t *) tcsc->JC_R;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_RC[k]; i < JA_RC[k + 1]; i++) {
      //for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        //for(uint32_t i = JA_R[k]; i < JA_R[k + 1]; i++) {
            //if(rows_sources[IR[IA[i]]] == 0)
                //printf("skip this one\n");
            y[IA[i]] += (A[i] * x_r[j]);
            noperations++;
            //
              //  break;
        }
    }
    
    
    
    printf("3_ SPMV noperations=%lu\n", noperations);
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

#endif
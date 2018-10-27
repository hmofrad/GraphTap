/*
 * bin2bin1.cpp: binary to binary without vertex 0 id converter
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 * Standalone compile commnad:
 * g++ -o bin2bin1 bin2bin1.cpp  -std=c++14
 */
 
#include <iostream>
#include <cstdlib> 
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
 
int main(int argc, char **argv) {
  	
  	std::cout << "Usage: " << argv[0] << " <filepath_in> <filepath_out> "  << std::endl;
	std::cout << "Blindly removes vertex id 0 from a binary graph by increasing vertex ids by 1" << std::endl;
	
	if (argc < 3) {
    	exit(1);
	}

	std::string filepath_in = argv[1];
  	std::string filepath_out = argv[2];
    
    // Open graph file.
    std::ifstream fin(filepath_in.c_str(), std::ios_base::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Unable to open input file");
        exit(1); 
    }
    std::ofstream fout(filepath_out.c_str(), std::ios_base::binary);
    
    // Obtain filesize
    uint64_t filesize, offset = 0;
    uint64_t num_edges = 0;
    uint32_t num_vertices = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    uint32_t i, j;
    while (offset < filesize)
    {
        fin.read(reinterpret_cast<char*>(&i), sizeof(uint32_t));
        i++;
        offset += sizeof(uint32_t);
        fin.read(reinterpret_cast<char*>(&j), sizeof(uint32_t));
        j++;
        offset += sizeof(uint32_t);
        //fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        //printf("%d %d\n", i, j);
        
        fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
        fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
        
        num_edges++;
        num_vertices = (num_vertices < i) ? i : num_vertices;
        num_vertices = (num_vertices < j) ? j : num_vertices;
    }
    fout.close();
    fin.close();
    assert(offset == filesize);
    printf("\n%s: Read %d vertices (excluding zero) and %lu edges\n", filepath_in.c_str(), num_vertices, num_edges);
}

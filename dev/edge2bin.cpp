/*
 * Edge list to binary converter
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include <iostream>
#include <cstdlib> 
#include <fstream>
#include <string>
#include <sstream>

int main(int argc, char **argv) {
  	
  	std::cout << "Usage: " << argv[0] << " <filepath_in> <filepath_out> "  << std::endl;
	std::cout << "Converts graph from edge list pairs (char i, char j) "
	             "to bianry pairs (uint32_t i, uint32_t j)" << std::endl;
	
	if (argc < 3) {
    	exit(1);
	}

	std::string filepath_in = argv[1];
  	std::string filepath_out = argv[2];

	std::ifstream  fin(filepath_in.c_str(),   std::ios_base::in);
	std::ofstream fout(filepath_out.c_str(), std::ios_base::binary);

	// Consume comments identified by '#' and '%' characters
	// Linebased read/write of pairs
	// Read pairs of (char i, char j) from filepath_in
	// Write bianry pairs of (uint32_t i, uint32_t j) to filepath_out
	uint32_t num_comments = 0;
	uint32_t num_edges = 0;
	bool is_comment = false;
	std::string line;
	uint32_t i, j;
	std::istringstream iss;
	while (std::getline(fin, line)) {
		if ((line[0] == '#') || (line[0] == '%')) {
			
			if(!is_comment) {
				is_comment = true;
				std::cout << "########################################" << std::endl;
			}
			std::cout << line << std::endl; //NoOP
			num_comments++;
		} else {
			iss.clear();
			iss.str(line);
			iss >> i >> j;
			//std::cout << "i=" << i << "j=" << j << std::endl;
			fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
    		fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
    		num_edges++;
		}
	}

	fin.close();
	fout.close();

	std::cout << "########################################" << std::endl;
	std::cout << "Read/write stats:" << std::endl;
	std::cout << num_comments << " line comments" << std::endl;
	std::cout << num_edges << " edges " << std::endl;
	std::cout << (num_comments + num_edges) << " number of lines" << std::endl;
	std::cout << "Verify using \"wc -l " << filepath_in << "\"" << std::endl;

	return(0);
}
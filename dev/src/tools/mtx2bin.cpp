#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <ctime>

using namespace std;

int main(int argc, char* argv[])
{
  cout
      << "Convert graph from Matrix Market (with optional header: n m nnz) "
      << "pairs (i j) or triples (i j w) "
      << "to binary (with optional header: uint:n uint:m ulong:nnz) "
      << "pairs (uint:i uint:j) or triples (uint:i uint:j uint:w). "
      << endl;

  cout << "Usage: " << argv[0] << " <filepath_in> <filepath_out> "
       << endl << "\t [-hi[o]]    read in first non-comment line as header [and write [o]ut]"
       << endl << "\t [-wi]       read in edge weights (must be int or double) "
       << endl << "\t [-wo{i|d}]  write out edge weights ({i}nt/{d}ouble) "
       << endl << "\t             (rand [1,128] if none given)"
       << endl;

  if (argc < 3)
    return 1;

  string fpath_in = argv[1];
  string fpath_out = argv[2];

  bool header_in      = false;
  bool header_out     = false;
  bool weights_in     = false;
  bool weights_out    = false;
  bool weights_int    = false;
  bool weights_rand   = false;
  bool weights_double = false;


  for (auto i = 3; i < argc; i++)
  {
    if (string(argv[i]) == "-hi" or string(argv[i]) == "-hio")
      header_in = true;
    if (string(argv[i]) == "-hio")
      header_out = true;
    if (string(argv[i]) == "-wi")
      weights_in = true;
    if (string(argv[i]) == "-wo")
      weights_rand = true;
    if (string(argv[i]) == "-woi")
      weights_int = true;
    if (string(argv[i]) == "-wod")
      weights_double = true;
  }
  
  // When there's input weights, they can be written in int and double 
  // When there's no input weight, only random weights will be written
  if(weights_in)
    weights_out = weights_int or weights_double;  
  else
    weights_out = weights_rand;

  ifstream fin(fpath_in.c_str());
  ofstream fout(fpath_out.c_str(), ios::binary);

  // Skip comments
  std::string line;
  int position; // Fallback position
  do {
    position = fin.tellg();
    std::getline(fin, line);
  } while ((line[0] == '#') || (line[0] == '%'));
  fin.seekg(position, ios_base::beg);

  // Read/write header
  uint32_t n, m;
  uint64_t nnz = 0;
  std::istringstream iss;
  if (header_in)
  {
    std::getline(fin, line);
    iss.str(line);
    iss >> n >> m >> nnz;
    cout << "Header: " << n << " " << m << " " << nnz << endl;

    // Write header
    if (header_out)
    {
      fout.write(reinterpret_cast<const char*>(&n),   sizeof(uint32_t));
      fout.write(reinterpret_cast<const char*>(&m),   sizeof(uint32_t));
      fout.write(reinterpret_cast<const char*>(&nnz), sizeof(uint64_t));
    }
  }

  // Read/write pairs/triples
  srand(time(NULL));
  uint32_t i, j;
  double wd = 0;
  uint32_t wi = 0;
  uint64_t nedges = 0;
  while (std::getline(fin, line))
  {
    iss.clear();
    iss.str(line);
    iss >> i >> j;  
    fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));

    if (weights_in)
      iss >> wd;  

    if (weights_out)
    {
      if (weights_int)
      {
        wi = (uint32_t) wd;
        fout.write(reinterpret_cast<const char*>(&wi), sizeof(uint32_t));
      }
      else
      {
        if (weights_rand)
          wd = 1 + (rand() % 128);  
        fout.write(reinterpret_cast<const char*>(&wd), sizeof(double));        
      }
    }

    nedges++;
    // Solely uncoment this for debug purpose 
     std::cout << "(i,j,w)=" << "(" << i << "," << j << "," << wd << ")" << std::endl;
  }

  fin.close();
  fout.close();

  cout << "\n" << argv[0] << ": " << nedges << " edges are written to "<<  argv[2] << endl;

  return(0);
}

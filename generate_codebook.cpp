#include <string>
#include <iostream>

#include "visual_codebook.h"
using namespace std;

#define MIN_CODEBOOK_ARGS 4
#define MAX_CODEBOOK_ARGS 5

int main(int argc, char** argv)
{
  if (argc < MIN_CODEBOOK_ARGS || argc > MAX_CODEBOOK_ARGS)
  {
    cerr << "usage: " << argv[0]
         << " path/to/images/ num_images num_centers [path/to/codebook]"
         << endl;
    return -1;
  }

  // Get command line arguments
  string img_path(argv[1]);
  int num_images = atoi(argv[2]);
  int num_centers = atoi(argv[3]);
  string cb_path;

  if(argc == MAX_CODEBOOK_ARGS)
  {
    cb_path = string(argv[4]);
  }
  else
  {
    cb_path = img_path;
    cb_path.append("default.codebook");
  }

  // Cluster the codewords
  VisualCodebook vc(num_centers, true);
  vc.constructCodebook(img_path, num_images);

  // Write cluster centers to file
  vc.saveCodebook(cb_path);

  return 0;
}

#include <string>
#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "visual_codebook.h"
using namespace std;

int main(int argc, char** argv)
{
  // Cluster the codewords
  VisualCodebook vc(1000);
  vc.constructCodebook("~/src/fab-data/city-centre/Images/", 2474);
  
  // Write cluster centers to file

  return 0;
}

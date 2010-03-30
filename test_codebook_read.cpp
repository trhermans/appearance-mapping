#include <string>
#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "visual_codebook.h"
using namespace std;


int main(int argc, char** argv)
{

  // Get command line arguments
  string cb_path(argv[1]);
  string img_path(argv[2]);

  VisualCodebook vc;
  vc.loadCodebook(cb_path);

  cout << "Number of centers: " << vc.numCenters()  << endl;
  string save_path = "./after_output";
  vc.saveCodebook(save_path);

  IplImage *img = cvLoadImage(img_path.c_str(), CV_8UC1);
  vector<int> feat;
  feat = vc.getCodewords(*img);

  cout << "Codewords at points: " << endl;
  int count = 0;
  for (unsigned int i = 0; i < feat.size(); ++i)
  {
    if (feat[i] > 0)
    {
      cout << "\t" << i << " has " << feat[i] << endl;
      ++count;
    }
  }
  cout << "Total words present: " << count << endl;
  return 0;
}

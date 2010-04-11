#include <string>
#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "fabmap/Timer.h"

#include "vis_codebook/visual_codebook.h"
using namespace std;

int main(int argc, char** argv)
{

  // Get command line arguments
  string cb_path(argv[1]);
  string img_path(argv[2]);
  int img_count = atoi(argv[3]);

  VisualCodebook vc;
  vc.loadCodebook(cb_path);
  int k = vc.numCenters();
  cout << "Number of centers: " << k  << endl;

  // Build distribution
  vector<long> dist(k, 0);
  vector<vector<int> > training_data;
  int padding_size = 4;
  TheTimer.start();
  for (int i = 1; i <= img_count; ++i)
  {
    stringstream image_name;
    // Create a zero padded list
    stringstream img_num;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = padding_size - img_num_str.size();
    img_num_str.insert(0, num_zeros, '0');

    // Build the full path to the image
    image_name << img_path << img_num_str << ".jpg";

    IplImage *img;
    img = cvLoadImage(image_name.str().c_str(), CV_8UC1);
    cout << image_name.str() << endl;
    vector<int> feat;
    feat = vc.getCodewords(*img);
	training_data.push_back(feat);

    for (unsigned int j = 0; j < feat.size(); ++j)
    {
      if (feat[j] > 0)
        dist[j] += feat[j];
    }
  }
  cout << "Histogram extraction needed " << TheTimer.getRuntime() << "s." << endl;

  cout << "Writing to disk" << endl;
  string save_path = "./word_dist.txt";
  fstream dist_file;
  dist_file.open(save_path.c_str(), ios::out);
  for (unsigned int i = 0; i < dist.size(); ++i)
  {
    dist_file << dist[i] << " ";
  }
  dist_file.close();

  save_path = "./_training_data.txt";
  dist_file.open(save_path.c_str(), ios::out);
  dist_file << training_data.size() << " " << training_data[0].size() << endl;;
  for (unsigned int i = 0; i < training_data.size(); ++i)
  {
    for (unsigned int j = 0; j < training_data[i].size(); ++j)
    {
      dist_file << training_data[i][j] << " ";
    }
	dist_file << endl;
  }
  dist_file.close();
  return 0;
}

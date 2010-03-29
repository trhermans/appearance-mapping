#include "visual_codebook.h"

#include <iostream>
#include <fstream>
using namespace std;

VisualCodebook::VisualCodebook(int k): k_(k)
{
}

void VisualCodebook::constructCodebook(string img_path, int img_count)
{
  int num_size = 4;
  for(int i = 1; i <= img_count; ++i)
  {
    IplImage* cur;
    stringstream image_name;

    // Create a zero padded list
    stringstream img_num;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = num_size - img_num_str.size();
    img_num_str.insert(0, num_zeros, '0');

    // Build the full path to the image
    image_name << img_path << img_num_str << ".jpg";
    cout << image_name.str() << endl;

    // Read the image into memory
    cur = cvLoadImage(image_name.str().c_str(), CV_8UC1);

    // Find the descriptors for the image and store them into memory
    CvSeq* keys = 0;
    CvSeq* descriptors = 0;
    CvMemStorage* mem  = cvCreateMemStorage(0);
    CvSURFParams params = cvSURFParams(500,1);

    cvExtractSURF(cur, NULL, &keys, &descriptors, mem, params);
  }

  // Cluster the codewords
}

void VisualCodebook::saveCodebook(string path)
{
  fstream output_file;
  output_file.open(path.c_str(), ios::out);
  for(unsigned int i = 0; i < centers_.size(); i++)
  {
    for(unsigned int j = 0; j < centers_[j].size(); j++)
    {
      output_file << centers_[j][i] << " ";
    }
    output_file << endl;
  }
}

void VisualCodebook::loadCodebook(string path)
{
}

vector<int> VisualCodebook::getCodewords(IplImage& img)
{
  vector<int> codewords;
  return codewords;
}


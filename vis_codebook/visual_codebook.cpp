#include "visual_codebook.h"

#include <iostream>
#include <fstream>
using namespace std;

VisualCodebook::VisualCodebook(int k): k_(k)
{
}

void VisualCodebook::constructCodebook(string img_path, int img_count)
{
  int padding_size = 4;
  vector<float*> raw_descriptors;
  for(int i = 1; i <= img_count; ++i)
  {
    IplImage* cur;
    stringstream image_name;

    // Create a zero padded list
    stringstream img_num;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = padding_size - img_num_str.size();
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

    cvExtractSURF(cur, 0, &keys, &descriptors, mem, params);

    // Add the descriptors to the current set
    CvSeqReader reader;
    cvStartReadSeq(descriptors, &reader, 0);
    for(int j = 0; j < descriptors->total; ++j)
    {
      const float * desc = (const float*)(reader.ptr);
      float* desc_array;
      desc_array = new float[128];
      for(int k = 0; k < 128; ++k)
      {
         desc_array[k] = desc[k];
      }
      raw_descriptors.push_back(desc_array);
    }
  }

  // Cluster the codewords

  // Convert raw descriptors
  CvMat* descriptors = cvCreateMat(raw_descriptors.size(), 128, CV_32FC1);
  for(int i = 0; i < raw_descriptors.size(); ++i)
  {
    for(int j = 0; j < 128; ++j)
    {
      descriptors->data.fl[i*128 + j] = raw_descriptors[i][j];
    }
  }

  CvMat* clusters = cvCreateMat(raw_descriptors.size(), 1, CV_32SC1);
  cv_centers_ = cvCreateMat(k_, 128, CV_32FC1);

  cout << "Clustering codewords" << endl;
  cvKMeans2(descriptors, k_, clusters,
            cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0 ),
            0, 0, 0, cv_centers_);

  // Record centers in our format
  for(int i = 0; i < k_; ++i)
  {
    vector<float> center;
    for (int j = 0; j < 128; ++j)
    {
      center.push_back(cv_centers_->data.fl[i*128 + j]);
    }
    centers_.push_back(center);
  }

  // Cleanup
  cvReleaseMat(&descriptors);
  cvReleaseMat(&clusters);
  for (int i = 0; i < raw_descriptors.size(); ++i)
  {
    delete[] raw_descriptors[i];
  }
}

void VisualCodebook::saveCodebook(string path)
{
  fstream output_file;
  output_file.open(path.c_str(), ios::out);
  for(unsigned int i = 0; i < centers_.size(); i++)
  {
    for(unsigned int j = 0; j < centers_[j].size(); j++)
    {
      output_file << centers_[i][j] << " ";
    }
    output_file << endl;
  }
}

void VisualCodebook::loadCodebook(string path)
{
  fstream input_file;
  input_file.open(path.c_str(), ios::in);
  centers_.clear();
  while( !input_file.eof() )
  {
    vector<float> center;
    for (int i = 0; i < 128 && !input_file.eof() ; ++i)
    {
      float f;
      input_file >> f;
      center.push_back(f);
    }
    if ( !input_file.eof() )
    {
      centers_.push_back(center);
    }
  }

  k_ = centers_.size();

  // Build opencv version of centers data
}

vector<int> VisualCodebook::getCodewords(IplImage& img)
{
  vector<int> codewords(k_, 0);

  if (centers_.size() < 1)
  {
    cerr << "Have not loaded a codebook, can not get codewords" << endl;
    return codewords;
  }

  // Get SURF descriptors from the image

  // Match the descriptors to their centers

  return codewords;
}


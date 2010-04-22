#include "visual_codebook.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include "../fabmap/Timer.h"
using namespace std;

VisualCodebook::VisualCodebook(int k, bool use128_surf):
    k_(k), desc_size_(128), hessian_thresh_(500), kmeans_iter_(5),
    use128_surf_(use128_surf)
{
  centers_.clear();
  if (!use128_surf_)
  {
    desc_size_ = 64;
  }
}

/**
 * Function to build a visual codebook from a set of images.
 * Currently produces a k-means clustered codebook with k_ centers, built from
 * SURF descriptors.
 *
 * @param img_path Directory storing all the images
 * @param img_count Number of images in img_path directory
 * @param mode Just for memory shortage on Richard's computer - separates the process into two pieces. mode=0 -> normal behavior, mode=1 -> normal behavior + all descriptors are saved to disk, mode=2 -> descriptors are loaded from disk and clustered
 */
void VisualCodebook::constructCodebook(string img_path, int img_count, int mode)
{
  vector<float*> raw_descriptors;
  if (mode<2)
  {
    int padding_size = 4;
    TheTimer.start();
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
      CvSURFParams params = cvSURFParams(hessian_thresh_, use128_surf_);

      cvExtractSURF(cur, 0, &keys, &descriptors, mem, params);

      // Add the descriptors to the current set
      CvSeqReader reader;
      cvStartReadSeq(descriptors, &reader, 0);
      for(int j = 0; j < descriptors->total; ++j)
      {
        const float * desc = (const float*)(reader.ptr);
        CV_NEXT_SEQ_ELEM(reader.seq->elem_size, reader);
        float* desc_array;
        desc_array = new float[desc_size_];
        for(int k = 0; k < desc_size_; ++k)
        {
          desc_array[k] = desc[k];
        }
        raw_descriptors.push_back(desc_array);
      }
      cvReleaseMemStorage(&mem);
    }
    cout << "SURF feature extraction needed "
         << TheTimer.getRuntime() << "s." << endl;
  }

  if (mode==1)
  {
    // save descriptors in file to avoid the SURF step in further versions
    fstream output_file;
    output_file.open("_surfDescriptors.txt", ios::out);
    output_file << raw_descriptors.size() << "\t" << desc_size_ << endl;
    for(unsigned int i = 0; i < raw_descriptors.size(); i++)
    {
      for(int j = 0; j < desc_size_; j++)
      {
        output_file << raw_descriptors[i][j];
        if (j < desc_size_ -1)
          output_file << "\t";
      }
      output_file << endl;
    }
    output_file.close();
  }

  int num_desc=0;
  CvMat* descriptors = 0;
  if (mode==2)
  {
    // load descriptors from file to avoid the SURF step in further versions
    cout << "Loading descriptor file..." << endl;
    fstream input_file;
    input_file.open("_surfDescriptors_CityCentre.txt", ios::in);
    int m=0;
    int n=0;
    input_file >> m >> n;
    num_desc = m;
    descriptors = cvCreateMat(num_desc, desc_size_, CV_32FC1);
    for(int i = 0; i < m; i++)
    {
      for(int j = 0; j < n; j++)
      {
        float temp = 0.0;
        input_file >> temp;
        cvSetReal2D(descriptors, i, j, temp);
      }
    }
    input_file.close();
    cout << "Descriptor file loaded." << endl;
  }

  if (mode<2)
  {
    // Convert raw descriptors
    num_desc = raw_descriptors.size();
    descriptors = cvCreateMat(num_desc, desc_size_, CV_32FC1);
    for(int i = 0; i < num_desc; ++i)
    {
      for(int j = 0; j < desc_size_; ++j)
      {
        descriptors->data.fl[i*desc_size_ + j] = raw_descriptors[i][j];
      }
      delete[] raw_descriptors[i];
    }
    raw_descriptors.clear();
  }


  CvMat* clusters = cvCreateMat(num_desc, 1, CV_32SC1);
  CvMat* cv_centers = cvCreateMat(k_, desc_size_, CV_32FC1);

  // Cluster the codewords
  cout << "Clustering codewords" << endl;
  TheTimer.start();
  cvKMeans2(descriptors, k_, clusters,
            cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                           kmeans_iter_, 1.0),
            1, 0, 0, cv_centers);
  cout << "Clustering needed " << TheTimer.getRuntime() << "s." << endl;
  cout << "Re-recording centers as vector" << endl;
  // Record centers in our format
  for(int i = 0; i < k_; ++i)
  {
    vector<float> center;
    for (int j = 0; j < desc_size_; ++j)
    {
      center.push_back(cv_centers->data.fl[i*desc_size_ + j]);
    }
    centers_.push_back(center);
  }

  // Cleanup
  cvReleaseMat(&descriptors);
  cvReleaseMat(&clusters);
  cvReleaseMat(&cv_centers);
}

/**
 * Generate a (binary) feature vector using the current codebook for the given
 * image.
 *
 * @param img image to find a feature vector for
 *
 * @return The (binary) feature vector of codewords if a codebook is present,
 *         zero vector otherwise.
 */
vector<int> VisualCodebook::getCodewords(IplImage& img)
{
  vector<int> codewords(k_, 0);

  if (centers_.size() < 1)
  {
    cerr << "Have not loaded a codebook, can not get codewords" << endl;
    return codewords;
  }

  // Get SURF descriptors from the image
  CvSeq* keys = 0;
  CvSeq* descriptors = 0;
  CvMemStorage* mem  = cvCreateMemStorage(0);
  CvSURFParams params = cvSURFParams(hessian_thresh_, use128_surf_);
  cvExtractSURF(&img, 0, &keys, &descriptors, mem, params);

  // Match the descriptors to their centers and record match
  CvSeqReader reader;
  cvStartReadSeq(descriptors, &reader, 0);
  for(int i = 0; i < descriptors->total; ++i)
  {
    const float * desc = (const float*)(reader.ptr);
    CV_NEXT_SEQ_ELEM(reader.seq->elem_size, reader);
    int c = nearestCenter(desc);
    if (c >= 0)
    {
      codewords[c] = 1;
    }
    else
    {
      cerr << "Found no nearest neighbor!" << endl;
    }
  }
  cvReleaseMemStorage(&mem);
  return codewords;
}

/**
 * Find the closest codeword for a given descriptor.
 *
 * @param desc The query descriptor.
 *
 * @return the index of the nearest center
 */
const int VisualCodebook::nearestCenter(const float * desc)
{
  double best_score = DBL_MAX;
  int best_index = -1;

  for(unsigned int i = 0; i < centers_.size(); ++i)
  {
    double score = 0;
    for(int j = 0; j < desc_size_; j += 4)
    {
      double diff0 = centers_[i][j] - desc[j];
      double diff1 = centers_[i][j+1] - desc[j+1];
      double diff2 = centers_[i][j+2] - desc[j+2];
      double diff3 = centers_[i][j+3] - desc[j+3];
      score += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
      if(score > best_score)
        break;
    }
    if (score < best_score)
    {
      best_score = score;
      best_index = i;
    }
  }

  return best_index;
}

//
// IO Methods
//

/**
 * Writes a plain text copy of the codebook to disk
 *
 * @param path file name to save the codebook as.
 */
void VisualCodebook::saveCodebook(string path)
{
  fstream output_file;
  output_file.open(path.c_str(), ios::out);
  for(unsigned int i = 0; i < centers_.size(); i++)
  {
    for(int j = 0; j < desc_size_; j++)
    {
      output_file << centers_[i][j];
      if (j < desc_size_ -1)
        output_file << " ";
    }
    output_file << endl;
  }
  output_file.close();
}

/**
 * Reads a plain text copy of the codebook from disk.
 *
 * @param path file name of the codebook on disk.
 */
void VisualCodebook::loadCodebook(string path)
{
  fstream input_file;
  input_file.open(path.c_str(), ios::in);
  if(!input_file.is_open())
  {
    std::cout << "VisualCodebook::loadCodebook: Error: could not open "
              << path.c_str() << "\n";
    return;
  }
  centers_.clear();
  while( !input_file.eof() )
  {
    vector<float> center;
    for (int i = 0; i < desc_size_ && !input_file.eof() ; ++i)
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
  input_file.close();
  // Set the number of clusters
  k_ = centers_.size();
}

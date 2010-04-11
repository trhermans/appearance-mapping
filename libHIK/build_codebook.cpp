#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include "Features.h"
#include "Codebook.h"
#include "util.h"
#include "UseSVM.h"

#include <omp.h>
#include <cv.h>

static int fsize = 0;
static const int resize_width = 512;
static int window_size = 16;
static std::vector<const char*> train_names;
static const int upperBound = 129;
static const int kmeans_stepsize = 24;
static const int splitlevel = 0;
static const int L1_norm = 128;
static bool normalize = false;
using namespace std;

void usage()
{
  cout<<"Usage: build_codebook path/to/training_images/ "
           << "num_training_images path/to/testing_images/ num_testing_images "
           << "K descriptor codebook train_output_path test_output_path use_harris" << endl;
  cout<<" descriptor: CENTRIST | SIFT"<<endl;
  cout<<" codebook: HIK | kmeans | kmedian"<<endl;
  cout<<" use_harris: 0 | 1"<<endl;
}

CodeBook::CODEBOOK_TYPE FindCodeBook(const char* codebook)
{
  string name = codebook;
  if(name=="HIK")
    return CodeBook::CODEBOOK_HIK;
  else if(name=="kmeans")
    return CodeBook::CODEBOOK_KMEANS;
  else if(name=="kmedian")
    return CodeBook::CODEBOOK_KMEDIAN;
  else
  {
    cout<<"Codebook name is not correct."<<endl;
    usage();
    exit(-1);
  }
}

CodeBook::DESCRIPTOR_TYPE FindDescriptor(const char* descriptor)
{
  string name = descriptor;
  if(name=="CENTRIST")
    return CodeBook::DESCRIPTOR_CENTRIST;
  else if(name=="SIFT")
    return CodeBook::DESCRIPTOR_SIFT;
  else
  {
    cout<<"Visual descriptor type is not correct."<<endl;
    usage();
    exit(-1);
  }
}

CodeBook* CodeBookEngine(const CodeBook::CODEBOOK_TYPE codebookType,
                         const CodeBook::DESCRIPTOR_TYPE featureType,
                         int K,
                         const bool useSobel,
                         const bool oneclassSVM,
                         int& fsize)
{
  CodeBook* codebook = NULL;
  switch(codebookType) {
    // NOTE: Use L1_norm
    case CodeBook::CODEBOOK_KMEANS:
      codebook = new LinearCodes(featureType,useSobel,resize_width,
                                 window_size,L1_norm,false);
      break;
    case CodeBook::CODEBOOK_KMEDIAN:
      codebook = new LinearCodes(featureType,useSobel,resize_width,
                                 window_size,L1_norm,true);
      break;
    case CodeBook::CODEBOOK_HIK:
      codebook = new HistogramCodes(featureType, useSobel, resize_width,
                                    window_size,L1_norm,upperBound);
      break;
    case CodeBook::CODEBOOK_LOCAL:
      codebook = new LocalCodes(useSobel,resize_width);
      break;
    default:
      cout<<"Codebook type is wrong."<<endl;
      break;
  }
  if(codebookType==CodeBook::CODEBOOK_KMEANS ||
     codebookType==CodeBook::CODEBOOK_KMEDIAN ||
     codebookType==CodeBook::CODEBOOK_HIK) {
    cout<<"Generating data for kernel k-means clustering."<<endl;
    codebook->GenerateClusterData(train_names, kmeans_stepsize);
    cout<<"Now do cluster and generate codewords."<<endl;
  }
  codebook->SetVerbose(true);
  int valid = codebook->GenerateCodeWords(K,oneclassSVM);
  if(splitlevel==0) fsize=valid;
  else if(splitlevel==1) fsize=valid*6;
  else if(splitlevel==2) fsize=valid*31;
  return codebook;
}

int main(int argc, char** argv)
{
  if (argc != 11)
  {
    usage();
    return -1;
  }
  string train_img_path(argv[1]);
  int num_train_images = atoi(argv[2]);
  string test_img_path(argv[3]);
  int num_test_images = atoi(argv[4]);
  int K = atoi(argv[5]); // Number of codeword centers to use

  const CodeBook::DESCRIPTOR_TYPE feature_type = FindDescriptor(argv[6]);
  const CodeBook::CODEBOOK_TYPE codebook_type = FindCodeBook(argv[7]);
  string train_out_path(argv[8]);
  string test_out_path(argv[9]);
  bool use_harris = atoi(argv[10]);

  int scaleChanges = 0; // We will densley compute the featuers at a single scale
  int stepSize = 8;
  bool useBoth = false;
  bool useSobel = false;
  bool one_class_SVM = false;
  double ratio = 1.0;
  int padding_size = 4;

  // Read the image names into a vector
  for (int i = 1; i <= num_train_images; i++)
  {
    // Create a zero padded list
    stringstream img_num;
    stringstream image_name;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = padding_size - img_num_str.size();
    img_num_str.insert(0, num_zeros, '0');
    // Build the full path to the image
    image_name << train_img_path << img_num_str << ".jpg";
    train_names.push_back(image_name.str().c_str());
  }
  // Build the Codebook
  CodeBook * codebook = CodeBookEngine(codebook_type, feature_type, K, useSobel,
                                       one_class_SVM, fsize);

  // Build the feature vectors for the train images
  cout<<"Now generating feature vectors for training data."<<endl;
  Array2d<double> train_data;
  train_data.Create(num_train_images, fsize);
  StartOfDuration();
#pragma omp parallel for
  for(int i=1; i <= num_train_images;i++) {
    // Create a zero padded list
    stringstream img_num;
    stringstream image_name;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = padding_size - img_num_str.size();
    img_num_str.insert(0, num_zeros, '0');
    // Build the full path to the image
    image_name << train_img_path << img_num_str << ".jpg";
    cout << image_name.str() << endl;
    if (use_harris)
      codebook->TranslateOneHarrisImage(image_name.str().c_str(),
                                        stepSize, splitlevel, train_data.p[i-1],
                                        fsize, normalize, ratio, scaleChanges,
                                        true);
    codebook->TranslateOneImage(image_name.str().c_str(),
                                stepSize, splitlevel, train_data.p[i-1],
                                fsize, normalize, ratio, scaleChanges, true);
    if(omp_get_thread_num()==0 && i%10==0) {
      cout<<".";
    }
    cout.flush();
  }

  cout << endl;
  cout << "Generated in "<< EndOfDuration()/1000 << " seconds."
       << endl;
  cout << "--------------------------------------------------------------------------"
       << endl;

    // Now write this stuff to disk
  ofstream train_out(train_out_path.c_str());
  for(int i = 0; i < train_data.nrow; i++)
  {
    for(int j = 0; j < train_data.ncol; j++)
      train_out << train_data.p[i][j] << " ";
    train_out << endl;
  }
  train_data.Clear();
  // Build the feature vectors for the test images
  cout<<"Now generating feature vectors for testing data."<<endl;
  Array2d<double> test_data;
  test_data.Create(num_test_images, fsize);
  StartOfDuration();
#pragma omp parallel for
  for(int i=1; i <= num_test_images;i++) {
    // Create a zero padded list
    stringstream img_num;
    stringstream image_name;
    img_num << i;
    string img_num_str = img_num.str();
    int num_zeros = padding_size - img_num_str.size();
    img_num_str.insert(0, num_zeros, '0');
    // Build the full path to the image
    image_name << test_img_path << img_num_str << ".jpg";
    cout << image_name.str() << endl;
    if (use_harris)
      codebook->TranslateOneHarrisImage(image_name.str().c_str(),
                                        stepSize, splitlevel, test_data.p[i-1],
                                        fsize, normalize, ratio, scaleChanges,
                                        true);
    codebook->TranslateOneImage(image_name.str().c_str(),
                                stepSize, splitlevel, test_data.p[i-1],
                                fsize, normalize, ratio, scaleChanges, true);
    if(omp_get_thread_num()==0 && i%10==0) {
      cout<<".";
    }
    cout.flush();
  }

  cout << endl;
  cout << "Generated in "<< EndOfDuration()/1000 << " seconds."
       << endl;
  cout << "--------------------------------------------------------------------------"
       << endl;

  // Now write this stuff to disk
  ofstream test_out(test_out_path.c_str());
  for(int i = 0; i < test_data.nrow; i++)
  {
    for(int j = 0; j < test_data.ncol; j++)
      test_out << test_data.p[i][j] << " ";
    test_out << endl;
  }
  test_data.Clear();
  return 0;
}

// Author: Jianxin Wu (wujx2001@gmail.com)

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <climits>

#include "mdarray.h"
#include "util.h"
#include "IntImage.h"
#include "HistogramKernel.h"
#include "LinearKernel.h"
#include "Codebook.h"
#include "Features.h"

using namespace std;

CodeBook::CodeBook(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,
                   const int _resizeWidth,const int _windowSize,
                   const int _L1_norm)
    :verbose(false),feature_type(_feature_type),useSobel(_useSobel),
     validcenters(0),resizeWidth(_resizeWidth),windowSize(_windowSize),
     L1_norm(_L1_norm)
{
  assert(ValidFeatureType(feature_type));
  assert(resizeWidth>=0 && windowSize>0 && L1_norm>0);
}

CodeBook::~CodeBook()
{
}

BaseFeature* CodeBook::FeatureEngine(const DESCRIPTOR_TYPE feature_type,
                                     const bool useSobel,const int L1_norm)
{
  assert(ValidFeatureType(feature_type) && L1_norm>0);
  BaseFeature* feature = NULL;
  switch(feature_type)
  {
    case DESCRIPTOR_CENTRIST:
      feature = new CENTRIST(useSobel,L1_norm);
      break;
    case DESCRIPTOR_SIFT:
      feature = new SiftFeature(useSobel,L1_norm);
      break;
    case DESCRIPTOR_LOCAL:
      feature = new LocalFeature(useSobel);
      break;
    case DESCRIPTOR_SYM_CENTRIST:
      feature = new SymCENTRIST(useSobel,L1_norm);
      break;
    default:
      std::cout<<"Feature type is wrong."<<std::endl;
      break;
  }
  return feature;
}

void CodeBook::TranslateOneImage(const char* filename,
                                 const int stepSize,const int splitlevel,
                                 double* p,const int fsize,const bool normalize,
                                 const double ratio,int scales, bool useBinary) const
{
  // we could put 'feature' as a member of the CodeBook class. However,
  // put it here makes parallel processing (OpenMP) possible

  // generate the right feature extractor
  BaseFeature* feature = FeatureEngine(feature_type,useSobel,L1_norm);

  // ratio to shrink the image to a smaller scale & weights at different
  // scaled version of the image
  const double scale_change = 0.75;
  // weight of codewords generated in different resized version of the image
  double scale = 1;

  // assign an image to the feature extractor
  const IntImage<double>* img = &feature->AssignFile(filename,resizeWidth);
  std::fill(p,p+fsize,0.0);
  while(scales>=0)
  {
    int match;
    for(int i=1;i+windowSize<img->nrow;i+=stepSize)
    {
      for(int j=1;j+windowSize<img->ncol;j+=stepSize)
      {
        // find the right codeword
        match = Find_Nearest(i,i+windowSize,j,j+windowSize,feature);
        // NOTE:special case -- this image patch is nearly uniform
        if(match<0) continue;
        if (useBinary)
          p[match] = 1;
        else
          p[match]+=scale; // depth 0 of the hierarchy
        int x,y;
        if(splitlevel>0) {
          x = (i+windowSize/2)*4/img->nrow;
          y = (j+windowSize/2)*4/img->ncol;
          if(x>0 && x<3 && y>0 && y<3) {
            // depth 1 of the hierarchy
            if (useBinary)
              p[validcenters+match] = 1;
            else
              p[validcenters+match] += scale;
          }
          x /= 2;
          y /= 2;
          // depth 1 of the hierarchy
          if (useBinary)
            p[validcenters*(2+x*2+y)+match] = 1;
          else
            p[validcenters*(2+x*2+y)+match] += scale;
        }
        if(splitlevel>1) {
          x = (i+windowSize/2)*8/img->nrow;
          y = (j+windowSize/2)*8/img->ncol;
          // depth 2 of the hierarchy
          if(x>0 && x<7 && y>0 && y<7){
            if (useBinary)
              p[validcenters*(6+(x-1)/2*3+ (y-1)/2)+match] = 1;
            else
              p[validcenters*(6+(x-1)/2*3+ (y-1)/2)+match] += scale;
          }
          x /= 2;
          y /= 2;
          // depth 2 of the hierarchy
          if(useBinary)
            p[validcenters*(15+x*4+y)+match] = 1;
          else
            p[validcenters*(15+x*4+y)+match] += scale;
        }
      }
    }
    // shrink the image if number of 'scales' is not reached
    img = &feature->ResizeImage(scale_change);
    scales--;
    // set weight of codewords generated in different resized images
    scale *= (1.0/scale_change/scale_change);
  }
  delete feature; feature = NULL;

  if(normalize) Normalize_L1(p,validcenters);
  // normalize if necessary, and weight the codewords differently depending on
  // depth in the hierarchy
  if(splitlevel>=1)
  {
    if(normalize) for(int j=0;j<5;j++) Normalize_L1(p+validcenters+
                                                    validcenters*j,
                                                    validcenters);
    for(int j=0;j<5*validcenters;j++) p[validcenters+j] *= ratio;
  }
  if(splitlevel>=2)
  {
    if(normalize) for(int j=0;j<25;j++) Normalize_L1(p+validcenters+
                                                     validcenters*5+
                                                     validcenters*j,
                                                     validcenters);
    for(int j=0;j<25*validcenters;j++) p[validcenters+validcenters*5+j] *=
                                               (ratio*ratio);
    }
}

void CodeBook::TranslateOneHarrisImage(const char* filename,
                                       const int stepSize,const int splitlevel,
                                       double* p,const int fsize,
                                       const bool normalize,
                                       const double ratio,
                                       int scales, bool useBinary) const
{
  // we could put 'feature' as a member of the CodeBook class. However,
  // put it here makes parallel processing (OpenMP) possible

  // generate the right feature extractor
  BaseFeature* feature = FeatureEngine(feature_type,useSobel,L1_norm);

  // ratio to shrink the image to a smaller scale & weights at different
  // scaled version of the image
  const double scale_change = 0.75;
  // weight of codewords generated in different resized version of the image
  double scale = 1;
  const int window_offset = windowSize / 2;
  // assign an image to the feature extractor
  const IntImage<double>* img = &feature->AssignFile(filename,resizeWidth);
  std::fill(p,p+fsize,0.0);
  // Extract keypoints from the image
  IplImage* cv_raw_img = cvLoadImage(filename, IPL_DEPTH_8U);
  int resizeHeight = (cv_raw_img->width/resizeWidth)*cv_raw_img->height;
  IplImage* cv_img = cvCreateImage(cvSize(img->ncol,img->nrow),
                                   IPL_DEPTH_8U, 1);
  cvResize(cv_raw_img,cv_img);
  cvReleaseImage(&cv_raw_img);
  IplImage* corner_img = cvCreateImage(cvSize(cv_img->width, cv_img->height),
                                       IPL_DEPTH_32F,1);;
  cvCornerHarris(cv_img, corner_img, windowSize);

  // Currently we create a corner at any point greater than 12.5% of the max
  const double threshold = 0.000001;
  vector<CvPoint> corners;
  float max_val = 0;
  // Threshold the image to find the corners
  for (int y = window_offset + 1; y + window_offset + 1 < corner_img->height;
       y++) {
    for (int x = window_offset + 1; x + window_offset + 1 < corner_img->width;
         x++) {
      const double val = cvGet2D(corner_img, y, x).val[0];
      if ( val > threshold) {

        if (val < cvGet2D(corner_img, y, x -1).val[0]) {
          continue;
        }
        if (val < cvGet2D(corner_img, y - 1, x -1).val[0]) {
            continue;
        }
        if (val < cvGet2D(corner_img, y + 1, x -1).val[0]) {
          continue;
        }
        if (val < cvGet2D(corner_img, y, x + 1).val[0]) {
          continue;
        }

        if (val < cvGet2D(corner_img, y - 1, x + 1).val[0]) {
          continue;
        }
        if (val < cvGet2D(corner_img, y + 1, x + 1).val[0]) {
          continue;
        }
        if (val < cvGet2D(corner_img, y - 1, x).val[0]) {
          continue;
        }
        if (val < cvGet2D(corner_img, y + 1, x).val[0]) {
          continue;
        }

        CvPoint p;
        p.x = x;
        p.y = y;
        corners.push_back(p);
        if (val > max_val)
          max_val = val;
      }
    }
  }
  cvReleaseImage(&corner_img);
  if (corners.size() < 10)
  {
    cout << "Found " << corners.size() << " corners."
         << "\tMax_val: " << max_val << endl;
  }
  // Get the descriptors for these keypoints
  for(unsigned int i = 0; i < corners.size(); i++)
  {
    int x1 = corners[i].x - window_offset;
    int x2 = corners[i].x + window_offset;
    int y1 = corners[i].y - window_offset;
    int y2 = corners[i].y + window_offset;
    if( x1 < 1 || x2 + 1 > img->ncol || y1 < 1 || y2 + 1 > img->nrow) continue;
    // find the right codeword
    int match = Find_Nearest(y1, y2, x1, x2,feature);
    // NOTE:special case -- this image patch is nearly uniform
    if(match<0) continue;
    if (useBinary)
      p[match] = 1;
    else
      p[match]+=scale; // depth 0 of the hierarchy
  }
  delete feature; feature = NULL;
  cvReleaseImage(&cv_img);

  if(normalize) Normalize_L1(p,validcenters);
  // normalize if necessary, and weight the codewords differently depending on
  // depth in the hierarchy
  if(splitlevel>=1)
  {
    if(normalize) for(int j=0;j<5;j++) Normalize_L1(p+validcenters+
                                                    validcenters*j,
                                                    validcenters);
    for(int j=0;j<5*validcenters;j++) p[validcenters+j] *= ratio;
  }
  if(splitlevel>=2)
  {
    if(normalize) for(int j=0;j<25;j++) Normalize_L1(p+validcenters+
                                                     validcenters*5+
                                                     validcenters*j,
                                                     validcenters);
    for(int j=0;j<25*validcenters;j++) p[validcenters+validcenters*5+j] *=
                                               (ratio*ratio);
    }
}

IplImage* CodeBook::TranslateOneImage(const char* filename,const int K) const
{
  CvScalar* colors = new CvScalar[K];
  my_srand(123456);
  for(int i=0;i<256;i++) colors[i] = cvScalar(my_rand()%K,my_rand()%K,my_rand()%K,my_rand()%K);

  BaseFeature* feature = FeatureEngine(feature_type,useSobel,L1_norm);
  const IntImage<double>& img = feature->AssignFile(filename,0);
  IplImage* result = cvCreateImage(cvSize(img.ncol,img.nrow),IPL_DEPTH_8U,3);
  const CvScalar black = cvScalar(0,0,0,0);
  for(int i=0;i<img.nrow;i++) for(int j=0;j<img.ncol;j++) cvSet2D(result,i,j,black);
  for(int i=1;i+windowSize<img.nrow;i++)
  {
    for(int j=1;j+windowSize<img.ncol;j++)
    {
      int match = Find_Nearest(i,i+windowSize,j,j+windowSize,feature);
      assert(match<K);
      cvSet2D(result,i+windowSize/2,j+windowSize/2,colors[match]);
    }
  }
  delete feature; feature = NULL;
  delete[] colors;
  return result;
}

LinearCodes::LinearCodes(const DESCRIPTOR_TYPE _feature_type,
                         const bool _useSobel,
                         const int _resizeWidth,
                         const int _windowSize,
                         const int _L1_norm,const bool _useMedian)
    :CodeBook(_feature_type,_useSobel,_resizeWidth,_windowSize,_L1_norm),
     useMedian(_useMedian)
{
}

LinearCodes::~LinearCodes()
{
}

void LinearCodes::GenerateClusterData(const std::vector<const char*>& files,
                                      const int stepSize)
{
  BaseFeature* feature = FeatureEngine(feature_type,useSobel,L1_norm);
  features.Create(1000,feature->Length());
  int added = 0;
  for(unsigned int imgindex=0; imgindex<files.size(); imgindex++)
  {
    const IntImage<double>& img = feature->AssignFile(files[imgindex],resizeWidth); // assign image to the feature extractor
    int num_subwin = ((img.nrow-2)/stepSize+1) * ((img.ncol-2)/stepSize+1); // number of features generated for this image
    if(added+num_subwin>features.nrow) // increase capacity of 'features' if necessary
      features.AdjustCapacity(max(features.nrow*3/2,added+num_subwin));
    for(int i=1;i+windowSize<img.nrow;i+=stepSize)
    {
      for(int j=1;j+windowSize<img.ncol;j+=stepSize)
      {   // LinearCodes use real valued ('D'--double) version of features
        const double* p = feature->D_feature(i,i+windowSize,j,j+windowSize);
        if(p[0]<0) continue; // NOTE: special case -- image patch nearly uniform
        std::copy(p,p+feature->Length(),features.p[added]);
        added++;
      }
    }
  }
  features.AdjustCapacity(added);
  delete feature; feature = NULL;
}

// set data for training purpose
void LinearCodes::SetData(Array2d<double>& data)
{
  features = data;
}

void LinearCodes::SetData(Array2d<int>& data)
{
  features.Create(data.nrow,data.ncol);
  for(int i=0;i<data.nrow;i++) for(int j=0;j<data.ncol;j++) features.p[i][j] = data.p[i][j];
}

int LinearCodes::GenerateCodeWords(const int K, const bool oneclassSVM,
                                   const int maxiteration)
{
  Array2dC<int> labels(1,features.nrow);
  Array2dC<int> counts(1,K);
  rho.Create(1,K);
  Linear_Kernel_Kmeans kmeans(features, K, 0.0001, maxiteration, false,
                              labels.buf, counts.buf, eval, rho.buf,
                              useMedian, verbose);
  validcenters = kmeans.KernelKmeans();
  if(oneclassSVM) validcenters = kmeans.OneClassSVM();
  features.Clear();
  return validcenters;
}

int LinearCodes::Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const
{
  const double* p = feature->D_feature(x1,x2,y1,y2);
  if(p[0]<0) return -1; // NOTE: special case
  else
    return Linear_Find_Nearest(p,eval,feature->Length(),validcenters,rho.buf,useMedian);
}

HistogramCodes::HistogramCodes(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,const int _resizeWidth,const int _windowSize,const int _L1_norm,const int _upper_bound)
    :CodeBook(_feature_type,_useSobel,_resizeWidth,_windowSize,_L1_norm),upper_bound(_upper_bound)
{
}

HistogramCodes::~HistogramCodes()
{
}

void HistogramCodes::GenerateClusterData(const std::vector<const char*>& files,const int stepSize)
{
  BaseFeature* feature = FeatureEngine(feature_type,useSobel,L1_norm);
  features.Create(1000,feature->Length());
  int added = 0;
  for(unsigned int imgindex=0;imgindex<files.size();imgindex++)
  {
    cout << files[imgindex] << endl;
    const IntImage<double>& img = feature->AssignFile(files[imgindex],resizeWidth);
    int num_subwin = ((img.nrow-2)/stepSize+1) * ((img.ncol-2)/stepSize+1);
    if(added+num_subwin>features.nrow) features.AdjustCapacity(max(features.nrow*3/2,added+num_subwin));
    for(int i=1;i+windowSize<img.nrow;i+=stepSize)
    {
      for(int j=1;j+windowSize<img.ncol;j+=stepSize)
      {   // LinearCodes use discrete valued ('I'--integer) version of features
        const int* p = feature->I_feature(i,i+windowSize,j,j+windowSize);
        if(p[0]<0) continue; // NOTE: special case
        std::copy(p,p+feature->Length(),features.p[added]);
        added++;
      }
    }
  }
  features.AdjustCapacity(added);
  delete feature; feature = NULL;
}

void HistogramCodes::SetData(Array2d<int>& data)
    // set data for training purpose
{
  features = data;
}

int HistogramCodes::GenerateCodeWords(const int K,const bool oneclassSVM,
                                      const int maxiteration)
{
  Array2dC<int> labels(1,features.nrow);
  Array2dC<int> counts(1,K);
  rho.Create(1,K);
  Histogram_Kernel_Kmeans kmeans(features,K,upper_bound,0.0001,maxiteration,
                                 false,labels.buf,counts.buf,eval,rho.buf,
                                 verbose);
  validcenters = kmeans.KernelKmeans();
  if(oneclassSVM) validcenters = kmeans.OneClassSVM();
  features.Clear();
  return validcenters;
}

int HistogramCodes::Find_Nearest(const int x1, const int x2, const int y1,
                                 const int y2,BaseFeature* feature) const
{
  const int* f = feature->I_feature(x1,x2,y1,y2);
  if(f[0]<0) return -1; // NOTE: special case
  else
    return Histogram_Find_Nearest(f,eval,feature->Length(),validcenters,
                                  upper_bound,rho.buf);
}

LocalCodes::LocalCodes(const bool _useSobel,const int _resizeWidth)
    :CodeBook(DESCRIPTOR_CENTRIST,_useSobel,_resizeWidth,3,INT_MAX)
{
  validcenters = 256;
}

LocalCodes::~LocalCodes()
{
}

int LocalCodes::Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const
{
  // we assuem 3x3 patches, so x2 & y2 are not useful
  return CensusTransform<double>(feature->Image().p,x1+1,y1+1);
}

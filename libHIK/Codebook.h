// Author: Jianxin Wu (wujx2001@gmail.com)
// visual codebook

#ifndef __CODEBOOK_H__
#define __CODEBOOK_H__

#include <vector>
#include <string>

#include "mdarray.h"
#include "util.h"
#include "IntImage.h"
#include "Features.h"

class CodeBook
{
 public:
  enum DESCRIPTOR_TYPE {DESCRIPTOR_CENTRIST, DESCRIPTOR_SIFT,
                        DESCRIPTOR_LOCAL, DESCRIPTOR_SYM_CENTRIST};
  enum CODEBOOK_TYPE { CODEBOOK_KMEANS, CODEBOOK_KMEDIAN, CODEBOOK_HIK,
                       CODEBOOK_LOCAL, CODEBOOK_TREE };
  // Create a visual descriptor of 'feature_type'
  static BaseFeature* FeatureEngine(const DESCRIPTOR_TYPE feature_type,
                                    const bool useSobel,const int L1_norm);
 public:
  static bool ValidFeatureType(const int _feature_type)
  {
    return (_feature_type==DESCRIPTOR_CENTRIST ||
            _feature_type==DESCRIPTOR_SIFT ||
            _feature_type==DESCRIPTOR_LOCAL ||
            _feature_type==DESCRIPTOR_SYM_CENTRIST);
  }
  int ValidCenters() const { return validcenters; }
  const Array2d<double>& Eval() { return eval; }
  const Array2dC<double>& Rho() { return rho; }
 protected:
  bool verbose;
 public:
  bool SetVerbose(const bool _verbose) {
    bool v = verbose; verbose = _verbose;
    return v;
  }
  bool GetVerbose() { return verbose; }
 protected:
  Array2d<double> eval;
  Array2dC<double> rho; // data structures that define codewords
  DESCRIPTOR_TYPE feature_type; // SIFT or CENTRIST, or?
  bool useSobel; // whether use Sobel image instead of original image
  int validcenters; // number of cluster centers that are not empty
  int resizeWidth; // resize the input image such that its width='resizeWidth' (if 'resizeWidth'>0)
  int windowSize; // a patch is 'windowSize' by 'windowSize'
  int L1_norm; // L1 norm of the feature vectors, see BaseFeature for detail
 public:
  CodeBook(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,const int _resizeWidth,const int _windowSize,const int _L1_norm);
  virtual ~CodeBook();
 public:
  // Generate set of features for clustering from 'files' (images), the sampling step size is 'stepSize'
  virtual void GenerateClusterData(const std::vector<const char*>& files,const int stepSize) = 0;
  // do clustering and generate data structures that define codewords
  virtual int GenerateCodeWords(const int K,const bool oneclassSVM,const int maxiteration = 10) = 0;
  // map an image ('filename') to a feature vector 'p' (with length 'fsize') in our spatial arrangement (with a 'splitlevel' depth hierarchy) -- see our ICCV 2009 paper
  // image dense sampling step size is 'stepSize', patches are sampled at 'scales' number of resized version of the image
  // histograms of codewords from different depth of the hierarchy is weighted differently specified by 'ratio'
  // the histogram of codewords is normalize to unit L1 norm if 'normalize'=true
  virtual void TranslateOneImage(const char* filename,const int stepSize,
                                 const int splitlevel,double* p,
                                 const int fsize,const bool normalize,
                                 const double ratio,int scales,
                                 bool useBinary=false) const;
  virtual void TranslateOneHarrisImage(const char* filename,const int stepSize,
                                       const int splitlevel,double* p,
                                       const int fsize,const bool normalize,
                                       const double ratio,int scales,
                                       bool useBinary=false) const;
  // Translate an image into its an image, where a color corresponds to a visual code word
  IplImage* TranslateOneImage(const char* filename,const int K) const;
  // with an image already assigned to 'feature', find the right codeword for patch [x1..x2)X[y1..y2)
  virtual int Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const = 0;
};

class LinearCodes: public CodeBook
{   // codebook generated through (usual, linear) k-means clustering
  friend class TreeCodes;
 private:
  Array2d<double> features; // feature vectors to be clustered, one row is a feature vector
  bool useMedian;
 public:
  LinearCodes(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,const int _resizeWidth,const int _windowSize,const int _L1_norm,const bool _useMedian);
  ~LinearCodes();
 public:
  void GenerateClusterData(const std::vector<const char*>& files,const int stepSize);
  int GenerateCodeWords(const int K,const bool oneclassSVM,const int maxiteration = 10);
  int Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const;
 public:
  void SetData(Array2d<double>& data);
  void SetData(Array2d<int>& data);
};

class HistogramCodes: public CodeBook
{   // codebook generated through kernel (HIK) k-means clustering
  friend class TreeCodes;
 protected:
  Array2d<int> features; // feature vectors to be clustered, one row is a feature vector
  const int upper_bound; // maximum value that any feature dimension can attain (if L1_norm is not big, it is convenient to set 'upperbound'='L1_norm+1'
 public:
  HistogramCodes(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,const int _resizeWidth,const int _windowSize,const int _L1_norm,const int _upper_bound);
  ~HistogramCodes();
 public:
  void GenerateClusterData(const std::vector<const char*>& files,const int stepSize);
  int GenerateCodeWords(const int K,const bool oneclassSVM,const int maxiteration = 10);
  int Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const;
 public:
  void SetData(Array2d<int>& data);
};

class LocalCodes: public CodeBook
{   // this is the code of local shape (Census Transform in 3x3 windows), K=256 is fixed
  // feature_type, eval, & rho are not used, since we do not need a feature extractor in this case
 public:
  LocalCodes(const bool _useSobel,const int _resizeWidth);
  ~LocalCodes();
 public:
  void GenerateClusterData(const std::vector<const char*>& files,const int stepSize) { }
  int GenerateCodeWords(const int K,const bool oneclassSVM,const int maxiteration = 10) { return 256; }
  int Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const;
};

class TreeCodes: public HistogramCodes
{
 private:
  static const int depth = 2;
  int total_evals_count;
  int fanout;

  Array2d<double>* total_evals;
  Array2d<double> rho;
 public:
  TreeCodes(const DESCRIPTOR_TYPE _feature_type,const bool _useSobel,const int _resizeWidth,const int _windowSize,const int _L1_norm,const int _d,
            const int _fanout);
  ~TreeCodes();
 public:
  int GenerateCodeWords(const int K,const bool oneclassSVM,const int maxiteration = 10);
  int Find_Nearest(const int x1,const int x2,const int y1,const int y2,BaseFeature* feature) const;
};

#endif // __CODEBOOK_H__

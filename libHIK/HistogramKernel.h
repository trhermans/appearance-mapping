// Author: Jianxin Wu (wujx2001@gmail.com)
// Clustering using the histogram intersection kernel
// And visual codebook with HIK
// Please refer to
//  Beyond the Euclidean distance: Creating effective visual codebooks\\using the histogram intersection kernel
//  Jianxin Wu and James M. Rehg
//  ICCV 2009

#ifndef __HISTOGRAM_KERNEL__
#define __HISTOGRAM_KERNEL__

#include "mdarray.h"
#include "util.h"

template<class T>
T HIK_Similarity(const T* p1,const T* p2,const int m)
{
    T sum = 0;
    for(int i=0;i<m;i++) sum += std::min(p1[i],p2[i]);
    return sum;
}

class Histogram_Kernel_Kmeans
{
// NOTE that we assume the input parameters are data are valid -- we are not checking them
// All features values should be between 0 and upper_bound - 1
// And upper_bound should not be very large -- otherwise maybe memory overflow
public:
    Histogram_Kernel_Kmeans(
                            const Array2d<int>& _data, // each row is a data point
                            const int _k, // total number of clusters (classes)
                            const int _upper_bound, // All feature values should be between 0 and upper_bound - 1
                            const double _t, // tolerance to terminate k-means iterations
                            const int _max_iteration, // maximum number of iterations
                            const bool _labelsInitialized, // whether labels for each data points has been initialized
                            int* _labels, // input/output of label (class/cluster membership)
                            int* _counts, // size of each clusters
                            Array2d<double>& _eval, // fast data structure to evaluate HIK
                            double* _rho, // negative half of squared L2 norm of cluster centers (in high dimensional feature space)
                            const bool _verbose = true // whether print clustering messages
                           );
    ~Histogram_Kernel_Kmeans();
    
    void Save(const char* filename);

    // Fill auxiliary data structure ('eval') such that histogram intersection kernel values can be quickly computed
    // If a set 'c' contains all data points with label 'c', then
    // the average HIK value between a query 'x' and all points in set 'c' can be quickly computed
    double BuildFastKernel();
    // Compute some statistics, used in Kernel k-means algorithm
    // return value is the average error in k-means algorithm (in the feature space spanned by histogram intersection kernel)
    // if changeRho==true then the boundary is changed. Otherwise, simply compute error rate.
    double ComputeConstants(const bool changeRho=true);
    void InitKmeans(const bool labelsInitialized);
    // Histogram Kernel k-means, the real clustering function
    int KernelKmeans();
    // Use one-class SVM to generate a visual code word
    int OneClassSVM();
    // remove the empty cluster centers
    void RemoveEmptyClusters();

public:
    bool SetVerbose(const bool _verbose) { bool v = verbose; verbose = _verbose; return v; }
    bool GetVerbose() { return verbose; }

private:
    const Array2d<int>& data;
    const int k;
    const int upper_bound;
    const double t;
    const int max_iteration;
    const bool labelsInitialized;
    int* labels;
    int* counts;
    Array2d<double>& eval;
    double* rho;
    bool verbose;

    const int n; // number of data points
    const int m; // number of bins in the histogram
    int emptycenters; // nubmer of empty clusters
    int validcenters; // number of non-empty clusters

    double* w_square; // square of L2 norm of cluster centers in feature space
    double* x_square; // L1 norm (square of L2 in feature space) of data points
};

// Given a row (visual code word) 'index' in the visual code book 'eval', find the HIK value for input 'f'
double Histogram_Fast_Similarity(const int* f,const Array2d<double>& eval,const int m,const int index,const int upper_bound);

// Find the nearest neighbor within the 'k' clusters/visual code words
int Histogram_Find_Nearest(
                           const int* f, // a data point (query)
                           const Array2d<double>& eval, // the fast data structure
                           const int m, // length of feature vector
                           const int k, // number of clusters, or, number of valid clusters
                           const int upper_bound,  // All feature values should be between 0 and upper_bound - 1
                           const double* rho // negative half of the squared L2 norm of cluster centers (in the feature space)
                          );

// NOTE: the difference is that query 'f' is in double format
// But you have to make sure that the data is valid: integers between 0 and upper_bound-1
int Histogram_Find_Nearest(const double* f,const Array2d<double>& eval,const int m,const int k,const int upper_bound,const double* rho);
                          
#endif // __HISTOGRAM_KERNEL__

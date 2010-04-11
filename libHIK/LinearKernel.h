// Author: Jianxin Wu (wujx2001@gmail.com)
// Clustering using the normal k-means AND k-median
// NOTE that data in linear, normal k-means is DOUBLE type
// which is different from the HIK clustering who uses INT type

#ifndef __LINEAR_KERNEL__
#define __LINEAR_KERNEL__

#include "mdarray.h"
#include "util.h"

class Linear_Kernel_Kmeans // just normal k-means
{
public:
    Linear_Kernel_Kmeans(
                            const Array2d<double>& _data, // each row is a data point
                            const int _k, // total number of clusters (classes)
                            const double _t, // tolerance to terminate k-means iterations
                            const int _max_iteration, // maximum number of iterations
                            const bool _labelsInitialized, // whether labels for each data points has been initialized
                            int* _labels, // input/output of label (class/cluster membership)
                            int* _counts, // size of each clusters
                            Array2d<double>& _eval, // fast data structure to evaluate dot product
                            double* _rho, // negative half of squared L2 norm of cluster centers
                            const bool _useMedian, // if this is set to true, then k-median instead of k-means
                            const bool _verbose = true // whether print clustering messages
                           );
    ~Linear_Kernel_Kmeans();
    void Save(const char* filename);

    // Fill auxiliary data structure ('eval') such that linear kernel values can be quickly computed
    double BuildFastKernel();
    // Compute some statistics, used in k-means algorithm
    // return value is the average error in k-means algorithm
    // if changeRho==true then the boundary is changed. Otherwise, simply compute error rate.
    double ComputeConstants(const bool changeRho=true);
    // Initialization
    void InitKmeans(const bool labelsInitialized);
    // Linear Kernel k-means, the real clustering function
    int KernelKmeans();
    // Use one-class SVM to generate clusters
    int OneClassSVM();
    // Remove empty clusters
    void RemoveEmptyClusters();

public:
    bool SetVerbose(const bool _verbose) { bool v = verbose; verbose = _verbose; return v; }
    bool GetVerbose() { return verbose; }

private:
    const Array2d<double>& data;
    const int k;
    const double t;
    const int max_iteration;
    const bool labelsInitialized;
    int* labels;
    int* counts;
    Array2d<double>& eval; // row i of 'eval' is the i-th cluster center
    double* rho;
    bool useMedian;
    bool verbose;

    const int n; // number of data points
    const int m; // number of bins in the histogram
    int emptycenters; // nubmer of empty centers
    int validcenters; // nubmer of non-empty centers

    double* x_square; // square of L2 norm of data points
    double* w_square; // square of L2 norm of centers
};

// Find the nearest neighbor within the 'k' clusters/codewords
int Linear_Find_Nearest(
                           const double* f, // a data point (query)
                           const Array2d<double>& eval, // the fast data structure
                           const int m, // length of feature vector
                           const int k, // number of clusters/codewords
                           const double* rho, // constant term in the decision boundary function
                           const bool useMedian // if this is true, then for k-median instead of k-means
                          );
// NOTE the difference is the type for 'f' is INT
int Linear_Find_Nearest(const int* f,const Array2d<double>& eval,const int m,const int k,const double* rho,const bool useMedian);

#endif // __LINEAR_KERNEL__

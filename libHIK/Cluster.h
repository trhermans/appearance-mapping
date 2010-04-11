// Author: Jianxin Wu (wujx2001@gmail.com)
// Cluster module using either HIK, k-means, or k-median
// We do NOT make extensive check on the validity of the input file -- please make sure your input file is correct

#ifndef CLUSTER_H
#define CLUSTER_H

#include "mdarray.h"

class ClusterModel
{
    enum ClusterType { HIK_CLUSTER, KMEANS_CLUSTER, KMEDIAN_CLUSTER };
public:
    ClusterModel() { }
    ~ClusterModel() { }
    
    void Load(const char* filename);
    
    // Find Cluster index for a query 'f'
    // Note that the returned index starts from 0 to 'k'-1
    int Map(const int* f);
    // you MUST make sure that the query 'f' have correct format
    int Map(const double* f);

    // If you do not want to save the clustered model to disk, just set "modelname=NULL"
    // Perform histogram intersection kernel clustering on integer input data
    void HIK_Clustering(Array2d<int>& features,const int _k,const int max_iteration,const int _upper_bound,const bool oneclass_svm,const bool verbose,const char* modelname);
    // Perform k-means or k-median clustering on real valued input data
    void Kmeans_Clustering(Array2d<double>& features,const int _k,const int max_iteration,const bool useMedian,const bool oneclass_svm,const bool verbose,const char* modelname);
    
    // Perform histogram intersection kernel clustering on integer input data
    void HIK_Clustering(const char* filename,const int _k,const int max_iteration,const int _upper_bound,const bool oneclass_svm,const bool verbose,const char* modelname);
    // Perform k-means or k-median clustering on real valued input data
    void Kmeans_Clustering(const char* filename,const int _k,const int max_iteration,const bool useMedian,const bool oneclass_svm,const bool verbose,const char* modelname);
private:
    ClusterType type;
    int m;
    int k;
    int upper_bound;
    Array2d<double> eval;
    Array2dC<double> rho;
};

#endif

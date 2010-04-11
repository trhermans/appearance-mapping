// Author: Jianxin Wu (wujx2001@gmail.com)

#include <fstream>
#include <string>
#include <cassert>

#include "Cluster.h"
#include "HistogramKernel.h"
#include "LinearKernel.h"

void ClusterModel::Load(const char* filename)
{
    std::string s;
    std::ifstream in(filename);
#if defined ( _WIN32 )
	getline(in,s,'\n');
#else
    in>>s; in.ignore(65536,'\n');
#endif
    if(s == "HIK")
        type = HIK_CLUSTER;
    else if(s == "kmeans")
        type = KMEANS_CLUSTER;
    else if(s == "kmedian")
        type = KMEDIAN_CLUSTER;
    else
    {
        std::cout<<"Invalid type of cluster model in file "<<filename<<std::endl;
        exit(-1);
    }
    
    char c;
    in>>c; assert(c=='m'); in>>m; assert(m>0); in.ignore(65536,'\n');
    in>>c; assert(c=='k'); in>>k; assert(k>0); in.ignore(65536,'\n');
    in>>c; assert(c=='u'); in>>upper_bound; assert(upper_bound>=0 || upper_bound==-1); in.ignore(65536,'\n');
    if(type==KMEANS_CLUSTER || type==KMEDIAN_CLUSTER) assert(upper_bound==-1);
    in>>c; assert(c=='r');
    rho.Create(1,k);
    for(int i=0;i<k;i++)
        in>>rho.buf[i];
    if(type==HIK_CLUSTER)
    {
        eval.Create(m*upper_bound,k);
        for(int i=0;i<m*upper_bound;i++)
            for(int j=0;j<k;j++)
                in>>eval.p[i][j];
    }
    else
    {
        eval.Create(k,m);
        for(int i=0;i<k;i++)
            for(int j=0;j<m;j++)
                in>>eval.p[i][j];
    }
    in.close();
}

// Perform histogram intersection kernel clustering on integer input data
void ClusterModel::HIK_Clustering(Array2d<int>& features,const int _k,const int max_iteration,const int _upper_bound,const bool oneclass_svm,const bool verbose,const char* modelname)
{
    type = HIK_CLUSTER;
    m = features.ncol;
    k = _k;
    upper_bound = _upper_bound;
    Array2dC<int> labels(1,features.nrow);
    Array2dC<int> counts(1,k);
    rho.Create(1,k);
    Histogram_Kernel_Kmeans kmeans(features,k,upper_bound,0.0001,max_iteration,false,labels.buf,counts.buf,eval,rho.buf,verbose);
    kmeans.KernelKmeans();
    if(oneclass_svm) kmeans.OneClassSVM();
    if(modelname) kmeans.Save(modelname);
}

// Perform k-means or k-median clustering on real valued input data
void ClusterModel::Kmeans_Clustering(Array2d<double>& features,const int _k,const int max_iteration,const bool useMedian,const bool oneclass_svm,const bool verbose,const char* modelname)
{
    if(useMedian)
        type = KMEDIAN_CLUSTER;
    else
        type = KMEANS_CLUSTER;
    m = features.ncol;
    k = _k;
    Array2dC<int> labels(1,features.nrow);
    Array2dC<int> counts(1,k);
    rho.Create(1,k);
    Linear_Kernel_Kmeans kmeans(features,k,0.0001,max_iteration,false,labels.buf,counts.buf,eval,rho.buf,useMedian,verbose);
    kmeans.KernelKmeans();
    if(oneclass_svm) kmeans.OneClassSVM();
    if(modelname) kmeans.Save(modelname);
}

// Perform histogram intersection kernel clustering on integer input data
// Data file format:
// First line: number of data points (rows), a space, feature dimensions (columns), a space, upper_bound (not used for k-means/k-median)
// Form 2nd line: one data point per line
// MAKE SURE your file format is valid -- since we do not check it
void ClusterModel::HIK_Clustering(const char* filename,const int _k,const int max_iteration,const int _upper_bound,const bool oneclass_svm,const bool verbose,const char* modelname)
{
    std::ifstream in(filename);
    int n;
    in>>n>>m;
    Array2d<int> features(n,m);
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) in>>features.p[i][j];
    in.close();

    upper_bound = _upper_bound;
    HIK_Clustering(features,_k,max_iteration,upper_bound,oneclass_svm,verbose,modelname);
}

void ClusterModel::Kmeans_Clustering(const char* filename,const int _k,const int max_iteration,const bool useMedian,const bool oneclass_svm,const bool verbose,const char* modelname)
{
    std::ifstream in(filename);
    int n;
    in>>n>>m;
    Array2d<double> features(n,m);
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) in>>features.p[i][j];
    in.close();
        
    Kmeans_Clustering(features,_k,max_iteration,useMedian,oneclass_svm,verbose,modelname);
}

// Find Cluster index for a query 'f'
// Note that the returned index starts from 0 to 'k'-1
int ClusterModel::Map(const int* f)
{
    if(type==HIK_CLUSTER)
        return Histogram_Find_Nearest(f,eval,m,k,upper_bound,rho.buf);
    else if(type==KMEANS_CLUSTER)
        return Linear_Find_Nearest(f,eval,m,k,rho.buf,false);
    else
        return Linear_Find_Nearest(f,eval,m,k,rho.buf,true);
}

// NOTE: you MUST make sure that the query 'f' in double format is valid
// if you use this function with HIK
// all feature values must be between 0 and upper_bound - 1
// Though it will work, this is not encouraged because it is slow.
int ClusterModel::Map(const double* f)
{
    if(type==HIK_CLUSTER)
        return Histogram_Find_Nearest(f,eval,m,k,upper_bound,rho.buf);
    else if(type==KMEANS_CLUSTER)
        return Linear_Find_Nearest(f,eval,m,k,rho.buf,false);
    else
        return Linear_Find_Nearest(f,eval,m,k,rho.buf,true);
}

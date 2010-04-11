// Fast computing of histogram intersection kernels
// including functions that convert LIBSVM & BSVM models to our method
// Author: Jianxin Wu (wujx2001@gmail.com)
// Please refer to
//  Beyond the Euclidean distance: Creating effective visual codebooks\\using the histogram intersection kernel
//  Jianxin Wu and James M. Rehg
//  ICCV 2009


#ifndef __USE_SVM__
#define __USE_SVM__

#include <set>
#include <string>

#include "svm.h"

// Interface function to LIBSVM
// Initialize parameter for SVM training to default values
// you need to change some parameters to values you desired after calling this function
void UseSVM_Init(svm_parameter& param,svm_problem& prob,svm_node* &x_space);

// Clean up all SVM related data structures, for next using
void UseSVM_CleanUp(svm_model* &model,svm_parameter& param,svm_problem& prob,svm_node* &x_space);

// Converting a 2d feature array to LIBSVM format, only data points whose class label in the set 'choose' is converted
// Function definition in later part of this .h file
template<class T>
int UseSVM_BuildProblem(
                        Array2d<T> data, // the data
                        const int* labels, // labels for data points
                        std::set<int>& choose, // specify which classes to use
                        svm_problem& prob, // output SVM problem, need to be EMPTY before calling this function
                        svm_node* &x_space, // SVM data cache, need to be NULL before calling this function
                        const bool oneclass, // whether we want to train a one class SVM?
                        const int maxElement = -1 // constrain SVM problem size to 'maxElement' if it is >0
                       );

// Data structure for fast computing of several histogram kernel discrimination functions (\phi(x) \dotproduct w) - \rho
// return value is -1.0*\rho
// for use in visual codebook only
double UseSVM_Histogram_FastEvaluationStructure(
                        const svm_model& model, // one SVM trained model
                        const int m, // length of feature vector
                        const int upper_bound, // maximum feature value can only be upper_bound - 1
                        Array2d<double>& eval, // fast data structure
                        const int index // which discrimination function is this? (or, which visual code word?)
                                               );

// Convert a HIK libsvm model into 'eval' so that testing is fast
double UseSVM_Histogram_FastEvaluationStructure(const svm_model& model,const int m,const int upper_bound,Array2dC<double>& eval,const bool normalize);
// Load libsvm model from file and convert to 'eval'
// If pmodel==NULL, then the loaded libsvm model is destroyed immediately, otherwise it is returned by pmodel -- you need to destroy it
double UseSVM_Histogram_FastEvaluationStructure(const char* modelfile,const int m,const int upper_bound,Array2dC<double>& eval,svm_model** pmodel=NULL,const bool normalize=true);

// Convert a HIK bsvm model into 'eval' so that testing is fast
double UseSVM_Histogram_FastEvaluationStructure(const char* modelfile,const int m,const int upper_bound,Array2dC<double>& eval,svm_model** pmodel,const bool normalize);
// Load bsvm model from file and convert to 'eval'
// If pmodel==NULL, then the loaded libsvm model is destroyed immediately, otherwise it is returned by pmodel -- you need to destroy it
// For the Crammer-Sinnger (-s 2) and HIK (-t 5) only
double UseSVM_CS_Histogram_FastEvaluationStructure(const char* modelfile,const int m,const int upper_bound,Array2dC<double>& eval,svm_model** pmodel);

// Data structure for fast computing of several linear kernel discrimination functions (\phi(x) \dotproduct w) - \rho
// return value is -1.0*\rho
double UseSVM_Linear_FastEvaluationStructure(
                        const svm_model& model, // one SVM trained model
                        const int m, // length of feature vector
                        Array2d<double>& result, // fast data structure
                        const int index // which discrimination function is this? 
                                            );

double UseSVM_Linear_FastEvaluationStructure(const svm_model& model,const int m,Array2dC<double>& eval);
double UseSVM_Linear_FastEvaluationStructure(const char* modelfile,const int m,Array2dC<double>& result);

// save a data set in the sparse SVM format to 'filename' with class labels in 'labels' and data in 'features' (1 row <==> a data point)
// NOTE that feature index will start from 1, not 0
void UseSVM_SaveSparse(const std::string& filename,const int* labels,Array2d<double>& features);
// 'split' should have same length as labels, and a point i is saved only if split[i]==value
void UseSVM_SaveSparse(const std::string& filename,const int* labels,Array2d<double>& features,const int* split,const int value);
// Same as above, but additionally record labels for which affordance values exist
void UseSVM_SaveSparseAffordance(const std::string& filename,const int* labels,
                                 Array2d<int>& affValues,
                                 Array2d<double>& features);
void UseSVM_SaveSparseAffordance(const std::string& filename,const int* labels,
                                 Array2d<int>& affValues,
                                 Array2d<double>& features, const int* split,
                                 const int value);

template<class T>
int UseSVM_BuildProblem(Array2d<T> data,const int* labels,std::set<int>& choose,svm_problem& prob,svm_node* &x_space,const bool oneclass,const int maxElement)
{
    assert(prob.l==0 && prob.y==NULL && prob.x==NULL && x_space==NULL);
    int size = 0;
    for(int i=0;i<data.nrow;i++) if(choose.find(labels[i])!=choose.end()) size++;
    if(size==0) return 0;
    if(maxElement>0 && size>maxElement) size = maxElement;

    prob.l = size;
    prob.y = new double[prob.l]; assert(prob.y!=NULL);
    prob.x = new svm_node*[prob.l]; assert(prob.x!=NULL);
    int totalfeatures = 0;
    int added_points = 0;
    for(int i=0;i<data.nrow;i++)
    {
        if(choose.find(labels[i])==choose.end()) continue;
        for(int j=0;j<data.ncol;j++) if(data.p[i][j]!=0) totalfeatures++;
        added_points++;
        if(maxElement>0 && added_points>=maxElement) break;
    }
    x_space = new svm_node[totalfeatures+prob.l]; assert(x_space!=NULL);

    added_points=0;
    int added_features=0;
    for(int i=0;i<data.nrow;i++)
    {
        if(choose.find(labels[i])==choose.end()) continue;
        if(oneclass)
            prob.y[added_points] = 1;
        else
            prob.y[added_points] = labels[i];
        prob.x[added_points] = &x_space[added_features];
        for(int j=0;j<data.ncol;j++)
        {
            if(data.p[i][j]==0) continue;
            x_space[added_features].index = j + 1;
            x_space[added_features].value = data.p[i][j];
            added_features++;
        }
        x_space[added_features].index = -1;
        added_features++;
        added_points++;
        if(maxElement>0 && added_points>=maxElement) break;
    }
    assert(added_points==size);
    assert(added_features==totalfeatures+prob.l);

    return size;
}

#endif // __USE_SVM__

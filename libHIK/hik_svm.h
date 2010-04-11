// Author: Jianxin Wu (wujx2001@gmail.com)
// Please refer to
//  Beyond the Euclidean distance: Creating effective visual codebooks\\using the histogram intersection kernel
//  Jianxin Wu and James M. Rehg
//  ICCV 2009

// revised from svm_node, replace the type of 'value' to int for HIK
struct svm_inode
{
	int index;
	int value;
};

// The below 3 functions are HIK version for the 1-vs-1 classification and others (adapted from LIBSVM)
// In 1-vs-1 classification, there are nr_class*(nr_class-1) classifiers
// each classifier takes 'm' (dimension of feature vector) rows in 'eval'
// These m rows corresponding to a table 'T' in our ICCV 2009 paper
// Refer to the paper for how to compute prediction values
// In other cases, there is only 1 classifier/regressor, which is all the rows of 'eval'
// NOTE that our revision supports the probabilistic version of LIBSVM ("-b 1"), which usually work better than default (-b 0)
void hik_predict_values(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x,double* dec_values);
double hik_predict(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x);
double hik_predict_probability(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x,double *prob_estimates);

// This function is our HIK version for BSVM
// NOTE that this function only supports "-s 2 -t 5", i.e. Crammer-Singer classification with HIK kernel
double hik_bsvm_predict(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x);

// NOTE that all above functions are assuming all features values are integers between 0 and upper_bound-1
// for real valued histograms, you could either use the "transform" utility command to convert to integer version
// and then use our functions. Or, you can use the original version of LIBSVM and BSVM

// Changing the SVM software so that histogram intersection kernel prediction is fast
// Author: Jianxin Wu (wujx2001@gmail.com)

#include "svm.h"
#include "mdarray.h"
#include "hik_svm.h"

// This function is revised from svm_predict_values for HIK 
void hik_predict_values(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x,double* dec_values)
{
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double sum = 0;
        while(x->index!=-1)
        {
            sum += eval.p[x->index-1][x->value];
            x++;
        }
		sum -= model->rho[0];
		*dec_values = sum;
	}
	else
	{
        svm_inode* xcopy;
		int nr_class = model->nr_class;
		int p=0;
		for(int i=0;i<nr_class;i++)
        {
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
                xcopy = x;
                while(xcopy->index!=-1)
                {
                    sum += eval.p[p*m+xcopy->index-1][xcopy->value];
                    xcopy++;
                }
				sum -= model->rho[p];
				dec_values[p] = sum;
				p++;
			}
        }
	}
}

// This function is revised from svm_predict, for HIK
double hik_predict(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x)
{
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double res;
		hik_predict_values(model, eval, m, upper_bound, x, &res);
		
		if(model->param.svm_type == ONE_CLASS)
			return (res>0)?1:-1;
		else
			return res;
	}
	else
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = new double[nr_class*(nr_class-1)/2];
		hik_predict_values(model, eval, m, upper_bound, x, dec_values);

		int *vote = new int[nr_class];
		for(i=0;i<nr_class;i++) vote[i] = 0;
		int pos=0;
		for(i=0;i<nr_class;i++)
        {
			for(int j=i+1;j<nr_class;j++)
			{
				if(dec_values[pos++] > 0)
					++vote[i];
				else
					++vote[j];
			}
        }

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;
        
        delete[] vote;
        delete[] dec_values;
		return model->label[vote_max_idx];
	}
}

// The below two function definitions drawn from svm.cpp
double sigmoid_predict(double decision_value, double A, double B);
void multiclass_probability(int k, double **r, double *p);
// Revised from function 'svm_predict_probability' for HIK
double hik_predict_probability(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x,double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = new double[nr_class*(nr_class-1)/2];
		hik_predict_values(model, eval, m, upper_bound, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=new double*[nr_class];
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=new double[nr_class];
		int k=0;
		for(i=0;i<nr_class;i++)
        {
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=std::min(std::max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
        }
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++) delete[] pairwise_prob[i];
        delete[] pairwise_prob;
		delete[] dec_values;
		return model->label[prob_max_idx];
	}
	else 
		return hik_predict(model, eval, m, upper_bound, x);
}

// Revised from svm_predict in bsvm.cpp for HIK
// Note that we only support C_SVC Crammer and Singer (-s 2 in bsvm) and HIK (-t 5)
// Other cases are removed
const int SPOC = 5; // for Crammer-Singer in bsvm, which corresponds to '-s 2'
double hik_bsvm_predict(const svm_model *model,Array2dC<double>& eval,const int m,const int upper_bound,svm_inode *x)
{
    assert(model->param.svm_type == SPOC);
    assert(model->param.kernel_type == 5);

    int l = model->l, nr_class = model->nr_class;

	double* f = new double[nr_class];
    for(int i=0;i<nr_class;i++) f[i] = 0;
    for(int i=0;i<nr_class;i++)
    {
        svm_inode* xcopy = x;
        while(xcopy->index!=-1)
        {
            f[i] += eval.p[i*m+xcopy->index-1][xcopy->value];
            xcopy++;
        }
    }
    
    int j = 0;
    for(int i=1;i<nr_class;i++)
        if (f[i] > f[j])
            j = i;

    delete[] f;
    return model->label[j];
}

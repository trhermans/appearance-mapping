// Author: Jianxin Wu (wujx2001@gmail.com)

#include <set>
#include <algorithm>
#include <cassert>

#include "svm.h"

#include "mdarray.h"
#include "HistogramKernel.h"

void UseSVM_Init(svm_parameter& param,svm_problem& prob,svm_node* &x_space)
{
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;    // 1/k
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    prob.l = 0;
    prob.x = NULL;
    prob.y = NULL;

    x_space = NULL;
}

void UseSVM_CleanUp(svm_model* &model,svm_parameter& param,svm_problem& prob,svm_node* &x_space)
{
    delete[] prob.y; prob.y = NULL;
    delete[] prob.x; prob.x = NULL;
    delete[] x_space; x_space=NULL;
    prob.l = 0;

    if(model)
    {
        svm_destroy_model(model);
        model = NULL;
    }
    svm_destroy_param(&param);
}

double UseSVM_Histogram_FastEvaluationStructure(const svm_model& model,const int m,const int upper_bound,Array2d<double>& eval,const int index)
// fill 'eval' such that the svm decision function can be quickly computed, refer to HistogramKernel.cpp
// for one-class SVM only, and for use in visual codebook ONLY
// The 'index' row corresponding to fast computing structure for the 'index'-th visual code word
// One row in 'eval' is a Table 'T' (refer to our ICCV 2009 paper) unfolded into a row, in row order
{
    assert(model.param.svm_type==ONE_CLASS);

    int size = 0;
    double coef_sum = 0;
    double* sv_coef = model.sv_coef[0];
    for(int i=0;i<model.l;i++)
    {
        if(sv_coef[i]>0)
        {
            size++;
            coef_sum += sv_coef[i];
        }
    }
    assert(size>0 && coef_sum>0); // if one class SVM, then size==model.l here
    for(int i=0;i<size;i++) assert(sv_coef[i]>=0);
    for(int i=size;i<model.l;i++) assert(sv_coef[i]<=0);
    for(int i=0;i<model.l;i++) sv_coef[i] /= coef_sum; // this is good since now alpha's adds up to 1. Require this in some cases.
    model.rho[0] /= coef_sum;

    Array2d<int> SVs(size,m);
    for(int i=0;i<model.l;i++) std::fill_n(SVs.p[i],m,0);
    for(int i=0;i<model.l;i++)
    {
        svm_node* sv = model.SV[i];
        while(sv->index!=-1)
        {
            SVs.p[i][sv->index-1] = (int)sv->value;
            sv++;
        }
    }

    Array2dC< sort_struct<int> > onecolumn(1,model.l);
    for(int fi=0;fi<m;fi++) // featureindex, ranging from 0 to 'm'-1
    {
        coef_sum = 1.0;
        for(int i=0;i<model.l;i++)
        {
            onecolumn.buf[i].value = SVs.p[i][fi];
            onecolumn.buf[i].id = i;
        }
        std::sort(onecolumn.buf,onecolumn.buf+model.l,Template_Less<int>);

        int curpos = 0;
        double cumsum = 0;
        for(int i=0;i<upper_bound;i++)
        {
            while(curpos<model.l && onecolumn.buf[curpos].value==i)
            {
                cumsum += sv_coef[onecolumn.buf[curpos].id]*onecolumn.buf[curpos].value;
                coef_sum -= sv_coef[onecolumn.buf[curpos].id];
                curpos++;
            }
            eval.p[fi*upper_bound+i][index] = cumsum + coef_sum * i;
        }
    }

    return -model.rho[0];
}

double UseSVM_Histogram_FastEvaluationStructure(const svm_model& model,const int m,const int upper_bound,Array2dC<double>& eval,const bool normalize)
// structure for fast average histogram similarity, NOT using codebook, to be used with LIBSVM classifiers
// This function should work with all cases (one-class, nu-SVM, C-SVC, binary, multi-class, etc)
{
    int nr_class = model.nr_class;
    int num_classifier = nr_class*(nr_class-1)/2;
    if(model.param.svm_type == ONE_CLASS || model.param.svm_type == EPSILON_SVR || model.param.svm_type == NU_SVR)
    {
        nr_class = 1;
        num_classifier = 1;
    }
    // 'start[i]': index of where the SVs for class i starts
    Array2dC<int> start(1,nr_class);
    start.buf[0] = 0;
    for(int i=1;i<nr_class;i++) start.buf[i] = start.buf[i-1]+model.nSV[i-1];
    // Check that the input is valid
    int max_index = -1;
    for(int i=0;i<nr_class;i++)
    {
        for(int j=0;j<model.nSV[i];j++)
        {
            svm_node* sv = model.SV[start.buf[i]+j];
            while(sv->index!=-1)
            {
                if(sv->index<=0 || (sv->index>m && m<0))
                {
                    std::cout<<"Invalid input: one feature index is out of range (<0 or >m)."<<std::endl;
                    exit(-1);
                }
                if(sv->index>max_index) max_index = sv->index;
                if(sv->value<0 || sv->value>=upper_bound)
                {
                    std::cout<<"Invalid input: one feature value is out of range (<0 or >=upper bound)"<<std::endl;
                    exit(-1);
                }
                sv++;
            }
        }
    }
    // eval: data structure for fast evalutation of SVMs
    // eval[(i*nr_class+j)*m : (i*nr_class+j+1)*m, :] is a sub-classifier of class i versus class j
    // each row is the lookup table value for a dimension of the feature vector, corresponding to feature value 0 to (upper_bound - 1)
    // refer to our ICCV 2009 paper
    if(m>max_index) max_index = m;
    eval.Create(num_classifier*max_index,upper_bound);
    int pos = 0; // index of the classifier
    for(int i=0;i<nr_class;i++)
    {
        for(int j=i+1;j<nr_class;j++)
        {   // sub-classifier for class i versus class j
            int size = model.nSV[i] + model.nSV[j]; // number of SVs in this sub-classifier
            Array2dC<double> sv_coef(1,size); // prepare \alpha values
            for(int k=0;k<model.nSV[i];k++) sv_coef.buf[k] = model.sv_coef[j-1][k+start.buf[i]];
            for(int k=0;k<model.nSV[j];k++) sv_coef.buf[model.nSV[i]+k] = model.sv_coef[i][k+start.buf[j]];
            // normalize these \alpha values and \rho
            double coef_sum = std::accumulate(sv_coef.buf,sv_coef.buf+model.nSV[i],0.0);
            coef_sum = 1.0/coef_sum;
            if(normalize) for(int k=0;k<size;k++) sv_coef.buf[k] *= coef_sum;
            if(normalize) model.rho[pos] *= coef_sum;
            // SVs of this sub-classifier, for sorting then filling the table
            Array2d<int> SVs(size,max_index);
            for(int k=0;k<size;k++) std::fill(SVs.p[k],SVs.p[k]+max_index,(int)0);
            for(int k=0;k<model.nSV[i];k++)
            {
                svm_node* sv = model.SV[start.buf[i]+k];
                while(sv->index!=-1)
                {
                    SVs.p[k][sv->index-1] = (int)sv->value;
                    sv++;
                }
            } // SV from class i
            for(int k=0;k<model.nSV[j];k++)
            {
                svm_node* sv = model.SV[start.buf[j]+k];
                while(sv->index!=-1)
                {
                    SVs.p[k+model.nSV[i]][sv->index-1] = (int)sv->value;
                    sv++;
                }
            } // SV from class j
            Array2dC< sort_struct<int> > onecolumn(1,size); // data structure for sorting
            for(int fi=0;fi<max_index;fi++) // feature index, ranging from 0 to 'm'-1
            {   // first sort a column
                coef_sum = std::accumulate(sv_coef.buf,sv_coef.buf+size,0.0); // should this be 0?
                for(int k=0;k<size;k++)
                {
                    onecolumn.buf[k].value = SVs.p[k][fi];
                    onecolumn.buf[k].id = k;
                }
                std::sort(onecolumn.buf,onecolumn.buf+size,Template_Less<int>);
                // Now fill the table
                int curpos = 0;
                double cumsum = 0; // cumulative partial sum
                for(int k=0;k<upper_bound;k++)
                {
                    while(curpos<size && onecolumn.buf[curpos].value==k)
                    {
                        cumsum += sv_coef.buf[onecolumn.buf[curpos].id]*onecolumn.buf[curpos].value;
                        coef_sum -= sv_coef.buf[onecolumn.buf[curpos].id];
                        curpos++;
                    }
                    eval.p[pos*m+fi][k] = cumsum + coef_sum * k;
                }
            }
            pos++;
        }
    }
    return -model.rho[0]; // this is not used, it's here for historical reasons
}

double UseSVM_Histogram_FastEvaluationStructure(const char* modelfile,const int m,const int upper_bound,Array2dC<double>& eval,svm_model** pmodel,const bool normalize)
// NOTE: If you want exactly the same result as libsvm (and usually better results) when -b 1 is used, set normalize to false
// if probability prediction of LIBSVM is not used (-b 0, which is default), the value of normalize does not have any effect.
// refer to hik-predict.cpp for example usage
{
    svm_model* model = svm_load_model(modelfile);
    if(model==0)
    {
        std::cout<<"SVM model "<<modelfile<<" can not be loaded."<<std::endl;
        exit(-1);
    }
    double rho = UseSVM_Histogram_FastEvaluationStructure(*model,m,upper_bound,eval,normalize);
    if(pmodel==NULL)
        svm_destroy_model(model);
    else
        *pmodel = model;
    return rho;
}

double UseSVM_CS_Histogram_FastEvaluationStructure(const svm_model& model,const int m,const int upper_bound,Array2dC<double>& eval)
// structure for fast average histogram similarity, NOT using codebook, for BSVM Crammer-Singer and t=5 (HIK) ONLY
// There are nr_class classifiers, each taking 'm' rows of 'eval'
{
    assert(model.param.svm_type == 5);
    assert(model.param.kernel_type == 5);

    int nr_class = model.nr_class;
    // Check that the input is valid
    int max_index = -1;
    for(int i=0;i<model.l;i++)
    {
        svm_node* sv = model.SV[i];
        while(sv->index!=-1)
        {
            if(sv->index<=0 || (sv->index>m && m<0))
            {
                std::cout<<"Invalid input: one feature index is out of range (<0 or >m)."<<std::endl;
                exit(-1);
            }
            if(sv->index>max_index) max_index = sv->index;
            if(sv->value<0 || sv->value>=upper_bound)
            {
                std::cout<<"Invalid input: one feature value is out of range (<0 or >=upper bound)"<<std::endl;
                exit(-1);
            }
            sv++;
        }
    }
    // eval: data structure for fast evalutation of SVMs
    // eval[i*m : (i+1)*m, :] is a boundary for class i
    // each row is the lookup table value for a dimension of the feature vector, corresponding to feature value 0 to (upper_bound - 1)
    if(m>max_index) max_index = m;
    eval.Create(nr_class*max_index,upper_bound);
    // SVs, for sorting then filling the table
    Array2d<int> SVs(model.l,max_index);
    for(int i=0;i<model.l;i++) std::fill(SVs.p[i],SVs.p[i]+max_index,(int)0);
    for(int i=0;i<model.l;i++)
    {
        svm_node* sv = model.SV[i];
        while(sv->index!=-1)
        {
            SVs.p[i][sv->index-1] = (int)sv->value;
            sv++;
        }
    }
    Array2dC< sort_struct<int> > onecolumn(1,model.l); // data structure for sorting
    Array2dC<double> sv_coef(1,model.l);
    for(int fi=0;fi<max_index;fi++) // feature index, ranging from 0 to 'm'-1
    {   // first sort a column
        for(int i=0;i<model.l;i++)
        {
            onecolumn.buf[i].value = SVs.p[i][fi];
            onecolumn.buf[i].id = i;
        }
        std::sort(onecolumn.buf,onecolumn.buf+model.l,Template_Less<int>);
        // Now fill the table
        for(int c=0;c<nr_class;c++)
        {
            for(int i=0;i<model.l;i++) sv_coef.buf[i] = model.sv_coef[c][i];
            double coef_sum = std::accumulate(sv_coef.buf,sv_coef.buf+model.l,0.0);
            int curpos = 0;
            double cumsum = 0; // cumulative partial sum
            for(int i=0;i<upper_bound;i++)
            {
                while(curpos<model.l && onecolumn.buf[curpos].value==i)
                {
                    cumsum += sv_coef.buf[onecolumn.buf[curpos].id]*onecolumn.buf[curpos].value;
                    coef_sum -= sv_coef.buf[onecolumn.buf[curpos].id];
                    curpos++;
                }
                eval.p[c*m+fi][i] = cumsum + coef_sum * i;
            }
        }
    }    
    return 0; // this is not used, it's here for historical reasons
}

double UseSVM_CS_Histogram_FastEvaluationStructure(const char* modelfile,const int m,const int upper_bound,Array2dC<double>& eval,svm_model** pmodel)
// This is to create fast structure for HIK in BSVM (Crammer-Singer), refer to bsvm-hik-predict.cpp for example usage
{
    svm_model* model = bsvm_load_model(modelfile);
    if(model==0)
    {
        std::cout<<"SVM model "<<modelfile<<" can not be loaded."<<std::endl;
        exit(-1);
    }
    double rho = UseSVM_CS_Histogram_FastEvaluationStructure(*model,m,upper_bound,eval);
    if(pmodel==NULL)
        svm_destroy_model(model);
    else
        *pmodel = model;
    return rho;
}

double UseSVM_Linear_FastEvaluationStructure(const svm_model& model,const int m,Array2d<double>& eval,const int index)
{   // collapse a set of linear SVM support vectors to a single vector
    assert(model.param.svm_type==ONE_CLASS || model.param.svm_type==C_SVC);
    assert(model.nr_class==1 || model.nr_class==2);

    int size = 0;
    double coef_sum = 0;
    double* sv_coef = model.sv_coef[0];
    for(int i=0;i<model.l;i++)
    {
        if(sv_coef[i]>0)
        {
            size++;
            coef_sum += sv_coef[i];
        }
    }
    assert(size>0 && coef_sum>0); // if one class SVM, then size==model.l here
    for(int i=0;i<size;i++) assert(sv_coef[i]>=0);
    for(int i=size;i<model.l;i++) assert(sv_coef[i]<=0);
    for(int i=0;i<model.l;i++) sv_coef[i] /= coef_sum; // this is good since now alpha's adds up to 1. Require this in some cases.
    model.rho[0] /= coef_sum;

    std::fill_n(eval.p[index],m,0.0);
    for(int i=0;i<model.l;i++)
    {
        svm_node* sv = model.SV[i];
        while(sv->index!=-1)
        {
            eval.p[index][sv->index-1] += sv->value * sv_coef[i];
            sv++;
        }
    }

    return -model.rho[0];
}

double UseSVM_Linear_FastEvaluationStructure(const svm_model& model,const int m,Array2dC<double>& result)
{
    Array2d<double> eval(1,m);
    result.Create(1,m);
    double rho = UseSVM_Linear_FastEvaluationStructure(model,m,eval,0);
    std::copy(eval.p[0],eval.p[0]+m,result.buf);
    return rho;
}

double UseSVM_Linear_FastEvaluationStructure(const char* modelfile,const int m,Array2dC<double>& result)
{
    svm_model* model = svm_load_model(modelfile);
    if(model==0)
    {
        std::cout<<"SVM model "<<modelfile<<" can not be loaded."<<std::endl;
        exit(-1);
    }
    double rho = UseSVM_Linear_FastEvaluationStructure(*model,m,result);
    svm_destroy_model(model);
    return rho;
}

void UseSVM_SaveSparse(const std::string& filename,const int* labels,Array2d<double>& features)
{
    std::ofstream out(filename.c_str());
    for(int i=0;i<features.nrow;i++)
    {
        out<<labels[i]+1<<" ";
        // we use "j+1" because we want feature index to start from 1, not 0
        for(int j=0;j<features.ncol;j++) if(features.p[i][j]>(1e-6)) out<<j+1<<":"<<features.p[i][j]<<" ";
        out<<std::endl;
    }
    out.close();
}

void UseSVM_SaveSparse(const std::string& filename,const int* labels,Array2d<double>& features,const int* split,const int value)
{
    std::ofstream out(filename.c_str());
    for(int i=0;i<features.nrow;i++)
    {
        if(split[i]!=value) continue;
        out<<labels[i]+1<<" ";
        for(int j=0;j<features.ncol;j++) if(features.p[i][j]>(1e-6)) out<<j+1<<":"<<features.p[i][j]<<" ";
        out<<std::endl;
    }
    out.close();
}

void UseSVM_SaveSparseAffordance(const std::string& filename, const int* labels,
                                 Array2d<int>& affValues,
                                 Array2d<double>& features)
{
    std::ofstream out(filename.c_str());
    for(int i=0;i<features.nrow;i++)
    {
        // Write the human category label
        out<<labels[i]+1<<" ";

        // Write the affordance labels
        for(int k = 0; k < affValues.ncol; ++k) {
            out << affValues.p[i][k] << " ";
        }

        // we use "j+1" because we want feature index to start from 1, not 0
        for(int j=0;j<features.ncol;j++)
            if(features.p[i][j]>(1e-6))
                out<< j+1 << ":" << features.p[i][j] << " ";
        out<<std::endl;
    }
    out.close();
}

void UseSVM_SaveSparseAffordance(const std::string& filename, const int* labels,
                                 Array2d<int>& affValues,
                                 Array2d<double>& features, const int* split,
                                 const int value)
{
    std::ofstream out(filename.c_str());
    for(int i=0;i<features.nrow;i++)
    {
        if(split[i]!=value) continue;
        out<<labels[i]+1<<" ";

        // Write the affordance labels
        for(int k = 0; k < affValues.ncol; ++k) {
            out << affValues.p[i][k] << " ";
        }

        for(int j=0;j<features.ncol;j++)
            if(features.p[i][j]>(1e-6))
                out << j+1 << ":" << features.p[i][j] << " ";
        out<<std::endl;
    }
    out.close();
}

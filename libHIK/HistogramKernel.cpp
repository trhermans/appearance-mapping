// Author: Jianxin Wu (wujx2001@gmail.com)

#if defined ( _WIN32 )
// Mircosoft is stupid, I have to put this line first to make sure cstdlib is not included before this macro
#define _CRT_RAND_S
#include <cstdlib>
#endif

#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <set>
#include <climits>
#include <cfloat>
#include <cmath>

#include <omp.h>
#include "svm.h"

#include "HistogramKernel.h"
#include "UseSVM.h"
#include "mdarray.h"

Histogram_Kernel_Kmeans::Histogram_Kernel_Kmeans(const Array2d<int>& _data,const int _k,const int _upper_bound,
                            const double _t,const int _max_iteration,const bool _labelsInitialized,
                            int* _labels,int* _counts,Array2d<double>& _eval,double* _rho,const bool _verbose)
    :data(_data),k(_k),upper_bound(_upper_bound),t(_t),max_iteration(_max_iteration),labelsInitialized(_labelsInitialized),
     labels(_labels),counts(_counts),eval(_eval),rho(_rho),verbose(_verbose),n(_data.nrow),m(_data.ncol),emptycenters(0),validcenters(_k)
{
    assert(n>=k && k>0);
    w_square = new double[k]; assert(w_square!=NULL);
    x_square = new double[n]; assert(x_square!=NULL);
    for(int i=0;i<n;i++) x_square[i] = std::accumulate(data.p[i],data.p[i]+m,0.0);
}

Histogram_Kernel_Kmeans::~Histogram_Kernel_Kmeans()
{
    delete[] w_square; w_square = NULL;
    delete[] x_square; x_square = NULL;
}

void Histogram_Kernel_Kmeans::Save(const char* filename)
{
    std::ofstream out(filename);
    out<<"HIK"<<std::endl;
    out<<"m "<<m<<std::endl;
    out<<"k "<<validcenters<<std::endl;
    out<<"u "<<upper_bound<<std::endl;
    out<<"r"<<std::endl;
    for(int i=0;i<validcenters;i++) out<<rho[i]<<' '; out<<std::endl;
    for(int i=0;i<m*upper_bound;i++)
    {
        for(int j=0;j<validcenters;j++)
        {
            out<<eval.p[i][j]<<' ';
        }
        out<<std::endl;
    }
    out.close();
}

double Histogram_Kernel_Kmeans::BuildFastKernel()
{   // fast method to evaluate for a query 'x', its distance to cluster centers, in mapped high dimensional feature space defined by histogram kernel
    // Given an input histogram f[1..m], the i-th component will contribute eval.p[upper_bound*i+f[i]][index] to the 'index'-th cluster center
    // A row in 'eval' is the table 'T' in our ICCV 2009 paper unfolded in the row major order
    emptycenters=0;
    eval.Zero();
    for(int index=0;index<k;index++)
    {
        Array2d<double> kernel(m,upper_bound);
        kernel.Zero();
        for(int i=0;i<n;i++)
        {
            if(labels[i]!=index) continue;
            for(int j=0;j<m;j++) kernel.p[j][data.p[i][j]]++;
        }
        if(counts[index]==0)
            emptycenters++;
        else
        {
            for(int i=0;i<m;i++)
            {
                double remain_sum = std::accumulate(kernel.p[i],kernel.p[i]+upper_bound,0.0);
                double cumsum = 0;
                for(int j=0;j<upper_bound;j++)
                {
                    cumsum += kernel.p[i][j]*j;
                    remain_sum -= kernel.p[i][j];
                    kernel.p[i][j] = cumsum + j*remain_sum;
                }
            }
            // Now save the result to 'eval', but in a different order for fast computing
            double r = 1.0/counts[index]; // we know that counts[index]>0
            for(int i=0;i<m;i++) for(int j=0;j<upper_bound;j++) eval.p[upper_bound*i+j][index] = kernel.p[i][j]*r;
        }
    }
    validcenters = k - emptycenters;
    RemoveEmptyClusters();
    return ComputeConstants();
}

double Histogram_Fast_Similarity(const int* f,const Array2d<double>& eval,const int m,const int index,const int upper_bound)
{
    double sum = 0;
    for(int i=0;i<m;i++) sum += eval.p[upper_bound*i+f[i]][index];
    return sum;
}

double Histogram_Kernel_Kmeans::ComputeConstants(const bool changeRho)
{   // compute error of the current clusetering, and compute square of L2 norm of cluster center (in feature space) for each cluster
    std::fill(w_square,w_square+k,0.0);
    Array2dC<double> e(1,n);
    double error = 0;
    for(int i=0;i<n;i++)
    {
        double value = Histogram_Fast_Similarity(data.p[i],eval,m,labels[i],upper_bound);
        w_square[labels[i]] += value;
        error += value;
        e.buf[i] = -2*value;
    }
    for(int i=0;i<k;i++) if(counts[i]>0) w_square[i] /= counts[i]; else w_square[i] = 1e200;
    if(changeRho) for(int i=0;i<k;i++) rho[i] = -w_square[i]*0.5;

    error *= -2;
    for(int i=0;i<n;i++) error += (w_square[labels[i]] + x_square[i]);
    error /= n;

    for(int i=0;i<n;i++) e.buf[i]=sqrt(e.buf[i]+w_square[labels[i]]+x_square[i]);
    std::nth_element(e.buf,e.buf+n/2,e.buf);
    //if(verbose) std::cout<<" Average L1 = "<<std::accumulate(e.buf,e.buf+n,0.0)/n<<", median L1 = "<<e.buf[n/2]<<std::endl;

    return error;
}

int Histogram_Find_Nearest(const int* f,const Array2d<double>& eval,const int m,const int k,const int upper_bound,const double* rho)
{
    double* values = new double[k];
    std::copy(rho,rho+k,values);
    for(int i=0;i<m;i++)
        for(int j=0;j<k;j++) values[j] += eval.p[i*upper_bound+f[i]][j];
    int v = int(std::max_element(values,values+k) - values);
	delete[] values;
	return v;
}

int Histogram_Find_Nearest(const double* f,const Array2d<double>& eval,const int m,const int k,const int upper_bound,const double* rho)
{
    double* values = new double[k];
    std::copy(rho,rho+k,values);
    for(int i=0;i<m;i++)
        for(int j=0;j<k;j++) values[j] += eval.p[i*upper_bound+int(f[i])][j];
    int v = int(std::max_element(values,values+k) - values);
	delete[] values;
	return v;
}

void Histogram_Kernel_Kmeans::InitKmeans(const bool labelsInitialized)
{
    if(labelsInitialized==false)
    {   // kmeans++ initialization, details in the kmeans++ paper
        Array2dC<int> centers(1,k); // index for initial centers
        Array2dC<double> portion(1,n);
		Array2dC<double> portion_backup(1,n);
        std::fill(portion.buf,portion.buf+n,DBL_MAX);
        int added = 0;
#if defined ( _WIN32 )
		unsigned int rr;
        rand_s(&rr); while(rr>=n) rand_s(&rr);
#else
        int rr;
        rr = random(); while(rr>=n) rr=random();
#endif
        centers.buf[0] = rr;
        std::fill(labels,labels+n,0);
        added = 1;
        while(added<=k)
        {
#pragma omp parallel for
            for(int i=0;i<n;i++)
            {
                double v = x_square[i] + x_square[centers.buf[added-1]] - 2*HIK_Similarity(data.p[i],data.p[centers.buf[added-1]],m);
                if(v<portion.buf[i])
                {
                    portion.buf[i] = v;
                    labels[i] = added - 1;
                }
            }
            if(added<k)
            {
                for(int i=1;i<n;i++) portion.buf[i] += portion.buf[i-1];
#if defined ( _WIN32 )
				if(portion.buf[n-1]<UINT_MAX)
					std::copy(portion.buf,portion.buf+n,portion_backup.buf);
				else
					for(int i=0;i<n;i++) portion_backup.buf[i] = portion.buf[i]/portion.buf[n-1]*UINT_MAX;
                rand_s(&rr); while(rr>=portion_backup.buf[n-1]) rand_s(&rr);
#else
				if(portion.buf[n-1]<RAND_MAX)
					std::copy(portion.buf,portion.buf+n,portion_backup.buf);
				else
					for(int i=0;i<n;i++) portion_backup.buf[i] = portion.buf[i]/portion.buf[n-1]*RAND_MAX;
                rr = random(); while(rr>portion.buf[n-1]) rr = random();
#endif
                int pos = 0;
                while(pos<n && portion_backup.buf[pos]<rr) pos++;
                centers.buf[added]=pos;
                for(int i=n-1;i>0;i--) portion.buf[i] -= portion.buf[i-1];
            }
            added++;
        }
    }

    std::fill(counts,counts+k,0);
    for(int i=0;i<n;i++) counts[labels[i]]++;
}

int Histogram_Kernel_Kmeans::KernelKmeans()
{
    if(verbose) std::cout<<"Initializing the histogram kernel k-means algorithm."<<std::endl;
    StartOfDuration();
    InitKmeans(labelsInitialized);
    if(verbose) std::cout<<"Initialization done for "<<data.nrow<<" features. Using "<<EndOfDuration()/1000<<" seconds."<<std::endl;
    // Initial matrix for fast evaluation of histogram kernel values
    StartOfDuration();
    eval.Create(m*upper_bound,k);
    double error = BuildFastKernel();
    if(verbose)
    {
        std::cout<<"Initial conditions: "<<emptycenters<<" empty clusters, error = "; std::cout.precision(12); std::cout<<error<<std::endl;
        std::cout<<"Initial fast evaluation matrix and constants computed. Using "<<EndOfDuration()/1000<<" seconds."<<std::endl;
    }

    double olderror;
    for(int round=1;round<=max_iteration;round++)
    {
        if(verbose) std::cout<<"Starting Round "<<round<<":"<<std::endl;
        StartOfDuration();
        olderror = error;
        std::fill(counts,counts+k,0);
#pragma omp parallel for
        for(int i=0;i<n;i++) labels[i] = Histogram_Find_Nearest(data.p[i],eval,m,validcenters,upper_bound,rho);
        std::fill(counts,counts+k,0);
        for(int i=0;i<n;i++) counts[labels[i]]++;
        if(verbose) std::cout<<"  Labels assigned in "<<EndOfDuration()/1000<<" seconds."<<std::endl;
        error = BuildFastKernel();
        if(verbose) std::cout<<"  Error = "<<error<<", with "<<emptycenters<<" empty clusters."<<std::endl;
        assert(error<=olderror);
        if(olderror-error<t) break;
    }
    return validcenters;
}

void Histogram_Kernel_Kmeans::RemoveEmptyClusters()
{
    Array2dC<int> mapping(1,k);
    int added = 0;
    for(int i=0;i<k;i++)
    {
        if(counts[i]==0) continue;
        if(added!=i)
        {
            for(int j=0;j<eval.nrow;j++) eval.p[j][added] = eval.p[j][i];
            counts[added] = counts[i];
            rho[added] = rho[i];
        }
        mapping.buf[i]=added;
        added++;
    }
    assert(added==validcenters);
    for(int i=validcenters;i<k;i++)
    {
        for(int j=0;j<eval.nrow;j++) eval.p[j][i] = 0;
        counts[i] = 0;
        rho[i] = -1e200;
    }
    for(int i=0;i<n;i++) labels[i] = mapping.buf[labels[i]];
}

int Histogram_Kernel_Kmeans::OneClassSVM()
{
    if(verbose) std::cout<<"  Training SVM for cluster : ";
#pragma omp parallel for
    for(int i=0;i<k;i++)
    {
        if(verbose) { std::cout<<i+1<<"."; std::cout.flush(); }
        if(counts[i]==0)
        {
            rho[i] = -(1e200);
            for(int j=0;j<eval.nrow;j++) eval.p[j][i]=0;
        }
        else
        {
            svm_parameter param;
            svm_problem prob;
            svm_model *model = NULL;
            svm_node *x_space = NULL;
            std::set<int> choose;

            UseSVM_Init(param,prob,x_space);
            param.svm_type = ONE_CLASS;
            param.kernel_type = 5; // '5' means histogram kernel
            param.nu = 0.2;
            choose.clear(); choose.insert(i); // only use points from class (cluster) 'i' -- so a one-class problem
            UseSVM_BuildProblem<int>(data,labels,choose,prob,x_space,true,5000);
            model = svm_train(&prob,&param);
            rho[i] = UseSVM_Histogram_FastEvaluationStructure(*model,m,upper_bound,eval,i);
            UseSVM_CleanUp(model,param,prob,x_space);
        }
    }
    if(verbose) std::cout<<std::endl;
    if(verbose)
    {
#pragma omp parallel for
        for(int i=0;i<n;i++) labels[i] = Histogram_Find_Nearest(data.p[i],eval,m,validcenters,upper_bound,rho);
        std::fill(counts,counts+k,0);
        for(int i=0;i<n;i++) counts[labels[i]]++;
        std::cout<<"  Error after 1-class SVM = "<<ComputeConstants(false)<<std::endl;
    }
    return validcenters;
}

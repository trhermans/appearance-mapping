// Author: Jianxin Wu (wujx2001@gmail.com)

#if defined ( _WIN32 )
// Mircosoft is stupid, I have to put this line first to make sure cstdlib is not included before this macro
#define _CRT_RAND_S
#include <cstdlib>
#endif

#include <algorithm>
#include <numeric>
#include <set>
#include <cmath>
#include <climits>
#include <cfloat>

#include "LinearKernel.h"
#include "UseSVM.h"
#include "mdarray.h"
#include "util.h"

#include "svm.h"

Linear_Kernel_Kmeans::Linear_Kernel_Kmeans(const Array2d<double>& _data,
                                           const int _k, const double _t,
                                           const int _max_iteration,
                                           const bool _labelsInitialized,
                                           int* _labels,int* _counts,
                                           Array2d<double>& _eval,double* _rho,
                                           const bool _useMedian,
                                           const bool _verbose) :
    data(_data), k(_k), t(_t), max_iteration(_max_iteration),
    labelsInitialized(_labelsInitialized), labels(_labels), counts(_counts),
    eval(_eval), rho(_rho), useMedian(_useMedian), verbose(_verbose),
    n(_data.nrow), m(_data.ncol), emptycenters(0), validcenters(k)
{
    assert(n>=k && k>0);
    x_square = new double[n]; assert(x_square!=NULL);
    w_square = new double[n]; assert(w_square!=NULL);
    for(int i=0;i<n;i++) x_square[i] = std::inner_product(data.p[i],data.p[i]+m,
                                                          data.p[i],0.0);
}

Linear_Kernel_Kmeans::~Linear_Kernel_Kmeans()
{
    delete[] x_square; x_square = NULL;
    delete[] w_square; w_square = NULL;
}

void Linear_Kernel_Kmeans::Save(const char* filename)
{
    std::ofstream out(filename);
    if(useMedian==false)
        out<<"kmeans"<<std::endl;
    else
        out<<"kmedian"<<std::endl;
    out<<"m "<<m<<std::endl;
    out<<"k "<<validcenters<<std::endl;
    out<<"u "<<-1<<std::endl;
    out<<"r"<<std::endl;
    for(int i=0;i<validcenters;i++) out<<rho[i]<<' '; out<<std::endl;
    for(int i=0;i<validcenters;i++)
    {
        for(int j=0;j<m;j++)
        {
            out<<eval.p[i][j]<<' ';
        }
        out<<std::endl;
    }
    out.close();
}

double Linear_Kernel_Kmeans::BuildFastKernel()
{   // Compute the cluster centers, which are the average of its data points
    emptycenters=0;
    eval.Zero();
    if(useMedian==false)
    {
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                eval.p[labels[i]][j] += data.p[i][j];
        for(int i=0;i<k;i++)
        {
            if(counts[i]>0)
            {
                double r = 1.0/counts[i];
                for(int j=0;j<m;j++) eval.p[i][j] *= r;
            }
            else
                emptycenters++;
        }
        validcenters = k - emptycenters;
        RemoveEmptyClusters();
    }
    else
    {
        Array2dC<double> t(k,*std::max_element(counts,counts+k));
        Array2dC<int> added(1,k);
        for(int i=0;i<m;i++) // each dimension
        {
            added.Zero();
            for(int j=0;j<n;j++)
            {
                t.p[labels[j]][added.buf[labels[j]]] = data.p[j][i];
                added.buf[labels[j]]++;
            }
            for(int j=0;j<k;j++)
            {
                assert(added.buf[j]==counts[j]);
                std::sort(t.p[j],t.p[j]+added.buf[j]);
                if(added.buf[j]%2==0)
                    eval.p[j][i] = (t.p[j][added.buf[j]/2-1]+t.p[j][added.buf[j]/2])/2.0;
                else
                    eval.p[j][i] = t.p[j][added.buf[j]/2];
            }
        }
    }
    return ComputeConstants();
}

double Linear_Kernel_Kmeans::ComputeConstants(const bool changeRho)
{   // compute error of the current clusetering, and compute norm of cluster center for each cluster
    double error = 0;
    if(useMedian==false)
    {
        std::fill_n(w_square,k,0.0);
        Array2dC<double> e(1,n);
        for(int i=0;i<k;i++)
            if(counts[i]>0)
                w_square[i] = std::inner_product(eval.p[i],eval.p[i]+m,eval.p[i],0.0);
            else
                w_square[i] = 1e200;
        for(int i=0;i<n;i++)
        {
            double t = std::inner_product(data.p[i],data.p[i]+m,eval.p[labels[i]],0.0);;
            e.buf[i] = -2*t;
            error += t;
        }
        if(changeRho) for(int i=0;i<k;i++) rho[i] = -w_square[i]*0.5;

        error *= -2;
        for(int i=0;i<n;i++) error += (w_square[labels[i]] + x_square[i]);
        error /= n;

        for(int i=0;i<n;i++) e.buf[i]=sqrt(e.buf[i]+w_square[labels[i]]+x_square[i]);
        std::nth_element(e.buf,e.buf+n/2,e.buf);
        //if(verbose) std::cout<<" Average L1 = "<<std::accumulate(e.buf,e.buf+n,0.0)/n<<", median L1 = "<<e.buf[n/2]<<std::endl;
    }
    else
    {   // for k-median
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++) error += fabs(data.p[i][j]-eval.p[labels[i]][j]);
        error /= n;
    }

    return error;
}

int Linear_Find_Nearest(const double* f,const Array2d<double>& eval,const int m,const int k,const double* rho,const bool useMedian)
{
    double* values = new double[k];
    if(useMedian==false)
    {
        for(int i=0;i<k;i++) values[i] = std::inner_product(f,f+m,eval.p[i],rho[i]);
    }
    else
    {
        for(int i=0;i<k;i++)
        {
            values[i] = 0;
            for(int j=0;j<m;j++) values[i] -= fabs(eval.p[i][j]-f[j]);
        }
    }
    int v = int(std::max_element(values,values+k) - values);
	delete[] values;
	return v;
}

int Linear_Find_Nearest(const int* f,const Array2d<double>& eval,const int m,const int k,const double* rho,const bool useMedian)
{
    double* values = new double[k];
    if(useMedian==false)
    {
        for(int i=0;i<k;i++)
        {
            values[i] = rho[i];
            for(int j=0;j<m;j++) values[i] += f[j]*eval.p[i][j];
        }
    }
    else
    {
        for(int i=0;i<k;i++)
        {
            values[i] = 0;
            for(int j=0;j<m;j++) values[i] -= fabs(eval.p[i][j]-f[j]);
        }
    }
    int v = int(std::max_element(values,values+k) - values);
	delete[] values;
	return v;
}

void Linear_Kernel_Kmeans::InitKmeans(const bool labelsInitialized)
{
    if(labelsInitialized==false)
    {   // kmeans++ initialization, details see the kmeans++ paper
        Array2dC<int> centers(1,k); // index for initial centers
        Array2dC<double> portion(1,n);
		Array2dC<double> portion_backup(1,n);
        std::fill_n(portion.buf,n,DBL_MAX);
        int added = 0;
#if defined ( _WIN32 )
		unsigned int rr;
        rand_s(&rr); while(rr>=n) rand_s(&rr);
#else
        int rr;
        rr = random(); while(rr>=n) rr=random();
#endif
        centers.buf[0] = rr;
        std::fill_n(labels,n,0);
        added = 1;
        while(added<=k)
        {
#pragma omp parallel for
            for(int i=0;i<n;i++)
            {
                double v;
                if(useMedian==false)
                    v = x_square[i] + x_square[centers.buf[added-1]] - 2*std::inner_product(data.p[i],data.p[i]+m,data.p[centers.buf[added-1]],0.0);
                else
                {
                    v = 0;
                    for(int j=0;j<m;j++) v += fabs(data.p[i][j]-data.p[centers.buf[added-1]][j]);
                }
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
                rand_s(&rr); while(rr>portion_backup.buf[n-1]) rand_s(&rr);
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

    std::fill_n(counts,k,0);
    for(int i=0;i<n;i++) counts[labels[i]]++;
}

int Linear_Kernel_Kmeans::KernelKmeans()
{
    if(verbose) std::cout<< "Initializing the linear kernel k-means algorithm."
                         << std::endl;
    StartOfDuration();
    InitKmeans(labelsInitialized);
    if(verbose) std::cout<< "Initialization done for " << data.nrow
                         << " features. Using " << EndOfDuration()/1000
                         <<" seconds."<< std::endl;
    // Initial fast evaluation of linear kernel values
    StartOfDuration();
    eval.Create(k,m);
    double error = BuildFastKernel();
    if(verbose)
    {
        std::cout << "Initial conditions: " << emptycenters
                  << " empty clusters, error = ";
        std::cout.precision(12);
        std::cout << error << std::endl;
        std::cout
            << "Initial fast evaluation matrix and constants computed. Using "
            << EndOfDuration()/1000 << " seconds." << std::endl;
    }

    double olderror;
    for(int round=1;round<=max_iteration;round++)
    {
        if(verbose) std::cout<<"Starting Round "<<round<<":"<<std::endl;
        StartOfDuration();
        olderror = error;
#pragma omp parallel for
        for(int i=0;i<n;i++) labels[i] = Linear_Find_Nearest(data.p[i],eval,m,k,rho,useMedian);
        std::fill_n(counts,k,0);
        for(int i=0;i<n;i++) counts[labels[i]]++;
        if(verbose) std::cout<<"  Labels assigned in "<<EndOfDuration()/1000<<" seconds."<<std::endl;
        error = BuildFastKernel();
        if(verbose) std::cout<<"  Error = "<<error<<", with "<<emptycenters<<" empty clusters."<<std::endl;
        assert(error<=olderror);
        if(olderror-error<t) break;
    }
    return validcenters;
}

void Linear_Kernel_Kmeans::RemoveEmptyClusters()
{
    Array2dC<int> mapping(1,k);
    int added = 0;
    for(int i=0;i<k;i++)
    {
        if(counts[i]==0) continue;
        if(added!=i)
        {
            std::copy(eval.p[i],eval.p[i]+m,eval.p[added]);
            counts[added] = counts[i];
            rho[added] = rho[i];
        }
        mapping.buf[i]=added;
        added++;
    }
    assert(added==validcenters);
    for(int i=validcenters;i<k;i++)
    {
        std::fill_n(eval.p[i],m,0.0);
        counts[i] = 0;
        rho[i] = -1e200;
    }
    for(int i=0;i<n;i++) labels[i] = mapping.buf[labels[i]];
}

int Linear_Kernel_Kmeans::OneClassSVM()
{
    if(verbose) std::cout<<"  Training SVM for cluster : ";
// #pragma omp parallel for num_threads(2)
    // NOTE: set num_threads = 2 or comment out to avoid memory overflow for
    // large datasets
    for(int i=0;i<k;i++)
    {
        if(verbose) { std::cout<<i+1<<"."; std::cout.flush(); }
        if(counts[i]==0)
        {
            rho[i] = -1e200;
            std::fill_n(eval.p[i],m,0.0);
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
            param.kernel_type = 0; // '0' -- linear kernel
            param.nu = 0.2;
            choose.clear(); choose.insert(i); // only use points from cluster 'i', so a one-class problem
            UseSVM_BuildProblem<double>(data,labels,choose,prob,x_space,true,5000);
            model = svm_train(&prob,&param);
            rho[i] = UseSVM_Linear_FastEvaluationStructure(*model,m,eval,i);
            UseSVM_CleanUp(model,param,prob,x_space);
        }
    }
    if(verbose) std::cout<<std::endl;
    if(verbose)
    {
#pragma omp parallel for
        for(int i=0;i<n;i++) labels[i] = Linear_Find_Nearest(data.p[i],eval,m,k,rho,useMedian);
        std::fill_n(counts,k,0);
        for(int i=0;i<n;i++) counts[labels[i]]++;
        std::cout<<"  Error after 1-class SVM = "<<ComputeConstants(false)<<std::endl;
    }
    return validcenters;
}

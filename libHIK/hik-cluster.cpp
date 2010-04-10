// Author: Jianxin Wu (wujx2001@gmail.com)
// This file performs clustering using histogram intersection kernel
// This is also an example of how to use the API in Cluster.h, etc
// Although the name is hik-cluster, k-means and k-median also supported

#include <string>

#include "Cluster.h"

void usage()
{
    std::cout<<"hik-cluster [options] train_filename [test_filename]"<<std::endl;
    std::cout<<"train_filename will be used to learn the clustering."<<std::endl;
    std::cout<<"Cluster results will be shown for test_filename if it exists."<<std::endl;
    std::cout<<"Options:"<<std::endl;
    std::cout<<"-t 1|2|3: 1 for HIK, 2 for k-means, 3 for k-median."<<std::endl;
    std::cout<<"-u v: only for HIK, set upper bound to v."<<std::endl;
    std::cout<<"-k v: generate v cluster centers."<<std::endl;
    std::cout<<"-m v: clustering will run at most m iterations."<<std::endl;
    std::cout<<"-o 0|1: set to 0 if not using one class svm."<<std::endl;
    std::cout<<"-h: print this message."<<std::endl;
    std::cout<<"Default options: -t 1 -k 200 -m 10 -o 0"<<std::endl;
    std::cout<<"Cluster results will be saved as train_filename.clusters."<<std::endl;
    exit(-1);
}

int main(int argc, char **argv)
{
    if(argc<2) usage();
    
    int type,k,upper_bound,max_iteration;
    bool oneclass_svm;
    int v;
    int i;
	// parse options
    type = 1;
    k = 200;
    max_iteration = 10;
    oneclass_svm = false;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 't':
				v = atoi(argv[i]);
                if(v<=0 || v>3)
                {
                    std::cout<<"The specified clustering method is not correct."<<std::endl;
                    exit(-1);
                }
                type = v;
				break;
            case 'u':
                // upper bound of feature values (all feature valeus should be less than (not equal to) upper_bound
                upper_bound = atoi(argv[i]);
                break;
            case 'k':
                k = atoi(argv[i]);
                break;
            case 'm':
                max_iteration = atoi(argv[i]);
                break;
            case 'o':
                v = atoi(argv[i]);
                if(v!=0) oneclass_svm = true;
                break;
			default:
				std::cout<<"Unknown option:"<<argv[i-1][1]<<std::endl;
				usage();
		}
	}
    if(type==1 && upper_bound<0)
    {
        std::cout<<"No upper bound specified for HIK clustering."<<std::endl;
        exit(-1);
    }
    if(upper_bound>1024)
        std::cout<<"You specified a large upper bound, be warned of insufficient memory."<<std::endl;
    if(k<0)
    {
        std::cout<<"You must specify a valid number of clusters."<<std::endl;
        exit(-1);
    }
	
    std::string input_name(argv[i]);
    if(FileExists(input_name.c_str())==false)
    {
        std::cout<<"The input file does not exist."<<std::endl;
        exit(-1);
    }

    ClusterModel cm;
    std::string model_name = input_name;
    model_name += ".clusters";
    if(type==1)
    {
        cm.HIK_Clustering(input_name.c_str(),k,max_iteration,upper_bound,oneclass_svm,true,model_name.c_str());
    }
    else if(type==2)
    {
        cm.Kmeans_Clustering(input_name.c_str(),k,max_iteration,false,oneclass_svm,true,model_name.c_str());
    }
    else
    {
        cm.Kmeans_Clustering(input_name.c_str(),k,max_iteration,true,oneclass_svm,true,model_name.c_str());
    }
    std::cout<<"Clustering done."<<std::endl;
    
    if(argc-1>i)
    {
        std::cout<<"Testing."<<std::endl;
        std::string test_name = argv[i+1];
        std::string test_out_name = test_name+".out";
        if(FileExists(test_name.c_str())==false)
        {
            std::cout<<"The testing file does not exist."<<std::endl;
            exit(-1);
        }
        std::ifstream in(test_name.c_str());
        std::ofstream out(test_out_name.c_str());
        int n2,m2,upper_bound2;
        in>>n2>>m2;
        // Use double for HIK is not very inefficient, but this is just an example.
        // In your own experiments, use either Array2dC<int> or Array2dC<double> when approapriately
        Array2dC<double> f(1,m2);
        for(int index=0;index<n2;index++)
        {
            for(int j=0;j<m2;j++) in>>f.buf[j];
            int v = cm.Map(f.buf);
            out<<v<<std::endl;
        }
        in.close();
        out.close();
        std::cout<<"Applying the cluster model to file"<<test_name<<", results in "<<test_out_name<<std::endl;
    }

	return 0;
}

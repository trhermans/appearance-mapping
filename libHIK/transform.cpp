// Author: Jianxin Wu (wujx2001@gmail.com)


// _____________________________________________________________________
// NOTE that the codes in between these two comments are extracted from
// libsvm for reading a problem in SVM format
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "libsvm-2.89/svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_problem prob;		// set by read_problem
struct svm_node *x_space;

static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in svmlight format)
// Changed by Jianxin Wu, to return number of elements

int read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t");
		prob.y[i] = strtod(label,&endptr);
		if(endptr == label)
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

// Removed by Jianxin Wu
	//if(param.gamma == 0 && max_index > 0)
		//param.gamma = 1.0/max_index;

// Removed by Jianxin Wu, we do not support pre-computed kernel
	//if(param.kernel_type == PRECOMPUTED)
		//for(i=0;i<prob.l;i++)
		//{
			//if (prob.x[i][0].index != 0)
			//{
				//fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				//exit(1);
			//}
			//if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			//{
				//fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				//exit(1);
			//}
		//}

	fclose(fp);
    
    return elements;
}
// _____________________________________________________________________

// Well I know people say do not use C and C++ i/o functions together
// But I am so attached to C++ stream I/O
// Also, change "const std::string prefix" to set where you want the output to be

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

void TransformOneFile(const char* train_filename,const char* test_filename,const int max_allowed)
{
    // Step 0: Read the inptu file
    const int elements = read_problem(train_filename);
    // Step 1: find the single maximum feature value
    double maxvalue = -1; // assuming non-negative feature values
    for(int i=0;i<elements;i++)
    {
        if(x_space[i].index==-1) continue;
        if(x_space[i].value<0)
        {
            std::cout<<"Non-negative feature values are required."<<std::endl;
            std::cout<<"Value = "<<x_space[i].value<<" is invalid."<<std::endl;
            exit(-1);
        }
        if(x_space[i].value>maxvalue) maxvalue = x_space[i].value;
    }
    // Step 2: build a histogram of these values
    const int bin_count = 1000;
    double bin_step = maxvalue/(bin_count-0.01);
    std::vector<int> histogram;
    histogram.resize(bin_count);
    std::fill(histogram.begin(),histogram.end(),(int)0);
    for(int i=0;i<elements;i++)
    {
        if(x_space[i].index==-1) continue;
        histogram[int(x_space[i].value/bin_step)]++;
    }
    // Step 3: Find the 97.5% percentile
    const int stopvalue = int(0.975*std::accumulate(histogram.begin(),histogram.end(),(int)0));
    int p = 0;
    int sum = histogram[0];
    while(sum<stopvalue)
    {
        p++;
        sum += histogram[p];
    }
    const double threshold = p*maxvalue/bin_count;
    const double threshold_step = threshold / (max_allowed-0.01);
    // Step 4: Process the output for train_filename
    std::string outname = train_filename;
    outname += ".int";
    std::ofstream out(outname.c_str());
    svm_node* pn;
    int v;
    for(int i=0;i<prob.l;i++)
    {
        out<<prob.y[i]<<' ';
        pn = prob.x[i];
        while(pn->index!=-1)
        {
            if(pn->value>=threshold) pn->value = threshold;
            v = int(pn->value/threshold_step);
            if(v) out<<pn->index<<':'<<v<<' ';
            // NOTE that value will be between 0 and max_allowed-1
            pn++;
        }
        out<<std::endl;
    }
    out.close();
    free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
    std::cout<<"Training file "<<train_filename<<" processed."<<std::endl;
    // Step 4: Process the output for test_filename
    if(strcmp(train_filename,test_filename)!=0)
    {
        read_problem(test_filename);
        outname = test_filename; outname += ".int";
        std::ofstream out2(outname.c_str());
        for(int i=0;i<prob.l;i++)
        {
            out2<<prob.y[i]<<' ';
            pn = prob.x[i];
            while(pn->index!=-1)
            {
                if(pn->value>=threshold) pn->value = threshold;
                v = int(pn->value/threshold_step);
                if(v) out2<<pn->index<<':'<<v<<' ';
                pn++;
            }
            out2<<std::endl;
        }
        out2.close();
        free(prob.y);
        free(prob.x);
        free(x_space);
        free(line);
        std::cout<<"Testing file "<<test_filename<<" processed."<<std::endl;
    }
}

int main(int argc,char* argv[])
{
    int max_allowed;
    
    if(argc<3 || argc>4)
    {
        std::cout<<"Usage: transform train_filename test_filename [upper_bound]"<<std::endl;
        std::cout<<"    where upper_bound is the maximum allowed integer feature value."<<std::endl;
        std::cout<<"    Output will be in train_filename.int and test_filename_int"<<std::endl;
        std::cout<<"    We only use training set to determine range and do transformation."<<std::endl;
        std::cout<<"    If the two file names are the same, only train_filename.int are generated."<<std::endl;
        exit(-1);
    }
    if(argc==3)
    {
        max_allowed = 128;
    }
    else
    {
        max_allowed = atoi(argv[3]);
        if(max_allowed<=0)
        {
            std::cout<<"upper bound is invalid, set to default value 128."<<std::endl;
            max_allowed = 128;
        }
        else if(max_allowed>1024)
        {
            std::cout<<"Warning: upper bound is too large (>1024)."<<std::endl;
            std::cout<<"You may not have enough memory (and may not have high accuracy."<<std::endl;
        }
    }
    
    TransformOneFile(argv[1],argv[2],max_allowed);
    return 0;
}

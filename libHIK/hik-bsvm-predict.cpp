// Revised from hik-predict.cpp
// Author: Jianxin Wu (wujx2001@gmail.com)

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"

#include "mdarray.h"
#include "UseSVM.h"
#include "hik_svm.h"

struct svm_inode *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;

int upper_bound = -1;
int m = -1;

// We want to compute the average accuracy across categories,
// i.e. avarage of diagonal entries in the confusion matrix
// This is a helper function for this purpose, by Jianxin Wu
int FindLabel(const int v,const int* labels)
{
	int i;
	for(i=0;i<svm_get_nr_class(model);i++)
		if(labels[i]==v)
			return i;
    printf("There are error(s) in the data (unknown category name).");
    exit(-1);
}

static char *line = NULL;
static int max_line_len;

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

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output,Array2dC<double>& eval,const int m,const int upper_bound)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	int j;
    
    // This block by Jianxin Wu, for average accuracy computation
    int ii,label_index;
    // number of correct predictions in each category
	int* correct_sub = (int *)malloc(nr_class*sizeof(int));
	for(ii=0;ii<nr_class;ii++) correct_sub[ii] = 0;
    // number of testing examples in each category
	int* total_sub = (int *)malloc(nr_class*sizeof(int));
	for(ii=0;ii<nr_class;ii++) total_sub[ii] = 0;
	int* labels_avg = (int*)malloc(nr_class*sizeof(int));
	svm_get_labels(model,labels_avg);

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        double tempdouble;

		label = strtok(line," \t");
		target_label = strtod(label,&endptr);
		if(endptr == label)
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_inode *) realloc(x,max_nr_attr*sizeof(struct svm_inode));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
            tempdouble = strtod(val,&endptr);
			x[i].value = int(tempdouble);
            if(x[i].value<0)
                x[i].value = 0;
            else if(x[i].value>=upper_bound)
                x[i].value = upper_bound - 1;
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;
        if(inst_max_index>m)
        {
            fprintf(stderr,"Error: feature indexes larger than m is not allowed.\n");
            exit(-1);
        }

		predict_label = hik_bsvm_predict(model,eval,m,upper_bound,x);
		fprintf(output,"%g\n",predict_label);

        // This block by Jianxin Wu, for average accuracy
        label_index = FindLabel((int)target_label,labels_avg);
		total_sub[label_index]++;
		if(predict_label == target_label) correct_sub[label_index]++;

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	printf("Accuracy = %g%% (%d/%d) (classification)\n",(double)correct/total*100,correct,total);
        
    // This block (till endo of function) by Jianxin WU
    // Print per-category accuracy and average accuracy of categories
    double sub_score = 0;
    int nonempty_category = 0;
	for(ii=0;ii<nr_class;ii++)
	{
		if(total_sub[ii]>0)
        {
            sub_score += (correct_sub[ii]*1.0/total_sub[ii]);
            nonempty_category++;
        }
	}
    printf("-----------\n");
    for(ii=0;ii<nr_class;ii++)
    {
        printf("%d / %d (Category %d)\n",correct_sub[ii],total_sub[ii],labels_avg[ii]);
    }
    printf("-----------\n");
	printf("Mean Accuray across classes = %g%%\n",sub_score*100.0/nonempty_category);
	free(correct_sub);
	free(total_sub);
	free(labels_avg);

}

void exit_with_help()
{
	printf(
	"Usage: hik-bsvm-predict -D dimension -u upper_bound test_file model_file output_file\n"
	"options:\n"
    "-D dimension of the feature vector\n"
    "-u the maximum possible feature index (minimum is 1) should be upper_bound-1"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
    int i;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
            case 'u':
                // upper bound of feature values (all feature valeus should be less than (not equal to) upper_bound
                upper_bound = atoi(argv[i]);
                break;
            case 'D':
                // dimension of the feature vector
                m = atoi(argv[i]);
                break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	if(i>=argc-2) exit_with_help();
    if(upper_bound<0) fprintf(stderr,"No upper bound specified, set to default (128).\n");
    if(upper_bound>1024) fprintf(stderr,"You specified a large upper bound, be warned of insufficient memory.\n");
    if(m<0)
    {
        fprintf(stderr,"You must specify a valid number of dimension in your feature vector.\n");
        fprintf(stderr,"For example, use -D 128 in your options.\n");
        exit(-1);
    }

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

    Array2dC<double> eval;
    UseSVM_CS_Histogram_FastEvaluationStructure(argv[i+1],m,upper_bound,eval,&model);
    free((void *)(model->SV[0]));
    model->free_sv = 0;
    printf("SVM model loaded.\n");
    
    if(model->param.svm_type!=5)
    {
        fprintf(stderr,"This program only support model trained BSVM Crammer-Singer formulation.\n");
        exit(-1);
    }
    if(model->param.kernel_type!=5)
    {
        fprintf(stderr,"This programm only support histogram intersection kernel.\n");
        exit(-1);
    }

	x = (struct svm_inode *) malloc(max_nr_attr*sizeof(struct svm_inode));
	predict(input,output,eval,m,upper_bound);
	svm_destroy_model(model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}

// Author: Jianxin Wu (wujx2001@gmail.com)

#ifndef __DATASETS_H__
#define __DATASETS_H__

const int DATASET_Caltech101 = 0; // Caltech 101 data set
const int DATASET_Scene15 = 1; // 15 natural scene category data set
const int DATASET_Event8 = 2; // Princeton 8 sports events data set
const int DATASET_JAFFE = 3; // Japanese Female Facial Expressions set
const int DATASET_ROVIO = 4; // Images from the rovio used for affordance learning
const int DATASET_CITY_CENTRE = 5;
const int DATASET_NEW_COLLEGE = 6;

const int CODEBOOK_Separate = 0; // generate codebooks from every training set in all cross validation splittings
const int CODEBOOK_Single = 1; // only generate 1 codebook for all the cross validation splittings

// generate training/testing data for 'dataset' -- Caltech 101, 15 scene, etc
// with 'multiplicity' -- whether 'CODEBOOK_Separate' or 'CODEBOOK_Single'
// using 'featuretype' -- CENTRIST? SIFT? or, ...
// 'codebooktype' -- linear clustering or histogram kernel k-means clustering? or ...
void GenerateDatasets(const int dataset,const int multiplicity,
                      const CodeBook::DESCRIPTOR_TYPE featuretype,
                      const CodeBook::CODEBOOK_TYPE codebooktype);

void TranslateImages(const int dataset,const int featuretype,
                     const int codebooktype);

#endif // __DATASETS_H__

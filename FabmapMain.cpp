// author: Richard

//#include <map>
//#include <vector>
//#include <iostream>
//#include <algorithm>

#include "fabmap/ChowLiuTree.h"
#include "fabmap/DetectorModel.h"
#include "fabmap/FABMAP.h"
#include "fabmap/Timer.h"

int main()
{
	// 1. initialize the fabmap algorithm
	// constructor parameters:
	// sigma is the smoothing factor (see equation (18)) - worked pretty bad, messed with the probabilities, I always set it to 1 (i.e. disabled)
	// numberOfSamples is the number of samples for the unknown place during the calculation of the normalization term (n_s in the paper in equation (17) in chapter 4.3.2)
	// approximationModel is one of the elements in enum ObservationLikelihoodModel (NAIVEBAYES, CHOWLIU)
	// falsePositiveProbability, falseNegativeProbability - the detector model probabilities (see detector model)
	// codebookFile is the file containing the codebook
	// trainingDataFile is the file name of the file with training data (vector<vector<int>>), if this parameter is "" you must provide a modelFile
	// modelFile - if the model was already generated, it can be loaded from a model file, if it does not exist already, it will be written into that file for later use
	// loadModel - if true, then the model will be loaded from file modelFile, if false, the model will be saved in file modelFile

	// -> either train a new model (further parameter infos in FABMAP.h)
	TheTimer.start();
//	FABMAP fm(1.0, 500, FABMAP::CHOWLIU, 0.0, 0.39, "_City_Centre_c10000.codebook", "_training_data_cbCityCentre_c10000.txt", "_ChowLiuTree_cbCityCentre_c10000.txt", false);
	FABMAP fm(1.0, 500, FABMAP::CHOWLIU, 0.0, 0.39, "_City_Centre_c100.codebook", "_training_data_cbCityCentre_c100.txt", "_ChowLiuTree_cbCityCentre_c100.txt", false);
	std::cout << "ChowLiu tree learning needed " << TheTimer.getRuntime() << "s." << std::endl;

	// -> or load an existing model from file (if the first option was run once before)
	//FABMAP fm(1.0, 500, FABMAP::CHOWLIU, 0.0, 0.39, "_City_Centre_c10000.codebook", "training_data_cbCityCentre_c10000.txt", "_ChowLiuTree_cbCityCentre_c10000.txt", true);  //0.99999999999999999999

	// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// 2. run the fabmap algorithm (do loop closure detection)
	
	TheTimer.start();

	// -> either you provide the path where all images can be found + the number of images to process + the probability at which a loop closure is accepted
	//fm.onlineApplication("../../../data/NewCollege_Images/Images/", 200, 0.99);

	// -> or you load the pre-built histogram data from file (be careful to build it from the test image set with the training set word clusters!),
	//    then provide the file with the histogram data + -1 (indicates pre-built file to the function) + the probability at which a loop closure is accepted
//	fm.onlineApplication("_test_data_NewCollege_cbCityCentre_c10000.txt", -1, 0.99);
	fm.onlineApplication("_test_data_NewCollege_cbCityCentre_c100.txt", -1, 0.99);

	std::cout << "Fabmap needed " << TheTimer.getRuntime() << "s." << std::endl;


	getchar();
	return 0;
}




// Tests (do not mind)

	//Test 1
	//int vals[] = {1,2,3,4,5,6,7,8};
	//std::vector<int> v(vals,vals+8);
	//std::make_heap(v.begin(), v.end());
	//std::cout << v.front() << "\n";
	//v[4] = 5;
	//std::make_heap(v.begin(), v.begin()+5);
	////std::make_heap(v.begin(), v.end());
	//std::cout << v.front();


	//Test 2
	//std::multimap<double, int> temp;
	//temp.insert(std::pair<double,int>(1.4, 0));
	//temp.insert(std::pair<double,int>(2.7, 1));
	//temp.insert(std::pair<double,int>(3.2, 2));
	//temp.insert(std::pair<double,int>(3.2, 3));
	//temp.insert(std::pair<double,int>(1.4, 4));
	//temp.insert(std::pair<double,int>(0.4, 5));

	//std::multimap<double,int>::iterator ittemp;
	//for (ittemp = temp.begin(); ittemp != temp.end(); ittemp++)
	//{
	//	std::cout << ittemp->second << " " << ittemp->first << "\n";
	//}

	//ittemp = temp.begin();
	//temp.erase(ittemp);

	//for (ittemp = temp.begin(); ittemp != temp.end(); ittemp++)
	//{
	//	std::cout << ittemp->second << " " << ittemp->first << "\n";
	//}

	//Test 3
	//std::vector< std::vector<int> > train;
	//std::vector<int> single;
	//single.push_back(0); single.push_back(0); single.push_back(0); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(0); single.push_back(1); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(1); single.push_back(0); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(0); single.push_back(0); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(0); single.push_back(1); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(1); single.push_back(1); single.push_back(0);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(1); single.push_back(0); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(0); single.push_back(1); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(0); single.push_back(0); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(1); single.push_back(0); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(0); single.push_back(1); single.push_back(1); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(0); single.push_back(1); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(1); single.push_back(0); single.push_back(1);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(1); single.push_back(1); single.push_back(0);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();
	//single.push_back(1); single.push_back(1); single.push_back(1); single.push_back(1);
	//train.push_back(single);
	//train.push_back(single);
	//single.clear();

	////ChowLiuTree CLT(train);

	//ChowLiuTree CLT("ChowLiuTree.txt");

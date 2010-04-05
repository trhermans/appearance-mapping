// author: Richard

//#include <map>
//#include <vector>
//#include <iostream>
//#include <algorithm>

#include "fabmap/ChowLiuTree.h"
#include "fabmap/DetectorModel.h"
#include "fabmap/FABMAP.h"

int main()
{
	// train a new model
	FABMAP fm(1.0, 20, FABMAP::CHOWLIU, 0.0, 0.39, "fabmap.codebook", "training_data.txt", "ChowLiuTree.txt");
	// or load an existing model from file
	//FABMAP fm(1.0, 20, FABMAP::CHOWLIU, 0.0, 0.39, "fabmap.codebook", "training_data.txt", "ChowLiuTree.txt");  //0.99999999999999999999

	fm.onlineApplication("../../../data/CityCentre_Images/Images/", 200);		//"../../../data/NewCollege_Images/Images/"

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

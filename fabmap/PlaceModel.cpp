// interface for the place appearance model p(e_i=1|L_j, Z^k) as defined in equation (19) chapter 4.3.5 in the paper
// author: Richard

#include "PlaceModel.h"

#include <iostream>

PlaceModel::PlaceModel()
{
	mLastLocation = -1;
}

PlaceModel::~PlaceModel()
{

}

double PlaceModel::getWordProbability(int ei, int es, int Lj)
{
	return mWordProbability[Lj][ei][es];
}

void PlaceModel::addLocation(std::vector<std::vector<double> >& marginalProbabilities)
{
	mWordProbability.push_back(marginalProbabilities);
	
	// add new location to the location adjacency list
	std::set<int> temp;
	int newLocationIndex = (int)mPlaceAdjancencyList.size();
	if (mLastLocation != -1)
	{
		temp.insert(mLastLocation);
		mPlaceAdjancencyList[mLastLocation].insert(newLocationIndex);
	}
	temp.insert(newLocationIndex);
	mPlaceAdjancencyList.push_back(temp);
}

void PlaceModel::updateLocation(std::vector<int> observations, int location, InterfaceDetectorModel *detectorModel, double p_Zk_Lj)
{
	//int attr=0;
	//for (ItPlaceModel = mWordProbability[location].begin(); ItPlaceModel != mWordProbability[location].end(); ItPlaceModel++, attr++)
	for (unsigned int attr=0; attr<mWordProbability[location].size(); attr++)
	{
		//double newValue0 = detectorModel->getDetectorProbability(observations[attr], 0) * mWordProbability[location][attr][0] / p_Zk_Lj;	// new value for e_i = 1
		//double newValue = detectorModel->getDetectorProbability(observations[attr], 1) * mWordProbability[location][attr][1] / p_Zk_Lj;	// new value for e_i = 1
		double newValue0 = detectorModel->getDetectorProbability(observations[attr], 0) * mWordProbability[location][attr][0];	// new value for e_i = 1
		double newValue = detectorModel->getDetectorProbability(observations[attr], 1) * mWordProbability[location][attr][1];	// new value for e_i = 1
		double sum=0.0;
		for (unsigned int s=0; s<mWordProbability[location][attr].size(); s++)
		{
			sum += (detectorModel->getDetectorProbability(observations[attr], s) * mWordProbability[location][attr][s]);
		}
		newValue0 /= sum;
		newValue /= sum;

		if (newValue>1.0)
		{
			newValue = 1.0;
			std::cout << "PlaceModel::updateLocation: newValue > 1 !\n";
		}
		//std::cout << "PlaceModel::updateLocation: Check sum of p(e_i=0 | L_j) and p(e_i=1 | L_j) = 1? " << newValue0+newValue << "\n";	//todo: check this sum

		mWordProbability[location][attr][0] = 1-newValue;
		mWordProbability[location][attr][1] = newValue;
	}

	// todo: this should only be done if location follows after mLastLocation, i.e. this function should only be called to update the identified current location (not all locations!) check that!
	if (mLastLocation != -1)
	{
		mPlaceAdjancencyList[mLastLocation].insert(location);
		mPlaceAdjancencyList[location].insert(mLastLocation);
	}
	mLastLocation = location;
}
// interface for the place appearance model p(e_i=1|L_j, Z^k) as defined in equation (19) chapter 4.3.5 in the paper
// author: Richard

#ifndef PLACEMODEL_H_
#define PLACEMODEL_H_

#include <vector>
#include <set>
#include "InterfacePlaceModel.h"
#include "InterfaceDetectorModel.h"

class PlaceModel : public InterfacePlaceModel
{
public:
	// 
	PlaceModel();
	~PlaceModel();

	// returns the place appearance model probability p(e_ei=es|L_j, Z^k) , i.e. the probability to have word e_ei=es on location L_j as in formula (19) in the paper
	double getWordProbability(int ei, int es, int Lj);

	// returns how many places are already established in the place model, i.e. how many places have been visited during operation
	int getNumberOfLocations() { return (int)mWordProbability.size(); };

	// returns the index of the location the robot has been the last time
	int getLastLocation() { return mLastLocation; };

	// returns true if location1 and location2 are adjacent places
	bool locationsAdjancent(int location1, int location2) { return (mPlaceAdjancencyList[location1].find(location2) != mPlaceAdjancencyList[location1].end()); };

	// returns the set of adjacent locations to location
	std::set<int>& getAdjacentLocations(int location) { return mPlaceAdjancencyList[location]; };

	// adds a new location, a new location is initialized with p(e_ei=es | L_j) = p(e_ei=es), these marginals have to be provided in marginalProbabilities[ei][es]
	void addLocation(std::vector<std::vector<double> >& marginalProbabilities);

	// updates the place model after an observation as defined in euqation (19),
	// observations is the vector of observations Z_k,
	// p_Zk_Lj = p(Z_k | L_j) which is determined by the ObservationLikelihood
	void updateLocation(std::vector<int> observations, int location, InterfaceDetectorModel* detectorModel, double p_Zk_Lj);

private:
	// stores the place appearance model probability p(e_ei=es|L_j, Z^k) , i.e. the probability to have word e_ei=es on location L_j as in formula (19) in the paper
	// mWordProbability[j][ei][es] = p(e_ei=es|L_j, Z^k)
	std::vector<std::vector<std::vector<double> > > mWordProbability;

	// the location the robot has been in during the time step before the current one
	int mLastLocation;

	// the list saves which places are adjacent to each other, i.e. mPlaceAdjacencyList[a] holds all indices of places which are adjacent to place a
	std::vector<std::set<int> > mPlaceAdjancencyList;
};

#endif /* PLACEMODEL_H_ */
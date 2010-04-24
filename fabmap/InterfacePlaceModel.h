// interface for the place appearance model p(e_i=1|L_j, Z^k) as defined in equation (19) chapter 4.3.5 in the paper
// author: Richard

#ifndef INTERFACEPLACEMODEL_H_
#define INTERFACEPLACEMODEL_H_

#include "InterfaceDetectorModel.h"
#include <vector>
#include <set>

class InterfacePlaceModel
{
public:
	// 
	InterfacePlaceModel() {};
	virtual ~InterfacePlaceModel() {};

	// returns the place appearance model probability p(e_ei=es|L_j, Z^k) , i.e. the probability to have word e_ei=es on location L_j as in formula (19) in the paper
	virtual double getWordProbability(int ei, int es, int Lj) = 0;

	// returns how many places are already established in the place model, i.e. how many places have been visited during operation
	virtual int getNumberOfLocations() = 0;

	// returns the index of the location the robot has been the last time
	virtual int getLastLocation() = 0;

	// returns true if location1 and location2 are adjacent places
	virtual bool locationsAdjancent(int location1, int location2) = 0;

	// returns the set of adjacent locations to location
	virtual std::set<int>& getAdjacentLocations(int location) = 0;

	// adds a new location, a new location is initialized with p(e_ei=es | L_j) = p(e_ei=es), these marginals have to be provided in marginalProbabilities[ei][es]
	virtual void addLocation(std::vector<std::vector<double> > marginalProbabilities) = 0;

	// updates the place model after an observation as defined in euqation (19),
	// observations is the vector of observations Z_k,
	// p_Zk_Lj = p(Z_k | L_j) which is determined by the ObservationLikelihood
	virtual void updateLocation(std::vector<int> observations, int location, InterfaceDetectorModel* detectorModel, double p_Zk_Lj) = 0;
};

#endif /* INTERFACEPLACEMODEL_H_ */
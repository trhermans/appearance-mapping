// implementation for the location prior p(L_i | Z^{k-1}) as defined in chapter 4.3.3 in the paper
// author: Richard

#ifndef SIMPLELOCATIONPRIOR_H_
#define SIMPLELOCATIONPRIOR_H_

#include "InterfacePlaceModel.h"
#include "InterfaceLocationPrior.h"

class SimpleLocationPrior : public InterfaceLocationPrior
{
public:
	// 
	SimpleLocationPrior();
	~SimpleLocationPrior();

	// returns the location prior p(L_i | Z^{k-1}) , i.e. the probability to be at location L_i, i = location, with the history of observations Z^{k-1}
	double getLocationPrior(int location, InterfacePlaceModel* placeModel);
};

#endif /* SIMPLELOCATIONPRIOR_H_ */
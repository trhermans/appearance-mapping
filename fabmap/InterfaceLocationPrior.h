// interface for the location prior p(L_i | Z^{k-1}) as defined in chapter 4.3.3 in the paper
// author: Richard

#ifndef INTERFACELOCATIONPRIOR_H_
#define INTERFACELOCATIONPRIOR_H_

#include "InterfacePlaceModel.h"

class InterfaceLocationPrior
{
public:
	// 
	InterfaceLocationPrior() {};
	virtual ~InterfaceLocationPrior() {};

	// returns the location prior p(L_i | Z^{k-1}) , i.e. the probability to be at location L_i, i = location, with the history of observations Z^{k-1}
	virtual double getLocationPrior(int location, InterfacePlaceModel* placeModel) = 0;
};

#endif /* INTERFACELOCATIONPRIOR_H_ */
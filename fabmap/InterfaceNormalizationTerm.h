// interface for the normalization term p(Z_k | Z^{k-1}) as defined in chapter 4.3.2 in the paper
// author: Richard

#ifndef INTERFACENORMALIZATIONTERM_H_
#define INTERFACENORMALIZATIONTERM_H_

#include "InterfaceObservationLikelihood.h"
#include "InterfaceLocationPrior.h"
#include "InterfacePlaceModel.h"
#include "InterfaceDetectorModel.h"
#include <vector>

class InterfaceNormalizationTerm
{
public:
	// 
	InterfaceNormalizationTerm() {};
	virtual ~InterfaceNormalizationTerm() {};

	// returns the normalization term p(Z_k | Z^{k-1}) as defined in chapter 4.3.2
	// p_Zk_Li contains the observation likelihoods p(Z_k | L_i) of all existing locations
	// the underlying observation likelihood model is used for sampling
	virtual double getNormalizationTerm(std::vector<double> p_Zk_Li, double& p_Zk_Lu, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior) = 0;
};

#endif /* INTERFACENORMALIZATIONTERM_H_ */
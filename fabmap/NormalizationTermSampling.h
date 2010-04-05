// implementation for the normalization term p(Z_k | Z^{k-1}) following the sampling method as defined in equation (17) in chapter 4.3.2 in the paper
// author: Richard

#ifndef NORMALIZATIONTERMSAMPLING_H_
#define NORMALIZATIONTERMSAMPLING_H_

#include "InterfaceObservationLikelihood.h"
#include "InterfaceLocationPrior.h"
#include "InterfacePlaceModel.h"
#include "InterfaceDetectorModel.h"
#include "InterfaceNormalizationTerm.h"
#include <vector>

class NormalizationTermSampling : public InterfaceNormalizationTerm
{
public:
	// 
	NormalizationTermSampling(int numberOfSamples);
	~NormalizationTermSampling();

	// returns the normalization term p(Z_k | Z^{k-1}) as defined in equation (17) in chapter 4.3.2
	// p_Zk_Li contains the observation likelihoods p(Z_k | L_i) of all existing locations
	// the underlying observation likelihood model is used for sampling
	double getNormalizationTerm(std::vector<double> p_Zk_Li, double& p_Zk_Lu, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior);

private:
	// defines how many samples should be taken during calculation of the normalization term
	int mNumberOfSamples;
};

#endif /* NORMALIZATIONTERMSAMPLING_H_ */
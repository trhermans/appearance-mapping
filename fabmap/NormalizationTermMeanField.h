// implementation for the normalization term p(Z_k | Z^{k-1}) following the sampling method as defined in equation (16) in chapter 4.3.2 in the paper
// author: Richard

#ifndef NORMALIZATIONTERMMEANFIELD_H_
#define NORMALIZATIONTERMMEANFIELD_H_

#include "InterfaceObservationLikelihood.h"
#include "InterfaceLocationPrior.h"
#include "InterfacePlaceModel.h"
#include "InterfaceDetectorModel.h"
#include "InterfaceNormalizationTerm.h"
#include <vector>
#include "math.h"

class NormalizationTermMeanField : public InterfaceNormalizationTerm
{
public:
	// 
	NormalizationTermMeanField();
	~NormalizationTermMeanField();

	// returns the normalization term p(Z_k | Z^{k-1}) as defined in equation (16) in chapter 4.3.2
	// observation is the current observation Z_k
	// p_Zk_Li contains the observation likelihoods p(Z_k | L_i) of all existing locations
	// the underlying observation likelihood model is used for sampling
	double getNormalizationTerm(std::vector<int>& observation, std::vector<double>& p_Zk_Li, double& p_Zk_Lu, double sigma, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior);
};

#endif /* NORMALIZATIONTERMMEANFIELD_H_ */
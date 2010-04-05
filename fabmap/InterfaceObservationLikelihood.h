// interface for the observation likelihood p(Z_k|L_i, Z^{k-1}) as defined in chapter 4.3.1 in the paper
// author: Martin, Richard

#ifndef INTERFACEOBSERVATIONLIKELIHOOD_H_
#define INTERFACEOBSERVATIONLIKELIHOOD_H_

#include <vector>
#include "InterfaceDetectorModel.h"
#include "InterfacePlaceModel.h"


class InterfaceObservationLikelihood {
public:
	//takes the training data in the form [image[histogram of image]] and generates the Chow Liu tree
	InterfaceObservationLikelihood(std::vector<std::vector<int> > training_data) {};
	
	// loads the stored Chow Liu Tree from file
	InterfaceObservationLikelihood(std::string pCLTreeFilename) {};
	
	virtual ~InterfaceObservationLikelihood() {};
	
	// returns P(Z_k | L_i) as defined in equations (5)-(8) for the naive Bayes model or (9)-(14) for the Chow Liu trees
	virtual double evaluate(std::vector<int> observations, int location, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel) = 0;

	// returns the marginal probability for observing a single attribute p(z_attr = val)
	virtual double getMarginalPriorProbability(int attr, int val) = 0;

	// returns the whole marginal probabilities vector
	virtual std::vector<std::vector<double> >& getMarginalPriorProbabilities() = 0;

	// returns p(Z_k | L_u) for a randomly sampled place L_u with randomly sampled obervations Z_k as needed in equation (17)
	virtual double sampleNewPlaceObservation(InterfaceDetectorModel* detectorModel) = 0;
};

#endif /* INTERFACEOBSERVATIONLIKELIHOOD_H_ */

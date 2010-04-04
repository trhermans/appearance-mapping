// implementation for the Naive Bayes Approach


#ifndef NAIVEBAYES_H_
#define NAIVEBAYES_H_
#include <iostream>
#include "InterfaceObservationLikelihood.h"
#include "InterfaceDetectorModel.h"
#include "InterfacePlaceModel.h"

class NaiveBayes : public InterfaceObservationLikelihood
{
public :
	NaiveBayes();
	~NaiveBayes();

	// returns p(Z_k | L_i) as defined in equations (5)-(8) for the NaiveBayes, i=location, Z_k=observations
	double evaluate(std::vector<int> observations, int location, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel);

private:

	//attribute sizes
	int mAttributeSize;

};

#endif /* NAIVEBAYES_H_ */

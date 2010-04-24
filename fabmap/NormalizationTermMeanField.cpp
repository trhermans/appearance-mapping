// author: Richard

#include "NormalizationTermMeanField.h"
#include "float.h"
NormalizationTermMeanField::NormalizationTermMeanField()
{

}


NormalizationTermMeanField::~NormalizationTermMeanField()
{

}


double NormalizationTermMeanField::getNormalizationTerm(std::vector<int>& observation, std::vector<double>& p_Zk_Li, double& p_Zk_Lu, double sigma, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior)
{
	double normalizationTerm = 0.0;
	double numericTrick = 0;		//now adaptive  //1e1;//1e300;			// increases log-likelihoods to avoid sums of 0 because of very small numbers
	double numericTrickMin = 0;
	double numericTrickMax = -DBL_MAX;

	for (int location=0; location<(int)p_Zk_Li.size(); location++)
	{
		if (p_Zk_Li[location] < numericTrickMin) numericTrickMin = p_Zk_Li[location];
		if (p_Zk_Li[location] > numericTrickMax) numericTrickMax = p_Zk_Li[location];
	}
	numericTrick = -(numericTrickMax+numericTrickMin)/2.0;

	// add probabilities for known (mapped) locations
	for (int location=0; location<(int)p_Zk_Li.size(); location++)
	{
		//normalizationTerm += p_Zk_Li[location] * locationPrior->getLocationPrior(location, placeModel);
		//normalizationTerm += exp(p_Zk_Li[location] + log(locationPrior->getLocationPrior(location, placeModel)));
		normalizationTerm += exp(p_Zk_Li[location] + log(locationPrior->getLocationPrior(location, placeModel)) + numericTrick);
	}

	// add the probabilities for the new place through mean field approximation from the training data distribution
	//p_Zk_Lu = pow(observationLikelihood->meanFieldNewPlaceObservation(detectorModel, observation), sigma) * exp((1-sigma)/std::max<int>(placeModel->getNumberOfLocations(),1));
	p_Zk_Lu = observationLikelihood->meanFieldNewPlaceObservation(detectorModel, observation)* sigma + (1-sigma)/std::max<int>(placeModel->getNumberOfLocations(),1);
	if ((int)p_Zk_Li.size() == 0) numericTrick = -p_Zk_Lu;

	//normalizationTerm += locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel) * p_Zk_Lu;
	//normalizationTerm += exp(log(locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel)) + p_Zk_Lu);
	normalizationTerm += exp(log(locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel)) + p_Zk_Lu + numericTrick);
	normalizationTerm = log(normalizationTerm) - numericTrick;

	return normalizationTerm;
}

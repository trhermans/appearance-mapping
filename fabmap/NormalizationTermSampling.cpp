// author: Richard

#include "NormalizationTermSampling.h"

NormalizationTermSampling::NormalizationTermSampling(int numberOfSamples)
{
	mNumberOfSamples = numberOfSamples;
}


NormalizationTermSampling::~NormalizationTermSampling()
{

}


double NormalizationTermSampling::getNormalizationTerm(std::vector<int>& observation, std::vector<double>& p_Zk_Li, double& p_Zk_Lu, double sigma, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior)
{
	double normalizationTerm = 0.0;

	// add probabilities for known (mapped) locations
	for (int location=0; location<(int)p_Zk_Li.size(); location++)
	{
		normalizationTerm += p_Zk_Li[location] * locationPrior->getLocationPrior(location, placeModel);
	}

	// add the probabilities for the new place through sampling from the training data distribution
	double sum=0.0;
	for (int sample=0; sample<mNumberOfSamples; sample++)
	{
		sum += observationLikelihood->sampleNewPlaceObservation(detectorModel, observation);
		//sum += pow(observationLikelihood->sampleNewPlaceObservation(detectorModel), sigma) * exp((1-sigma)/std::max<int>(placeModel->getNumberOfLocations(),1));
	}
	sum = pow(sum/(double)mNumberOfSamples, sigma) * exp((1-sigma)/std::max<int>(placeModel->getNumberOfLocations(),1));
	normalizationTerm += locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel) * sum;
	//normalizationTerm += locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel) * sum/(double)mNumberOfSamples;
	//p_Zk_Lu = sum/(double)mNumberOfSamples;
	p_Zk_Lu = sum;

	return normalizationTerm;
}
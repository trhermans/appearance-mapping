// author: Richard

#include "NormalizationTermSampling.h"

NormalizationTermSampling::NormalizationTermSampling(int numberOfSamples)
{
	mNumberOfSamples = numberOfSamples;
}


NormalizationTermSampling::~NormalizationTermSampling()
{

}


double NormalizationTermSampling::getNormalizationTerm(std::vector<double> p_Zk_Li, InterfaceObservationLikelihood* observationLikelihood, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel, InterfaceLocationPrior* locationPrior)
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
		sum += observationLikelihood->sampleNewPlaceObservation(detectorModel);
	}
	normalizationTerm += locationPrior->getLocationPrior((int)p_Zk_Li.size(), placeModel) * sum/mNumberOfSamples;

	return normalizationTerm;
}
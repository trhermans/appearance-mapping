// author: Richard

#include "FABMAP.h"

FABMAP::FABMAP(double sigma, int numberOfSamples, int approximationModel, std::string modelFile)
{
	if (modelFile.size() == 0) offlinePreparation(approximationModel);
	else offlinePreparation(approximationModel, modelFile);
	mSigma = sigma;

	//mDetectorModel = new ;
	//mPlaceModel = new ;
	mNormalizationTerm = new NormalizationTermSampling(numberOfSamples);
	mLocationPrior = new SimpleLocationPrior;
}

FABMAP::~FABMAP()
{
}

void FABMAP::offlinePreparation(int approximationModel)
{
	//create bag of words model
	std::vector<std::vector<int> > trainingData;

	//learn naive Bayes or Chow Liu tree
	if (approximationModel == NAIVEBAYES)
	{
		mObservationLikelihood = new NaiveBayes(trainingData);
	}
	if (approximationModel == CHOWLIU)
	{
		mObservationLikelihood = new ChowLiuTree(trainingData);
	}
}

void FABMAP::offlinePreparation(int approximationModel, std::string modelFile)
{
	// load naive Bayes or Chow Liu tree
	if (approximationModel == NAIVEBAYES)
	{
		mObservationLikelihood = new NaiveBayes(modelFile);
	}
	if (approximationModel == CHOWLIU)
	{
		mObservationLikelihood = new ChowLiuTree(modelFile);
	}
}

void FABMAP::onlineApplication()
{
	// get histogram from a certain image
	std::vector<int> observation;

	// calculate p_Zk_Li = p(Z_k | L_i) for all locations
	// p_Zk_Li is the observation likelihood p(Z_k | L_i)
	std::vector<double> p_Zk_Li;
	for (int location=0; location<mPlaceModel->getNumberOfLocations(); location++)
	{
		double p_Zk_Li_ = mObservationLikelihood->evaluate(observation, location, mDetectorModel, mPlaceModel);
		// do smoothing as in equation (18)
		p_Zk_Li.push_back(mSigma * p_Zk_Li_ + (1-mSigma)/mPlaceModel->getNumberOfLocations());
	}
	//// calculate p_Zk_Li for the new place, too --- no, that is done by sampling in the normalization term class
	//double p_Zk_Li_ = 0.0;
	//// do smoothing as in equation (18)
	//p_Zk_Li.push_back(mSigma * p_Zk_Li_ + (1-mSigma)/mPlaceModel->getNumberOfLocations());
	
	// calculate normalization term
	double normalizationTerm = mNormalizationTerm->getNormalizationTerm(p_Zk_Li, mObservationLikelihood, mDetectorModel, mPlaceModel, mLocationPrior);

	// find the maximum likelihood estimate for p(L_i | Z^k)
	// iterate over all locations (including the new place (?)) to get the most probable location estimate
	// place with index mPlaceModel->getNumberOfLocations() is the new place
	int mostProbableLocation = 0;
	double mostProbableLocationProbability = 0.0;
	for (int location=0; location<mPlaceModel->getNumberOfLocations()+1; location++)
	{
		// calculate p_Li_Zk = p(L_i | Z^k) as in equation (4)
		double p_Li_Zk = p_Zk_Li[location] * mLocationPrior->getLocationPrior(location, mPlaceModel) / normalizationTerm;
		
		// maximum likelihood estimation of the location
		if (p_Li_Zk > mostProbableLocationProbability)
		{
			mostProbableLocation = location;
			mostProbableLocationProbability = p_Li_Zk;
		}
	}

	// place update
	// if new place found, add the new place to the place model
	if (mostProbableLocation == mPlaceModel->getNumberOfLocations())
	{
		mPlaceModel->addLocation(mObservationLikelihood->getMarginalPriorProbabilities());
		mPlaceModel->updateLocation(observation, mostProbableLocation, mDetectorModel, mObservationLikelihood->evaluate(observation, mostProbableLocation, mDetectorModel, mPlaceModel));
	}
	// update
	else
	{
		mPlaceModel->updateLocation(observation, mostProbableLocation, mDetectorModel, p_Zk_Li[mostProbableLocation]);
	}
}
// author: Richard

#include "FABMAP.h"

FABMAP::FABMAP(double sigma, int numberOfSamples, int approximationModel, double falsePositiveProbability, double falseNegativeProbability, std::string codebookFile, std::string trainingDataFile, std::string modelFile)
{
	if (modelFile.size() == 0) offlinePreparationTrain(approximationModel, trainingDataFile);
	else offlinePreparationLoad(approximationModel, modelFile);
	mSigma = sigma;

	mDetectorModel = new DetectorModel(falsePositiveProbability, falseNegativeProbability);
	mPlaceModel = new PlaceModel;
	mNormalizationTerm = new NormalizationTermSampling(numberOfSamples);
	mLocationPrior = new SimpleLocationPrior;
	mVisualCodebook.loadCodebook(codebookFile);
}

FABMAP::~FABMAP()
{
}

void FABMAP::offlinePreparationTrain(int approximationModel, std::string trainingDataFile)
{
	// load training data preprocessed with the bag of words model
	std::vector<std::vector<int> > trainingData;

	std::ifstream in(trainingDataFile.c_str());
	if(!in.is_open())
	{
		std::cout << "FABMAP::offlinePreparation: Error: could not open " << trainingDataFile.c_str() << "\n";
		return;
	}
	int samples, attributes;
	in >> samples;
	in >> attributes;
	trainingData.resize(samples, std::vector<int>(attributes, 0));
	for (unsigned int i = 0; i < trainingData.size(); ++i)
	{
		for (unsigned int j = 0; j < trainingData[i].size(); ++j)
		{
			in >> trainingData[i][j];
		}
	}
	in.close();

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

void FABMAP::offlinePreparationLoad(int approximationModel, std::string modelFile)
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

void FABMAP::onlineApplication(std::string imagePath, int numberOfImages)
{
	// iterate over image sequence
	for (int image=1; image<=numberOfImages; image++)
	{
		// get histogram from a certain image
		std::vector<int> observation;

		int paddingSize = 4;
		std::stringstream imgNum;
		imgNum << image;
		std::string imgNumStr = imgNum.str();
		int numZeros = paddingSize - imgNumStr.size();
		imgNumStr.insert(0, numZeros, '0');
		std::stringstream imageName;
		imageName << imagePath << imgNumStr << ".jpg";
		std::cout << imageName.str() << std::endl;

		IplImage *img;
		img = cvLoadImage(imageName.str().c_str(), CV_8UC1);
		observation = mVisualCodebook.getCodewords(*img);

		// calculate p_Zk_Li = p(Z_k | L_i) for all mapped locations
		// p_Zk_Li is the observation likelihood p(Z_k | L_i)
		std::vector<double> p_Zk_Li;
		for (int location=0; location<mPlaceModel->getNumberOfLocations(); location++)
		{
			double p_Zk_Li_ = mObservationLikelihood->evaluate(observation, location, mDetectorModel, mPlaceModel);
			// do smoothing as in equation (18)
			p_Zk_Li.push_back(mSigma * p_Zk_Li_ + (1-mSigma)/mPlaceModel->getNumberOfLocations());
		}

		// calculate p_Zk_Li for the new place, that is done by sampling in the normalization term class
		double p_Zk_Lu = 0.0;	
		// calculate normalization term
		double normalizationTerm = mNormalizationTerm->getNormalizationTerm(p_Zk_Li, p_Zk_Lu, mObservationLikelihood, mDetectorModel, mPlaceModel, mLocationPrior);
		// do smoothing as in equation (18)
		p_Zk_Li.push_back(mSigma * p_Zk_Lu + (1-mSigma)/std::max<int>(mPlaceModel->getNumberOfLocations(),1));

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
		if (mostProbableLocation == mPlaceModel->getNumberOfLocations())
		{
			// if new place found, add the new place to the place model
			mPlaceModel->addLocation(mObservationLikelihood->getMarginalPriorProbabilities());
			//mPlaceModel->updateLocation(observation, mostProbableLocation, mDetectorModel, mObservationLikelihood->evaluate(observation, mostProbableLocation, mDetectorModel, mPlaceModel));
		}
		//else
		//{
			// update existing place
			mPlaceModel->updateLocation(observation, mostProbableLocation, mDetectorModel, p_Zk_Li[mostProbableLocation]);
		//}
	}
}
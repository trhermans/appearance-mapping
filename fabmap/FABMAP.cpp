// author: Richard

#include "FABMAP.h"

FABMAP::FABMAP(double sigma, int numberOfSamples, int approximationModel, double falsePositiveProbability, double falseNegativeProbability, std::string codebookFile, std::string trainingDataFile, std::string modelFile, boolean loadModel)
{
	if (loadModel==false) offlinePreparationTrain(approximationModel, trainingDataFile, modelFile);
	else offlinePreparationLoad(approximationModel, modelFile);
	mSigma = sigma;

	mDetectorModel = new DetectorModel(falsePositiveProbability, falseNegativeProbability);
	mPlaceModel = new PlaceModel;
	mNormalizationTerm = new NormalizationTermSampling(numberOfSamples);
	mLocationPrior = new SimpleLocationPrior;
	std::cout << "1";
	mVisualCodebook.loadCodebook(codebookFile);
	std::cout << "2";
}

FABMAP::~FABMAP()
{
}

void FABMAP::offlinePreparationTrain(int approximationModel, std::string trainingDataFile, std::string modelFile)
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
		mObservationLikelihood = new NaiveBayes(trainingData, modelFile);
	}
	if (approximationModel == CHOWLIU)
	{
		mObservationLikelihood = new ChowLiuTree(trainingData, modelFile);
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

void FABMAP::onlineApplication(std::string imagePath_or_trainingDataFile, int numberOfImages, double loopClosureThreshold)
{
	// images already preprocessed, load histogram data from file
	bool preloaded = false;
	std::vector<std::vector<int> > loadedObservations;
	if (numberOfImages == -1)
	{
		std::fstream input;
		input.open(imagePath_or_trainingDataFile.c_str(), std::ios::in);
		int columns = 0;
		input >> numberOfImages >> columns;
		std::vector<int> temp;
		temp.resize(columns, 0);
		loadedObservations.resize(numberOfImages, temp);
		for (int i = 0; i < numberOfImages; ++i)
		{
			for (int j = 0; j < columns; ++j)
			{
				input >> loadedObservations[i][j];
			}
		}
		input.close();
		preloaded = true;
	}

	// file for statistics
	std::fstream output;
	output.open("_fabmap_onlinerun_statistics.txt", std::ios::out);

	// list of last 10 visited places; those are not allowed to close a loop with the current image
	std::vector<int> lastTenPlaces;

	// iterate over image sequence
	for (int image=1; image<=numberOfImages; image++)
	{
		if (image%100==0) std::cout << image << " images processed. There are " << mPlaceModel->getNumberOfLocations() << " different locations found." << std::endl;
		// get histogram from a certain image
		std::vector<int> observation;

		if (preloaded)
		{
			observation = loadedObservations[image-1];
		}
		else
		{
			int paddingSize = 4;
			std::stringstream imgNum;
			imgNum << image;
			std::string imgNumStr = imgNum.str();
			int numZeros = paddingSize - imgNumStr.size();
			imgNumStr.insert(0, numZeros, '0');
			std::stringstream imageName;
			imageName << imagePath_or_trainingDataFile << imgNumStr << ".jpg";
			std::cout << imageName.str() << std::endl;

			IplImage *img;
			img = cvLoadImage(imageName.str().c_str(), CV_8UC1);
			observation = mVisualCodebook.getCodewords(*img);
		}

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
		// only accept loop closures with p(L_i | Z_k) > threshold and do not close with last 10 images
		bool inLastTenPlaces = false;
		for (unsigned int i=0; i<lastTenPlaces.size(); i++) if (lastTenPlaces[i]==mostProbableLocation) inLastTenPlaces=true;
		if (((mostProbableLocation != mPlaceModel->getNumberOfLocations()) && (mostProbableLocationProbability<loopClosureThreshold))  ||  (inLastTenPlaces))
		{
			mostProbableLocation = mPlaceModel->getNumberOfLocations();
			mostProbableLocationProbability *= -1;		// to indicate that this probability stems from a not accepted loop closure and not from a regular new place
		}
		// update last 10 visited places list
		if (lastTenPlaces.size() > 9) lastTenPlaces.erase(lastTenPlaces.begin());
		lastTenPlaces.push_back(mostProbableLocation);

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

		// save image number, location decided by the algorithm, probability for that decision
		output << image << "\t" << mostProbableLocation << "\t" << mostProbableLocationProbability << std::endl;
	}
	std::cout << numberOfImages << " images processed. There are " << mPlaceModel->getNumberOfLocations() << " different locations found." << std::endl;


	output.close();
}
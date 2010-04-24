// author: Richard

#include "FABMAP.h"

FABMAP::FABMAP(double sigma, int numberOfSamples, int approximationModel, double falsePositiveProbability, double falseNegativeProbability, std::string codebookFile, std::string trainingDataFile, std::string modelFile, bool loadModel)
{
		// load training data preprocessed with the bag of words model
	std::vector<std::vector<int> >* trainingData = new std::vector<std::vector<int> >;

	std::ifstream in(trainingDataFile.c_str());
	
	// original fabmap data (2859 images)	
	//std::vector<int> temp;
	//while(in.eof()==false)
	//{
	//	char line[70000];
	//	std::istrstream istr(line, 70000);
	//	in.getline(line, 70000);
	//	temp.clear();

	//	int j=0;
	//	int column=0;
	//	while (istr >> j)
	//	{
	//		for (int k=column; k<j; k++) temp.push_back(0);
	//		temp.push_back(1);
	//		column = j+1;
	//	}
	//	for (int k=column; k<10987; k++) temp.push_back(0);
	//	trainingData->push_back(temp);
	//	for (int k=0; k<4; k++) in.getline(line, 70000);
	//}
	//in.close();

	//std::ofstream out("_training_data_ori_cbOri_c11000.txt");
	//out << trainingData->size() << "\t" << trainingData[0].size() << "\n";
	//for (unsigned int i = 0; i < trainingData->size(); ++i)
	//{
	//	for (unsigned int j = 0; j < (*trainingData)[i].size(); ++j)
	//	{
	//		out << (*trainingData)[i][j] << "\t";
	//	}
	//	out << "\n";
	//}
	//out.close();

	if(!in.is_open())
	{
		std::cout << "FABMAP::offlinePreparation: Error: could not open " << trainingDataFile.c_str() << "\n";
		return;
	}
	int samples, attributes;
	in >> samples;
	in >> attributes;
	trainingData->resize(samples, std::vector<int>(attributes, 0));
	for (unsigned int i = 0; i < trainingData->size(); ++i)
	{
		for (unsigned int j = 0; j < (*trainingData)[i].size(); ++j)
		{
			in >> (*trainingData)[i][j];
		}
	}
	in.close();

	if (loadModel==false) offlinePreparationTrain(approximationModel, *trainingData, modelFile);
	else offlinePreparationLoad(approximationModel, *trainingData, modelFile);
	mSigma = sigma;

	mDetectorModel = new DetectorModel(falsePositiveProbability, falseNegativeProbability);
	mPlaceModel = new PlaceModel;
	mNormalizationTerm = new NormalizationTermMeanField; //NormalizationTermSampling(numberOfSamples);
	mLocationPrior = new SimpleLocationPrior;
	mCodebookFile = codebookFile;
}

FABMAP::~FABMAP()
{
}

void FABMAP::offlinePreparationTrain(int approximationModel, std::vector<std::vector<int> >& trainingData, std::string modelFile)
{
	//learn naive Bayes or Chow Liu tree
	if (approximationModel == NAIVEBAYES)
	{
		mObservationLikelihood = new NaiveBayes(trainingData, modelFile, true);
	}
	if (approximationModel == CHOWLIU)
	{
		mObservationLikelihood = new ChowLiuTree(trainingData, modelFile, true);
	}
}

void FABMAP::offlinePreparationLoad(int approximationModel, std::vector<std::vector<int> >& trainingData, std::string modelFile)
{
	// load naive Bayes or Chow Liu tree
	if (approximationModel == NAIVEBAYES)
	{
		mObservationLikelihood = new NaiveBayes(trainingData, modelFile, false);
	}
	if (approximationModel == CHOWLIU)
	{
		mObservationLikelihood = new ChowLiuTree(trainingData, modelFile, false);
	}
}

void FABMAP::onlineApplication(std::string imagePath_or_testDataFile, int numberOfImages, double loopClosureThreshold)
{
	// images already preprocessed, load histogram data from file
	bool preloaded = false;
	std::vector<std::vector<int> > loadedObservations;
	if (numberOfImages == -1)
	{
		std::cout << "Loading test data file." << std::endl;
		std::fstream input;
		input.open(imagePath_or_testDataFile.c_str(), std::ios::in);
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

		// original fabmap data format (2146 images) preprocessed with fabmap Surf and original clusters
		//std::fstream input;
		//input.open(imagePath_or_testDataFile.c_str(), std::ios::in);
		//std::vector<int> temp;
		//temp.resize(10987, 0);
		//numberOfImages=2146;
		//loadedObservations.resize(numberOfImages, temp);
		//for (int i = 0; i < numberOfImages; ++i)
		//{
		//	char line[70000];
		//	std::istrstream istr(line, 70000);
		//	input.getline(line, 70000);

		//	int j=0;
		//	while (istr >> j)
		//	{
		//		loadedObservations[i][j] = 1;
		//	}
		//	for (int k=0; k<4; k++) input.getline(line, 70000);
		//}
		//input.close();

		//std::ofstream out("_training_data_NewCollegeOri_cbOri_c11000.txt");
		//out << loadedObservations.size() << "\t" << loadedObservations[0].size() << "\n";
		//for (unsigned int i = 0; i < loadedObservations.size(); ++i)
		//{
		//	for (unsigned int j = 0; j < loadedObservations[i].size(); ++j)
		//	{
		//		out << loadedObservations[i][j] << "\t";
		//	}
		//	out << "\n";
		//}
		//out.close();

		preloaded = true;
	}
	else
	{
		mVisualCodebook.loadCodebook(mCodebookFile);
	}


	// file for statistics
	std::fstream output;
	output.open("_fabmap_onlinerun_statistics.txt", std::ios::out);

	// file for statistics
	std::fstream outputProbMatrix;
	outputProbMatrix.open("_fabmap_prob_matrix.txt", std::ios::out);

	// list of last 10 visited places; those are not allowed to close a loop with the current image
	std::vector<int> lastTenPlaces;

	// iterate over image sequence
	for (int image=1; image<=numberOfImages; image++)
	{
		// get histogram from a certain image
		std::vector<int>* observation = 0;

		if (preloaded)
		{
			observation = &(loadedObservations[image-1]);
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
			imageName << imagePath_or_testDataFile << imgNumStr << ".jpg";
			std::cout << imageName.str() << std::endl;

			IplImage *img;
			img = cvLoadImage(imageName.str().c_str(), CV_8UC1);
			observation = &(mVisualCodebook.getCodewords(*img));
		}

		// calculate p_Zk_Li = p(Z_k | L_i) for all mapped locations
		// p_Zk_Li is the observation likelihood p(Z_k | L_i)
		std::vector<double> p_Zk_Li;
		for (int location=0; location<mPlaceModel->getNumberOfLocations(); location++)
		{
			double p_Zk_Li_ = mObservationLikelihood->evaluate(*observation, location, mDetectorModel, mPlaceModel);
			// do smoothing as in equation (18) (please notice that eq. (18) assumes log-likelihoods, this formula adopted that)
			p_Zk_Li.push_back(p_Zk_Li_ * mSigma + (1-mSigma)/mPlaceModel->getNumberOfLocations());
			//p_Zk_Li.push_back(pow(p_Zk_Li_, mSigma) * exp((1-mSigma)/mPlaceModel->getNumberOfLocations()));
		}

		// calculate p_Zk_Li for the new place, that is done by sampling in the normalization term class
		double p_Zk_Lu = 0.0;	
		// calculate normalization term
		std::cout << "<";
		double normalizationTerm = mNormalizationTerm->getNormalizationTerm(*observation, p_Zk_Li, p_Zk_Lu, mSigma, mObservationLikelihood, mDetectorModel, mPlaceModel, mLocationPrior);
		// do smoothing as in equation (18) (please notice that eq. (18) assumes log-likelihoods, this formula adopted that)
		//p_Zk_Li.push_back(pow(p_Zk_Lu, mSigma) * exp((1-mSigma)/std::max<int>(mPlaceModel->getNumberOfLocations(),1)));
		p_Zk_Li.push_back(p_Zk_Lu);
		std::cout << ">";

		// find the maximum likelihood estimate for p(L_i | Z^k)
		// iterate over all locations (including the new place (?)) to get the most probable location estimate
		// place with index mPlaceModel->getNumberOfLocations() is the new place
		int mostProbableLocation = 0;
		double mostProbableLocationProbability = 0.0;
		//std::cout << "Image " << image << std::endl;
		for (int location=0; location<mPlaceModel->getNumberOfLocations()+1; location++)
		{
			// calculate p_Li_Zk = p(L_i | Z^k) as in equation (4)
			//std::cout << location << ": " << /*p_Zk_Li[location]/sqrt(normalizationTerm) << "\t" << mLocationPrior->getLocationPrior(location, mPlaceModel)/sqrt(normalizationTerm) << "\t" <<*/ p_Zk_Li[location] << "\t" << mLocationPrior->getLocationPrior(location, mPlaceModel) << "\t" << normalizationTerm << "\t" << "\n";
			//double p_Li_Zk = p_Zk_Li[location] * mLocationPrior->getLocationPrior(location, mPlaceModel) / normalizationTerm;
			double p_Li_Zk = exp(p_Zk_Li[location] + log(mLocationPrior->getLocationPrior(location, mPlaceModel)) - normalizationTerm);

			outputProbMatrix << p_Li_Zk << "\t";
			
			// maximum likelihood estimation of the location
			bool inLastTenPlaces = false;
			for (unsigned int i=0; i<lastTenPlaces.size(); i++) 
			{
				if (lastTenPlaces[i]==location) inLastTenPlaces=true;
			}
			if ((p_Li_Zk > mostProbableLocationProbability) && !inLastTenPlaces)
			{
				mostProbableLocation = location;
				mostProbableLocationProbability = p_Li_Zk;
			}
		}

		for (int i=mPlaceModel->getNumberOfLocations(); i<=numberOfImages; i++) outputProbMatrix << "0\t";
		outputProbMatrix << "\n";
		
		// only accept loop closures with p(L_i | Z_k) > threshold and do not close with last 10 images
		bool inLastTenPlaces = false;
		for (unsigned int i=0; i<lastTenPlaces.size(); i++) if (lastTenPlaces[i]==mostProbableLocation) inLastTenPlaces=true;
		int formermostProbableLocation = mostProbableLocation;
		if (((mostProbableLocation != mPlaceModel->getNumberOfLocations()) && (mostProbableLocationProbability<loopClosureThreshold))  ||  (inLastTenPlaces))
		{
			mostProbableLocation = mPlaceModel->getNumberOfLocations();
			mostProbableLocationProbability *= -1;		// to indicate that this probability stems from a not accepted loop closure and not from a regular new place
		}
		// update last 10 visited places list
		if (lastTenPlaces.size() > 9) lastTenPlaces.erase(lastTenPlaces.begin());
		lastTenPlaces.push_back(mostProbableLocation);

		// place update
		if (mostProbableLocation != mPlaceModel->getNumberOfLocations()) std::cout << "c";
		//if (mostProbableLocation == mPlaceModel->getNumberOfLocations())
		//{
			// if new place found, add the new place to the place model
			mPlaceModel->addLocation(mObservationLikelihood->getMarginalPriorProbabilities());
			//mPlaceModel->updateLocation(*observation, mostProbableLocation, mDetectorModel, mObservationLikelihood->evaluate(*observation, mostProbableLocation, mDetectorModel, mPlaceModel));
		//}
		//else
		//{
		// update existing place
		mPlaceModel->updateLocation(*observation, mostProbableLocation, mDetectorModel, p_Zk_Li[mostProbableLocation]);
		//}

		// save image number, location decided by the algorithm, probability for that decision
		output << image << "\t" << mostProbableLocation << "\t" << mostProbableLocationProbability << "\t" << formermostProbableLocation << std::endl;

		if (image%10==0)
		{
			std::cout << image << " images processed. There are " << mPlaceModel->getNumberOfLocations() << " different locations found." << std::endl;
			outputProbMatrix.flush();
		}
	}
	std::cout << numberOfImages << " images processed. There are " << mPlaceModel->getNumberOfLocations() << " different locations found." << std::endl;


	output.close();
	outputProbMatrix.close();
}

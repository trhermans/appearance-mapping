// implements the FABMAP algorithm for probabilistic localization and mapping in the space of appearance
// author: Richard

#ifndef FABMAP_H_
#define FABMAP_H_

#include <vector>
#include <fstream>
#include "InterfaceObservationLikelihood.h"
#include "ChowLiuTree.h"
#include "NaiveBayes.h"
#include "InterfaceDetectorModel.h"
#include "InterfacePlaceModel.h"
#include "InterfaceNormalizationTerm.h"
#include "InterfaceLocationPrior.h"
#include "NormalizationTermSampling.h"
#include "SimpleLocationPrior.h"
#include "DetectorModel.h"
#include "PlaceModel.h"
#include "../vis_codebook/visual_codebook.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"

class FABMAP
{
public:
	// constructor for building the model from the image files (if no modelFile is provided) or constructor for loading an existing model from file (if modelFile is provided)
	// sigma is the smoothing factor (see equation (18))
	// numberOfSamples is the number of samples for the unknown place during the calculation of the normalization term (n_s in the paper in equation (17) in chapter 4.3.2)
	// approximationModel is one of the elements in enum ObservationLikelihoodModel
	// falsePositiveProbability, falseNegativeProbability - the detector model probabilities (see detector model)
	// trainingDataFile is the file name of the file with training data (vector<vector<int>>), if this parameter is "" you must provide a modelFile
	// modelFile - if the model was already generated, it can be loaded from a model file
	FABMAP(double sigma, int numberOfSamples, int approximationModel, double falsePositiveProbability, double falseNegativeProbability, std::string codebookFile, std::string trainingDataFile, std::string modelFile="");

	~FABMAP();

	// does probabilistic localization and mapping in the space of appearance with the provided data
	// imagePath is the path where the images can be found, e.g. "../../data/images/"
	void onlineApplication(std::string imagePath, int numberOfImages);

	enum ObservationLikelihoodModel {NAIVEBAYES, CHOWLIU};

private:
	// accomplishes all the offline learning (starts from image files) which is necessary before the algorithm can be applied
	void offlinePreparationTrain(int approximationModel, std::string trainingDataFile);

	// loads existing offline-learned models from file which is necessary before the algorithm can be applied
	void offlinePreparationLoad(int approximationModel, std::string modelFile);

	// the smoothing factor applied in equation (18)
	double mSigma;

	InterfaceObservationLikelihood* mObservationLikelihood;
	InterfaceDetectorModel* mDetectorModel;
	InterfacePlaceModel* mPlaceModel;
	InterfaceNormalizationTerm* mNormalizationTerm;
	InterfaceLocationPrior* mLocationPrior;
	VisualCodebook mVisualCodebook;
};

#endif /* FABMAP_H_ */
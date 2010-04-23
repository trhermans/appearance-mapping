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
	// codebookFile is the file containing the codebook
	// trainingDataFile is the file name of the file with training data (vector<vector<int>>), if this parameter is "" you must provide a modelFile
	// modelFile - if the model was already generated, it can be loaded from a model file, if it does not exist already, it will be written into that file for later use
	// loadModel - if true, then the model will be loaded from file modelFile, if false, the model will be saved in file modelFile
	FABMAP(double sigma, int numberOfSamples, int approximationModel, double falsePositiveProbability, double falseNegativeProbability, std::string codebookFile, std::string trainingDataFile, std::string modelFile, bool loadModel);

	~FABMAP();

	// does probabilistic localization and mapping in the space of appearance with the provided data
	// imagePath is the path where the images can be found, e.g. "../../data/images/", then you must set numberOfImages to the number of images you want to process
	// or
	// trainingDataFile is the path where the histograms for the images are already stored (i.e. the bag-of-words stage of the image features is already done), then choose numberOfImages=-1 to indicate that
	// loopClosureThreshold is the probability which must be exceeded by p(L_i | Z_k) in order to accept a location L_i as loop closure
	void onlineApplication(std::string imagePath_or_trainingDataFile, int numberOfImages, double loopClosureThreshold);

	enum ObservationLikelihoodModel {NAIVEBAYES, CHOWLIU};

private:
	// accomplishes all the offline learning (starts from image files) which is necessary before the algorithm can be applied
	void offlinePreparationTrain(int approximationModel, std::string trainingDataFile, std::string modelFile);

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

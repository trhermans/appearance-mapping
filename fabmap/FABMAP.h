// implements the FABMAP algorithm for probabilistic localization and mapping in the space of appearance
// author: Richard

#ifndef FABMAP_H_
#define FABMAP_H_

#include <vector>
#include "InterfaceObservationLikelihood.h"
#include "ChowLiuTree.h"
#include "NaiveBayes.h"
#include "InterfaceDetectorModel.h"
#include "InterfacePlaceModel.h"
#include "InterfaceNormalizationTerm.h"
#include "InterfaceLocationPrior.h"
#include "NormalizationTermSampling.h"
#include "SimpleLocationPrior.h"

class FABMAP
{
public:
	// constructor for building the model from the image files (if no modelFile is provided) or constructor for loading an existing model from file (if modelFile is provided)
	// sigma is the smoothing factor (see equation (18))
	// numberOfSamples is the number of samples for the unknown place during the calculation of the normalization term (n_s in the paper in equation (17) in chapter 4.3.2)
	// approximationModel is one of the elements in enum ObservationLikelihoodModel
	FABMAP(double sigma, int numberOfSamples, int approximationModel, std::string modelFile="");

	~FABMAP();

	// does probabilistic localization and mapping in the space of appearance with the provided data
	void onlineApplication();

	enum ObservationLikelihoodModel {NAIVEBAYES, CHOWLIU};

private:
	// accomplishes all the offline learning (starts from image files) which is necessary before the algorithm can be applied
	void offlinePreparation(int approximationModel);

	// loads existing offline-learned models from file which is necessary before the algorithm can be applied
	void offlinePreparation(int approximationModel, std::string modelFile);

	// the smoothing factor applied in equation (18)
	double mSigma;

	InterfaceObservationLikelihood* mObservationLikelihood;
	InterfaceDetectorModel* mDetectorModel;
	InterfacePlaceModel* mPlaceModel;
	InterfaceNormalizationTerm* mNormalizationTerm;
	InterfaceLocationPrior* mLocationPrior;
};

#endif /* FABMAP_H_ */
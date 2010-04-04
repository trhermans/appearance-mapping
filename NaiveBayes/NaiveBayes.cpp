
#include "NaiveBayes.h"

NaiveBayes::NaiveBayes()
: InterfaceObservationLikelihood()
{
	//set number of discrete attribute values per attribute (which are 0,1,2,...,n)
	mAttributeSize = 2;

}

NaiveBayes::~NaiveBayes()
{
}

double NaiveBayes::evaluate(std::vector<int> observations, int location, InterfaceDetectorModel* detectorModel, InterfacePlaceModel* placeModel)
{
	double p_Zk_Li = 1.0;	// = p(Z_k | L_i), i.e. the return value of this function
	
	double p_zj_Li = 0.0;	// = p(z_j | L_i), i.e. the probability of the observation attribute z_j given location L_i
	for (unsigned int j = 0;j < observations.size(); ++j){
		p_zj_Li = 0.0;
		for (int s=0; s<mAttributeSize; s++){
			p_zj_Li += detectorModel->getDetectorProbability(observations[j], s) * placeModel->getWordProbability(j, s, location);	// equation (8)
		}
		p_Zk_Li *= p_zj_Li;
	}

	return p_Zk_Li;
}

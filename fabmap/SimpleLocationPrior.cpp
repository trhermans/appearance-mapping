// author: Richard

#include "SimpleLocationPrior.h"

SimpleLocationPrior::SimpleLocationPrior()
{

}


SimpleLocationPrior::~SimpleLocationPrior()
{

}


double SimpleLocationPrior::getLocationPrior(int location, InterfacePlaceModel* placeModel)
{
	int lastLocation = placeModel->getLastLocation();
	// initialization case
	if (lastLocation == -1)
	{
		return 1.0;
	}
	int numberOfLocations = placeModel->getNumberOfLocations();
	int numberOfAdjancentLocations = (int)placeModel->getAdjacentLocations(lastLocation).size();

	double locationPrior = 0.0;
	double probabilityMassHigh = 0.8;
	// if adjacency list has less than two places, we are at an end point; then the probability of new place should be high
	if (numberOfAdjancentLocations < 2)
	{
		if (location == numberOfLocations)
		{
			locationPrior = 0.9;
		}
		else
		{
			locationPrior = (0.1)/((double)numberOfLocations);
		}
	}
	else
	{
		// places which are in the adjacency list of lastLocation or the new place have a high likelihood (equal for all these places)
		if (placeModel->locationsAdjancent(lastLocation, location) || location == numberOfLocations)
		{
			locationPrior = probabilityMassHigh/((double)numberOfAdjancentLocations+1.0);	// divided by all existing adjacent locations + the new place
		}
		// all other places share a small probability mass
		else
		{
			locationPrior = (1.0-probabilityMassHigh)/((double)numberOfLocations-numberOfAdjancentLocations);	// divided by all existing locations + the new place - all existing adjacent locations - the new place
		}
	}

	return locationPrior;
}
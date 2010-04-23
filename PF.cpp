#include "PF.h"
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND
#include <cmath>

using namespace std;

#define M_PI_FLOAT M_PI*1.0f
#define MAX_CHANGE_X 5.0f
#define MAX_CHANGE_Y 5.0f
#define MAX_CHANGE_H M_PI_FLOAT / 16.0f
#define MAX_CHANGE_F 5.0f
#define MAX_CHANGE_L 5.0f
#define MAX_CHANGE_R M_PI_FLOAT / 16.0f
#define UNIFORM_1_NEG_1 (2.0f*(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)+1)) - 1.0f)
#define RESEED_SPREAD 1.0f

/**
 * Initializes the sampel sets so that the first update works appropriately
 */
PF::PF(int _M) : useBest(false), lastOdo(0,0,0), frameCounter(0), M(_M)
{
  // Initialize particles to be randomly spread about the field...
  srand(time(NULL));
  for (int m = 0; m < M; ++m) {
    //Particle p_m;
    // X bounded by width of the field
    // Y bounded by height of the field
    // H between +-pi
    PoseEst x_m(static_cast<float>(rand() % static_cast<int>(RESEED_SPREAD)),
                static_cast<float>(rand() % static_cast<int>(RESEED_SPREAD)),
                UNIFORM_1_NEG_1 * (M_PI_FLOAT / 2.0f));
    Particle p_m(x_m, 1.0f);
    X_t.push_back(p_m);
  }

  updateEstimates();
}

PF::~PF()
{
}

/**
 * Method to reset the particle filter to an initial state.
 */
void PF::reseed(PoseEst est, float spread)
{
  for (int m = 0; m < M; ++m) {
    PoseEst x_m(est.x + sampleUniformDistribution(spread),
                est.y + sampleUniformDistribution(spread),
                UNIFORM_1_NEG_1 * (M_PI_FLOAT / 2.0f));
    Particle p_m(x_m, 1.0f);
    X_t.push_back(p_m);
  }

  updateEstimates();
}

/**
 * Method updates the set of particles and estimates the robots position.
 * Called every frame.
 *
 * @param u_t The motion (odometery) change since the last update.
 * @param z_t The set of landmark observations in the current frame.
 * @param resample Should we resample during this update
 * @return The set of particles representing the estimate of the current frame.
 */
void PF::updateOdometry(MotionModel u_t)
{
  frameCounter++;
  // Set the current particles to be of time minus one.
  vector<Particle> X_t_1 = X_t;
  // Clar the current set
  X_t.clear();
  X_bar_t.clear();
  float totalWeights = 0.; // Must sum all weights for zture use

  // Run through the particles
  for (int m = 0; m < M; ++m) {
    Particle x_t_m;

    // Update motion model for the particle
    x_t_m.pose = updateMotionModel(X_t_1[m].pose, u_t);
    totalWeights += x_t_m.weight;
    // Add the particle to the current frame set
    X_bar_t.push_back(x_t_m);
  }
}

void PF::updateLoopClosure(LoopClosure z_t)
{
  // Resample the particles
  if (z_t.loop_detected_) {
    reseed(z_t.detected_closure_location_, RESEED_SPREAD);
  } else {
    noResample(&X_bar_t);
  }

  // Update pose and uncertainty estimates
  updateEstimates();
}

/**
 * Update a particle's pose based on the last motion model.
 * We sample the pose with noise proportional to the odometery update.
 *
 * @param x_t_1 The robot pose from the pervious position distribution
 * @param u_t The odometry update from the last frame
 *
 * @return A new robot pose sampled on the odometry update
 */
PoseEst PF::updateMotionModel(PoseEst x_t, MotionModel u_t)
{
  lastOdo = u_t;
  u_t.deltaF -= sampleNormalDistribution(fabs(u_t.deltaF));
  u_t.deltaL -= sampleNormalDistribution(fabs(u_t.deltaL));
  u_t.deltaR -= sampleNormalDistribution(fabs(u_t.deltaR));

  // u_t.deltaF -= sampleTriangularDistribution(fabs(u_t.deltaF));
  // u_t.deltaL -= sampleTriangularDistribution(fabs(u_t.deltaL));
  // u_t.deltaR -= sampleTriangularDistribution(fabs(u_t.deltaR));

  x_t += u_t;

  return x_t;
}

/**
 * Method to resample the particles based on a straight proportion of their
 * weights. Adds copies of the paritcle jittered proportional to the weight
 * of the particle
 *
 * @param X_bar_t the set of particles before being resampled
 * @param totalWeights the totalWeights of the particle set X_bar_t
 */
void PF::resample(std::vector<Particle> * X_bar_t, float totalWeights) {
  for (int m = 0; m < M; ++m) {
    // Normalize the particle weights
    (*X_bar_t)[m].weight /= totalWeights;

    int count = int(round(float(M) * (*X_bar_t)[m].weight));
    for (int i = 0; i < count; ++i) {
      // Random walk the particles
      X_t.push_back(randomWalkParticle((*X_bar_t)[m]));
      //X_t.push_back(X_bar_t[m]);
    }
  }
}

void PF::lowVarianceResample(std::vector<Particle> * X_bar_t,
                             float totalWeights) {
  float r = ((static_cast<float>(rand()) /
              static_cast<float>(RAND_MAX)) *
             (1.0f/static_cast<float>(M) ));
  float c = (*X_bar_t)[0].weight / totalWeights;
  int i = 0;
  for (int m = 0; m < M; ++m) {

    float U = r = static_cast<float>(m) * (1.0f/static_cast<float>(M));

    // Normalize the particle weights
    (*X_bar_t)[m].weight /= totalWeights;

    while ( U > c) {
      i++;
      c += (*X_bar_t)[m].weight;
    }
    X_t.push_back(randomWalkParticle((*X_bar_t)[i]));
  }
}

/**
 * Prepare for the next update step without resampling the particles
 *
 * @param X_bar_t the set of updated particles
 */
void PF::noResample(std::vector<Particle> * X_bar_t) {
  X_t = *X_bar_t;
}

/**
 * Method to update the robot pose and uncertainty estimates.
 * Calculates the weighted mean and biased standard deviations of the particles.
 */
void PF::updateEstimates()
{
  float weightSum = 0.;
  PoseEst wMeans(0.,0.,0.);
  PoseEst bSDs(0., 0., 0.);
  PoseEst best(0.,0.,0.);
  float maxWeight = 0;

  // Calculate the weighted mean
  for (unsigned int i = 0; i < X_t.size(); ++i) {
    // Sum the values
    wMeans.x += X_t[i].pose.x*X_t[i].weight;
    wMeans.y += X_t[i].pose.y*X_t[i].weight;
    wMeans.h += X_t[i].pose.h*X_t[i].weight;
    // Sum the weights
    weightSum += X_t[i].weight;

    if (X_t[i].weight > maxWeight) {
      maxWeight = X_t[i].weight;
      best = X_t[i].pose;
    }
  }

  wMeans.x /= weightSum;
  wMeans.y /= weightSum;
  wMeans.h /= weightSum;


  // Calculate the biased variances
  for (unsigned int i=0; i < X_t.size(); ++i) {
    bSDs.x += X_t[i].weight *
        (X_t[i].pose.x - wMeans.x)*
        (X_t[i].pose.x - wMeans.x);
    bSDs.y += X_t[i].weight *
        (X_t[i].pose.y - wMeans.y)*
        (X_t[i].pose.y - wMeans.y);
    bSDs.h += X_t[i].weight *
        (X_t[i].pose.h - wMeans.h)*
        (X_t[i].pose.h - wMeans.h);
  }

  bSDs.x /= weightSum;
  bSDs.x = sqrt(bSDs.x);

  bSDs.y /= weightSum;
  bSDs.y = sqrt(bSDs.y);

  bSDs.h /= weightSum;
  bSDs.h = sqrt(bSDs.h);

  // Set class variables to reflect newly calculated values
  curEst = wMeans;
  curBest = best;
  curUncert = bSDs;
}

//Helpers

/**
 * Move a particle randomly in the x, y, and h directions proportional
 * to its weight, within a certian bounds.
 *
 * @param p The particle to be random walked
 *
 * @return The walked particle
 */
Particle PF::randomWalkParticle(Particle p)
{
  p.pose.x += sampleNormalDistribution(MAX_CHANGE_X * (1.0f - p.weight));
  p.pose.y += sampleNormalDistribution(MAX_CHANGE_Y * (1.0f - p.weight));
  p.pose.h += sampleNormalDistribution(MAX_CHANGE_H * (1.0f - p.weight));

  // p.pose.x += sampleTriangularDistribution(MAX_CHANGE_X * (1.0f - p.weight));
  // p.pose.y += sampleTriangularDistribution(MAX_CHANGE_Y * (1.0f - p.weight));
  // p.pose.h += sampleTriangularDistribution(MAX_CHANGE_H * (1.0f - p.weight));

  return p;
}

float PF::sampleNormalDistribution(float sd)
{
  float samp = 0;
  for(int i = 0; i < 12; i++) {
    samp += (2*(static_cast<float>(rand()) /
                static_cast<float>(RAND_MAX)) * sd) - sd;
  }
  return 0.5f*samp;
}

float PF::sampleTriangularDistribution(float sd)
{
  return std::sqrt(6.0f)*0.5f * ((2.0f*sd*(static_cast<float>(rand()) /
                                           static_cast<float>(RAND_MAX))) - sd +
                                 (2.0f*sd*(static_cast<float>(rand()) /
                                           static_cast<float>(RAND_MAX))) - sd);
}

float PF::sampleUniformDistribution(float width)
{
  return UNIFORM_1_NEG_1*width;
}

// Particle
Particle::Particle(PoseEst _pose, float _weight) :
    pose(_pose), weight(_weight)
{
}

Particle::Particle(const Particle& other) :
    pose(other.pose), weight(other.weight)
{
}

Particle::Particle(){}

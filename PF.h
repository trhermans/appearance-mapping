#ifndef PF_h_DEFINED
#define PF_h_DEFINED

// Includes
// STL
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>

#ifdef __APPLE__
static inline float sincosf(float h, float* sinh, float* cosh)
{
  *sinh = std::sin(h);
  *cosh = std::cos(h);
}
#endif

// Odometery change
class MotionModel
{
public:
    MotionModel(float f = 0.0f, float l = 0.0f, float r = 0.0f)
        : deltaF(f), deltaL(l), deltaR(r) { }
    MotionModel(const MotionModel& other)
        : deltaF(other.deltaF), deltaL(other.deltaL), deltaR(other.deltaR) { }
    float deltaF;
    float deltaL;
    float deltaR;

    friend std::ostream& operator<< (std::ostream &o, const MotionModel &u) {
        return o << "(" << u.deltaF << ", " << u.deltaL << ", " << u.deltaR
                 << ")";
    }

};

/**
 * @brief Estimate of the robot's (x,y,heading) pose.
 */
class PoseEst
{
 public:
  // Constructors
  PoseEst(float _x, float _y, float _h) :
      x(_x), y(_y), h(_h) {}
  PoseEst(const PoseEst& other) :
      x(other.x), y(other.y), h(other.h) {}
  PoseEst() {}
  float x;
  float y;
  float h;

  PoseEst operator+ (const PoseEst o)
  {
    return PoseEst(o.x + x,
                   o.y + y,
                   o.h + h);
  }
  void operator+= (const PoseEst o)
  {
    x += o.x;
    y += o.y;
    h += o.h;
  }
  PoseEst operator+ (const MotionModel u_t)
  {
    // Translate the relative change into the global coordinate system
    // And add that to the current estimate
    float sinh, cosh;
    sincosf(h, &sinh, &cosh);
    return PoseEst(x + u_t.deltaF * cosh -
                   u_t.deltaL * sinh,
                   y + u_t.deltaF * sinh +
                   u_t.deltaL * cosh,
                   h + u_t.deltaR);
  }
  void operator+= (const MotionModel u_t)
  {
    float sinh, cosh;
    sincosf(h, &sinh, &cosh);

    // Translate the relative change into the global coordinate system
    // And add that to the current estimate
    x += u_t.deltaF * cosh - u_t.deltaL * sinh;
    y += u_t.deltaF * sinh + u_t.deltaL * cosh;
    h += u_t.deltaR;
  }

  friend std::ostream& operator<< (std::ostream &o, const PoseEst &c)
  {
    return o << "(" << c.x << ", " << c.y << ", " << c.h << ")";
  }


};
class LoopClosure
{
 public:
  LoopClosure(bool detected, PoseEst est) :
      loop_detected_(detected), detected_closure_location_(est)
  {
  }

  bool loop_detected_;
  PoseEst detected_closure_location_;
};

// Particle
class Particle
{
 public:
  Particle(PoseEst _pose, float _weight);
  Particle(const Particle& other);
  Particle();
  PoseEst pose;
  float weight;

  friend std::ostream& operator<< (std::ostream &o, const Particle &c) {
    return o << c.pose.x << " " << c.pose.y << " " << c.pose.h << " "
             << c.weight;
  }

};

// Constants
 // Minimum possible similarity
static const float MIN_SIMILARITY = static_cast<float>(1.0e-20);

// The Monte Carlo Localization class
class PF
{
 public:
  // Constructors & Destructors
  PF(int M=100);
  virtual ~PF();

  // Core Functions
  virtual void updateOdometry(MotionModel u_t);
  virtual void updateLoopClosure(LoopClosure z_t);
  virtual void reseed(PoseEst est, float spread);

  // Getters
  const PoseEst getCurrentEstimate() const { return curEst; }
  const PoseEst getCurrentUncertainty() const { return curUncert; }
  const PoseEst getCurrentBest() const { return curBest; }

  /**
   * @return The current x esitamte of the robot
   */
  const float getXEst() const {
    if (useBest) return curBest.x;
    else return curEst.x;
  }

  /**
   * @return The current y esitamte of the robot
   */
  const float getYEst() const {
    if (useBest) return curBest.y;
    else return curEst.y;
  }

  /**
   * @return The current heading esitamte of the robot in radians
   */
  const float getHEst() const {
    if (useBest) return curBest.h;
    else return curEst.h;
  }

  /**
   * @return The uncertainty associated with the x estimate of the robot.
   */
  const float getXUncert() const { return curUncert.x * 2;}

  /**
   * @return The uncertainty associated with the y estimate of the robot.
   */
  const float getYUncert() const { return curUncert.y * 2;}

  /**
   * @return The uncertainty associated with the robot's heading estimate.
   */
  const float getHUncert() const { return curUncert.h * 2;}

  /**
   * @return The uncertainty associated with the robot's heading estimate.
   */
  const MotionModel getLastOdo() const { return lastOdo; }

  /**
   * @return The current set of particles in the filter
   */
  const std::vector<Particle> getParticles() const { return X_t; }

  // Setters
  /**
   * @param xEst The current x esitamte of the robot
   */
  void setXEst(float xEst) { curEst.x = xEst;}

  /**
   * @param yEst The current y esitamte of the robot
   */
  void setYEst(float yEst) { curEst.y = yEst;}

  /**
   * @param hEst The current heading esitamte of the robot
   */
  void setHEst(float hEst) { curEst.h = hEst;}

  /**
   * @param uncertX The uncertainty of the x estimate of the robot.
   */
  void setXUncert(float uncertX) { curUncert.x = uncertX;}

  /**
   * @return uncertY The uncertainty of the y estimate of the robot.
   */
  void setYUncert(float uncertY) { curUncert.y = uncertY;}

  /**
   * @param uncertH The uncertainty of the robot's heading estimate.
   */
  void setHUncert(float uncertH) { curUncert.h = uncertH;}

  void setUseBest(bool _new) { useBest = _new; }

 private:
  // Class variables
  PoseEst curEst; // Current {x,y,h} esitamates
  PoseEst curBest; // Current {x,y,h} esitamate of the highest weighted particle
  PoseEst curUncert; // Associated {x,y,h} uncertainties (standard deviations)
  std::vector<Particle> X_t; // Current set of particles
  std::vector<Particle> X_bar_t; // A priori estimates
  bool useBest;
  MotionModel lastOdo;

  // Core Functions
  PoseEst updateMotionModel(PoseEst x_t, MotionModel u_t);
  void resample(std::vector<Particle> * X_bar_t, float totalWeights);
  void lowVarianceResample(std::vector<Particle> * X_bar_t,
                           float totalWeights);
  void noResample(std::vector<Particle> * X_bar_t);
  void updateEstimates();

  // Helpers
  Particle randomWalkParticle(Particle p);
  float sampleNormalDistribution(float sd);
  float sampleTriangularDistribution(float sd);
  float sampleUniformDistribution(float sd);

 public:
  // friend std::ostream& operator<< (std::ostream &o, const PF &c) {
  //     return o << "Est: " << c.curEst << "\nUnct: " << c.curUncert;
  // }
  int frameCounter;
  const int M; // Number of particles
};

#endif // _PF_H_DEFINED

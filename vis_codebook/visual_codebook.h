#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <vector>
#include <string>

class VisualCodebook
{
 public:
  // Constructors and Destructors
  VisualCodebook(int k=0, bool use128_surf = true);

  // Core Methods
  void constructCodebook(std::string img_path, int img_count, int mode=0);
  std::vector<int> getCodewords(IplImage& img);
  const int nearestCenter(const float * desc);

  // IO
  void saveCodebook(std::string path);
  void loadCodebook(std::string path);

  // Getters and Setters
  const int numCenters() const { return static_cast<const int>(k_); }
  void setHessianThresh(int thresh) { hessian_thresh_ = thresh; }
  void setKmeansIter(int iters) { kmeans_iter_ = iters; }

 protected:
  int k_;
  int desc_size_;
  int hessian_thresh_;
  int kmeans_iter_;
  bool use128_surf_;
  std::vector<std::vector<float> > centers_;
};

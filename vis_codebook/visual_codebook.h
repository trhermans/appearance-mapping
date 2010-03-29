#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <vector>
#include <string>

class VisualCodebook
{
 public:
  VisualCodebook(int k);
  void constructCodebook(std::string img_path, int img_count);
  void saveCodebook(std::string path);
  void loadCodebook(std::string path);
  std::vector<int> getCodewords(IplImage& img);

 protected:
  int k_;
  std::vector<std::vector< int> > centers_;
};

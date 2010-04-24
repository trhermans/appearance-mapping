// Author: Jianxin Wu (wujx2001@gmail.com)

#include <vector>
#include <string>
#include <sstream>
#include <climits>
#include <cassert>

#include <omp.h>

#include "util.h"
#include "Codebook.h"
#include "UseSVM.h"
#include "Datasets.h"

// fold of cross validation
static const int sizeCV = 5;
// a patch is 'windowSize' times 'windowSize'
static int windowSize = 16;
// maximum possible value of any component of a feature vector
static const int upper_bound = 129;
// desired (approximate ) L1 norm of a feature vector
static const int L1_norm = 128;
static bool normalize = false;

// do we use the Sobel gradient image instead of original image?
static bool useSobel = false;
// do we use Sobel + original image?
static bool useBoth = false;
// do we use one class SVM to generate code words?
static bool oneclassSVM = false;
// number of code words (clusters)
static int K = 200;
// resize image to have width='resizeWidth' if 'resizeWidth'>0
static int resizeWidth = 256;
// dense sample step size when generate features for clustering into codewords
static int kmeans_stepsize = 16;
// depth of sPACT hierarchy
static int splitlevel = 0;
// sample the original image, plus 'scaleChanges' of resized version of
// the images
static int scaleChanges = 0;
// relative importance in different layers of the spatial hierarchy
static double ratio = 1.0;
// dense sample step size in generating training/testing data
static int stepSize = 8;

static int trainsize = -1; // number of training images in each category
static int testsize = -1; // number of testing images in each category
static int totalsize = -1; // total length of feature vector
static int numClass = -1; // number of categories
static int numAffordances = -1; // number of categories
static bool usingAffordances = false;
static CodeBook::DESCRIPTOR_TYPE featureType = CodeBook::DESCRIPTOR_CENTRIST;
static CodeBook::CODEBOOK_TYPE codebookType = CodeBook::CODEBOOK_HIK;

static Array2dC<int> labels; // labels of all images (training, testing, ignored, ...)
static Array2dC<int> affValues; // affordance values of all images
static std::vector<std::string> filenames; // disk file names of all images

// assuming images in the same category are contiguous starting index and number
// of images in each category
static Array2dC<int> classStartings, classSizes;

static std::ostringstream name;
// list of file names of training images
static std::vector<const char*> train_names;
static Array2dC<int> split; // -1: not using; 0: training; 1:testing
// feature vector length of 1st & 2nd codebook, and their sum
static int fsize=0, fsize2=0, fsizetotal=0;
static CodeBook* codebook=NULL; // 1st codebook
static CodeBook* codebook2=NULL; // 2nd codebook, if useBoth is set to true

CodeBook* CodeBookEngine(const CodeBook::CODEBOOK_TYPE codebookType,
                         const CodeBook::DESCRIPTOR_TYPE featureType,
                         const bool useSobel,
                         const bool oneclassSVM,
                         int& fsize)
{
  CodeBook* codebook = NULL;
  switch(codebookType) {
    // NOTE: Use L1_norm
    case CodeBook::CODEBOOK_KMEANS:
      codebook = new LinearCodes(featureType,useSobel,resizeWidth,
                                 windowSize,L1_norm,false);
      break;
    case CodeBook::CODEBOOK_KMEDIAN:
      codebook = new LinearCodes(featureType,useSobel,resizeWidth,
                                 windowSize,L1_norm,true);
      break;
    case CodeBook::CODEBOOK_HIK:
      codebook = new HistogramCodes(featureType,useSobel,resizeWidth,
                                    windowSize,L1_norm,upper_bound);
      break;
    case CodeBook::CODEBOOK_LOCAL:
      codebook = new LocalCodes(useSobel,resizeWidth);
      break;
    default:
      std::cout<<"Codebook type is wrong."<<std::endl;
      break;
  }
  if(codebookType==CodeBook::CODEBOOK_KMEANS ||
     codebookType==CodeBook::CODEBOOK_KMEDIAN ||
     codebookType==CodeBook::CODEBOOK_HIK) {
    std::cout<<"Generating data for kernel k-means clustering."<<std::endl;
    codebook->GenerateClusterData(train_names,kmeans_stepsize);
    std::cout<<"Now do cluster and generate codewords."<<std::endl;
  }
  codebook->SetVerbose(true);
  int valid = codebook->GenerateCodeWords(K,oneclassSVM);
  if(splitlevel==0) fsize=valid;
  else if(splitlevel==1) fsize=valid*6;
  else if(splitlevel==2) fsize=valid*31;
  return codebook;
}

void SplitData()
{   // split data into training/testing/ignored
  split.Create(1,totalsize);
  train_names.clear();
  for(int c=0;c<numClass;c++)
  {   // randomly permute a category, and decide training/testing/ignored examples
    Array2dC<int> perm(1,classSizes.buf[c]);
    GenerateRandomPermutation(perm.buf,perm.ncol);
    assert(trainsize<=classSizes.buf[c]);
    for(int i=0;i<trainsize;i++) // training
    {
      split.buf[classStartings.buf[c]+perm.buf[i]] = 0;
      train_names.push_back(filenames[classStartings.buf[c]+
                                      perm.buf[i]].c_str());
    }
    for(int i=trainsize;i<std::min<int>(classSizes.buf[c],trainsize+testsize);i++) {
      split.buf[classStartings.buf[c]+perm.buf[i]] = 1; // testing
    }
    for(int i=std::min<int>(classSizes.buf[c],trainsize+testsize);
        i<classSizes.buf[c];i++) {
      split.buf[classStartings.buf[c]+perm.buf[i]] = -1; // ignored
    }
  }
}

void GenerateDatasetsSeparateCodebook()
{   // Generate training/testing data using CODEBOOK_Separate
  unsigned seed = 1194575810;
  //unsigned seed=(unsigned)time(NULL);
  my_srand(seed);
  Array2d<double> traindata,testdata;
  Array2d<int> trainAffVals,testAffVals;
  Array2dC<int> trainlabels,testlabels,trainmapping,testmapping;
  int trainsetsize,testsetsize;
  for(int r=0;r<sizeCV;r++)
  {
    SplitData();
    trainsetsize = trainsize*numClass;
    testsetsize = 0;
    for(int c=0;c<numClass;c++) {
      testsetsize += std::min<int>(classSizes.buf[c]-trainsize,testsize);
    }
    trainlabels.Create(1,trainsetsize); testlabels.Create(1,testsetsize);
    trainmapping.Create(1,trainsetsize); testmapping.Create(1,testsetsize);
    int trainadded = 0, testadded = 0;
    for(int i=0;i<totalsize;i++) {
      // the i-th training image is number 'trainmapping[i]' in the entire
      // image list -- similar for testing images
      if(split.buf[i]==0) trainmapping.buf[trainadded++]=i;
      if(split.buf[i]==1) testmapping.buf[testadded++]=i;
    }
    assert(trainadded==trainsetsize && testadded==testsetsize);

    fsize = fsize2 = fsizetotal = 0;
    codebook = CodeBookEngine(codebookType,featureType,useSobel,oneclassSVM,
                              fsize);
    if(useBoth) {
      codebook2 = CodeBookEngine(codebookType,featureType,true,
                                 oneclassSVM,fsize2);
    }
    fsizetotal = fsize + fsize2; // total length of the feature vector
    traindata.Create(trainsetsize,fsizetotal);
    testdata.Create(testsetsize,fsizetotal);

    if(usingAffordances) {
      trainAffVals.Create(trainsetsize,numAffordances);
      testAffVals.Create(testsetsize,numAffordances);
    }

    std::cout<<"Now generating training and testing examples."<<std::endl;
    StartOfDuration();
#pragma omp parallel for
    for(int i=0;i<trainsetsize;i++) {
      codebook->TranslateOneImage(filenames[trainmapping.buf[i]].c_str(),
                                  stepSize,splitlevel,traindata.p[i],
                                  fsize,normalize,ratio,scaleChanges);
      if(useBoth) {
        codebook2->TranslateOneImage(filenames[trainmapping.buf[i]].c_str(),
                                     stepSize,splitlevel,
                                     traindata.p[i]+fsize,fsize2,
                                     normalize,ratio,scaleChanges);
      }
      trainlabels.buf[i] = labels.buf[trainmapping.buf[i]];

      if(usingAffordances) {
        for(int j = 0; j < trainAffVals.ncol; ++j) {
          trainAffVals.p[i][j] = affValues.p[trainmapping.buf[i]][j];
        }
      }

      if(omp_get_thread_num()==0 && i%10==0) {
        std::cout<<".";
      }
      std::cout.flush();
    }
#pragma omp parallel for
    for(int i=0; i < testsetsize; i++) {
      codebook->TranslateOneImage(filenames[testmapping.buf[i]].c_str(),
                                  stepSize, splitlevel, testdata.p[i],
                                  fsize, normalize, ratio, scaleChanges);

      if(useBoth) codebook2->TranslateOneImage(
             filenames[testmapping.buf[i]].c_str(), stepSize, splitlevel,
             testdata.p[i] + fsize, fsize2, normalize, ratio, scaleChanges);
      testlabels.buf[i] = labels.buf[testmapping.buf[i]];

      if(usingAffordances) {
        for(int j = 0; j < testAffVals.ncol; ++j) {
          testAffVals.p[i][j] = affValues.p[testmapping.buf[i]][j];
        }
      }

      if(omp_get_thread_num()==0 && i%10==0)
        std::cout<<".";
      std::cout.flush();
    }
    delete codebook; codebook = NULL;
    if(codebook2) { delete codebook2; codebook2 = NULL; }

    name.str(""); name << "train" << r+1 << ".txt";

    if(usingAffordances) {
      UseSVM_SaveSparseAffordance(name.str(),
                                  trainlabels.buf,
                                  trainAffVals,
                                  traindata);
    } else {
      UseSVM_SaveSparse(name.str(),trainlabels.buf,traindata);
    }

    name.str(""); name << "test" << r+1 << ".txt";

    if(usingAffordances) {
      UseSVM_SaveSparseAffordance(name.str(),
                                  testlabels.buf,
                                  testAffVals,
                                  testdata);
    } else {
      UseSVM_SaveSparse(name.str(), testlabels.buf, testdata);
    }

    std::cout << std::endl;
    std::cout << "Generated in "<< EndOfDuration()/1000 << " seconds."
              << std::endl;
    std::cout << "--------------------------------------------------------------------------"
              << std::endl;
  }
}

void GenerateDatasetsSingleCodebook()
{   // Generate training/testing data using CODEBOOK_Single
  // since the same codebook is used, an image will have the same output vector in different cross validation splittings
  unsigned seed = 1194575810;
  //unsigned seed=(unsigned)time(NULL);
  my_srand(seed);
  Array2dC<int> split(1,totalsize); // -1: not using; 0: training; 1:testing
  Array2d<double> data;
  Array2dC<bool> dataGenerated(1,totalsize); // whether the output for an image has been generated
  std::fill_n(dataGenerated.buf,totalsize,false);
  Array2dC<int> mapping;
  int usefulsize = 0;
  codebook = NULL;
  codebook2 = NULL;
  for(int r=0;r<sizeCV;r++)
  {
    SplitData();
    usefulsize = 0; // number of images used in this splitting
    for(int c=0;c<numClass;c++) usefulsize += (trainsize+std::min<int>(classSizes.buf[c]-trainsize,testsize));
    mapping.Create(1,usefulsize);
    int mapped = 0;
    for(int i=0;i<totalsize;i++)
    {   // the i-th useful image is number 'mapping[i]' in the entire list of images
      if(split.buf[i]==0) mapping.buf[mapped++]=i;
      if(split.buf[i]==1) mapping.buf[mapped++]=i;
    }
    assert(mapped==usefulsize);

    if(r==0)
    {   // generate the codebook if it is not ready
      fsize = fsize2 = fsizetotal = 0;
      codebook = CodeBookEngine(codebookType,featureType,useSobel,oneclassSVM,fsize);
      if(useBoth) codebook2 = CodeBookEngine(codebookType,featureType,true,oneclassSVM,fsize2);
      fsizetotal = fsize + fsize2;
      data.Create(totalsize,fsizetotal);
    }

    std::cout<<"Now generating training and testing examples."<<std::endl;
    StartOfDuration();
    int newsize = 0; // number of images need to be processed in this splitting
    // only if an image is useful in this splitting and it has not been generated, we process it in this splitting
    for(int i=0;i<mapping.ncol;i++) if(dataGenerated.buf[mapping.buf[i]]==false) newsize++;
    if(newsize>0)
    {
      Array2dC<int> newMapped(1,newsize);
      mapped = 0; for(int i=0;i<mapping.ncol;i++) if(dataGenerated.buf[mapping.buf[i]]==false) newMapped.buf[mapped++] = mapping.buf[i];
      assert(mapped==newMapped.ncol);
#pragma omp parallel for
      for(int i=0;i<newMapped.ncol;i++)
      {
        codebook->TranslateOneImage(filenames[newMapped.buf[i]].c_str(),stepSize,splitlevel,data.p[newMapped.buf[i]],fsize,normalize,ratio,scaleChanges);
        if(useBoth) codebook2->TranslateOneImage(filenames[newMapped.buf[i]].c_str(),stepSize,splitlevel,data.p[newMapped.buf[i]]+fsize,fsize2,normalize,ratio,scaleChanges);
        dataGenerated.buf[newMapped.buf[i]] = true;
        if(omp_get_thread_num()==0 && i%10==0) std::cout<<"."; std::cout.flush();
      }
    }

    name.str(""); name<<"train"<<r+1<<".txt";
    UseSVM_SaveSparse(name.str(),labels.buf,data,split.buf,0);
    name.str(""); name<<"test"<<r+1<<".txt";
    UseSVM_SaveSparse(name.str(),labels.buf,data,split.buf,1);

    std::cout<<std::endl;
    std::cout<<"Generated in "<<EndOfDuration()/1000<<" seconds."<<std::endl;
    std::cout<<"--------------------------------------------------------------------------"<<std::endl;
  }
  delete codebook; codebook = NULL;
  if(codebook2) { delete codebook2; codebook2 = NULL; }
}

void GenerateFilesForCaltech101()
    // Generate 5-fold random splitting of training/testing data for Caltech 101 dataset
    // Note that functions in this file uses a lot of global parameters
    // so you MUST prepare all parameters correctly BEFORE call functions to generate data
    // data are saved in SVM format for use with HIK version SVM software
{
  totalsize = 8677;
  numClass = 101;
  trainsize = 15;
  testsize = 20;

  K = 200;
  resizeWidth = 256;
  kmeans_stepsize = 16;
  splitlevel = 2;
  scaleChanges = 4;
  ratio = 2.0;
  normalize = false;
  stepSize = 8;
  useSobel = false;
  useBoth = false;
  oneclassSVM = false;

  filenames.resize(totalsize);
  labels.Create(1,totalsize);
  classStartings.Create(1,numClass);
  classSizes.Create(1,numClass);

  std::ifstream dlist("../Data/images_101.txt");
  int added = 0;
  std::ofstream out("../Data/images_101_files.txt");
  out<<totalsize<<" 101"<<std::endl;
  for(int c=0;c<numClass;c++)
  {
    std::ostringstream dirname; std::string temp;
    dirname.str(""); dirname<<"../Data/101_ObjectCategories/";
    dlist>>temp;
    dirname<<temp; dirname<<"/";

    dlist>>classSizes.buf[c]; assert(classSizes.buf[c]>0);
    for(int i=0;i<classSizes.buf[c];i++)
    {
      name.str(""); name<<dirname.str()<<"image_"; name.fill('0');
      name.width(4); name<<i+1<<".jpg";
      labels.buf[added] = c;
      filenames[added] = name.str();
      out<<filenames[added]<<' '<<c<<std::endl;
      added++;
    }
  }
  out.close();
  assert(added==totalsize);
  dlist.close();
  classStartings.buf[0]=0; for(int i=1;i<numClass;i++) classStartings.buf[i] = classStartings.buf[i-1] + classSizes.buf[i-1];
}

void GenerateFilesForScene15()
{   // prepare the parameters and inputs for 15 scene category dataset
  const int imagesizes[] = {216,289,241, 311,210,360, 328,260,308, 374,410,292, 356,215,315};
  const char* imagenames[] = {"../Data/scene/bedroom/",
                              "../Data/scene/livingroom/",
                              "../Data/scene/CALsuburb/",
                              "../Data/scene/industrial/",
                              "../Data/scene/kitchen/",
                              "../Data/scene/MITcoast/",
                              "../Data/scene/MITforest/",
                              "../Data/scene/MIThighway/",
                              "../Data/scene/MITinsidecity/",
                              "../Data/scene/MITmountain/",
                              "../Data/scene/MITopencountry/",
                              "../Data/scene/MITstreet/",
                              "../Data/scene/MITtallbuilding/",
                              "../Data/scene/PARoffice/",
                              "../Data/scene/store/"};

  totalsize = 4485;
  numClass = 15;
  trainsize = 100;
  testsize = totalsize; // set 'testsize=totalsize' such that all non-training examples are used in testing

  K = 200;
  resizeWidth = 0;    // '0' means not resize the input image
  kmeans_stepsize = 16;
  splitlevel = 2;
  scaleChanges = 4;
  ratio = 2;
  normalize = false;
  stepSize = 2;
  useSobel = false;
  useBoth = false;
  oneclassSVM = false;

  assert(numClass==sizeof(imagesizes)/sizeof(int) && numClass==sizeof(imagenames)/sizeof(const char*));
  assert(std::accumulate(imagesizes,imagesizes+numClass,0)==totalsize);
  filenames.resize(totalsize);
  labels.Create(1,totalsize);
  classStartings.Create(1,numClass);
  classSizes.Create(1,numClass);

  std::copy(imagesizes,imagesizes+numClass,classSizes.buf);
  for(int i=0;i<totalsize;i++)
  {
    int curindex = i + 1; // base 0 and base 1 conversion
    int curClass = 0;
    while(curindex>imagesizes[curClass])
    {
      curindex -= imagesizes[curClass];
      curClass++;
    }
    std::ostringstream buf;
    buf.str(""); buf<<imagenames[curClass]; buf<<"image_"; buf.fill('0'); buf.width(4); buf<<curindex; buf<<".jpg";
    filenames[i] = buf.str();
    labels.buf[i] = curClass;
  }
  classStartings.buf[0]=0; for(int i=1;i<numClass;i++) classStartings.buf[i] = classStartings.buf[i-1] + classSizes.buf[i-1];
}

void GenerateFilesForEvent8()
{   // prepare the parameters and inputs for 8 sports events dataset
  const int imagesizes[] = {200,137,236,182,194,250,190,190};
  const char* imagenames[] = {"../Data/event_dataset/badminton/",
                              "../Data/event_dataset/bocce/",
                              "../Data/event_dataset/croquet/",
                              "../Data/event_dataset/polo/",
                              "../Data/event_dataset/RockClimbing/",
                              "../Data/event_dataset/rowing/",
                              "../Data/event_dataset/sailing/",
                              "../Data/event_dataset/snowboarding/"};

  totalsize = 1579;
  numClass = 8;
  trainsize = 70;
  testsize = 60;

  K = 200;
  resizeWidth = 512;
  kmeans_stepsize = 24;
  splitlevel = 2;
  scaleChanges = 4;
  ratio = 2.0;
  normalize = false;
  stepSize = 8;
  useSobel = false;
  useBoth = false;
  oneclassSVM = false;

  assert(numClass==sizeof(imagesizes)/sizeof(int) &&
         numClass==sizeof(imagenames)/sizeof(const char*));
  assert(std::accumulate(imagesizes,imagesizes+numClass,0)==totalsize);
  filenames.resize(totalsize);
  labels.Create(1,totalsize);
  classStartings.Create(1,numClass);
  classSizes.Create(1,numClass);

  std::copy(imagesizes,imagesizes+numClass,classSizes.buf);
  std::ifstream flist("../Data/event_files.txt");

  int added = 0;
  std::string filename;
  for(int c=0;c<numClass;c++)
  {
    for(int i=0;i<classSizes.buf[c];i++)
    {
      flist>>filename;
      name.str(""); name<<imagenames[c]<<filename;
      labels.buf[added] = c;
      filenames[added] = name.str();
      added++;
    }
  }
  assert(added==totalsize);
  flist.close();
  classStartings.buf[0]=0;
  for(int i=1;i<numClass;i++) {
    classStartings.buf[i] = classStartings.buf[i-1] + classSizes.buf[i-1];
  }
}

void GenerateFilesForJAFFE()
{   // prepare the parameters and inputs for 8 sports events dataset
  const int imagesizes[] = {30, 29, 32, 31, 30, 31, 30};
  const char* imagenames[] = {"../Data/JAFFE/ang/",
                              "../Data/JAFFE/dis/",
                              "../Data/JAFFE/fea/",
                              "../Data/JAFFE/hap/",
                              "../Data/JAFFE/nut/",
                              "../Data/JAFFE/sad/",
                              "../Data/JAFFE/sur/"};

  totalsize = 0;
  numClass = 7;
  for(int c = 0; c < numClass; ++c) {
    totalsize += imagesizes[c];
  }
  trainsize = 20;
  testsize = totalsize;

  K = 2000;
  resizeWidth = 512;
  kmeans_stepsize = 24;
  splitlevel = 2;
  scaleChanges = 4;
  ratio = 2.0;
  normalize = false;
  stepSize = 8;
  useSobel = false;
  useBoth = false;
  oneclassSVM = false;

  assert(numClass==sizeof(imagesizes)/sizeof(int) &&
         numClass==sizeof(imagenames)/sizeof(const char*));
  assert(std::accumulate(imagesizes,imagesizes+numClass,0)==totalsize);
  filenames.resize(totalsize);
  labels.Create(1,totalsize);
  classStartings.Create(1,numClass);
  classSizes.Create(1,numClass);

  std::copy(imagesizes,imagesizes+numClass,classSizes.buf);
  std::ifstream flist("../Data/JAFFE/jaffe_files.txt");
  int added = 0;
  std::string filename;
  for(int c=0; c < numClass; c++)
  {
    for(int i=0; i < classSizes.buf[c]; i++)
    {
      flist >> filename;
      name.str("");
      name << imagenames[c] << filename;
      labels.buf[added] = c;
      filenames[added] = name.str();
      added++;
    }
  }
  assert(added==totalsize);
  flist.close();
  classStartings.buf[0]=0;
  for(int i=1;i<numClass;i++) {
    classStartings.buf[i] = classStartings.buf[i-1] + classSizes.buf[i-1];
  }
}
void GenerateFilesForRovio()
{   // prepare the parameters and inputs for 5 different object categories
  const int imagesizes[] = {40, 75, 77, 75, 48};
  const char* imagenames[] = {"../Data/rovio/ball/",
                              "../Data/rovio/box/",
                              "../Data/rovio/container/",
                              "../Data/rovio/towel/",
                              "../Data/rovio/carpet/"};
  totalsize = 315;
  numClass = 5;
  trainsize = 25; // We will only use some of these as labeled data
  testsize = totalsize;
  numAffordances = 5;
  usingAffordances = true;

  K = 100;
  resizeWidth = 512;
  kmeans_stepsize = 24;
  splitlevel = 2;
  scaleChanges = 4;
  ratio = 2.0;
  normalize = false;
  stepSize = 8;
  useSobel = false;
  useBoth = false;
  oneclassSVM = false;

  assert(numClass==sizeof(imagesizes)/sizeof(int) &&
         numClass==sizeof(imagenames)/sizeof(const char*));
  assert(std::accumulate(imagesizes,imagesizes+numClass,0)==totalsize);
  filenames.resize(totalsize);
  labels.Create(1,totalsize);
  classStartings.Create(1,numClass);
  classSizes.Create(1,numClass);

  // Make a 2d array of store the affordance values
  affValues.Create(totalsize,numAffordances);

  std::copy(imagesizes,imagesizes+numClass,classSizes.buf);
  std::ifstream flist("../Data/rovio/rovio_files.txt");
  std::ifstream aflist("../Data/rovio/affordance_labels.txt");

  int added = 0;
  std::string filename;
  for(int c=0; c < numClass; c++)
  {
    for(int i=0; i < classSizes.buf[c]; i++)
    {
      int d;
      // Record affordance values
      for(int k = 0; k < numAffordances; ++k) {
        aflist >> d;
        affValues.p[added][k] = d;
      }

      flist >> filename;
      name.str("");
      name << imagenames[c] << filename;
      labels.buf[added] = c;
      filenames[added] = name.str();
      added++;
    }
  }
  assert(added==totalsize);
  flist.close();
  classStartings.buf[0]=0;
  for(int i=1;i<numClass;i++) {
    classStartings.buf[i] = classStartings.buf[i-1] + classSizes.buf[i-1];
  }
}

void GenerateDatasets(const int dataset,
                      const int multiplicity,
                      const CodeBook::DESCRIPTOR_TYPE featuretype,
                      const CodeBook::CODEBOOK_TYPE codebooktype)
{   // As the function names says it -- generate the training/testing data
  assert(multiplicity==CODEBOOK_Separate || multiplicity==CODEBOOK_Single);
  assert(CodeBook::ValidFeatureType(featuretype));

  featureType = featuretype;
  codebookType = codebooktype;

  switch(dataset) {
    case DATASET_Caltech101:
      GenerateFilesForCaltech101();
      break;
    case DATASET_Scene15:
      GenerateFilesForScene15();
      break;
    case DATASET_Event8:
      GenerateFilesForEvent8();
      break;
    case DATASET_JAFFE:
      GenerateFilesForJAFFE();
      break;
    case DATASET_ROVIO:
      GenerateFilesForRovio();
      usingAffordances = true;
      break;
    case DATASET_CITY_CENTRE:
      break;
    case DATASET_NEW_COLLEGE:
      break;
    default:
      std::cout<<"The index for datasets is not correct."<<std::endl;
      break;
  }
  if(useBoth==true) useSobel = false;
  if(codebookType==CodeBook::CODEBOOK_LOCAL) stepSize = 1;

  if(multiplicity==CODEBOOK_Separate)
    GenerateDatasetsSeparateCodebook();
  else
    GenerateDatasetsSingleCodebook();
}

void TranslateImages(const int dataset,
                     const CodeBook::DESCRIPTOR_TYPE featuretype,
                     const CodeBook::CODEBOOK_TYPE codebooktype)
{
  featureType = featuretype;
  codebookType = codebooktype;

  switch(dataset) {
    case DATASET_Caltech101:
      GenerateFilesForCaltech101();
      break;
    case DATASET_Scene15:
      GenerateFilesForScene15();
      break;
    case DATASET_Event8:
      GenerateFilesForEvent8();
      break;
    case DATASET_JAFFE:
      GenerateFilesForJAFFE();
      break;
    default:
      std::cout<<"The index for datasets is not correct."<<std::endl;
      break;
  }
  K = 256;
  oneclassSVM = false;

  int fsize;
  SplitData();
  CodeBook* codebook = CodeBookEngine(codebookType,featureType,
                                      useSobel,oneclassSVM,fsize);
  cvNamedWindow("image");
  cvNamedWindow("translated");
  resizeWidth = 0;
  while(true)
  {
    std::cout<<"filename: ";
    std::string filename;
    std::cin>>filename;
    IntImage<double> img;
    if(filename=="exit") break;
    if(img.Load(filename)==false)
    {
      std::cout<<"File "<<filename<<" loading failed."<<std::endl;
      continue;
    }
    img.Show("image");
    cvWaitKey();
    IplImage* translated = codebook->TranslateOneImage(filename.c_str(),K);
    cvShowImage("translated",translated);
    cvSaveImage((filename+".png").c_str(),translated);
    cvWaitKey();
    cvReleaseImage(&translated);
  }
  delete codebook; codebook = NULL;
}


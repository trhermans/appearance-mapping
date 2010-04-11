// Author: Jianxin Wu (wujx2001@gmail.com)
// This is 1) how to use visual codebook and visual descriptor included in our ICCV 2009 paper
//         2) an example of how to use the APIs included in this package

#include <iostream>
#include <string>
#include <cstdlib>

#include "Codebook.h"
#include "Datasets.h"

void usage()
{
  std::cout<<"Usage: generate_data dataset  descriptor codebook"<<std::endl;
  std::cout<<" dataset: caltech | scene | sports | jaffe | rovio | city-centre | new-college"<<std::endl;
  std::cout<<" descriptor: CENTRIST | SIFT"<<std::endl;
  std::cout<<" codebook: HIK | kmeans | kmedian"<<std::endl;
}

int FindDataset(const char* dataset)
{
  const char* names[] = {"caltech","scene","sports","jaffe","rovio",
                         "city-centre", "new-college"};
  const int num_names = 7;
  std::string name = dataset;
  for(int i=0; i< num_names; i++)
  {
    if(name==names[i])
      return i;
  }
  std::cout<<"Dataset name is not correct."<<std::endl;
  usage();
  exit(-1);
}

CodeBook::CODEBOOK_TYPE FindCodeBook(const char* codebook)
{
  std::string name = codebook;
  if(name=="HIK")
    return CodeBook::CODEBOOK_HIK;
  else if(name=="kmeans")
    return CodeBook::CODEBOOK_KMEANS;
  else if(name=="kmedian")
    return CodeBook::CODEBOOK_KMEDIAN;
  else
  {
    std::cout<<"Codebook name is not correct."<<std::endl;
    usage();
    exit(-1);
  }
}

CodeBook::DESCRIPTOR_TYPE FindDescriptor(const char* descriptor)
{
  std::string name = descriptor;
  if(name=="CENTRIST")
    return CodeBook::DESCRIPTOR_CENTRIST;
  else if(name=="SIFT")
    return CodeBook::DESCRIPTOR_SIFT;
  else
  {
    std::cout<<"Visual descriptor type is not correct."<<std::endl;
    usage();
    exit(-1);
  }
}

int main(int argc, char** argv)
{
  if(argc!=4)
  {
    usage();
    exit(-1);
  }

  GenerateDatasets(FindDataset(argv[1]),
                   CODEBOOK_Separate,
                   FindDescriptor(argv[2]),
                   FindCodeBook(argv[3]));

  return 0;
}

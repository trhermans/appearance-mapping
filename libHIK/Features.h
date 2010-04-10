// Author: Jianxin Wu (wujx2001@gmail.com)
// Various visual descriptors

#ifndef __FEATURES_H
#define __FEATURES_H

#include "mdarray.h"
#include "IntImage.h"

template<class T>
inline int CensusTransform(T** p,const int x,const int y)
{
    int index = 0;
    if(p[x][y]<=p[x-1][y-1]) index += 0x80;
    if(p[x][y]<=p[x-1][y]) index += 0x40;
    if(p[x][y]<=p[x-1][y+1]) index += 0x20;
    if(p[x][y]<=p[x][y-1]) index += 0x10;
    if(p[x][y]<=p[x][y+1]) index += 0x08;
    if(p[x][y]<=p[x+1][y-1]) index += 0x04;
    if(p[x][y]<=p[x+1][y]) index += 0x02;
    if(p[x][y]<=p[x+1][y+1]) index += 0x01;
    return index;
}

template<class T>
inline int CensusTransform3(T** p,const int x,const int y)
{
// Census transform separating a==b from a<b, there are 3^8 = 6561 bins
    int index = 0;
    if(p[x][y]<p[x-1][y-1]) index++; else if(p[x][y]>p[x-1][y-1])   index+=2; index *= 3;
    if(p[x][y]<p[x-1][y])   index++; else if(p[x][y]>p[x-1][y])     index+=2; index *= 3;
    if(p[x][y]<p[x-1][y+1]) index++; else if(p[x][y]>p[x-1][y+1])   index+=2; index *= 3;
    if(p[x][y]<p[x][y-1])   index++; else if(p[x][y]>p[x][y-1])     index+=2; index *= 3;
    if(p[x][y]<p[x][y+1])   index++; else if(p[x][y]>p[x][y+1])     index+=2; index *= 3;
    if(p[x][y]<p[x+1][y-1]) index++; else if(p[x][y]>p[x+1][y-1])   index+=2; index *= 3;
    if(p[x][y]<p[x+1][y])   index++; else if(p[x][y]>p[x+1][y])     index+=2; index *= 3;
    if(p[x][y]<p[x+1][y+1]) index++; else if(p[x][y]>p[x+1][y+1])   index+=2;
    return index;
}


template<class T>
inline int SymmetricCensusTransform(T** p,const int x,const int y)
// Similar to CT, however, only compare pixels diagonally, there are 16 bins
{
    int index = 0;
    if(p[x-1][y]<=p[x+1][y]) index += 0x08;
    if(p[x-1][y+1]<=p[x+1][y-1]) index += 0x04;
    if(p[x][y-1]<=p[x][y+1]) index += 0x02;
    if(p[x-1][y-1]<=p[x+1][y+1]) index += 0x01;
    return index;
}

class BaseFeature
{
protected:
    const bool useSobel; // whether use original image or Sobel gradient image
    const int L1_norm; // we want the feature vector to (at least approximately) have L1 norm equal 'L1_norm'
    int length; // length of the feature vector
    Array2dC<double> d_feature; // real valued ('D' -- double) version of feature vectors
    Array2dC<int> i_feature; // discrete version ('I' -- integer)
    IntImage<double> img,original; // currently processed image (if useSobel==true, 'img' is Sobel gradient of 'original'
public:
    BaseFeature(const bool _useSobel,const int _L1_norm)
        :useSobel(_useSobel),L1_norm(_L1_norm),length(0)
    { }
    virtual ~BaseFeature()
    {
        length = 0; img.Clear(); d_feature.Clear(); i_feature.Clear();
    }
public:
    static const int MagicNumber = 65536; // for special case when a region is almost uniform
public:
    int Length() { return length; }
    double* D_pointer() { return d_feature.buf; }
    int* I_pointer() { return i_feature.buf; }
    const IntImage<double>& Image() { return img; }
    const IntImage<double>& OriginalImage() { return original; }
public:
    // assign an image ('filename') to the feature vector. if 'resizeWidth'>0, the image will be resized to have width equal it
    virtual const IntImage<double>& AssignFile(const char* filename,const int resizeWidth);
    // if features from multiple resolution of the image is needed, we need to resize the image by 'ratio'
    virtual const IntImage<double>& AssignImage(IntImage<double>& image,const int resizeWidth);
    virtual const IntImage<double>& ResizeImage(const double ratio);
    // generate real valued version of the feature vector for patch [x1..x2) X [y1..y2)
    virtual double* D_feature(const int x1,const int x2,const int y1,const int y2);
    // generate discrete version of the feature vector for patch [x1..x2) X [y1..y2)
    virtual int* I_feature(const int x1,const int x2,const int y1,const int y2) { return NULL; }
};

class SiftFeature: public BaseFeature
{   // SIFT visual descriptor
private:
    Array2dC<double> grad;
    Array2dC<int> orient; // magnitude and orientation of the image graidents
private:
    void GradientAndOrientation(); // compute 'grad' and 'orient' for an image
    void Accumulate_Gradient(const int x1,const int x2,const int y1,const int y2);
public:
    SiftFeature(const bool _useSobel,const int _L1_norm);
    ~SiftFeature() { }
public:
    const IntImage<double>& AssignFile(const char* filename,const int resizeWidth);
    const IntImage<double>& AssignImage(IntImage<double>& image,const int resizeWidth);
    const IntImage<double>& ResizeImage(const double ratio);
    int* I_feature(const int x1,const int x2,const int y1,const int y2);
};

class CENTRIST: public BaseFeature
{   // CENTRIST visual descriptor (CENsus TRansform hISTogram)
private:
    Array2dC<int> ct; // image of census transform values
private:
    void CTimage(); // compute 'ct'
public:
    CENTRIST(const bool _useSobel,const int _L1_norm);
    ~CENTRIST() { }
public:
    const IntImage<double>& AssignFile(const char* filename,const int resizeWidth);
    const IntImage<double>& AssignImage(IntImage<double>& image,const int resizeWidth);
    const IntImage<double>& ResizeImage(const double ratio);
    int* I_feature(const int x1,const int x2,const int y1,const int y2);
};

class SymCENTRIST: public BaseFeature
{
    // symmetric census feature with 4x4 spatial arrangement
private:
    Array2dC<int> ct;
private:
    void CTimage();
public:
    SymCENTRIST(const bool _useSobel,const int _L1_norm);
    ~SymCENTRIST() { }
public:
    const IntImage<double>& AssignFile(const char* filename,const int resizeWidth);
    const IntImage<double>& AssignImage(IntImage<double>& image,const int resizeWidth);
    const IntImage<double>& ResizeImage(const double ratio);
    int* I_feature(const int x1,const int x2,const int y1,const int y2);
};

class LocalFeature: public BaseFeature
{   // use Census Transform as a "codebook" directly
public:
    LocalFeature(const bool _useSobel);
    ~LocalFeature() { }
public:
    int* I_feature(const int x1,const int x2,const int y1,const int y2) { return NULL;}
};

#endif // __FEATURES_H

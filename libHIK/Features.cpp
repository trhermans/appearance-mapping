// Author: Jianxin Wu (wujx2001@gmail.com)

#include <opencv/cv.h>

#if defined ( _WIN32)
#define _USE_MATH_DEFINES
int round(double x) {
  return (int)(x + 0.5);
}
#endif
#include <cmath>

#include "Features.h"

const IntImage<double>& BaseFeature::AssignFile(const char* filename,const int resizeWidth)
{
    img.Load(filename);
    AssignImage(img,resizeWidth);
    return img; // return a reference to the processed image
}

const IntImage<double>& BaseFeature::AssignImage(IntImage<double>& image,const int resizeWidth)
{
    IntImage<double> resize;
    img = image;
    if(img.ncol>resizeWidth && resizeWidth>0)
    {
        img.Resize(resize,img.nrow*resizeWidth/img.ncol,resizeWidth);
        img.Swap(resize); // now 'img' is the resized version
    }
    original = img;
    if(useSobel)
    {
        img.Sobel(resize,false,false);
        img.Swap(resize); // now 'img' is the Sobel version
    }
    return img; // return a reference to the processed image
}

const IntImage<double>& BaseFeature::ResizeImage(const double ratio)
{
    IntImage<double> resize;
    original.Resize(img,ratio);
    original = img;
    if(useSobel)
    {
        img.Sobel(resize,false,false);
        img.Swap(resize);
    }
    return img;
}

double* BaseFeature::D_feature(const int x1,const int x2,const int y1,const int y2)
{   // D_feature is just the 'double' version of the 'integer' values in 'I_feature'
    I_feature(x1,x2,y1,y2);
    for(int i=0;i<length;i++) d_feature.buf[i] = i_feature.buf[i];
    return d_feature.buf;
}

SiftFeature::SiftFeature(const bool _useSobel,const int _L1_norm)
    :BaseFeature(_useSobel,_L1_norm)
{
    length = 128;
    d_feature.Create(1,length);
    i_feature.Create(1,length);
}

void SiftFeature::GradientAndOrientation()
{
    grad.Create(img.nrow,img.ncol);
    grad.Zero();
    orient.Create(img.nrow,img.ncol);
    orient.Zero();
    for(int i=1;i<img.nrow-1;i++)
    {
        for(int j=1;j<img.ncol-1;j++)
        {
            double diff1 = img.p[i+1][j]-img.p[i-1][j];
            double diff2 = img.p[i][j+1]-img.p[i][j-1];
            grad.p[i][j] = sqrt(diff1*diff1+diff2*diff2);
            orient.p[i][j] = int(round(atan2(diff1,diff2)/3.1415926535897932384626433832795)+4);
            if(orient.p[i][j]>=8) orient.p[i][j]=0;
        }
    }
}

const IntImage<double>& SiftFeature::AssignFile(const char* filename,const int resizeWidth)
{
    BaseFeature::AssignFile(filename,resizeWidth);
    GradientAndOrientation(); 
    return img;
}

const IntImage<double>& SiftFeature::AssignImage(IntImage<double>& image,const int resizeWidth)
{
    BaseFeature::AssignImage(image,resizeWidth);
    GradientAndOrientation();
    return img;
}

const IntImage<double>& SiftFeature::ResizeImage(const double ratio)
{
    BaseFeature::ResizeImage(ratio);
    GradientAndOrientation();
    return img;
}

void SiftFeature::Accumulate_Gradient(const int x1,const int x2,const int y1,const int y2)
{
    // As you notice, we do not weigh the gradients according to their distance to the center pixel
    // These two ways seems to have no difference in a densely sample descriptor representation
    // So this simpler way is used here for faster speed
    // You can implement your own version with weights, and please do notify me when you see large differences in performance
    const int h = (x2-x1)/4; const int w = (y2-y1)/4;
    std::fill(d_feature.buf,d_feature.buf+length,0.0);
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
            for(int x=x1+h*i;x<x1+h*i+h;x++)
                for(int y=y1+w*j;y<y1+w*j+w;y++)
                    d_feature.buf[(i*4+j)*8+orient.p[x][y]] += grad.p[x][y];
}

int* SiftFeature::I_feature(const int x1,const int x2,const int y1,const int y2)
{
    Accumulate_Gradient(x1,x2,y1,y2);
    // use L1 to normalize the histogram -- default
    double sum = std::accumulate(d_feature.buf,d_feature.buf+length,0.0);
    double r;
    if(sum>0)
        r = L1_norm*1.0/sum; // again, there will be round-off error so L1 norm is only approximate
    else
        r = 0;
    for(int i=0;i<length;i++) i_feature.buf[i] = int(round(d_feature.buf[i]*r));
    if(sum<600)   // all components in 'd_feature' are close to 0 -- image patch is almost uniform
    {
        std::fill(i_feature.buf,i_feature.buf+length,L1_norm/length); // this is only approximate, if L1_norm<128 or L1_norm%128>0, then it is not L1 normalized
        i_feature.buf[0] -= MagicNumber; // NOTE: special case / make sure i_feature.buf[0] is <0 after this operation
    }

    return i_feature.buf;
}

CENTRIST::CENTRIST(const bool _useSobel,const int _L1_norm)
    :BaseFeature(_useSobel,_L1_norm)
{
    length = 256;
    d_feature.Create(1,length);
    i_feature.Create(1,length);
}

void CENTRIST::CTimage()
{
    ct.Create(img.nrow,img.ncol);
    ct.Zero();
    for(int i=1;i<img.nrow-1;i++)
        for(int j=1;j<img.ncol-1;j++)
            ct.p[i][j] = CensusTransform<double>(img.p,i,j);
}

const IntImage<double>& CENTRIST::AssignFile(const char* filename,const int resizeWidth)
{
    BaseFeature::AssignFile(filename,resizeWidth);
    CTimage();
    return img;
}

const IntImage<double>& CENTRIST::AssignImage(IntImage<double>& image,const int resizeWidth)
{
    BaseFeature::AssignImage(image,resizeWidth);
    CTimage();
    return img;
}

const IntImage<double>& CENTRIST::ResizeImage(const double ratio)
{
    BaseFeature::ResizeImage(ratio);
    CTimage();
    return img;
}

int* CENTRIST::I_feature(const int x1,const int x2,const int y1,const int y2)
{
    double s1=0,s2=0;
    if(L1_norm>0)
    {
        for(int x=x1;x<x2;x++) for(int y=y1;y<y2;y++) { s1+=img.p[x][y]; s2+=img.p[x][y]*img.p[x][y]; }
        s1 /= ((x2-x1)*(y2-y1)); s2 /= ((x2-x1)*(y2-y1));
        s2 -= s1 * s1;
    }

    std::fill(i_feature.buf,i_feature.buf+length,0);
    for(int x=x1;x<x2;x++) for(int y=y1;y<y2;y++) i_feature.buf[ct.p[x][y]]++;
    if(L1_norm==0) return i_feature.buf;
    // Use L1 to normalize the histogram -- default way
    int sum = std::accumulate(i_feature.buf,i_feature.buf+length,0);
    double r = L1_norm*1.0/sum; // again, L1 norm is only approximately 'L1_norm'
    for(int i=0;i<length;i++) i_feature.buf[i] = int(round(i_feature.buf[i]*r));
    if(s2<5) i_feature.buf[0] -= MagicNumber; // NOTE: special case / make sure i_feature.buf[0] is <0 after this operation
    return i_feature.buf;
}

SymCENTRIST::SymCENTRIST(const bool _useSobel,const int _L1_norm)
    :BaseFeature(_useSobel,_L1_norm)
{
    length = 256;
    d_feature.Create(1,length);
    i_feature.Create(1,length);
}

void SymCENTRIST::CTimage()
{
    ct.Create(img.nrow,img.ncol);
    for(int i=1;i<img.nrow-1;i++)
        for(int j=1;j<img.ncol-1;j++)
            ct.p[i][j] = SymmetricCensusTransform<double>(img.p,i,j);
}

const IntImage<double>& SymCENTRIST::AssignFile(const char* filename,const int resizeWidth)
{
    BaseFeature::AssignFile(filename,resizeWidth);
    CTimage();
    return img;
}

const IntImage<double>& SymCENTRIST::AssignImage(IntImage<double>& image,const int resizeWidth)
{
    BaseFeature::AssignImage(image,resizeWidth);
    CTimage();
    return img;
}

const IntImage<double>& SymCENTRIST::ResizeImage(const double ratio)
{
    BaseFeature::ResizeImage(ratio);
    CTimage();
    return img;
}

int* SymCENTRIST::I_feature(const int x1,const int x2,const int y1,const int y2)
{
    std::fill(i_feature.buf,i_feature.buf+length,0);
    const int h = (x2-x1)/4; const int w = (y2-y1)/4;
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
            for(int x=x1+h*i;x<x1+h*i+h;x++)
                for(int y=y1+w*j;y<y1+w*j+w;y++)
                    i_feature.buf[(i*4+j)*16+ct.p[x][y]]++;
    int sum = std::accumulate(i_feature.buf,i_feature.buf+length,int(0));
    double r = L1_norm*1.0/sum; // again, L1 norm is only approximately 'L1_norm'
    for(int i=0;i<length;i++) i_feature.buf[i] = int(round(i_feature.buf[i]*r));
    return i_feature.buf;
}

LocalFeature::LocalFeature(const bool _useSobel)
    :BaseFeature(_useSobel,-1) // L1_norm not useful in this type of feature
{
}

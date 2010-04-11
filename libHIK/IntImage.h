// Author: Jianxin Wu (wujx2001@gmail.com)

#ifndef __INTIMAGE_H__
#define __INTIMAGE_H__

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "mdarray.h"

template<class T>
class IntImage:public Array2dC<T>
{
private:
    IntImage(const IntImage<T> &source) { } // prohibit copy constructor

public:
    IntImage()
    { }
    virtual ~IntImage()
    { Clear(); }

    virtual void Clear(void);
    inline void SetSize(const int h, const int w);
    bool Load(const std::string& filename,const char channel='I');
    void Show(const char* windowName) const;
    void Swap(IntImage<T>& image2);

    void CalcIntegralImageInPlace(void);
    void Resize(IntImage<T> &result,const REAL ratio) const;
    void Resize(IntImage<T>& result,const int height,const int width) const;

    IntImage<T>& operator=(const IntImage<T>& source);
    void SubImage(const int top,const int bottom,const int left,const int right,IntImage<T>& result) const;

    void Sobel(IntImage<REAL>& result,const bool useSqrt,const bool normalize);
public:
    using Array2dC<T>::nrow;
    using Array2dC<T>::ncol;
    using Array2dC<T>::buf;
    using Array2dC<T>::p;
};

template<class T>
void IntImage<T>::Clear(void)
{
    Array2dC<T>::Clear();
}

template<class T>
bool IntImage<T>::Load(const std::string& filename,const char channel)
{
    IplImage* img;
    IplImage* img2;

    img = cvLoadImage(filename.c_str());
    if(img==NULL) return false;

    if(channel=='R' || channel=='G' || channel=='B')
    {
        int c;
        if(channel=='B') c=0; else if(channel=='G') c=1; else c=2; // OpenCV is 'BGR' ordering
        img2 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
        for(int i=0;i<img->height;i++)
            for(int j=0;j<img->width;j++)
                *(img2->imageData+img2->widthStep*i+j)=*(img->imageData+img->widthStep*i+3*j+c);
    }
    else // use gray scale for all others
    {
        img2 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
        cvCvtColor(img,img2,CV_BGR2GRAY);
    }
    cvReleaseImage(&img);
    img = img2; img2 = NULL;
    SetSize(img->height,img->width);
    for(int i=0,ih=img->height,iw=img->width;i<ih;i++)
    {
        T* pdata = p[i];
        unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
        for(int j=0;j<iw;j++) pdata[j] = pimg[j];
    }
    cvReleaseImage(&img);

    return true;
}

template<class T>
void IntImage<T>::SetSize(const int h,const int w)
{
    if((h == nrow) && (w == ncol)) return;
    Clear();
    Array2dC<T>::Create(h,w);
}

template<class T>
IntImage<T>& IntImage<T>::operator=(const IntImage<T>& source)
{
    if(&source==this) return *this;
    SetSize(source.nrow,source.ncol);
    std::copy(source.buf,source.buf+nrow*ncol,buf);
    return *this;
}

template<class T>
void IntImage<T>::Resize(IntImage<T> &result,const REAL ratio) const
{
    Resize(result,int(nrow*ratio),int(ncol*ratio));
}

template<class T>
void IntImage<T>::Resize(IntImage<T>& result,const int height,const int width) const
{
    assert(height>0 && width>0);
    result.SetSize(height,width);
    REAL ixratio = nrow*1.0/height, iyratio = ncol*1.0/width;

    REAL* p_y = new REAL[result.ncol]; assert(p_y!=NULL);
    int* p_y0 = new int[result.ncol]; assert(p_y0!=NULL);
    for(int i=0;i<width;i++)
    {
        p_y[i] = i*iyratio;
        p_y0[i] = (int)p_y[i];
        if(p_y0[i]==ncol-1) p_y0[i]--;
        p_y[i] -= p_y0[i];
    }

    for(int i=0;i<height;i++)
    {
        int x0; REAL x;
        x = i*ixratio;
        x0 = (int)x;
        if(x0==nrow-1) x0--;
        x -= x0;
        for(int j=0;j<width;j++)
        {
            int y0=p_y0[j];
            REAL y=p_y[j],fx0,fx1;

            fx0 = REAL(p[x0][y0] + y*(p[x0][y0+1]-p[x0][y0]));
            fx1 = REAL(p[x0+1][y0] + y*(p[x0+1][y0+1]-p[x0+1][y0]));

            result.p[i][j] = T(fx0 + x*(fx1-fx0));
        }
    }

    delete[] p_y; p_y=NULL;
    delete[] p_y0; p_y0=NULL;
}

template<class T>
void IntImage<T>::CalcIntegralImageInPlace(void)
// We pad a zero column and a zero row, so 24*24 image will be 25*25 in size
// if the input image is not padded, the results on 1st row will be problematic
{
    for(int i=1;i<ncol;i++)     // process the first line
    {
        buf[i] = buf[i-1] + buf[i];
    }
    for(int i=1;i<nrow;i++)
    {
        REAL partialsum = 0;
        for(int j=0;j<ncol;j++)
        {
            partialsum += REAL(p[i][j]);
            p[i][j] = p[i-1][j] + partialsum;
        }
    }
}

template<class T>
void IntImage<T>::Show(const char* windowName) const
{
    IplImage* image = cvCreateImage(cvSize(ncol,nrow),IPL_DEPTH_8U,1);
    for(int i=0;i<nrow;i++)
        for(int j=0;j<ncol;j++)
            *(image->imageData+image->widthStep*i+j) = (unsigned char)p[i][j];
    cvShowImage(windowName,image);
    cvReleaseImage(&image);
}

template<class T>
void IntImage<T>::Swap(IntImage<T>& image2)
{
    Array2dC<T>::Swap(image2);
}

template<class T>
void IntImage<T>::SubImage(const int top,const int bottom,const int left,const int right,IntImage<T>& result) const
{
    // inclusive of top & left, exclusive of bottom & right
    result.Create(bottom-top,right-left);
    for(int i=0;i<bottom-top;i++)
        for(int j=0;j<right-left;j++)
            result.p[i][j] = p[i+top][j+left];
}

template<class T>
void IntImage<T>::Sobel(IntImage<REAL>& result,const bool useSqrt,const bool normalize)
{// compute the Sobel gradient. For now, we just use the very inefficient way. Optimization can be done later
// if useSqrt = true, we compute the real Sobel gradient; otherwise, the square of it
// if normalize = true, the numbers are normalized to be in 0..255
    result.Create(nrow,ncol);
    for(int i=0;i<nrow;i++) result.p[i][0] = result.p[i][ncol-1] = 0;
    for(int i=0;i<ncol;i++) result.p[0][i] = result.p[nrow-1][i] = 0;
    for(int i=1;i<nrow-1;i++)
    {
        for(int j=1;j<ncol-1;j++)
        {
            REAL gx =     p[i-1][j-1] - p[i-1][j+1]
                     + 2*(p[i][j-1]   - p[i][j+1])
                     +    p[i+1][j-1] - p[i+1][j+1];
            REAL gy =     p[i-1][j-1] - p[i+1][j-1]
                     + 2*(p[i-1][j]   - p[i+1][j])
                     +    p[i-1][j+1] - p[i+1][j+1];
           if(useSqrt || normalize ) // if we want to normalize the result image, we'd better use the true Sobel gradient
                result.p[i][j] = sqrt(gx*gx+gy*gy);
           else
                result.p[i][j] = gx*gx+gy*gy;
        }
    }
    if(normalize)
    {
        REAL minv = 1e20, maxv = -minv;
        for(int i=1;i<nrow-1;i++)
        {
            for(int j=1;j<ncol-1;j++)
            {
                if(result.p[i][j]<minv)
                    minv = result.p[i][j];
                else if(result.p[i][j]>maxv)
                    maxv = result.p[i][j];
            }
        }
        for(int i=0;i<nrow;i++) result.p[i][0] = result.p[i][ncol-1] = minv;
        for(int i=0;i<ncol;i++) result.p[0][i] = result.p[nrow-1][i] = minv;
        REAL s = 255.0/(maxv-minv);
        for(int i=0;i<nrow*ncol;i++) result.buf[i] = (result.buf[i]-minv)*s;
    }
}

#endif //__INTIMAGE_H__

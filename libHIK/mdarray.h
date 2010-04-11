// Author: Jianxin Wu (wujx2001@gmail.com)
// Two dimensional array

#ifndef __MDARRAY_H__
#define __MDARRAY_H__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cassert>

#include "util.h"

#define USE_DOUBLE

#ifdef USE_DOUBLE
    typedef double REAL;
#else
    typedef float REAL;
#endif

template<class T> class Array2dC;

template<class T>
class Array2d
{
private:
    void IncreaseCapacity(const int newrow);
    void DecreaseCapacity(const int newrow);
public:
    int nrow;
    int ncol;
    T** p;
public:
    Array2d():nrow(0),ncol(0),p(NULL) { }
    Array2d(const int nrow,const int ncol):nrow(0),ncol(0),p(NULL) { Create(nrow,ncol); }
    Array2d(const Array2d<T>& source);
    virtual ~Array2d() { Clear(); }

    Array2d<T>& operator=(const Array2d<T>& source);
    void Create(const int _nrow,const int _ncol);
    void Swap(Array2d<T>& array2);
    void Clear();
    void Zero(const T t = 0);
    void AdjustCapacity(const int newrow);
};

template<class T>
class Array2dC
{
public:
    int nrow;
    int ncol;
    T** p;
    T* buf;
public:
    Array2dC():nrow(0),ncol(0),p(NULL),buf(NULL) {}
    Array2dC(const int nrow,const int ncol):nrow(0),ncol(0),p(NULL),buf(NULL) { Create(nrow,ncol); }
    Array2dC(const Array2dC<T>& source);
    virtual ~Array2dC() { Clear(); }

    Array2dC<T>& operator=(const Array2dC<T>& source);
    void Create(const int _nrow,const int _ncol);
    void Swap(Array2dC<T>& array2);
    void Zero(const T t = 0);
    void Clear();
};

double MatrixInversion(double** a,const int n,const double diagonal_increment);
bool Inv3x3(double a[3][3],double b[3][3],const double thresh = 1E-200);
void SVD(Array2d<double>& a,double* w,Array2d<double>& v);

void Normalize_L1(double* p,const int size);
void Normalize_L2(double* p,const int size);
void Normalize_01(double* p,const int size);
void ComputeMeanAndVariance(Array2d<double>& data,Array2dC<double>& avg,Array2d<double>& cov,const bool subtractMean);

template<class T>
void Array2d<T>::IncreaseCapacity(const int newrow)
{
    assert(newrow>nrow);
    T** newp = new T*[newrow]; assert(newp!=NULL);
    std::copy(p,p+nrow,newp);
    for(int i=nrow;i<newrow;i++)
    {
        newp[i] = new T[ncol]; assert(newp[i]!=NULL);
    }
    delete[] p; p = NULL;
    p = newp; newp = NULL;
    nrow = newrow;
}

template<class T>
void Array2d<T>::DecreaseCapacity(const int newrow)
{
    assert(newrow<nrow);
    T** newp = new T*[newrow]; assert(newp!=NULL);
    std::copy(p,p+newrow,newp);
    for(int i=newrow;i<nrow;i++)
    {
        delete[] p[i]; p[i] = NULL;
    }
    delete[] p; p = NULL;
    p = newp; newp = NULL;
    nrow = newrow;
}

template<class T>
Array2d<T>::Array2d(const Array2d<T>& source):nrow(0),ncol(0),p(NULL)
{
    if(source.p!=NULL)
    {
        Create(source.nrow,source.ncol);
        for(int i=0;i<nrow;i++) std::copy(source.p[i],source.p[i]+ncol,p[i]);
    }
}

template<class T>
Array2d<T>& Array2d<T>::operator=(const Array2d<T>& source)
{
    if(source.p!=NULL)
    {
        Create(source.nrow,source.ncol);
        for(int i=0;i<nrow;i++) std::copy(source.p[i],source.p[i]+ncol,p[i]);
    }
    else
        Clear();
    return *this;
}

template<class T>
void Array2d<T>::Create(const int _nrow,const int _ncol)
{
    assert(_nrow>0 && _ncol>0);
    if(ncol==_ncol) return AdjustCapacity(_nrow);
    Clear();
    nrow = _nrow; ncol = _ncol;
    p = new T*[nrow]; assert(p!=NULL);
    for(int i=0;i<nrow;i++)
    {
        p[i] = new T[ncol]; assert(p[i]!=NULL);
    }
}

template<class T>
void Array2d<T>::Swap(Array2d<T>& array2)
{
    std::swap(nrow,array2.nrow);
    std::swap(ncol,array2.ncol);
    std::swap(p,array2.p);
}

template<class T>
void Array2d<T>::Zero(const T t)
{
    if(nrow>0)
    {
        for(int i=0;i<nrow;i++) std::fill(p[i],p[i]+ncol,t);
    }
}

template<class T>
void Array2d<T>::Clear()
{
    for(int i=0;i<nrow;i++) { delete[] p[i]; p[i] = NULL; }
    delete[] p; p = NULL;
    nrow = ncol = 0;
}

template<class T>
void Array2d<T>::AdjustCapacity(const int newrow)
{
    assert(newrow>0);
    if(newrow == nrow)
        return;
    else if(newrow>nrow)
        IncreaseCapacity(newrow);
    else // newrow < nrow
        DecreaseCapacity(newrow);
}

template<class T>
Array2dC<T>::Array2dC(const Array2dC<T>& source):nrow(0),ncol(0),p(NULL),buf(NULL)
{
    if(source.buf!=NULL)
    {
        Create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
}

template<class T>
Array2dC<T>& Array2dC<T>::operator=(const Array2dC<T>& source)
{
    if(source.buf!=NULL)
    {
        Create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
    else
        Clear();
    return *this;
}

template<class T>
void Array2dC<T>::Create(const int _nrow,const int _ncol)
{
    assert(_nrow>0 && _ncol>0);
    if(nrow==_nrow && ncol==_ncol) return;
    Clear();
    nrow = _nrow; ncol = _ncol;
    buf = new T[nrow*ncol]; assert(buf!=NULL);
    p = new T*[nrow]; assert(p!=NULL);
    for(int i=0;i<nrow;i++) p[i] = buf + i * ncol;
}

template<class T>
void Array2dC<T>::Swap(Array2dC<T>& array2)
{
    std::swap(nrow,array2.nrow);
    std::swap(ncol,array2.ncol);
    std::swap(p,array2.p);
    std::swap(buf,array2.buf);
}

template<class T>
void Array2dC<T>::Zero(const T t)
{
    if(nrow>0) std::fill(buf,buf+nrow*ncol,t);
}

template<class T>
void Array2dC<T>::Clear()
{
    delete[] buf; buf = NULL;
    delete[] p; p = NULL;
    nrow = ncol = 0;
}

#endif // __MDARRAY_H__

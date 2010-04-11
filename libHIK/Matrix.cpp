// Author: Jianxin Wu (wujx2001@gmail.com)
// A few helper function for simple mathematical operations
#include <cmath>

#include "mdarray.h"

void Normalize_L1(double* p,const int size)
{
    double sum = std::accumulate(p,p+size,0.0);
    if(fabs(sum)<1e-30) sum = 0; else sum = 1.0/sum;
    for(int i=0;i<size;i++) p[i] *= sum;
}

void Normalize_L2(double* p,const int size)
{
    double sum = 0;
    for(int i=0;i<size;i++) sum += p[i]*p[i];
    if(fabs(sum)<1e-60) sum = 0; else sum = 1.0/sqrt(sum);
    for(int i=0;i<size;i++) p[i] *= sum;
}

void Normalize_01(double* p,const int size)
{
    double avg = std::accumulate(p,p+size,0.0)/size;
    for(int i=0;i<size;i++) p[i] -= avg;
    Normalize_L2(p,size);
}

void ComputeMeanAndVariance(Array2d<double>& data,Array2dC<double>& avg,Array2d<double>& cov,const bool subtractMean)
{
    avg.Create(1,data.ncol); avg.Zero();
    cov.Create(data.ncol,data.ncol); cov.Zero();
    for(int i=0;i<data.nrow;i++)
    {
        for(int j=0;j<data.ncol;j++) avg.buf[j] += data.p[i][j];
        for(int j=0;j<data.ncol;j++)
            for(int k=0;k<data.ncol;k++)
                cov.p[j][k] += data.p[i][j]*data.p[i][k];
    }
    const double r = 1.0/data.nrow;
    for(int i=0;i<data.ncol;i++) avg.buf[i] *= r;
    if(subtractMean)
    {
        for(int i=0;i<data.ncol;i++)
            for(int j=0;j<data.ncol;j++)
                cov.p[i][j] = cov.p[i][j]*r-avg.buf[i]*avg.buf[j];
    }
    else
    {
        for(int i=0;i<data.ncol;i++)
            for(int j=0;j<data.ncol;j++)
                cov.p[i][j] = cov.p[i][j]*r;
    }
}

bool Inv3x3(double a[3][3],double b[3][3],const double thresh)
{
    double det = a[0][0]*(a[2][2]*a[1][1]-a[2][1]*a[1][2])-a[1][0]*(a[2][2]*a[0][1]-a[2][1]*a[0][2])+a[2][0]*(a[1][2]*a[0][1]-a[1][1]*a[0][2]);
    bool invertible = true;

    if(fabs(det)<thresh)
    {
        a[0][0] += 1E-60; a[1][1] += 1E-60; a[2][2] += 1E-60;
        det = a[0][0]*(a[2][2]*a[1][1]-a[2][1]*a[1][2])-a[1][0]*(a[2][2]*a[0][1]-a[2][1]*a[0][2])+a[2][0]*(a[1][2]*a[0][1]-a[1][1]*a[0][2]);
        invertible = false;
    }
    det = 1.0/det;

    b[0][0] = (a[2][2]*a[1][1] - a[2][1]*a[1][2]) * det;
    b[0][1] = (-a[2][2]*a[0][1] + a[2][1]*a[0][2]) * det;
    b[0][2] = (a[1][2]*a[0][1] - a[1][1]*a[0][2]) * det;
    b[1][0] = (-a[2][2]*a[1][0] + a[2][0]*a[1][2]) * det;
    b[1][1] = (a[2][2]*a[0][0] - a[2][0]*a[0][2]) * det;
    b[1][2] = (-a[1][2]*a[0][0] + a[1][0]*a[0][2]) * det;
    b[2][0] = (a[2][1]*a[1][0] - a[2][0]*a[1][1]) * det;
    b[2][1] = (-a[2][1]*a[0][0] + a[2][0]*a[0][1]) * det;
    b[2][2] = (a[1][1]*a[0][0] - a[1][0]*a[0][1]) * det;

    return invertible;
}

// Matrix inversion code, revised from Numerical recipe in C, and the graphviz Documentation
int lu_decompose(double** a,const int n,double** lu,int* ps)
{
    int i,j,k;
    int even_odd = 1; // 1 if number of row exchanges is even, -1 if odd
    int pivotindex=0;
    double pivot,biggest,mult,tempf;
    double* scales;

    scales = new double[n]; assert(scales!=NULL);
    for(i=0;i<n;i++)
    {
        biggest=0.0;
        for(j=0;j<n;j++) if(biggest<(tempf=fabs(lu[i][j]=a[j][i]))) biggest=tempf;
        if(biggest!=0.0) 
            scales[i] = 1.0/biggest;
        else 
        {
            delete[] scales; return 0; //zero row: singular matrix
        }
        ps[i]=i;
    }

    for(k=0;k<n-1;k++)
    {
        biggest=0.0;
        for(i=k;i<n;i++) if(biggest<(tempf=fabs(lu[ps[i]][k])*scales[ps[i]])) { biggest=tempf; pivotindex=i; }
        if(biggest==0.0)
        {
            delete[] scales; return 0; //zero row: singular matrix
        }
        if(pivotindex!=k)
        {
            even_odd *= -1;
            j=ps[k];
            ps[k] = ps[pivotindex];
            ps[pivotindex]=j;
        }

        pivot=lu[ps[k]][k];
        for(i=k+1;i<n;i++)
        {
            lu[ps[i]][k]=mult=lu[ps[i]][k]/pivot;
            if(mult!=0.0) { for(j=k+1;j<n;j++) lu[ps[i]][j] -= mult*lu[ps[k]][j]; }
        }
    }
    if(lu[ps[n-1]][n-1]==0.0)
    {
        delete[] scales; scales=NULL;
        return 0; // signular matrix
    }

    delete[] scales; scales=NULL;
    return even_odd;
}

void lu_solve(double* x,double* b,int n,double** lu,int* ps)
{
    int i,j;
    double dot;

    for(i=0;i<n;i++) 
    {
        dot=0.0; for(j=0;j<i;j++) dot+=lu[ps[i]][j]*x[j];
        x[i]=b[ps[i]]-dot;
    }

    for(i=n-1;i>=0;i--)
    {
        dot=0.0; for(j=i+1;j<n;j++) dot+=lu[ps[i]][j]*x[j];
        x[i]=(x[i]-dot)/lu[ps[i]][i];
    }
}

double MatrixInversion(double** a,const int n,const double diagonal_increment)
{ // return the determined of matrix 'a'
    double** lu;
    int i,j;
    double* col;
    int* ps;

    if(diagonal_increment!=0) for(int i=0;i<n;i++) a[i][i] += diagonal_increment;
    lu = new double*[n]; assert(lu!=NULL);
    for(i=0;i<n;i++) { lu[i] = new double[n]; assert(lu[i]!=NULL); }
    ps = new int[n]; assert(ps!=NULL);
    double det;
    if((det=lu_decompose(a,n,lu,ps))==0) std::cerr<<"Singular Matrix!"<<std::endl;
    for(i=0;i<n;i++) det *= lu[i][i];

    col = new double[n]; assert(col!=NULL);
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++) col[j]=0.0;
        col[i]=1.0;
        lu_solve(a[i],col,n,lu,ps);
    }
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            double temp;
            temp=a[i][j];
            a[i][j]=a[j][i];
            a[j][i]=temp;
        }
    }
    for(i=0;i<n;i++) { delete[] lu[i]; lu[i]=NULL; }
    delete[] lu; lu=NULL;
    delete[] ps; ps=NULL;
    delete[] col; col=NULL;
    
    return det;
}
// Matrix inversion code finished.

// SVD code, by Numerical Recipes in C
static double at, bt, ct;
#define PYTHAG(a,b) ((at=fabs(a)) > (bt=fabs(b)) ? \
(ct=bt/at,at*sqrt(1.0+ct*ct)) : (bt ? (ct=at/bt,bt*sqrt(1.0+ct*ct)): 0.00))

static float maxarg1, maxarg2;
#define NR_MAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1)>(maxarg2) ?\
    (maxarg1) : (maxarg2))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

void svdcmp(double **a, int m, int n, double *w, double **v)
{
    assert(m>=n);

    int flag,i,its,j,jj,k,l,nm;
    double c,f,h,s,x,y,z;
    double anorm=0.0,g=0.0,scale=0.0;
    double *rv1=new double [n]; assert(rv1!=NULL); rv1--;

    for(i=1;i<=n;i++)
    {
        l=i+1;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if(i<=m)
        {
            for(k=i;k<=m;k++) scale+=fabs(a[k][i]);
            if(scale)
            {
                for(k=i;k<=m;k++)
                {
                    a[k][i]/=scale;
                    s+=a[k][i]*a[k][i];
                }
                f=a[i][i];
                g=-SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                if(i!=n)
                {
                    for(j=l;j<=n;j++)
                    {
                        for(s=0.0,k=i;k<=m;k++) s+=a[k][i]*a[k][j];
                        f=s/h;
                        for(k=i;k<=m;k++) a[k][j]+=f*a[k][i];
                    }
                }
                for(k=i;k<=m;k++) a[k][i]*=scale;
            }
        }
        w[i]=scale*g;
        g=s=scale=0.0;
        if((i<=m)&&(i!=n))
        {
            for(k=l;k<=n;++k) scale+=fabs(a[i][k]);
            if(scale)
            {
                for(k=l;k<=n;k++)
                {
                    a[i][k]/=scale;
                    s+=a[i][k]*a[i][k];
                }
                f=a[i][l];
                g=-SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l]=f-g;
                for(k=l;k<=n;k++) rv1[k]=a[i][k]/h;
                if(i!=m)
                {
                    for(j=l;j<=m;j++)
                    {
                        for(s=0.0,k=l;k<=n;k++) s+=a[j][k]*a[i][k];
                        for(k=l;k<=n;k++) a[j][k]+=s*rv1[k];
                    }
                }
                for(k=l;k<=n;k++) a[i][k]*=scale;
            }
        }
        anorm=NR_MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
    }

    for(i=n;i>=1;--i)
    {
        if(i<n)
        {
            if(g)
            {
                for(j=l;j<=n;j++) v[j][i]=(a[i][j]/a[i][l])/g;
                for(j=l;j<=n;j++)
                {
                    for(s=0.0,k=l;k<=n;k++) s+=a[i][k]*v[k][j];
                    for(k=l;k<=n;k++) v[k][j]+=s*v[k][i];
                }
            }
            for(j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }

    for(i=n;i>=1;i--)
    {
        l=i+1;
        g=w[i];
        if(i<n) for(j=l;j<=n;j++) a[i][j]=0.0;
        if(g)
        {
            g=1.0/g;
            if(i!=n)
            {
                for(j=l;j<=n;j++)
                {
                    for(s=0.0,k=l;k<=m;k++) s+=a[k][i]*a[k][j];
                    f=(s/a[i][i])*g;
                    for(k=i;k<=m;k++) a[k][j]+=f*a[k][i];
                }
            }
            for(j=i;j<=m;j++) a[j][i]*=g;
        }
        else 
            for(j=i;j<=m;j++) a[j][i]=0.0;
        ++a[i][i];
    }
    for(k=n;k>=1;k--)
    {
        for(its=1;its<=30;its++)
        {
            flag=1;
            for(l=k;l>=1;l--)
            {
                nm=l-1;
                if((double)(fabs(rv1[l])+anorm)==anorm)
                {
                    flag=0;
                    break;
                }
                if((double)(fabs(w[nm])+anorm)==anorm) break;
            }
            if(flag)
            {
                c=0.0;
                s=1.0;
                for(i=l;i<=k;i++)
                {
                    f=s*rv1[i];
                    rv1[i]=c*rv1[i];
                    if((double)(fabs(f)+anorm)==anorm) break;
                    g=w[i];
                    h=PYTHAG(f,g);
                    w[i]=h;
                    h=1.0/h;
                    c=g*h;
                    s=-f*h;
                    for(j=1;j<=m;j++)
                    {
                        y=a[j][nm];
                        z=a[j][i];
                        a[j][nm]=y*c+z*s;
                        a[j][i]=z*c-y*s;
                    }
                }
            }
            z=w[k];
            if(l==k)
            {
                if(z<0.0)
                {
                    w[k]=-z;
                    for(j=1;j<=n;j++) v[j][k]=-v[j][k];
                }
                break;
            }
            if(its==30) exit(fprintf(stderr,"no convergence in 30 svdcmp iterations"));
            x=w[l];
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=PYTHAG(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c=s=1.0;
            for(j=l;j<=nm;j++)
            {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=PYTHAG(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g=g*c-x*s;
                h=y*s;
                y=y*c;
                for(jj=1;jj<=n;jj++)
                {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=PYTHAG(f,h);
                w[j]=z;
                if(z)
                {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=c*g+s*y;
                x=c*y-s*g;
                for(jj=1;jj<=m;jj++)
                {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
    rv1++;
    delete[] rv1;
}

void SVD(Array2d<double>& a,double* w,Array2d<double>& v)
{   // w -- signular values; a = UWV', where U is stored in a
    // NOTE: a.nrow >= a.ncol is required
    Array2dC<double*> nr_a(1,a.nrow+1),nr_v(1,a.ncol+1);
    v.Create(a.ncol,a.ncol);
    for(int i=0;i<a.nrow;i++) nr_a.buf[i+1] = a.p[i] - 1;
    for(int i=0;i<a.ncol;i++) nr_v.buf[i+1] = v.p[i] - 1;
    svdcmp(nr_a.buf,a.nrow,a.ncol,w-1,nr_v.buf);
}
// end of SVD code

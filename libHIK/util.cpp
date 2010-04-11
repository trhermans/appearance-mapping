// Author: Jianxin Wu (wujx2001@gmail.com)

#if defined ( _WIN32 )
    #include <io.h>
#else
    #include <unistd.h>
#endif
#include <sys/types.h>
#include <sys/timeb.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cfloat>
#include <cmath>

#include "util.h"

bool FileExists(const char* filename)
{
#if defined ( _WIN32 )
    return (_access(filename,0)!=-1);
#else
    return (access(filename,F_OK)!=-1);
#endif
}

#if defined ( _WIN32 )
    static __int64 TimingMilliSeconds;
#else
    static timeb TimingMilliSeconds;
#endif

void StartOfDuration()
{
#if defined ( _WIN32 )
    struct _timeb timebuffer;
    _ftime64_s(&timebuffer);
    TimingMilliSeconds = timebuffer.time * 1000 + timebuffer.millitm;
#else
    ftime(&TimingMilliSeconds);
#endif
}

int EndOfDuration()
{
#if defined ( _WIN32 )
    struct _timeb timebuffer;
    _ftime64_s(&timebuffer);
    __int64 now = timebuffer.time * 1000 + timebuffer.millitm;
    return int(now-TimingMilliSeconds);
#else
    struct timeb now;
    ftime(&now);
    return int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) );
#endif
}

KNN_Vote::KNN_Vote(const int _K,const bool _findmax):K(_K),findmax(_findmax)
{
    assert(K>=1);
    scores = new double[K]; assert(scores!=NULL);
    votes = new int[K]; assert(votes!=NULL);
    indexes = new int[K]; assert(indexes!=NULL);
}

KNN_Vote::~KNN_Vote()
{
    delete[] scores; scores=NULL;
    delete[] votes; votes=NULL;
    delete[] indexes; indexes=NULL;
}

void KNN_Vote::Init() // MAKE SURE call Init() before evaluate EVERY SINGLE example
{
    if(findmax)
        for(int i=0;i<K;i++) scores[i] = (-1E10)*(i+1); // scores[0] is the largest, find maximum
    else
        for(int i=0;i<K;i++) scores[i] = (1E10)*(i+1); // scores[0] is the smallest, find minimum
    for(int i=0;i<K;i++) votes[i] = indexes[i] = -1; 
}

void KNN_Vote::Examine(const double newscore,const int newvote,const int newindex)
{
    int pos = K-1;
    if(findmax)
        while(pos>=0 && scores[pos]<=newscore) pos--;
    else
        while(pos>=0 && scores[pos]>=newscore) pos--;
    pos++;
    if(pos==K) return;
    for(int i=K-2;i>=pos;i--) 
    {
        scores[i+1] = scores[i];
        votes[i+1] = votes[i];
        indexes[i+1] = indexes[i];
    }
    scores[pos] = newscore;
    votes[pos] = newvote;
    indexes[pos] = newindex;
}

double KNN_Vote::GetBestScore() const
{
    return scores[0]; 
}

int KNN_Vote::GetBestScoreClass() const
{
    return votes[0]; 
}

int KNN_Vote::GetBestScoreIndex() const
{
    return indexes[0]; 
}

int KNN_Vote::GetLabel(const int pos) const
{
    assert(pos>=0 && pos<K);
    return votes[pos];
}

int KNN_Vote::GetIndex(const int pos) const
{
    assert(pos>=0 && pos<K);
    return indexes[pos];
}

double KNN_Vote::GetScore(const int pos) const
{
    assert(pos>=0 && pos<K);
    return scores[pos];
}

int KNN_Vote::GetVotedClass(const int nclass) const
{ // assume all class labels (votes[i]) are integers between {0,1,2,..,nclass-1}
    int* classvotes = new int[nclass]; assert(classvotes!=NULL);
    for(int i=0;i<nclass;i++) classvotes[i]=0;
    for(int i=0;i<K;i++) classvotes[votes[i]]++;
    int temp = int(std::max_element(classvotes,classvotes+nclass)-classvotes);
    delete[] classvotes; classvotes=NULL;
    return temp;
}

int KNN_Vote::GetWeightVotedClass(const int nclass) const
{
    double* classvotes = new double[nclass]; assert(classvotes!=NULL);
    for(int i=0;i<nclass;i++) classvotes[i]=0;
    for(int i=0;i<K;i++) classvotes[votes[i]]+=scores[i];
    int temp = int(std::max_element(classvotes,classvotes+nclass)-classvotes);
    delete[] classvotes; classvotes=NULL;
    return temp;
}

// Linux (gcc) and Windows (VC++) uses different rand() function
// below are an implementation of the VC++ c library rand() function
// use my_srand() and my_rand() to get the same result in both Linux and Windows
#if defined ( _WIN32 )
void my_srand(unsigned int newseed)
{
    srand(newseed);
}

int my_rand()
{
    return rand();
}
#else
static unsigned long seed = 1234567890;

void my_srand(unsigned int newseed)
{
    seed = (unsigned long)newseed;
 }

int my_rand()
{
    return( ((seed = seed * 214013L + 2531011L) >> 16) & 0x7fff );
}
#endif 
// rand() and srand() finished

void GenerateRandomPermutation(int* p,const int length,const unsigned int seed)
// return a rand permutation of 1..length-1, assume p is 'length' long vector with allocated memory
// 'seed' is random number generator seed, if 0xffffffff then not re-init seed
{
    if(seed != 0xffffffff) my_srand(seed);
    assert(length>0);

    sort_struct<int>* elements = new sort_struct<int>[length]; 
    assert(elements!=NULL);
    for(int i=0;i<length;i++) 
    {
        elements[i].id = i;
        elements[i].value = my_rand();
    }
    std::sort(elements,elements+length,Template_Less<int>);
    for(int i=0;i<length;i++) p[i] = elements[i].id;
    delete[] elements; elements=NULL;
}

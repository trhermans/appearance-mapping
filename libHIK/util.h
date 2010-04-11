// Author: Jianxin Wu (wujx2001@gmail.com)

#ifndef __UTIL_H__
#define __UTIL_H__

bool FileExists(const char* filename);

void StartOfDuration();
int EndOfDuration();

void my_srand(unsigned int newseed);
int my_rand();

template<class T> 
struct sort_struct
{
    int id;
    T value;
} ;

template<class T>
bool Template_Less(const sort_struct<T>& elem1, const sort_struct<T>& elem2)
{
    return elem1.value < elem2.value;
}

void GenerateRandomPermutation(int* p,const int length,const unsigned int seed=0xffffffff);

class KNN_Vote
{
private:
    int K;
    double* scores; // scores of the K nearest neighbors
    int* votes;  // class labels of the K Nearest Neighbors
    int* indexes; // index of the nearest K examples in the training set
    bool findmax;

public:
    KNN_Vote(const int _K,const bool _findmax);
    virtual ~KNN_Vote();
    void Init(); // MAKE SURE call Init() before evaluate EVERY SINGLE example
    void Examine(const double newscore,const int newvote,const int newindex);
    double GetBestScore() const;
    int GetBestScoreClass() const;
    int GetBestScoreIndex() const;
    int GetVotedClass(const int nclass) const; // assume all class labels (votes[i]) are integers between {0,1,2,..,nclass-1}
    int GetWeightVotedClass(const int nclass) const;
    int GetLabel(const int pos) const;
    int GetIndex(const int pos) const;
    double GetScore(const int pos) const;
};

class KNN_FindMax:public KNN_Vote
{
public:
    KNN_FindMax(const int _K):KNN_Vote(_K,true) {}
};

class KNN_FindMin:public KNN_Vote
{
public:
    KNN_FindMin(const int _K):KNN_Vote(_K,false) {}
};

#endif // __UTIL_H__

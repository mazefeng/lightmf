#ifndef __MATRIX_FACTORIZATION__
#define __MATRIX_FACTORIZATION__

#include "SparseMatrix.h"
#include "LatentFactorModel.h"
#include <pthread.h>
#include <time.h>
#include <string>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

using namespace std;

#define T_REGRESS  1
#define T_CLASSIFY 2
#define LIMITED_EPOCHES  100
#define LIMITED_VALIDATE 10

class LatentFactorModel;

class MfParams{
public:
    int type;
    int num_factor;
    double sigma;
    double lambda;
    int max_epoch;
    double alpha;
    int validate;

public:
    MfParams();
    MfParams(int _type, int _num_factor, double _sigma, double _lambda, int _max_epoch, double _alpha, int _validate);
};

class MatrixFactorization{

public:
    static pthread_rwlock_t rwlock;

private:
    
    typedef struct{
        int row;
        int col;
        double val;
    }Triplet;
    
    Triplet* data;

    double loss_train[LIMITED_EPOCHES];
    double loss_validate[LIMITED_EPOCHES];

    double clf_err_train[LIMITED_EPOCHES];
    double clf_err_validate[LIMITED_EPOCHES];   
 
    int num_train;
    int num_validate;

    MfParams* _params;
    SparseMatrix* _matrix;
    LatentFactorModel* _model;

public:

    MatrixFactorization();
    LatentFactorModel* train(MfParams* params, SparseMatrix* matrix, string path);
    void validate(int start, int end, int epoch);
    double predict(Triplet* p);
    void test(string in, string out);

    MfParams* getMfParams(){ return _params; }
    void setMfParams(MfParams* params){
        _params = params;
    }

    SparseMatrix* getSparseMatrix(){return _matrix;}
    void setSparseMatrix(SparseMatrix* matrix){
        _matrix = matrix;
    }

    LatentFactorModel* getLatentFactorModel(){return _model;}
    void setLatentFactorModel(LatentFactorModel* model){
        _model = model;
    }

    ~MatrixFactorization();
    
    static void* run_helper(void *params);
};

#endif

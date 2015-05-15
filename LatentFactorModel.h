#ifndef __LATENT_FACTOR_MODEL__
#define __LATENT_FACTOR_MODEL__

#include "SparseMatrix.h"
#include "MatrixFactorization.h"
#include <strstream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <stdexcept>
#define PI 3.141592654

using namespace std;

class MfParams;

class LatentFactorModel{
private:
    SparseMatrix* matrix;
    int type;
    int num_factor;
    double sigma;
    bool init_params(double* v, int size);

public:
    double *P, *P0, *P1;
    double *Q, *Q0, *Q1;
    double *bu, *bu0, *bu1;
    double *bi, *bi0, *bi1;
    double mu;

public:
    LatentFactorModel(SparseMatrix* _matrix, int _type, int _num_factor, double _sigma);
    LatentFactorModel();
    ~LatentFactorModel();
    bool dump(string path);
    bool load(string path);
    bool swap();
    
    SparseMatrix* getSparseMatrix(){ return matrix; }
    void setSparseMatrix(SparseMatrix* _matrix){ matrix = _matrix; }

    int getType(){ return type; }
    void setType(int _type){ type = _type; }

    int getNumFactor(){ return num_factor; }
    void setNumFactor(int _num_factor){ num_factor = _num_factor; }

    double getSigma(){ return sigma; }
    void setSigma(double _sigma){ sigma = _sigma; }
    
};

#endif

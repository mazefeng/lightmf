#include "LatentFactorModel.h"

LatentFactorModel::LatentFactorModel(SparseMatrix* _matrix, int _type, int _num_factor, double _sigma){
    matrix = _matrix;
    type = _type;
    num_factor = _num_factor;
    sigma = _sigma;

    mu = 1.0 * matrix->sum() / matrix->size();
    P0 = (double*) malloc(matrix->rows()->size() * num_factor * sizeof(double));
    P1 = (double*) malloc(matrix->rows()->size() * num_factor * sizeof(double));
    Q0 = (double*) malloc(matrix->cols()->size() * num_factor * sizeof(double));
    Q1 = (double*) malloc(matrix->cols()->size() * num_factor * sizeof(double));
    bu0 = (double*) malloc(matrix->rows()->size() * sizeof(double));
    bu1 = (double*) malloc(matrix->rows()->size() * sizeof(double));
    bi0 = (double*) malloc(matrix->cols()->size() * sizeof(double));
    bi1 = (double*) malloc(matrix->cols()->size() * sizeof(double));

    P = P0;
    Q = Q0;
    bu = bu0;
    bi = bi0;

    init_params(P,  matrix->rows()->size() * num_factor);
    init_params(Q,  matrix->cols()->size() * num_factor);
    init_params(bu, matrix->rows()->size());
    init_params(bi, matrix->cols()->size());
}

LatentFactorModel::LatentFactorModel(){
    matrix = NULL;
    type = 0;
    num_factor = 0;
    sigma = 0.0;
    mu = 0.0;
    P = P0 = P1 = NULL;
    Q = Q0 = Q1 = NULL;
    bu = bu0 = bu1 = NULL;
    bi = bi0 = bi1 = NULL;
}

bool LatentFactorModel::swap(){
    if(P == P0 && Q == Q0 && bu == bu0 && bi == bi0){
        memcpy(P1,  P0,  matrix->rows()->size() * num_factor * sizeof(double));
        memcpy(Q1,  Q0,  matrix->cols()->size() * num_factor * sizeof(double));
        memcpy(bu1, bu0, matrix->rows()->size() * sizeof(double));
        memcpy(bi1, bi0, matrix->cols()->size() * sizeof(double));
        P = P1;
        Q = Q1;
        bu = bu1;
        bi = bi1;    
        // cerr << "swap 0 <==> 1" << endl;
        return true;
    }
    else if(P == P1 && Q == Q1 && bu == bu1 && bi == bi1){
        memcpy(P0,  P1,  matrix->rows()->size() * num_factor * sizeof(double));
        memcpy(Q0,  Q1,  matrix->cols()->size() * num_factor * sizeof(double));
        memcpy(bu0, bu1, matrix->rows()->size() * sizeof(double));
        memcpy(bi0, bi1, matrix->cols()->size() * sizeof(double));
        P = P0;
        Q = Q0;
        bu = bu0;
        bi = bi0;    
        // cerr << "swap 1 <==> 0" << endl;
        return true;
    }
    return false;
}

LatentFactorModel::~LatentFactorModel(){
    free(P0);  free(P1);
    free(Q0);  free(Q1);
    free(bu0); free(bu1);
    free(bi0); free(bi1);
}

bool LatentFactorModel::init_params(double* v, int size){

    double n;
    for(int k = 0; k < size; k++){
        // v[k] = (rand() % 2001 - 1000) / 100000.0;
        // generate Gaussian distribution by Box-Muller method
        n = sqrt(-2.0 * log(1.0 * rand() / RAND_MAX)) * sin(2.0 * PI * (1.0 * rand() / RAND_MAX));
        v[k] = sigma * n;
    }
}

bool LatentFactorModel::dump(string path){
   
    int s, k;
    double *v, *b;

    ofstream os;
    ostringstream oss;
    map<string, int>::iterator iter;  

    string meta_file = path;
    string row_file  = path + ".row";
    string col_file  = path + ".col";

    os.open(meta_file.c_str());

    os << type << " " << num_factor << " " << sigma << " " << matrix->rows()->size() << " " << matrix->cols()->size() << endl;
    os << row_file << endl;
    os << col_file << endl;
    os << mu << endl;
    
    os.close();

    if(P == P0 && bu == bu0){
        v = P1;
        b = bu1;
    }
    else if(P == P1 && bu == bu1){
        v = P0;
        b = bu0;
    }
    else{
        return false;
    }

    os.open(row_file.c_str());

    for(iter = matrix->rows()->begin(); iter != matrix->rows()->end(); iter++){
        oss.str("");
        oss << iter->first << " " << iter->second << " " << b[iter->second];
        for(k = 0; k < num_factor; k++){
            oss << " " << v[iter->second * num_factor + k];
        }
        os << oss.str() << endl;
    }

    os.close();
    
    if(Q == Q0 && bi == bi0){
        v = Q1;
        b = bi1;
    }
    else if(Q == Q1 && bi == bi1){
        v = Q0;
        b = bi0;
    }
    else{
        return false;
    }

    os.open(col_file.c_str());

    for(iter = matrix->cols()->begin(); iter != matrix->cols()->end(); iter++){
        oss.str("");
        oss << iter->first << " " << iter->second << " " << b[iter->second];
        for(k = 0; k < num_factor; k++){
            oss << " " << v[iter->second * num_factor + k];
        }
        os << oss.str() << endl;
    }

    os.close();

    return true;
}

bool LatentFactorModel::load(string path){
    
    string meta_file = path;
    string row_file;
    string col_file;
    ifstream is;
    int row_size, col_size, i, j, val;
    string key;
    
    map<string, int>* rows = new map<string, int>();
    map<string, int>* cols = new map<string, int>();


    free(P0);  free(P1);
    free(Q0);  free(Q1);
    free(bu0); free(bu1);
    free(bi0); free(bi1);

    is.open(meta_file.c_str(), ios::in);

    if(!is.good()){ 
        throw runtime_error(meta_file + " not exist"); 
    }

    is >> type >> num_factor >> sigma >> row_size >> col_size;
    is >> row_file;
    is >> col_file;
    is >> mu;
    is.close();

    P0 = (double*) malloc(row_size * num_factor * sizeof(double));
    P1 = (double*) malloc(row_size * num_factor * sizeof(double));
    Q0 = (double*) malloc(col_size * num_factor * sizeof(double));
    Q1 = (double*) malloc(col_size * num_factor * sizeof(double));
    bu0 = (double*) malloc(row_size * sizeof(double));
    bu1 = (double*) malloc(row_size * sizeof(double));
    bi0 = (double*) malloc(col_size * sizeof(double));
    bi1 = (double*) malloc(col_size * sizeof(double));

    is.open(row_file.c_str(), ios::in);

    if(!is.good()){ 
        throw runtime_error(row_file + " not exist"); 
    }

    for(i = 0; i < row_size; i++){
        is >> key >> val >> bu0[val];
        rows->insert(pair<string, int>(key, val));
        for(j = 0; j < num_factor; j++){
            is >> P0[val * num_factor + j];
        }
    }
    is.close();

    is.open(col_file.c_str(), ios::in);
    
    if(!is.good()){ 
        throw runtime_error(col_file + " not exist"); 
    }
    
    for(i = 0; i < col_size; i++){
        is >> key >> val >> bi0[val];
        cols->insert(pair<string, int>(key, val));
        for(j = 0; j < num_factor; j++){
            is >> Q0[val * num_factor + j];
        }
    }
    is.close();

    P = P0;
    Q = Q0;
    bu = bu0;
    bi = bi0;

    if(matrix != NULL){
        delete matrix;
    }
    matrix = new SparseMatrix(rows, cols);

    return true;
}



#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(){
    _rows = new map<string, int>();
    _cols = new map<string, int>();
    _kv_dict = new map<pair<int, int>, double>();
    _sum = 0.0;
    _mean = 0.0;
}
    
SparseMatrix::SparseMatrix(map<string, int>* rows, map<string, int>* cols){
    _rows = rows;
    _cols = cols;
    _kv_dict = new map<pair<int, int>, double>();
    _sum = 0.0;
    _mean = 0.0;
}

SparseMatrix::~SparseMatrix(){
    delete _rows;
    delete _cols;
    delete _kv_dict;
}

bool SparseMatrix::get_rating(const string& row, const string& col, double& rating){
    if(_rows->find(row) == _rows->end()){
        rating = 0.0;
        return false;
    }

    if(_cols->find(col) == _cols->end()){
        rating = 0.0;
        return false;
    }

    pair<int, int> k(_rows->find(row)->second, _cols->find(col)->second);

    if(_kv_dict->find(k) == _kv_dict->end()){
        rating = 0.0;
        return false;
    }

    rating = _kv_dict->find(k)->second;
    return true;

}

bool SparseMatrix::set_rating(const string& row, const string& col, double rating){
    if(_rows->find(row) == _rows->end()){
        _rows->insert(pair<string, int>(row, _rows->size()));
    }
    
    if(_cols->find(col) == _cols->end()){
        _cols->insert(pair<string, int>(col, _cols->size()));
    }
    
    pair<int, int> k(_rows->find(row)->second, _cols->find(col)->second);

    if(_kv_dict->find(k) == _kv_dict->end()){
        _sum += rating;
    }
    else{
        _sum += rating - _kv_dict->find(k)->second;
    }
    
    _kv_dict->insert(pair<pair<int, int>, double>(k, rating));

    if(_kv_dict->size() == 0){
        _mean = 0.0;
    }
    else{
        _mean = _sum / _kv_dict->size();
    }
} 

void SparseMatrix::print_info(){
    cerr << "[INFO]: kv_dict size = " << _kv_dict->size() << endl;
    cerr << "[INFO]: rows size = " << _rows->size() << endl;
    cerr << "[INFO]: cols size = " << _cols->size() << endl;
    cerr << "[INFO]: mean = " << _mean << endl;
    cerr << "[INFO]: sum = " << _sum << endl;
}

SparseMatrix* SparseMatrix::load(string path, map<string, int>* rows, map<string, int>* cols){
    
    SparseMatrix* matrix = new SparseMatrix(rows, cols);
    ifstream is(path.c_str());
    string row, col;
    double rating;
    char line[MAX_LINE_LENGTH];

    while(is.getline(line, MAX_LINE_LENGTH)){
        row = string(strtok(line, " "));
        col = string(strtok(NULL, " "));
        rating = atof(strtok(NULL, " "));
        matrix->set_rating(row, col, rating);
    }

    is.close();
    return matrix;

}

SparseMatrix* SparseMatrix::load(string path){
    SparseMatrix* matrix = new SparseMatrix();
    return SparseMatrix::load(path, matrix->rows(), matrix->cols());
}

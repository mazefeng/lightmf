#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(){
    _rows = new map<string, int>();
    _cols = new map<string, int>();
    _kv_dict = new map<pair<int, int>, vector<double>*>();
    _sum = 0.0;
    _size = 0;
}

SparseMatrix::SparseMatrix(map<string, int>* rows, map<string, int>* cols){
    _rows = rows;
    _cols = cols;
    _kv_dict = new map<pair<int, int>, vector<double>*>();
    _sum = 0.0;
    _size = 0;
}

SparseMatrix::~SparseMatrix(){
    delete _rows;
    delete _cols;
    delete _kv_dict;
}

bool SparseMatrix::get_rating(const string& row, const string& col, vector<double>* rating_list){
    if(_rows->find(row) == _rows->end()){
        rating_list = NULL;
        return false;
    }

    if(_cols->find(col) == _cols->end()){
        rating_list = NULL;
        return false;
    }

    pair<int, int> k(_rows->find(row)->second, _cols->find(col)->second);

    if(_kv_dict->find(k) == _kv_dict->end()){
        rating_list = NULL;
        return false;
    }

    rating_list = _kv_dict->find(k)->second;
    return true;

}

bool SparseMatrix::set_rating(const string& row, const string& col, double rating){

    _sum += rating;
    _size += 1;

    if(_rows->find(row) == _rows->end()){
        _rows->insert(pair<string, int>(row, _rows->size()));
    }
    
    if(_cols->find(col) == _cols->end()){
        _cols->insert(pair<string, int>(col, _cols->size()));
    }
    
    pair<int, int> k(_rows->find(row)->second, _cols->find(col)->second);
    vector<double>* rating_list = NULL;
    
    if(_kv_dict->find(k) == _kv_dict->end()){
        rating_list = new vector<double>();
        _kv_dict->insert(pair<pair<int, int>, vector<double>*>(k, rating_list));
    }    

    rating_list = _kv_dict->find(k)->second;
    rating_list->push_back(rating);
    
    // _kv_dict->insert(pair<pair<int, int>, double>(k, rating));
    return (0);

} 

void SparseMatrix::print_info(){
    cerr << "[INFO]: kv_dict size = " << _kv_dict->size() << endl;
    cerr << "[INFO]: rows size = " << _rows->size() << endl;
    cerr << "[INFO]: cols size = " << _cols->size() << endl;
    cerr << "[INFO]: sum = " << _sum << endl;
    cerr << "[INFO]: size = " << _size << endl;
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

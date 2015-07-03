#ifndef __SPARSE_MATRIX__
#define __SPARSE_MATRIX__

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#define MAX_LINE_LENGTH 256

using namespace std;

class SparseMatrix{
private:
    map<string, int>* _rows;
    map<string, int>* _cols;
    map<pair<int, int>, vector<double>*>* _kv_dict;
    double _sum;
    int _size;
public:

    SparseMatrix();
    SparseMatrix(map<string, int>* rows, map<string, int>* cols);
    ~SparseMatrix();

    bool get_rating(const string& row, const string& col, vector<double>* rating_list);
    bool set_rating(const string& row, const string& col, double rating);
    
    int kv_size(){ return _kv_dict->size(); }
    int size(){ return _size; }
    double sum(){ return _sum; }

    map<string, int>* rows(){return _rows;}
    map<string, int>* cols(){return _cols;}
    map<pair<int, int>, vector<double>*>* kv_dict(){return _kv_dict;}

    static SparseMatrix* load(string path);
    static SparseMatrix* load(string path, map<string, int>* rows, map<string, int>* cols);

    void print_info();
 
};

#endif

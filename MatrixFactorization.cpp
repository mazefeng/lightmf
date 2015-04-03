#include "MatrixFactorization.h"

MfParams::MfParams(){
    num_factor = 25;
    sigma = 0.01;
    lambda = 0.005;
    max_epoch = 10;
    alpha = 0.01;
    validate = 0;
}

MfParams::MfParams(int _num_factor, double _sigma, double _lambda, int _max_epoch, double _alpha, int _validate){

    num_factor = _num_factor;
    sigma = _sigma;
    lambda = _lambda;
    max_epoch = _max_epoch;
    alpha = _alpha;
    validate = _validate;

    if(max_epoch > LIMITED_EPOCHES)
        max_epoch = LIMITED_EPOCHES;

    if(validate < 0)  validate = 0;
    if(validate == 1 || validate > LIMITED_VALIDATE){
        validate = 10;
    }

}

MatrixFactorization::MatrixFactorization()
{
    pthread_rwlock_init(&rwlock, NULL);
    data = NULL;
    
    memset(rmse_train, 0, sizeof(rmse_train));
    memset(rmse_validate, 0, sizeof(rmse_validate));

    num_train = 0;
    num_validate = 0;

    _params = NULL;
    _matrix = NULL;
    _model  = NULL;

}

void* MatrixFactorization::run_helper(void* params){
    char buffer[MAX_LINE_LENGTH];
    // (LatentFactorModel*)((void**)params)[0];
    string path = *(string*)((void**)params)[1];
    snprintf(buffer, sizeof(buffer), "%.4d", *(int*)((void**)params)[2]);
    pthread_rwlock_t rwlock = *(pthread_rwlock_t*)((void**)params)[3];
    
    pthread_rwlock_rdlock(&rwlock);
    bool ret = ((LatentFactorModel*)((void**)params)[0])->dump(path + buffer);
    pthread_rwlock_unlock(&rwlock); 
    return (void*) ret;
}


LatentFactorModel* MatrixFactorization::train(MfParams* params, SparseMatrix* matrix, string path){

    int k, s, n;

    _params = params;
    _matrix = matrix;
    _model = new LatentFactorModel(_matrix, _params->num_factor, _params->sigma);

    if(path.compare(path.size() - 1, 1, "/")){
        path = path + "/";
    }

    data = (Triplet*) malloc(_matrix->length() * sizeof(Triplet));
    map<pair<int, int>, double>::iterator iter;

    for(iter = _matrix->kv_dict()->begin(), k = 0; iter != _matrix->kv_dict()->end(); iter++, k++){
        data[k].row = iter->first.first;
        data[k].col = iter->first.second;
        data[k].val = iter->second;
    }

    random_shuffle(data, data + _matrix->length());

    if(_params->validate == 0){
        num_train = _matrix->length();
        num_validate = 0;
    }
    else{
        num_validate = _matrix->length() / _params->validate;
        num_train = _matrix->length() - num_validate;
    }

    clock_t start, terminal;
    pthread_t pt;
    double dot, err, sum_sqr_err, duration;
    double bu_update, bi_update;
    double p[_params->num_factor], q[_params->num_factor], swap[_params->num_factor];

    for(s = 0; s < _params->max_epoch; s++){
        random_shuffle(data, data + num_train);
        sum_sqr_err = 0.0;
        start = clock();

        for(n = 0; n < num_train; n++){
    
            Triplet t = data[n];

            memcpy(p, _model->P + (t.row * _params->num_factor), sizeof(p));
            memcpy(q, _model->Q + (t.col * _params->num_factor), sizeof(q));
            bu_update = _model->bu[t.row];
            bi_update = _model->bi[t.col];
            
            dot = 0.0;
            for(k = 0; k < _params->num_factor; k++){
                dot += p[k] * q[k];
            }
    
            err = t.val - (_model->mu + bu_update + bi_update + dot);
    
            for(k = 0; k < _params->num_factor; k++){
                swap[k] = p[k] + _params->alpha * (err * q[k] - _params->lambda * p[k]);
                q[k] += _params->alpha * (err * p[k] - _params->lambda * q[k]);
            }
            
            memcpy(p, swap, sizeof(swap));
    
            bu_update += _params->alpha * (err - _params->lambda * bu_update);
            bi_update += _params->alpha * (err - _params->lambda * bi_update);
           
            memcpy(_model->P + (t.row * _params->num_factor), p, sizeof(p));
            memcpy(_model->Q + (t.col * _params->num_factor), q, sizeof(q));
            _model->bu[t.row] = bu_update;
            _model->bi[t.col] = bi_update;
            sum_sqr_err += err * err;
    
        }

        pthread_rwlock_wrlock(&rwlock);
        _model->swap();
        pthread_rwlock_unlock(&rwlock); 
      
        int epoch = s;
        void* params[4] = {_model, &path, &epoch, &rwlock};
        // void* params[4] = {_model, &s, &rwlock};
        pthread_create(&pt, NULL, &MatrixFactorization::run_helper, params);
        
        // _model->dump("test");

        terminal = clock();
        duration = (double) (terminal - start) / (double) (CLOCKS_PER_SEC);

        rmse_train[s] = sqrt(sum_sqr_err / _matrix->length());
        rmse_validate[s] = validate(num_train, num_train + num_validate);

        cerr << setiosflags(ios::fixed);

        cerr << "[Iter] = " << setw(4) << setfill('0') << s << " " \
             << "[RMSE] = " << setprecision(6) << rmse_train[s] << " ";
        if(_params->validate != 0){
            cerr << "[ValidateRMSE] = " << setprecision(6) << rmse_validate[s] << " ";
        }
        cerr << "[Duration] = " << setprecision(2) << duration << " Sec."  << endl;      
    }

    pthread_join(pt, NULL);
    return _model;
}


double MatrixFactorization::validate(int start, int end){
    double pred, err, sum_sqr_err;
    sum_sqr_err = 0.0;
    for(int k = start; k < end; k++){
        pred = predict(data + k);
        err = (data + k)->val - pred;
        sum_sqr_err += err * err;
    }
    return sqrt(sum_sqr_err / (end - start));
}

double MatrixFactorization::predict(Triplet* p){
    int k, check;
    double pred = _model->mu;

    check = 0;
    if(p->row > 0 && p->row < _matrix->rows()->size()){
        pred += _model->bu[p->row];
        check++;
    }

    if(p->col > 0 && p->col < _matrix->cols()->size()){
        pred += _model->bi[p->col];
        check++;
    }
    
    if(check == 0 || check == 1) return pred;

    for(k = 0; k < _params->num_factor; k++){
        pred += _model->P[p->row * _params->num_factor + k] * _model->Q[p->col * _params->num_factor + k];
    }
    return pred;
}

double MatrixFactorization::test(string in, string out){
   
    ifstream is(in.c_str());
    ofstream os(out.c_str());
    string row, col;
    double rating, pred, err, sum_sqr_err;
    char line[MAX_LINE_LENGTH];
    int n;
    Triplet* p = new Triplet();
    

    sum_sqr_err = 0.0;
    n = 0;

    while(is.getline(line, MAX_LINE_LENGTH)){
        string strline(line);
        row = string(strtok(line, " "));
        col = string(strtok(NULL, " "));
        rating = atof(strtok(NULL, " "));
       
        if(_matrix->rows()->find(row) == _matrix->rows()->end()){
            p->row = -1;
        }
        else{
            p->row = _matrix->rows()->find(row)->second;
        }

        if(_matrix->cols()->find(col) == _matrix->cols()->end()){
            p->col = -1;
        }
        else{
            p->col = _matrix->cols()->find(col)->second;
        }

        p->val = rating;

        pred = predict(p);
        err = p->val - pred;
        sum_sqr_err += err * err;
        os << pred << "\t" << strline << endl;
        n++;
    }

    is.close();
    os.close();
    return sqrt(1.0 * sum_sqr_err / n);
}

MatrixFactorization::~MatrixFactorization(){
    free(data);
    pthread_rwlock_destroy(&rwlock);
}


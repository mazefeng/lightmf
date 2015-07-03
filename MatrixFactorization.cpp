#include "MatrixFactorization.h"

double sigmoid(double s){
    return 1.0 / (1.0 + exp(-s));
}

MfParams::MfParams(){
    // type = T_REGRESS;
    type = T_CLASSIFY;
    num_factor = 25;
    sigma = 0.01;
    lambda = 0.005;
    max_epoch = 10;
    alpha = 0.01;
    validate = 0;
}

MfParams::MfParams(int _type, int _num_factor, double _sigma, double _lambda, int _max_epoch, double _alpha, int _validate){

    type = _type;
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

pthread_rwlock_t MatrixFactorization::rwlock;

MatrixFactorization::MatrixFactorization()
{
    pthread_rwlock_init(&MatrixFactorization::rwlock, NULL);
    data = NULL;
    
    memset(loss_train, 0, sizeof(loss_train));
    memset(loss_validate, 0, sizeof(loss_validate));

    memset(clf_err_train, 0, sizeof(clf_err_train));
    memset(clf_err_validate, 0, sizeof(clf_err_validate));

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
    snprintf(buffer, sizeof(buffer), "pass-%.4d", *(int*)((void**)params)[2]);
    // pthread_rwlock_t rwlock = *(pthread_rwlock_t*)((void**)params)[3];
 
    pthread_rwlock_wrlock(&MatrixFactorization::rwlock);
    bool ret = ((LatentFactorModel*)((void**)params)[0])->dump(path + buffer);
    pthread_rwlock_unlock(&MatrixFactorization::rwlock); 
    return (void*) ret;
}


LatentFactorModel* MatrixFactorization::train(MfParams* params, SparseMatrix* matrix, string path){

    bool ret;
    int j, k, s, n, row, col;
    pair<int, int> index;

    _params = params;
    _matrix = matrix;
    _model = new LatentFactorModel(_matrix, _params->type, _params->num_factor, _params->sigma);

    data = (Triplet*) malloc(_matrix->size() * sizeof(Triplet));
    map<pair<int, int>, vector<double>*>::iterator iter;

    j = 0;
    for(iter = _matrix->kv_dict()->begin(); iter != _matrix->kv_dict()->end(); iter++){
        index = iter->first;
        row = index.first;
        col = index.second;
        vector<double>* rating_list = iter->second;
        if(rating_list != NULL){
            for(k = 0; k < rating_list->size(); k++){
                
                data[j].row = row;
                data[j].col = col;
                data[j].val = rating_list->at(k);
                j += 1;
            }
        }
    }

    random_shuffle(data, data + _matrix->size());

    if(_params->validate == 0){
        num_train = _matrix->size();
        num_validate = 0;
    }
    else{
        num_validate = _matrix->size() / _params->validate;
        num_train = _matrix->size() - num_validate;
    }

    clock_t start, terminal;
    pthread_t pt;
    double dot, err, duration, prob;
    double bu_update, bi_update;
    double p[_params->num_factor], q[_params->num_factor], swap[_params->num_factor];

    for(s = 0; s < _params->max_epoch; s++){
        random_shuffle(data, data + num_train);
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

            if(_params->type == T_REGRESS){
                err = t.val - (_model->mu + bu_update + bi_update + dot);
            }
            else if(_params->type == T_CLASSIFY){
                prob = sigmoid(_model->mu + bu_update + bi_update + dot);
                err = t.val - prob;
            }

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
            
            if(_params->type == T_REGRESS){
                loss_train[s] += err * err;
            }
            else if(_params->type == T_CLASSIFY){
                loss_train[s] -= t.val * log(prob) + (1.0 - t.val)  * log(1.0 - prob);
                if((prob >= 0.5 && t.val == 0.0) || (prob < 0.5 && t.val == 1.0))
                    clf_err_train[s] += 1.0;
            }
        }

        pthread_rwlock_wrlock(&MatrixFactorization::rwlock);
        _model->swap();
        pthread_rwlock_unlock(&MatrixFactorization::rwlock); 
      
        int epoch = s;
        // void* params[4] = {_model, &path, &epoch, &rwlock};
        void* params[3] = {_model, &path, &epoch};

        pthread_create(&pt, NULL, &MatrixFactorization::run_helper, params);

        // _model->dump("test");

        terminal = clock();
        duration = (double) (terminal - start) / (double) (CLOCKS_PER_SEC);

        if(_params->type == T_REGRESS){
            loss_train[s] = sqrt(loss_train[s] / num_train);
        }
        else if(_params->type == T_CLASSIFY){
            loss_train[s] = loss_train[s] / num_train;
            clf_err_train[s] = clf_err_train[s] / num_train;
        }
        validate(num_train, num_train + num_validate, s);

        cerr << setiosflags(ios::fixed);

        cerr << "[Iter]: " << setw(4) << setfill('0') << s << " " \
             << "\t[Train]: Loss=" << setprecision(4) << loss_train[s] << " ";
    
        if(_params->type == T_CLASSIFY){
            cerr << "Accuracy=" << setprecision(4) << 100.0 * (1.0 - clf_err_train[s]) << "% ";
        }

        if(_params->validate != 0){
            cerr << "\t[Validate]: Loss=" << setprecision(4) << loss_validate[s] << " ";
            if(_params->type == T_CLASSIFY){
                cerr << "Accuracy=" << setprecision(4) << 100.0 * (1.0 - clf_err_validate[s]) << "% ";
            }
        }
        cerr << "\t[Duration]: " << setprecision(2) << duration << " Sec."  << endl;      
    }

    pthread_join(pt, NULL);
    return _model;
}


void MatrixFactorization::validate(int start, int end, int epoch){
    double pred, err, val;
    for(int k = start; k < end; k++){
        val = (data + k)->val;
        pred = predict(data + k);
        err = val - pred;
        if(_params->type == T_REGRESS){
            loss_validate[epoch] += err * err;            
        }
        else if(_params->type == T_CLASSIFY){
            loss_validate[epoch] -= val * log(pred) + (1.0 - val) * log(1.0 - pred);
            if((pred >= 0.5 && val == 0.0) || (pred < 0.5 && val == 1.0))
                clf_err_validate[epoch] += 1.0;
        }
    }
    if(_params->type == T_REGRESS){
        loss_validate[epoch] = sqrt(loss_validate[epoch] / (end - start));
    }
    else if(_params->type == T_CLASSIFY){
        loss_validate[epoch] = loss_validate[epoch] / (end - start);
        clf_err_validate[epoch] = clf_err_validate[epoch] / (end - start);
    }
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
    
    if(check != 0 && check != 1){
        for(k = 0; k < _params->num_factor; k++){
            pred += _model->P[p->row * _params->num_factor + k] * _model->Q[p->col * _params->num_factor + k];
        }
    }

    if(_params->type == T_CLASSIFY){
        pred = sigmoid(pred);
    }

    return pred;
}

void MatrixFactorization::test(string in, string out){
   
    ifstream is(in.c_str());
    ofstream os(out.c_str());
    string row, col;
    double val, pred, err, loss, clf_err;
    char line[MAX_LINE_LENGTH];
    int n;
    Triplet* p = new Triplet();
    
    loss = 0.0;
    clf_err = 0.0;
    n = 0;

    while(is.getline(line, MAX_LINE_LENGTH)){
        string strline(line);
        row = string(strtok(line, " "));
        col = string(strtok(NULL, " "));
        val = atof(strtok(NULL, " "));
       
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

        p->val = val;

        // cerr << row << " " << p->row << " " << col << " " << p->col << " " << p->val << endl;
        pred = predict(p);
        // cerr << pred << endl;
        err = p->val - pred;

        if(_params->type == T_REGRESS){
            loss += err * err;
            os << pred << "\t" << strline << endl;
        }

        else if(_params->type == T_CLASSIFY){
            loss -= p->val * log(pred) + (1.0 - p->val) * log(1.0 - pred);
            if((pred >= 0.5 && p->val == 0.0) || (pred < 0.5 && p->val == 1.0))
                clf_err += 1.0;
            os << (pred >= 0.5 ? "1" : "0") << "\t" << setprecision(4) << pred << "\t" << strline << endl;
        }

        n++;
    }

    is.close();
    os.close();

    if(_params->type == T_REGRESS){
        loss = sqrt(1.0 * loss / n);
        cerr << "[Test]: Loss=" << setprecision(4) << loss << endl;
    }
    else if(_params->type == T_CLASSIFY){
        loss = 1.0 * loss / n;
        clf_err = 1.0 * clf_err / n;
        cerr << "[Test]: Loss=" << setprecision(4) << loss << " ";
        cerr << "Accuracy=" << setprecision(4) << 100.0 * (1.0 - clf_err) << "% " << endl;
    }

}

MatrixFactorization::~MatrixFactorization(){
    free(data);
    pthread_rwlock_destroy(&MatrixFactorization::rwlock);
}


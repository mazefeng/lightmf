#include "SparseMatrix.h"
#include "MatrixFactorization.h"
#include "LatentFactorModel.h"

#include <iostream>  
#include <iomanip>  
#include <fstream>
#include <vector>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

#define FLAG_TRAIN      (1<<0)
#define FLAG_MODEL      (1<<1)
#define FLAG_TYPE       (1<<2)
#define FLAG_NUM_FACTOR (1<<3)
#define FLAG_SIGMA      (1<<4)
#define FLAG_LAMBDA     (1<<5)
#define FLAG_MAX_EPOCH  (1<<6)
#define FLAG_ALPHA      (1<<7)
#define FLAG_VALIDATE   (1<<8)
#define FLAG_HELP       (1<<9)
#define MAX_PATH_LENGTH 256

typedef struct{
    unsigned int flags;
    char train[MAX_PATH_LENGTH];
    char model[MAX_PATH_LENGTH];
    MfParams* params;
}Opt;

Opt* opt;


int parse_cmd_line(int argc, char *const argv[]){
    int option;
    const char *optstring = "";
    struct option longopts[] = {
        {"train",       1, NULL, 0},
        {"model",       1, NULL, 1},
        {"type",        1, NULL, 2},
        {"num_factor",  1, NULL, 3},
        {"sigma",       1, NULL, 4},
        {"lambda",      1, NULL, 5},
        {"max_epoch",   1, NULL, 6},
        {"alpha",       1, NULL, 7},
        {"validate",    1, NULL, 8},
        {"help",        0, NULL, 9},
        {0, 0, 0, 0}
    };
 
    while ((option = getopt_long_only(argc, argv, optstring, longopts, NULL)) != -1){
        switch (option) {
            case 0:
                opt->flags |= FLAG_TRAIN;
                memcpy(opt->train, optarg, strlen(optarg));
                break;
            case 1:
                opt->flags |= FLAG_MODEL;
                memcpy(opt->model, optarg, strlen(optarg));
                break;
            case 2:
                opt->flags |= FLAG_TYPE;
                opt->params->type = atoi(optarg);
                break;
            case 3:
                opt->flags |= FLAG_NUM_FACTOR;
                opt->params->num_factor = atoi(optarg);
                break;
            case 4:
                opt->flags |= FLAG_SIGMA;
                opt->params->sigma = atof(optarg);
                break;
            case 5:
                opt->flags |= FLAG_LAMBDA;
                opt->params->lambda = atof(optarg);
                break;
            case 6:
                opt->flags |= FLAG_MAX_EPOCH;
                opt->params->max_epoch = atoi(optarg);
                break;
            case 7:
                opt->flags |= FLAG_ALPHA;
                opt->params->alpha = atof(optarg);
                break;
            case 8:
                opt->flags |= FLAG_VALIDATE;
                opt->params->validate = atoi(optarg);
                break;
            case 9:
                opt->flags |= FLAG_HELP;
                break;
        }
    }

    if (!(opt->flags & FLAG_TRAIN)) return FLAG_TRAIN;
    if (!(opt->flags & FLAG_MODEL)) return FLAG_MODEL;
    if (!(opt->flags & FLAG_TYPE))  return FLAG_TYPE;
    return 0;
}
 
void print_help(void)
{
    cerr <<
        "Usage: lightmf-train [OPTIONS]\n"
        "Options are:\n"
        " -train        (required)          Filename for training data \n"
        " -model        (required)          Output path for model \n"
        " -type         (required)          Output type, " << T_REGRESS << " for regression, " << T_CLASSIFY << " for classification \n"
        " -num_factor   (default = 25)      Number of latent factors \n"
        " -sigma        (default = 0.01)    Initial std of normal distribution for latent factors \n"
        " -lambda       (default = 0.005)   L2 regularizaton parameter \n"
        " -max_epoch    (default = 10)      Max training iterations \n"
        " -alpha        (default = 0.01)    Learning rate of SGD \n"
        " -validate     (default = 0)       Proportion of training data for validation \n"
        " -help                             Show this help \n"
    << endl;
}
 
int main(int argc, char **argv){
    
    opt = new Opt();
    opt->params = new MfParams();
    int ret = parse_cmd_line(argc, argv);

    if(opt->flags & FLAG_HELP){
        print_help();
        exit(0);
    }
    
    if(ret & FLAG_TRAIN) cerr << "Train path parameter is required" << endl;
    if(ret & FLAG_MODEL) cerr << "Model path parameter is required" << endl;
    if(ret & FLAG_TYPE)  cerr << "Type parameter is required" << endl;

    if(ret != 0){
        print_help();
        exit(ret);
    }

    struct stat dir_stat;
    if (!((stat(opt->model, &dir_stat) == 0) && S_ISDIR(dir_stat.st_mode))){
        cerr << opt->model << " directory not exist" << endl;
        exit(-1);
    }

    cerr << "[Train]: " << opt->train << endl;
    cerr << "[Model]: " << opt->model << endl;
    cerr << "[Type]: " << opt->params->type << endl;
    cerr << "[NumFactor]: " << opt->params->num_factor << endl;
    cerr << "[Sigma]: " << opt->params->sigma << endl;
    cerr << "[Lambda]: " << opt->params->lambda << endl;
    cerr << "[MaxEpoch]: " << opt->params->max_epoch << endl;
    cerr << "[Alpha]: " << opt->params->alpha << endl;
    cerr << "[Validate]: " << opt->params->validate << endl;
 
    SparseMatrix* matrix = SparseMatrix::load(opt->train);
    MatrixFactorization* mf_model = new MatrixFactorization();
 
    matrix->print_info();

    string path(opt->model);
    if(path.compare(path.size() - 1, 1, "/")){
        path = path + "/";
    }
 
    LatentFactorModel* model = mf_model->train(opt->params, matrix, path);

    return 0;
}



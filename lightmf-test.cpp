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

using namespace std;

#define FLAG_MODEL      (1<<0)
#define FLAG_TEST       (1<<1)
#define FLAG_OUTPUT     (1<<2)
#define FLAG_HELP       (1<<8)
#define MAX_PATH_LENGTH 256

typedef struct{
    unsigned int flags;
    char model[MAX_PATH_LENGTH];
    char test[MAX_PATH_LENGTH];
    char output[MAX_PATH_LENGTH];
}Opt;

Opt* opt;

int parse_cmd_line(int argc, char *const argv[]){
    int option;
    const char *optstring = "";
    struct option longopts[] = {
        {"model",   1, NULL, 0},
        {"test",    1, NULL, 1},
        {"output",  1, NULL, 2},
        {"help",    0, NULL, 3},
        {0, 0, 0, 0}
    };
 
    while ((option = getopt_long_only(argc, argv, optstring, longopts, NULL)) != -1){
        switch (option) {
            case 0:
                opt->flags |= FLAG_MODEL;
                memcpy(opt->model, optarg, strlen(optarg));
                break;
            case 1:
                opt->flags |= FLAG_TEST;
                memcpy(opt->test, optarg, strlen(optarg));
                break;
            case 2:
                opt->flags |= FLAG_OUTPUT;
                memcpy(opt->output, optarg, strlen(optarg));
                break;
            case 3:
                opt->flags |= FLAG_HELP;
                break;
        }
    }

    if (!(opt->flags & FLAG_MODEL))  return FLAG_MODEL;
    if (!(opt->flags & FLAG_TEST))   return FLAG_TEST;
    if (!(opt->flags & FLAG_OUTPUT)) return FLAG_OUTPUT;
    return 0;
}
 
void print_help(void)
{
    cerr <<
        "Usage: lightmf-test [OPTIONS]\n"
        "Options are:\n"
        " -model    (required)  Latent factor model path \n"
        " -test     (required)  Filename for test data \n"
        " -output   (required)  Filename for output data \n"
        " -help                 Show this help \n"
    << endl;
}
 
int main(int argc, char **argv){

    opt = new Opt();

    int ret = parse_cmd_line(argc, argv);
    
    if(opt->flags & FLAG_HELP){
        print_help();
        exit(0);
    }

    if(ret & FLAG_MODEL)  cerr << "Model path parameter is required" << endl;
    if(ret & FLAG_TEST)   cerr << "Test path parameter is required" << endl;
    if(ret & FLAG_OUTPUT) cerr << "Output path parameter is required" << endl;

    if(ret != 0){
        print_help();
        exit(ret);
    }

    cerr << "[Model]: " << opt->model << endl;
    cerr << "[Test]: " << opt->test << endl;
    cerr << "[Output]: " << opt->output << endl;
 
    LatentFactorModel* model = new LatentFactorModel();   
    model->load(opt->model);

    MfParams* params = new MfParams();
    params->num_factor = model->getNumFactor();
    params->sigma = model->getSigma();

    SparseMatrix* matrix = model->getSparseMatrix();

    MatrixFactorization* mf = new MatrixFactorization();
    mf->setMfParams(params);
    mf->setSparseMatrix(matrix);
    mf->setLatentFactorModel(model);

    mf->test(opt->test, opt->output);


    return 0;
}



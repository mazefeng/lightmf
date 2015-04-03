# lightmf
A light-weight matrix factorization tool

##Introduction

##Useage

###lightmf-train
`
./lightmf-train [OPTIONS]
Options are:
 -train        (required)          Filename for training data 
 -model        (required)          Output path for model 
 -num_factor   (default = 25)      Number of latent factors 
 -sigma        (default = 0.01)    Initial std of normal distribution for latent factors 
 -lambda       (default = 0.005)   L2 regularizaton parameter 
 -max_epoch    (default = 10)      Max training iterations 
 -alpha        (default = 0.01)    Learning rate of SGD 
 -validate     (default = 0)       Proportion of training data for validation 
 -help                             Show this help 
`

###lightmf-test

`
./lightmf-test [OPTIONS]
Options are:
 -model    (required)  Latent factor model path 
 -test     (required)  Filename for test data 
 -output   (required)  Filename for output data 
 -help                 Show this help 
`

##Evaluation


##Todo




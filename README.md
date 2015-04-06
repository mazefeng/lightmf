# lightmf
A light-weight matrix factorization tool

##Introduction

*lightmf*是一个轻量级的矩阵分解工具,实现了推荐系统中其中一类重要的模型--隐因子模型的训练和预测.

从功能上看lightmf使用了SGD实现了对带偏置的隐因子模型(BiasMF)的训练.
尽管没有像SVDFeature一样提供包括SVD++,Learning to Rank等衍生模型,
也没有libmf提供除了SGD外的ALS,MCMC等多种模型训练方式,lightmf的易用性也是前两者无法比拟的,体现在以下几个方面:

1. 训练和预测工具简单易用,只需要提供2-3个必要的参数即可运行
2. 输入数据格式简洁,无需做任何预处理.支持文本id数据
3. 支持模型验证,可以指定训练过程,训练数据中用于验证的比例
4. 训练模型明文保存,支持对模型进行分析和二次开发
5. 采用多线程加速,模型训练速度快

##Useage

###lightmf-train

隐因子模型的训练数据包括有uid,iid和rating组成的三元组,三元组之间以一个空格隔开："uid iid rating 附加信息".
附加信息不会影响到模型的训练.

除了训练数据的路径,另外一个必须提供的参数是模型保存的路径,该路径必须是一个目录,用于保存每一轮迭代后的模型.

除了train和model这两个参数外,其他参数都是可选的,包括：

1. num_factor: 隐因子的数量.隐因子数量越多,训练速度越慢,同时过拟合的可能性更高.默认为25,一般隐因子数量的选择范围是20~50
2. sigma: 随机初始化隐因子模型的正态分布的方差.使用以0为均值的正态分布对隐因子模型进行初始化化是一种常见做法.正态分布方差的选择对结果会有一定的影响.默认值是0.01.使用太小的方差会影响模型的收敛速度
3. lambda: L2正则化数据,控制模型复杂度和泛化能力之间的权衡.默认值是0.005
4. max_epoch: 训练迭代轮数.默认值是10.每一轮训练完成后,中间结果都会保存到model指定的路径下
5. alpha: SGD的学习率.默认值为0.01.对于矩阵分解这类非凸模型来说,学习率直接影响了模型最终是否能够收敛
6. validate: 验证数据的比例.设置为n意味着1/n的训练数据将用不会用于训练,而是单独用于验证模型的效果.validate默认值为0,即不进行验证.推荐的做法是,使用一定比例的数据用于验证,挑选出最优的参数后,在使用最优参数和全量训练数据重新训练模型. 



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



###lightmf-test

    ./lightmf-test [OPTIONS]
    Options are:
     -model    (required)  Latent factor model path 
     -test     (required)  Filename for test data 
     -output   (required)  Filename for output data 
     -help                 Show this help 

##Evaluation


##Todo




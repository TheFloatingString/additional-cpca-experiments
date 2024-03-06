# Additional cPCA Experiments

## Claims that We're Trying to Make

1. cPCA-preprocessed data yields better model performance downstream, compared to PCA
2. cPCA-preprocessed data yields better model performance downstream compared to no preprocessing
3. We idenify which types of backgrounds are most effective for cPCA
4. We identify the cPCA parameters which are optimal (for both alpha and number of dimensions)

## Motivation

1. Datasets with high-dimensionality are expensive to run models on
2. cPCA provides better target label separation relative to PCA or no preprocessing

## Limitations

1. cPCA-compressed data is difficult to explain

## Set of Experiments

### Evaluating Numerical Datasets

Metrics: f1, precision and recall

+ Mouse gene expression dataset
+ Beans dataset

### Evaluating Natural Language Datasets

Metrics: f1, precision and recall

+ Sentiment analysis (sst)

### Evaluating Image Datasets

Metrics: f1, precision and recall

+ CIFAR-10 classification

### Evaluating Effective Backgrounds

+ Ablation study of having unlabelled beans
+ Comparing different backgrounds

- models: tabpfn, tabicl, xgboost, svc
- preprocessing: None, cPCA, PCA
- dimensions (for PCA and cPCA): 2, 10, 20, 50
- alphas (for cPCA): 1, 10, 100
- random seed (for numpy): 42
- random state (for sklearn train test split): 42
- metrics: f1-score, accuracy, standard error, standard deviation, time elapsed per run
- 80% train, 20% split
- dataset: micro-mass from OpenML

run on modal, log results on wandb in tabular format

# cPCA Performance Analysis

## Dataset: Micro-Mass (OpenML ID: 1514)

Total experiments: 204
Number of data splits: 3 (seeds: 8, 42, 123)

## Summary

**No cases found where cPCA outperformed both PCA and no preprocessing.**

This suggests that for this particular dataset (micro-mass), standard PCA or no dimensionality reduction may be sufficient.

## Notes

- Alpha values tested: 0, 1, 10, 100
- Dimensionality: 2, 10, 20, 50 components
- Classifiers: TabPFN, TabICL, SVC, XGBoost
- Metric: Accuracy (averaged across 3 random train/test splits)

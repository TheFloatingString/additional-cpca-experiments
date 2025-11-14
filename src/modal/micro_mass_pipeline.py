"""
Modal pipeline for micro-mass dataset with cPCA and TabPFN classification.

This script:
1. Loads the micro-mass dataset from OpenML (id=1301)
2. Splits data 50-50 by target class (background/foreground)
3. Runs cPCA dimension reduction (2, 10, 20, 50 dimensions)
4. Runs TabPFN classification with 5-fold validation
5. Logs results to W&B in tabular format
"""

import modal
import os

# Create Modal app
app = modal.App("micro-mass-cpca-pipeline")

# Define Docker image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Install git first for git-based pip installs
    .pip_install(
        "openml",
        "numpy",
        "scikit-learn",
        "wandb",
        "tqdm",
        "xgboost",
    )
    .pip_install(
        "contrastive @ git+https://github.com/abidlabs/contrastive.git",
        "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git",
        "tabicl @ git+https://github.com/soda-inria/tabicl.git",
    )
)

# Create secrets for W&B
wandb_secret = modal.Secret.from_name("wandb-secret")


def load_dataset(dataset_id=1514):
    """Load dataset from OpenML."""
    import openml
    import numpy as np

    print(f"Loading micro-mass dataset (id={dataset_id})...")
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique classes: {np.unique(y)}")

    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def split_foreground_background(X, y, seed):
    """Split data 50-50 by target class into foreground/background."""
    import numpy as np

    np.random.seed(seed)

    # Get unique classes and shuffle them
    unique_classes = np.unique(y)
    shuffled_classes = unique_classes.copy()
    np.random.shuffle(shuffled_classes)

    # Split classes 50-50
    num_classes = len(shuffled_classes)
    split_point = num_classes // 2

    foreground_classes = shuffled_classes[:split_point]
    background_classes = shuffled_classes[split_point:]

    print(f"Total classes: {num_classes}")
    print(f"Foreground classes: {sorted(foreground_classes.tolist())}")
    print(f"Background classes: {sorted(background_classes.tolist())}")

    # Assign samples to foreground or background based on their class
    foreground_mask = np.isin(y, foreground_classes)
    background_mask = np.isin(y, background_classes)

    X_foreground = X[foreground_mask]
    y_foreground = y[foreground_mask]
    X_background = X[background_mask]
    y_background = y[background_mask]

    print(f"Foreground shape: {X_foreground.shape}")
    print(f"Background shape: {X_background.shape}")
    print(f"Foreground class distribution: {np.unique(y_foreground, return_counts=True)}")
    print(f"Background class distribution: {np.unique(y_background, return_counts=True)}")

    return X_foreground, y_foreground, X_background, y_background, foreground_classes, background_classes


def preprocess_for_cpca(X_foreground, X_background):
    """Pre-process with PCA to reduce dimensions for cPCA compatibility."""
    import numpy as np
    from sklearn.decomposition import PCA

    # cPCA library has issues with high-dimensional data where features >> samples
    # Reduce to min(n_samples, n_features) - 1 to be safe
    max_initial_dims = min(X_foreground.shape[0], X_foreground.shape[1],
                          X_background.shape[0], X_background.shape[1]) - 10

    if X_foreground.shape[1] > max_initial_dims:
        print(f"\nPre-processing for PCA/cPCA: Reducing dimensions from {X_foreground.shape[1]} to {max_initial_dims} with PCA...")
        pca_preprocessor = PCA(n_components=max_initial_dims)
        # Fit on combined data
        X_combined = np.vstack([X_foreground, X_background])
        pca_preprocessor.fit(X_combined)
        X_foreground = pca_preprocessor.transform(X_foreground)
        X_background = pca_preprocessor.transform(X_background)
        print(f"After PCA preprocessing - Foreground shape: {X_foreground.shape}, Background shape: {X_background.shape}")

    return X_foreground, X_background


def get_classifier(classifier_name):
    """Initialize and return a classifier by name."""
    from sklearn.svm import SVC
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    import xgboost as xgb

    if classifier_name == "tabpfn":
        return TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
    elif classifier_name == "tabicl":
        return TabICLClassifier()
    elif classifier_name == "svc":
        return SVC()
    elif classifier_name == "xgboost":
        return xgb.XGBClassifier()
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


def run_classification(X_data, y_data, method_name, classifier_name, n_comp,
                       split_idx, seed, fg_classes_str, bg_classes_str,
                       foreground_samples, background_samples, description=""):
    """Run classification with cross-validation and return results."""
    import numpy as np
    import time
    import wandb
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.preprocessing import LabelEncoder

    print(f"Running {classifier_name} classification with 5-fold validation...")

    # Initialize classifier
    clf = get_classifier(classifier_name)

    # Encode labels to be contiguous (required by XGBoost and some other classifiers)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    # Determine if this is a binary or multiclass problem
    unique_classes_fg = np.unique(y_encoded)
    is_binary = len(unique_classes_fg) == 2
    f1_average = 'binary' if is_binary else 'weighted'

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average=f1_average, zero_division=0)
    }

    start_time = time.time()
    cv_results = cross_validate(clf, X_data, y_encoded,
                               cv=5, scoring=scoring, return_train_score=False)
    classification_time = time.time() - start_time

    acc_scores = cv_results['test_accuracy']
    f1_scores = cv_results['test_f1']

    mean_acc = float(np.mean(acc_scores))
    std_acc = float(np.std(acc_scores))
    stderr_acc = float(np.std(acc_scores) / np.sqrt(len(acc_scores)))

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    stderr_f1 = float(np.std(f1_scores) / np.sqrt(len(f1_scores)))

    print(f"Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}, stderr: {stderr_acc:.4f})")
    print(f"F1 Score: {mean_f1:.4f} (+/- {std_f1:.4f}, stderr: {stderr_f1:.4f})")
    print(f"Classification completed in {classification_time:.2f} seconds")

    # Create description if not provided
    if not description:
        if method_name == "no_reduction":
            description = f"Split{split_idx+1}: No dim reduction, {X_data.shape[1]} features, {classifier_name}"
        elif method_name == "pca":
            description = f"Split{split_idx+1}: PCA to {n_comp} dims, {classifier_name}"
        elif method_name == "cpca":
            description = f"Split{split_idx+1}: cPCA to {n_comp} dims, {classifier_name}"

    result = {
        "split_idx": split_idx + 1,
        "random_seed": seed,
        "foreground_classes": fg_classes_str,
        "background_classes": bg_classes_str,
        "method": method_name,
        "classifier": classifier_name,
        "description": description,
        "n_components": n_comp if n_comp is not None else X_data.shape[1],
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "stderr_accuracy": stderr_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "stderr_f1": stderr_f1,
        "cv_accuracy_scores": acc_scores.tolist(),
        "cv_f1_scores": f1_scores.tolist(),
        "classification_time_seconds": classification_time,
        "data_shape": list(X_data.shape),
        "foreground_samples": foreground_samples,
        "background_samples": background_samples,
    }

    wandb.log(result)
    return result


def run_no_reduction_baseline(X_foreground_original, y_foreground, classifiers,
                               split_idx, seed, fg_classes_str, bg_classes_str,
                               foreground_samples, background_samples):
    """Run baseline with no dimensionality reduction."""
    import wandb

    results_data = []

    for clf_name in classifiers:
        print(f"\n{'='*60}")
        print(f"Baseline: No dimensionality reduction - {clf_name}")
        print(f"{'='*60}")
        try:
            result = run_classification(
                X_foreground_original, y_foreground, "no_reduction", clf_name, None,
                split_idx, seed, fg_classes_str, bg_classes_str,
                foreground_samples, background_samples
            )
            results_data.append(result)
        except Exception as e:
            print(f"Error with no dimensionality reduction ({clf_name}): {e}")
            result = {"method": "no_reduction", "classifier": clf_name, "error": str(e)}
            results_data.append(result)
            wandb.log(result)

    return results_data


def run_pca_experiments(X_foreground, X_background, y_foreground, n_components_list,
                        classifiers, split_idx, seed, fg_classes_str, bg_classes_str,
                        foreground_samples, background_samples):
    """Run PCA-only experiments for each target dimension."""
    import numpy as np
    import wandb
    from sklearn.decomposition import PCA

    results_data = []

    for n_components in n_components_list:
        print(f"\n{'='*60}")
        print(f"PCA-only with {n_components} components")
        print(f"{'='*60}")
        try:
            pca_model = PCA(n_components=n_components)
            # Fit on combined foreground+background data
            X_combined = np.vstack([X_foreground, X_background])
            pca_model.fit(X_combined)
            # Transform only foreground for classification
            X_pca = pca_model.transform(X_foreground)
            print(f"PCA fitted on combined data (shape: {X_combined.shape})")
            print(f"PCA transformed foreground data shape: {X_pca.shape}")

            for clf_name in classifiers:
                result = run_classification(
                    X_pca, y_foreground, "pca", clf_name, n_components,
                    split_idx, seed, fg_classes_str, bg_classes_str,
                    foreground_samples, background_samples
                )
                results_data.append(result)
        except Exception as e:
            print(f"Error with PCA {n_components} components: {e}")
            result = {"method": "pca", "n_components": n_components, "error": str(e)}
            results_data.append(result)
            wandb.log(result)

    return results_data


def run_cpca_experiments(X_foreground, X_background, y_foreground, n_components_list,
                         classifiers, split_idx, seed, fg_classes_str, bg_classes_str,
                         foreground_samples, background_samples, alpha_list=None):
    """Run cPCA experiments for each dimension and alpha value."""
    import numpy as np
    import time
    import wandb
    from contrastive import CPCA

    if alpha_list is None:
        alpha_list = [0, 1, 10, 100]

    results_data = []

    for n_components in n_components_list:
        for alpha in alpha_list:
            print(f"\n{'='*60}")
            print(f"cPCA with {n_components} components, alpha={alpha}")
            print(f"{'='*60}")

            try:
                # Run cPCA
                print(f"Running cPCA with n_components={n_components}, alpha={alpha}...")
                start_time = time.time()
                cpca_model = CPCA(n_components=n_components)
                X_cpca = cpca_model.fit_transform(X_foreground, X_background,
                                                  alpha_selection="manual", alpha_value=alpha)
                cpca_time = time.time() - start_time
                print(f"cPCA completed in {cpca_time:.2f} seconds")

                # Handle potential list output from cPCA
                if isinstance(X_cpca, list):
                    X_cpca_transformed = np.asarray(X_cpca)[0]
                else:
                    X_cpca_transformed = X_cpca

                print(f"Transformed data shape: {X_cpca_transformed.shape}")

                # Run classification with all classifiers
                for clf_name in classifiers:
                    description = f"Split{split_idx+1}: cPCA to {n_components} dims (alpha={alpha}), {clf_name}"
                    result = run_classification(
                        X_cpca_transformed, y_foreground, "cpca", clf_name, n_components,
                        split_idx, seed, fg_classes_str, bg_classes_str,
                        foreground_samples, background_samples, description=description
                    )
                    result["cpca_alpha"] = alpha
                    result["cpca_time_seconds"] = cpca_time
                    result["total_time_seconds"] = cpca_time + result["classification_time_seconds"]

                    # Log the updated result with cpca timing
                    wandb.log({"cpca_alpha": alpha, "cpca_time_seconds": cpca_time, "total_time_seconds": result["total_time_seconds"]})

                    results_data.append(result)

            except Exception as e:
                print(f"Error running cPCA with {n_components} components, alpha={alpha}: {e}")
                result = {
                    "method": "cpca",
                    "n_components": n_components,
                    "cpca_alpha": alpha,
                    "error": str(e),
                }
                results_data.append(result)
                wandb.log(result)

    return results_data


def create_wandb_table(results_data):
    """Create and log W&B table with all results."""
    import wandb

    print(f"\n{'='*60}")
    print("Creating W&B table with all results...")
    print(f"{'='*60}")

    columns = [
        "split_idx",
        "random_seed",
        "foreground_classes",
        "background_classes",
        "method",
        "classifier",
        "description",
        "n_components",
        "cpca_alpha",
        "mean_accuracy",
        "std_accuracy",
        "stderr_accuracy",
        "mean_f1",
        "std_f1",
        "stderr_f1",
        "classification_time_seconds",
        "cpca_time_seconds",
        "total_time_seconds"
    ]

    table_data = [
        [
            r.get("split_idx", 1),
            r.get("random_seed", 8),
            r.get("foreground_classes", ""),
            r.get("background_classes", ""),
            r.get("method", "unknown"),
            r.get("classifier", "unknown"),
            r.get("description", ""),
            r["n_components"],
            r.get("cpca_alpha", None),
            r.get("mean_accuracy", None),
            r.get("std_accuracy", None),
            r.get("stderr_accuracy", None),
            r.get("mean_f1", None),
            r.get("std_f1", None),
            r.get("stderr_f1", None),
            r.get("classification_time_seconds", None),
            r.get("cpca_time_seconds", None),
            r.get("total_time_seconds", None),
        ]
        for r in results_data if "error" not in r
    ]

    results_table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"results_summary": results_table})

    # Log final summary
    wandb.summary["total_experiments"] = len(results_data)
    wandb.summary["dataset_id"] = 1514
    wandb.summary["dataset_name"] = "micro-mass"

    return results_data


def analyze_and_write_findings(results_data, output_path="cpca_analysis.md"):
    """Analyze results and write findings to markdown file."""
    import numpy as np
    from collections import defaultdict

    print(f"\n{'='*60}")
    print("Analyzing results...")
    print(f"{'='*60}")

    # Filter out errors
    valid_results = [r for r in results_data if "error" not in r]

    # Group by method, classifier, n_components, and alpha
    grouped = defaultdict(list)
    for r in valid_results:
        key = (r["method"], r["classifier"], r["n_components"], r.get("cpca_alpha"))
        grouped[key].append(r["mean_accuracy"])

    # Calculate average accuracy across splits
    avg_results = {}
    for key, accuracies in grouped.items():
        avg_results[key] = {
            "mean_acc": np.mean(accuracies),
            "std_acc": np.std(accuracies),
            "n_splits": len(accuracies)
        }

    # Find cases where cPCA outperforms PCA and no preprocessing
    findings = []

    for classifier in ["tabpfn", "tabicl", "svc", "xgboost"]:
        for n_comp in [2, 10, 20, 50]:
            # Get baseline (no preprocessing)
            no_prep_key = ("no_reduction", classifier, valid_results[0].get("n_components", 1300), None)
            no_prep_acc = avg_results.get(no_prep_key, {}).get("mean_acc", 0)

            # Get PCA result
            pca_key = ("pca", classifier, n_comp, None)
            pca_acc = avg_results.get(pca_key, {}).get("mean_acc", 0)

            # Check each alpha value for cPCA
            for alpha in [0, 1, 10, 100]:
                cpca_key = ("cpca", classifier, n_comp, alpha)
                cpca_acc = avg_results.get(cpca_key, {}).get("mean_acc", 0)

                # Check if cPCA outperforms both baselines
                if cpca_acc > pca_acc and cpca_acc > no_prep_acc:
                    improvement_over_pca = ((cpca_acc - pca_acc) / pca_acc * 100) if pca_acc > 0 else 0
                    improvement_over_no_prep = ((cpca_acc - no_prep_acc) / no_prep_acc * 100) if no_prep_acc > 0 else 0

                    findings.append({
                        "classifier": classifier,
                        "n_components": n_comp,
                        "alpha": alpha,
                        "cpca_acc": cpca_acc,
                        "pca_acc": pca_acc,
                        "no_prep_acc": no_prep_acc,
                        "improvement_over_pca": improvement_over_pca,
                        "improvement_over_no_prep": improvement_over_no_prep
                    })

    # Write findings to markdown
    with open(output_path, "w") as f:
        f.write("# cPCA Performance Analysis\n\n")
        f.write("## Dataset: Micro-Mass (OpenML ID: 1514)\n\n")
        f.write(f"Total experiments: {len(valid_results)}\n")
        f.write(f"Number of data splits: 3 (seeds: 8, 42, 123)\n\n")

        if findings:
            f.write("## Cases Where cPCA Outperforms Both PCA and No Preprocessing\n\n")
            f.write("| Classifier | n_components | Alpha | cPCA Acc | PCA Acc | No Prep Acc | Improvement over PCA | Improvement over No Prep |\n")
            f.write("|------------|--------------|-------|----------|---------|-------------|----------------------|-------------------------|\n")

            for finding in sorted(findings, key=lambda x: -x["improvement_over_pca"]):
                f.write(f"| {finding['classifier']} | {finding['n_components']} | {finding['alpha']} | "
                       f"{finding['cpca_acc']:.4f} | {finding['pca_acc']:.4f} | {finding['no_prep_acc']:.4f} | "
                       f"{finding['improvement_over_pca']:.2f}% | {finding['improvement_over_no_prep']:.2f}% |\n")

            f.write("\n## Summary\n\n")
            f.write(f"cPCA outperformed both baselines in **{len(findings)}** configurations.\n\n")

            # Best performing configuration
            if findings:
                best = max(findings, key=lambda x: x["cpca_acc"])
                f.write(f"**Best cPCA configuration:**\n")
                f.write(f"- Classifier: {best['classifier']}\n")
                f.write(f"- Components: {best['n_components']}\n")
                f.write(f"- Alpha: {best['alpha']}\n")
                f.write(f"- Accuracy: {best['cpca_acc']:.4f}\n")
                f.write(f"- Improvement over PCA: {best['improvement_over_pca']:.2f}%\n")
                f.write(f"- Improvement over No Preprocessing: {best['improvement_over_no_prep']:.2f}%\n\n")
        else:
            f.write("## Summary\n\n")
            f.write("**No cases found where cPCA outperformed both PCA and no preprocessing.**\n\n")
            f.write("This suggests that for this particular dataset (micro-mass), standard PCA or no dimensionality reduction may be sufficient.\n\n")

        f.write("## Notes\n\n")
        f.write("- Alpha values tested: 0, 1, 10, 100\n")
        f.write("- Dimensionality: 2, 10, 20, 50 components\n")
        f.write("- Classifiers: TabPFN, TabICL, SVC, XGBoost\n")
        f.write("- Metric: Accuracy (averaged across 3 random train/test splits)\n")

    print(f"Analysis written to {output_path}")
    return findings


@app.function(
    image=image,
    secrets=[wandb_secret],
    timeout=3600,  # 1 hour timeout
    memory=8192,  # 8GB memory
    gpu="any",  # Enable GPU for TabPFN
)
def run_micro_mass_pipeline():
    """Main pipeline function that runs on Modal."""
    import wandb

    # Initialize W&B
    wandb.init(project="micro-mass-cpca", name="micro-mass-experiment")

    # Load dataset
    X, y = load_dataset(dataset_id=1514)

    # Dimensions to test
    n_components_list = [2, 10, 20, 50]

    # Random seeds for 3 different foreground/background splits
    random_seeds = [8, 42, 123]

    # List of classifiers to test
    classifiers = ["tabpfn", "tabicl", "svc", "xgboost"]

    results_data = []

    # Run experiments for each random split
    for split_idx, seed in enumerate(random_seeds):
        print(f"\n{'#'*60}")
        print(f"# SPLIT {split_idx + 1}/3 - Random Seed: {seed}")
        print(f"{'#'*60}\n")

        # Split data into foreground/background
        X_foreground, y_foreground, X_background, y_background, foreground_classes, background_classes = \
            split_foreground_background(X, y, seed)

        # Save original data for no-reduction baseline
        X_foreground_original = X_foreground.copy()

        # Pre-process for cPCA compatibility
        X_foreground, X_background = preprocess_for_cpca(X_foreground, X_background)

        # Convert class arrays to strings for logging
        fg_classes_str = str(sorted(foreground_classes.tolist()))
        bg_classes_str = str(sorted(background_classes.tolist()))

        foreground_samples = X_foreground.shape[0]
        background_samples = X_background.shape[0]

        # Baseline 1: No dimensionality reduction
        results_data.extend(run_no_reduction_baseline(
            X_foreground_original, y_foreground, classifiers,
            split_idx, seed, fg_classes_str, bg_classes_str,
            foreground_samples, background_samples
        ))

        # Baseline 2: PCA-only
        results_data.extend(run_pca_experiments(
            X_foreground, X_background, y_foreground, n_components_list,
            classifiers, split_idx, seed, fg_classes_str, bg_classes_str,
            foreground_samples, background_samples
        ))

        # cPCA experiments
        results_data.extend(run_cpca_experiments(
            X_foreground, X_background, y_foreground, n_components_list,
            classifiers, split_idx, seed, fg_classes_str, bg_classes_str,
            foreground_samples, background_samples
        ))

    # Create W&B table for final results
    create_wandb_table(results_data)

    # Analyze results and write findings
    analyze_and_write_findings(results_data, output_path="/root/cpca_analysis.md")

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Total experiments: {len(results_data)}")
    print(f"Results logged to W&B project: micro-mass-cpca")
    print(f"Analysis written to /root/cpca_analysis.md")

    wandb.finish()

    return results_data


@app.local_entrypoint()
def main():
    """Local entrypoint to run the pipeline."""
    results = run_micro_mass_pipeline.remote()

    # Generate analysis locally as well
    print("\nGenerating local analysis...")
    analyze_and_write_findings(results, output_path="cpca_analysis.md")

    print("\nFinal Results:")
    for result in results:
        if "error" not in result:
            print(f"  Split {result.get('split_idx', '?')} - {result.get('method', '?')} - "
                  f"{result.get('classifier', '?')} - {result.get('n_components', '?')} dims: "
                  f"acc={result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
        else:
            n_comp = result.get('n_components', '?')
            method = result.get('method', '?')
            clf = result.get('classifier', '?')
            print(f"  {method} - {clf} - {n_comp} dims: ERROR - {result['error']}")

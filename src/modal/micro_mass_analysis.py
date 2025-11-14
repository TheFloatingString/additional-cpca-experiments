"""
DNA Analysis Pipeline with cPCA and Multiple Models

This script:
1. Loads the micro-mass dataset from OpenML
2. Applies preprocessing: None, PCA, or cPCA with various dimensions and alphas
3. Trains models: TabPFN, TabICL, XGBoost, SVC
4. Logs results to W&B in tabular format

Based on requirements in 1_dna_analysis.md:
- Models: tabpfn, tabicl, xgboost, svc
- Preprocessing: None, cPCA, PCA
- Dimensions (for PCA and cPCA): 2, 10, 20, 50
- Alphas (for cPCA): 1, 10, 100
- Random seed (for numpy): 42
- Random state (for sklearn train test split): 42
- Metrics: f1-score, accuracy, standard error, standard deviation, time elapsed per run
- 80% train, 20% split
- Dataset: micro-mass from OpenML
"""

import modal

# Create Modal app
app = modal.App("dna-analysis")

# Define Docker image with CUDA support for TabPFN
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "openml",
        "numpy",
        "scipy",
        "scikit-learn",
        "wandb",
        "tqdm",
        "torch",
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


@app.function(
    image=image,
    secrets=[wandb_secret],
    timeout=3600,  # 1 hour timeout
    memory=16384,  # 16GB memory for TabPFN
    gpu="any",  # Enable GPU for TabPFN
)
def run_dna_analysis():
    """Main analysis function that runs on Modal."""
    import numpy as np
    import wandb
    import openml
    import time
    from scipy.stats import sem
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score, accuracy_score
    from contrastive import CPCA
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    import warnings

    warnings.filterwarnings('ignore')

    # Initialize W&B (will update config with class splits after loading data)
    wandb.init(
        project="dna-analysis",
        name="micro-mass-cpca-pca-comparison",
        config={
            "dataset": "micro-mass",
            "models": ["tabpfn", "tabicl", "xgboost", "svc"],
            "preprocessing": ["None", "PCA", "cPCA"],
            "dimensions": [2, 10, 20, 50],
            "alphas": [1, 10, 100],
            "random_seed": 42,
            "train_test_split": 0.8,
        }
    )

    # Set random seed
    np.random.seed(42)

    # Load dataset from OpenML
    print("Loading micro-mass dataset from OpenML...")
    dataset = openml.datasets.get_dataset(1515)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique classes: {np.unique(y)}")

    # Split classes 50-50 for cPCA foreground/background
    unique_classes = np.unique(y)
    shuffled_classes = unique_classes.copy()
    np.random.shuffle(shuffled_classes)

    split_point = len(shuffled_classes) // 2
    foreground_classes = shuffled_classes[:split_point]
    background_classes = shuffled_classes[split_point:]

    print(f"\nForeground classes: {sorted(foreground_classes.tolist())}")
    print(f"Background classes: {sorted(background_classes.tolist())}")

    # Log class splits to wandb
    wandb.config.update({
        "foreground_classes": sorted(foreground_classes.tolist()),
        "background_classes": sorted(background_classes.tolist()),
    })

    # Create foreground and background datasets
    foreground_mask = np.isin(y, foreground_classes)
    background_mask = np.isin(y, background_classes)

    X_foreground = X[foreground_mask]
    y_foreground = y[foreground_mask]
    X_background = X[background_mask]

    print(f"\nForeground shape: {X_foreground.shape}")
    print(f"Background shape: {X_background.shape}")

    # Pre-process for cPCA compatibility (reduce dimensions if needed)
    max_initial_dims = min(X_foreground.shape[0], X_foreground.shape[1],
                          X_background.shape[0], X_background.shape[1]) - 10

    X_foreground_preprocessed = X_foreground.copy()
    X_background_preprocessed = X_background.copy()

    if X_foreground.shape[1] > max_initial_dims:
        print(f"\nPre-processing: Reducing dimensions from {X_foreground.shape[1]} to {max_initial_dims} with PCA...")
        pca_preprocessor = PCA(n_components=max_initial_dims, random_state=42)
        X_combined = np.vstack([X_foreground, X_background])
        pca_preprocessor.fit(X_combined)
        X_foreground_preprocessed = pca_preprocessor.transform(X_foreground)
        X_background_preprocessed = pca_preprocessor.transform(X_background)
        print(f"After preprocessing - Foreground: {X_foreground_preprocessed.shape}, Background: {X_background_preprocessed.shape}")

    # Helper function to get classifier
    def get_classifier(classifier_name):
        """Initialize and return a classifier by name."""
        if classifier_name == "tabpfn":
            return TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
        elif classifier_name == "tabicl":
            return TabICLClassifier()
        elif classifier_name == "svc":
            return SVC()
        elif classifier_name == "xgboost":
            return xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    # Helper function to train and evaluate
    def train_and_evaluate(X_data, y_data, model_name, preprocessing_method, n_components=None, alpha=None):
        """Train a model with single 80/20 split and evaluate with metrics using scipy.sem."""
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Preprocessing: {preprocessing_method} | Components: {n_components} | Alpha: {alpha}")
        print(f"{'='*60}")

        # Encode labels to be contiguous
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_data)

        print(f"Data shape: {X_data.shape}")

        # Determine if binary or multiclass
        unique_classes = np.unique(y_encoded)
        is_binary = len(unique_classes) == 2
        f1_average = 'binary' if is_binary else 'weighted'

        # Train/test split (80/20) with random_state=42
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Train model
        start_time = time.time()
        clf = get_classifier(model_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=f1_average, zero_division=0)

        # Calculate standard deviation and standard error from predictions
        correct_predictions = (y_pred == y_test).astype(float)
        std_accuracy = float(np.std(correct_predictions))
        se_accuracy = float(sem(correct_predictions))

        # For F1, we calculate per-sample "correctness" as binary
        # Note: F1 is a global metric, so std/se are based on prediction correctness
        std_f1 = std_accuracy  # Same as accuracy std
        se_f1 = se_accuracy  # Same as accuracy se

        print(f"Accuracy: {accuracy:.4f} (std: {std_accuracy:.4f}, SE: {se_accuracy:.4f})")
        print(f"F1 Score: {f1:.4f}")
        print(f"Time elapsed: {elapsed_time:.2f}s")

        result = {
            "model": model_name,
            "preprocessing": preprocessing_method,
            "n_components": n_components if n_components is not None else X_data.shape[1],
            "alpha": alpha if alpha is not None else "N/A",
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "std_accuracy": std_accuracy,
            "std_f1": std_f1,
            "se_accuracy": se_accuracy,
            "se_f1": se_f1,
            "time_elapsed": float(elapsed_time),
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
        }

        wandb.log(result)
        return result

    # Store all results
    results = []

    # Configuration
    models = ["tabpfn", "tabicl", "xgboost", "svc"]
    dimensions = [2, 10, 20, 50]
    alphas = [1, 10, 100]

    # 1. No preprocessing (baseline)
    print("\n" + "#"*60)
    print("# BASELINE: No Preprocessing")
    print("#"*60)

    for model_name in models:
        try:
            result = train_and_evaluate(
                X_foreground, y_foreground, model_name, "None",
                n_components=None, alpha=None
            )
            results.append(result)
        except Exception as e:
            print(f"Error with {model_name} (no preprocessing): {e}")
            error_result = {
                "model": model_name,
                "preprocessing": "None",
                "error": str(e)
            }
            results.append(error_result)
            wandb.log(error_result)

    # 2. PCA preprocessing
    print("\n" + "#"*60)
    print("# PCA PREPROCESSING")
    print("#"*60)

    for n_comp in dimensions:
        print(f"\n--- PCA with {n_comp} components ---")
        try:
            # Fit PCA on combined foreground + background
            pca = PCA(n_components=n_comp, random_state=42)
            X_combined = np.vstack([X_foreground_preprocessed, X_background_preprocessed])
            pca.fit(X_combined)
            X_pca = pca.transform(X_foreground_preprocessed)

            print(f"PCA transformed shape: {X_pca.shape}")

            for model_name in models:
                try:
                    result = train_and_evaluate(
                        X_pca, y_foreground, model_name, "PCA",
                        n_components=n_comp, alpha=None
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error with {model_name} (PCA {n_comp}): {e}")
                    error_result = {
                        "model": model_name,
                        "preprocessing": "PCA",
                        "n_components": n_comp,
                        "error": str(e)
                    }
                    results.append(error_result)
                    wandb.log(error_result)

        except Exception as e:
            print(f"Error with PCA {n_comp} components: {e}")

    # 3. cPCA preprocessing
    print("\n" + "#"*60)
    print("# cPCA PREPROCESSING")
    print("#"*60)

    for n_comp in dimensions:
        for alpha in alphas:
            print(f"\n--- cPCA with {n_comp} components, alpha={alpha} ---")
            try:
                # Fit cPCA
                cpca_start = time.time()
                cpca = CPCA(n_components=n_comp, standardize=True)
                X_cpca = cpca.fit_transform(
                    X_foreground_preprocessed,
                    X_background_preprocessed,
                    alpha_selection="manual",
                    alpha_value=alpha
                )
                cpca_time = time.time() - cpca_start

                # Handle potential list output from cPCA
                if isinstance(X_cpca, list):
                    X_cpca = np.asarray(X_cpca)[0]

                print(f"cPCA transformed shape: {X_cpca.shape}")
                print(f"cPCA transformation time: {cpca_time:.2f}s")

                for model_name in models:
                    try:
                        result = train_and_evaluate(
                            X_cpca, y_foreground, model_name, "cPCA",
                            n_components=n_comp, alpha=alpha
                        )
                        result["cpca_time"] = cpca_time
                        result["total_time"] = cpca_time + result["time_elapsed"]
                        wandb.log({"cpca_time": cpca_time, "total_time": result["total_time"]})
                        results.append(result)
                    except Exception as e:
                        print(f"Error with {model_name} (cPCA {n_comp}, alpha={alpha}): {e}")
                        error_result = {
                            "model": model_name,
                            "preprocessing": "cPCA",
                            "n_components": n_comp,
                            "alpha": alpha,
                            "error": str(e)
                        }
                        results.append(error_result)
                        wandb.log(error_result)

            except Exception as e:
                print(f"Error with cPCA {n_comp} components, alpha={alpha}: {e}")

    # Create W&B results table
    print("\n" + "="*60)
    print("Creating W&B results table...")
    print("="*60)

    columns = [
        "model",
        "preprocessing",
        "n_components",
        "alpha",
        "accuracy",
        "f1_score",
        "std_accuracy",
        "std_f1",
        "se_accuracy",
        "se_f1",
        "time_elapsed",
    ]

    table_data = []
    for r in results:
        if "error" not in r:
            row = [
                r["model"],
                r["preprocessing"],
                r["n_components"],
                str(r["alpha"]),  # Convert to string for wandb Table consistency
                r["accuracy"],
                r["f1_score"],
                r["std_accuracy"],
                r["std_f1"],
                r["se_accuracy"],
                r["se_f1"],
                r["time_elapsed"],
            ]
            table_data.append(row)

    results_table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"results_summary": results_table})

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Preprocessing':<15} {'Dims':<6} {'Alpha':<8} {'Accuracy':<20} {'F1 Score':<10} {'Time':<8}")
    print("-" * 95)

    for r in results:
        if "error" not in r:
            model = r["model"]
            preprocessing = r["preprocessing"]
            n_comp = r["n_components"]
            alpha = r["alpha"]
            acc = r["accuracy"]
            se_acc = r["se_accuracy"]
            f1 = r["f1_score"]
            time_val = r["time_elapsed"]
            acc_str = f"{acc:.4f}±{se_acc:.4f}"
            f1_str = f"{f1:.4f}"
            print(f"{model:<12} {preprocessing:<15} {n_comp:<6} {str(alpha):<8} {acc_str:<20} {f1_str:<10} {time_val:.2f}s")
        else:
            print(f"{r['model']:<12} {r['preprocessing']:<15} ERROR: {r.get('error', 'Unknown')}")

    wandb.summary["total_experiments"] = len([r for r in results if "error" not in r])
    wandb.summary["total_errors"] = len([r for r in results if "error" in r])

    wandb.finish()
    print("\nAnalysis complete! Results logged to W&B.")

    return results


@app.local_entrypoint()
def main():
    """Local entrypoint to run the analysis."""
    results = run_dna_analysis.remote()

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Total successful experiments: {len([r for r in results if 'error' not in r])}")
    print(f"Total errors: {len([r for r in results if 'error' in r])}")
    print("Results logged to W&B project: dna-analysis")

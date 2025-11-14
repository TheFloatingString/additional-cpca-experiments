"""
TTT Analysis - Multi-class Classification on Micro-Mass Dataset

This script:
1. Loads the micro-mass dataset from OpenML (id=1514)
2. Performs multi-class classification on ALL classes (no preprocessing)
3. Uses only TabPFN model
4. Logs results to W&B

Configuration:
- Model: TabPFN only
- Preprocessing: None
- Random seed: 42
- Random state (for sklearn train test split): 42
- Metrics: f1-score, accuracy, standard error, standard deviation, time elapsed
- 80% train, 20% test split
- Dataset: micro-mass from OpenML (id=1514)
- All classes included in classification
"""

import modal

# Create Modal app
app = modal.App("ttt-analysis")

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
        "torch",
    )
    .pip_install(
        "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git",
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
def run_ttt_analysis():
    """Main analysis function that runs on Modal."""
    import numpy as np
    import wandb
    import openml
    import time
    from scipy.stats import sem
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score
    from tabpfn import TabPFNClassifier
    from sklearn.preprocessing import LabelEncoder
    import warnings

    warnings.filterwarnings('ignore')

    # Initialize W&B
    wandb.init(
        project="ttt-analysis",
        name="micro-mass-tabpfn-multiclass",
        config={
            "dataset": "micro-mass",
            "dataset_id": 1514,
            "model": "tabpfn",
            "preprocessing": "None",
            "random_seed": 42,
            "train_test_split": 0.8,
        }
    )

    # Set random seed
    np.random.seed(42)

    # Load dataset from OpenML
    print("Loading micro-mass dataset from OpenML (id=1514)...")
    dataset = openml.datasets.get_dataset(1514)

    # Sanity check: verify this is the micro-mass dataset
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset description: {dataset.description[:200] if dataset.description else 'No description'}...")

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    # Log dataset info to wandb
    wandb.config.update({
        "n_classes": len(np.unique(y)),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "classes": sorted(np.unique(y).tolist()),
    })

    # Encode labels to be contiguous
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nEncoded classes: {np.unique(y_encoded)}")

    # Train/test split (80/20) with random_state=42
    print("\nPerforming 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train classes distribution: {np.bincount(y_train)}")
    print(f"Test classes distribution: {np.bincount(y_test)}")

    # Train TabPFN model
    print("\n" + "="*60)
    print("Training TabPFN model...")
    print("="*60)

    start_time = time.time()
    clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    elapsed_time = time.time() - start_time

    print(f"Training and prediction completed in {elapsed_time:.2f}s")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Calculate standard deviation and standard error from predictions
    correct_predictions = (y_pred == y_test).astype(float)
    std_accuracy = float(np.std(correct_predictions))
    se_accuracy = float(sem(correct_predictions))

    # For F1, std/se are based on prediction correctness
    std_f1 = std_accuracy
    se_f1 = se_accuracy

    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} (std: {std_accuracy:.4f}, SE: {se_accuracy:.4f})")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Time elapsed: {elapsed_time:.2f}s")

    # Log results to wandb
    result = {
        "model": "tabpfn",
        "preprocessing": "None",
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

    # Create confusion matrix analysis
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Log confusion matrix to wandb
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred,
            class_names=[str(i) for i in range(len(np.unique(y_encoded)))]
        )
    })

    # Create results table
    print("\n" + "="*60)
    print("Creating W&B results table...")
    print("="*60)

    columns = [
        "model",
        "preprocessing",
        "accuracy",
        "f1_score",
        "std_accuracy",
        "std_f1",
        "se_accuracy",
        "se_f1",
        "time_elapsed",
    ]

    table_data = [[
        result["model"],
        result["preprocessing"],
        result["accuracy"],
        result["f1_score"],
        result["std_accuracy"],
        result["std_f1"],
        result["se_accuracy"],
        result["se_f1"],
        result["time_elapsed"],
    ]]

    results_table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"results_summary": results_table})

    wandb.summary["accuracy"] = accuracy
    wandb.summary["f1_score"] = f1
    wandb.summary["time_elapsed"] = elapsed_time

    wandb.finish()
    print("\nAnalysis complete! Results logged to W&B.")

    return result


@app.local_entrypoint()
def main():
    """Local entrypoint to run the analysis."""
    result = run_ttt_analysis.remote()

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Accuracy: {result['accuracy']:.4f} ± {result['se_accuracy']:.4f}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"Time: {result['time_elapsed']:.2f}s")
    print("Results logged to W&B project: ttt-analysis")

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from tabicl import TabICLClassifier
from sklearn.svm import SVC
from contrastive import CPCA
import openml
import tqdm
import numpy as np
import yaml
import xgboost as xgb
import time

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

from tabpfn import TabPFNClassifier

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="myapp.log", level=logging.INFO)


def run_classifier_and_append_results(
    clf, X_foreground, y_foreground, output_data_local, clf_name, exp_name
):
    logger.info(f"running: {exp_name} - {clf_name}")
    # Initialize a classifier
    start = time.time()
    scores = cross_val_score(clf, X_foreground, y_foreground, cv=5)
    end = time.time()
    output_data_local["results"].append(
        {
            "acc": float(float(round(np.mean(scores), 3))),
            "std": float(float(round(np.std(scores), 3))),
            "classifier": clf_name,
            "exp": exp_name,
            "time_in_seconds": round(end - start, 3),
        }
    )
    return output_data_local


def run_single_experiment(
    TASK_ID: int = 3560, output_filepath: str = "outfile.yaml", ndim: int = 2
):
    output_data = {"task_id": TASK_ID, "results": [], "ndim": ndim}
    # suite = openml.study.get_suite(99)

    # TASK_ID=11 # balance scale
    # TASK_ID=167140 # dna
    # TASK_ID= 53 # vehicle
    # TASK_ID=2074 # SAT IMAgE
    # TASK_ID = 167140 #DNA
    # TASK_ID = 3560  # dmft
    # TASK_ID = 3573 # mnist
    # TASK_ID=12

    task = openml.tasks.get_task(TASK_ID)

    X, y = task.get_X_and_y()
    X = np.asarray(X)
    y = np.asarray(y)

    X_foreground = []
    y_foreground = []

    X_background = []
    y_background = []

    for i in tqdm.trange(X.shape[0]):
        if y[i] not in [0, 1]:
            X_background.append(X[i])
            y_background.append(y[i])
        else:
            X_foreground.append(X[i])
            y_foreground.append(y[i])

    X_foreground = np.asarray(X_foreground)
    X_background = np.asarray(X_background)

    # --- NO PCA OR CPCA --- #

    output_data = run_classifier_and_append_results(
        clf=TabPFNClassifier(ignore_pretraining_limits=True),
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="tabpfn",
        exp_name="no_cpca",
    )
    output_data = run_classifier_and_append_results(
        clf=SVC(),
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="svc",
        exp_name="no_cpca",
    )
    output_data = run_classifier_and_append_results(
        clf=TabICLClassifier(),
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="tabicl",
        exp_name="no_cpca",
    )
    output_data = run_classifier_and_append_results(
        clf=xgb.XGBClassifier(),
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="xgboost",
        exp_name="no_cpca",
    )
    output_data = run_classifier_and_append_results(
        clf=DecisionTreeClassifier(),
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="decision_tree",
        exp_name="no_cpca",
    )

    # --- PCA --- #

    pca_model = PCA(n_components=ndim)
    X_data_original_compress = pca_model.fit_transform(X_foreground)

    output_data = run_classifier_and_append_results(
        clf=TabPFNClassifier(ignore_pretraining_limits=True),
        X_foreground=X_data_original_compress,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="tabpfn",
        exp_name="pca",
    )
    output_data = run_classifier_and_append_results(
        clf=SVC(),
        X_foreground=X_data_original_compress,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="svc",
        exp_name="pca",
    )
    output_data = run_classifier_and_append_results(
        clf=TabICLClassifier(),
        X_foreground=X_data_original_compress,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="tabicl",
        exp_name="pca",
    )
    output_data = run_classifier_and_append_results(
        clf=xgb.XGBClassifier(),
        X_foreground=X_data_original_compress,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="xgboost",
        exp_name="pca",
    )
    output_data = run_classifier_and_append_results(
        clf=DecisionTreeClassifier(),
        X_foreground=X_data_original_compress,
        y_foreground=y_foreground,
        output_data_local=output_data,
        clf_name="decision_tree",
        exp_name="pca",
    )

    # --- CPCA --- #

    mdl = CPCA(n_components=ndim)
    projected_data = mdl.fit_transform(X_foreground, X_background)

    for i in range(np.asarray(projected_data).shape[0]):
        logger.info(f"cpca at {i}")

        output_data = run_classifier_and_append_results(
            clf=TabPFNClassifier(ignore_pretraining_limits=True),
            X_foreground=np.asarray(projected_data)[i],
            y_foreground=y_foreground,
            output_data_local=output_data,
            clf_name="tabpfn",
            exp_name=f"cpca-choice-of-alpha-{i}",
        )
        output_data = run_classifier_and_append_results(
            clf=SVC(),
            X_foreground=np.asarray(projected_data)[i],
            y_foreground=y_foreground,
            output_data_local=output_data,
            clf_name="svc",
            exp_name=f"cpca-choice-of-alpha-{i}",
        )
        output_data = run_classifier_and_append_results(
            clf=TabICLClassifier(),
            X_foreground=np.asarray(projected_data)[i],
            y_foreground=y_foreground,
            output_data_local=output_data,
            clf_name="tabicl",
            exp_name=f"cpca-choice-of-alpha-{i}",
        )
        output_data = run_classifier_and_append_results(
            clf=xgb.XGBClassifier(),
            X_foreground=np.asarray(projected_data)[i],
            y_foreground=y_foreground,
            output_data_local=output_data,
            clf_name="xgb",
            exp_name=f"cpca-choice-of-alpha-{i}",
        )
        output_data = run_classifier_and_append_results(
            clf=DecisionTreeClassifier(),
            X_foreground=np.asarray(projected_data)[i],
            y_foreground=y_foreground,
            output_data_local=output_data,
            clf_name="decision_tree",
            exp_name=f"cpca-choice-of-alpha-{i}",
        )
    #     # Initialize a classifier
    #     clf = TabPFNClassifier()
    #     scores = cross_val_score(clf, np.asarray(projected_data)[i], y_foreground, cv=5)
    #     output_data["results"].append(
    #         {
    #             "acc": float(round(np.mean(scores), 3)),
    #             "std": float(round(np.std(scores), 3)),
    #             "classifier": "tabpfn",
    #             "exp": f"cpca-choice-of-alpha-{i}",
    #         }
    #     )

    #     clf = SVC()
    #     scores = cross_val_score(clf, X_data_original_compress, y_foreground, cv=5)
    #     output_data["results"].append(
    #         {
    #             "acc": float(round(np.mean(scores), 3)),
    #             "std": float(round(np.std(scores), 3)),
    #             "classifier": "svc",
    #             "exp": f"cpca-choice-of-alpha-{i}",
    #         }
    #     )

    #     clf = TabICLClassifier()
    #     scores = cross_val_score(clf, X_data_original_compress, y_foreground, cv=5)
    #     output_data["results"].append(
    #         {
    #             "acc": float(round(np.mean(scores), 3)),
    #             "std": float(round(np.std(scores), 3)),
    #             "classifier": "tabicl",
    #             "exp": f"cpca-choice-of-alpha-{i}",
    #         }
    #     )

    with open(output_filepath, "w") as outputfile:
        yaml.dump(output_data, outputfile)

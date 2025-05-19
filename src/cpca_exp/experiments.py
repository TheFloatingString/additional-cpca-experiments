from sklearn.svm import SVC
from tabicl import TabICLClassifier
from sklearn.svm import SVC
from contrastive import CPCA
import openml
import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier


def run_single_experiment(TASK_ID: int = 3560):
    suite = openml.study.get_suite(99)
    print(suite)

    # TASK_ID=11 # balance scale
    # TASK_ID=167140 # dna
    # TASK_ID= 53 # vehicle
    # TASK_ID=2074 # SAT IMAgE
    # TASK_ID = 167140 #DNA
    # TASK_ID = 3560  # authorship
    # TASK_ID=12

    task = openml.tasks.get_task(TASK_ID)
    print(task)

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
    # Load data
    # task = openml.tasks.get_task(TASK_ID)
    # X, y = task.get_X_and_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X_foreground, y_foreground, test_size=0.2, random_state=42
    )

    # Initialize a classifier
    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)

    print("Accuracy with no PCA or cPCA:")

    # Predict probabilities
    prediction_probabilities = clf.predict_proba(X_test)
    print("ROC AUC:", round(roc_auc_score(y_test, prediction_probabilities[:, 1]), 3))

    # Predict labels
    predictions = clf.predict(X_test)
    print("Accuracy", round(accuracy_score(y_test, predictions), 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X_foreground, y_foreground, test_size=0.2, random_state=42
    )

    # Initialize a classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    print("Accuracy with no PCA or cPCA:")

    # Predict labels
    predictions = clf.predict(X_test)
    print("Accuracy", round(accuracy_score(y_test, predictions), 3))

    clf = TabICLClassifier()
    clf.fit(X_train, y_train)  # this is cheap
    clf.predict(X_test)  # in-context learning happens here

    from sklearn.decomposition import PCA

    pca_model = PCA(n_components=2)
    X_data_original_compress = pca_model.fit_transform(X_foreground)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data_original_compress, y_foreground, test_size=0.2, random_state=42
    )

    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    print("Accuracy with PCA:")

    # Predict probabilities
    prediction_probabilities = clf.predict_proba(X_test)
    print("ROC AUC:", round(roc_auc_score(y_test, prediction_probabilities[:, 1]), 3))

    # Predict labels
    predictions = clf.predict(X_test)
    print("Accuracy", round(accuracy_score(y_test, predictions), 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X_data_original_compress, y_foreground, test_size=0.2, random_state=42
    )

    # Initialize a classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    print("Accuracy with PCA:")

    # Predict labels
    predictions = clf.predict(X_test)
    print("Accuracy", round(accuracy_score(y_test, predictions), 3))

    mdl = CPCA(n_components=2)
    projected_data = mdl.fit_transform(X_foreground, X_background)

    # returns a set of 2-dimensional projections of the foreground data stored in the list 'projected_data', for several different values of 'alpha' that are automatically chosen (by default, 4 values of alpha are chosen)

    print("Accuracy with cPCA:")
    print("-------------------")

    for i in range(np.asarray(projected_data).shape[0]):
        X_train, X_test, y_train, y_test = train_test_split(
            np.asarray(projected_data)[i], y_foreground, test_size=0.2, random_state=42
        )

        # Initialize a classifier
        clf = TabPFNClassifier()
        clf.fit(X_train, y_train)

        print(f"choice {i + 1} of alpha:")
        # Predict probabilities
        # prediction_probabilities = clf.predict_proba(X_test)
        # print("ROC AUC:", round(roc_auc_score(y_test, prediction_probabilities[:, 1]),3))

        # Predict labels
        predictions = clf.predict(X_test)
        print("tabpfn Accuracy", round(accuracy_score(y_test, predictions), 3))

        # Initialize a classifier
        clf = SVC()
        clf.fit(X_train, y_train)

        # print("Accuracy with PCA:")

        # Predict labels
        predictions = clf.predict(X_test)
        print("svc Accuracy", round(accuracy_score(y_test, predictions), 3))
        print()

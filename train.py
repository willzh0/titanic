import pandas as pd
import numpy as np
import re
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

SV_classifier = SVC()
KN_classifier = KNeighborsClassifier()
RF_classifier = RandomForestClassifier()
ridge_classifier = RidgeClassifier()
logit_regression = LogisticRegression()
MLP_classifier = MLPClassifier()
bag_classifier = BaggingClassifier()
NB_classifier = GaussianNB()
XGB_classifier = XGBClassifier()
gradient_boost_classifier = GradientBoostingClassifier()

SVC_param_grid = {'kernel': ['rbf', "linear", "polynomial"], 'decision_function_shape': ['ovo', 'ovr'], 'gamma': ["scale", "auto", 1e-3, 1e-4],
                  'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
KNN_param_grid = {"n_neighbors": [2 * i for i in range(1, 41)], "weights": ["uniform", "distance"], "p": [1, 2], }
RFC_param_grid = {"n_estimators": [10 * i for i in range(5, 41)], "max_depth": [3, 8, 16, 32],
                  "min_samples_split": [2, 4, 6, 8], "min_samples_leaf": [1, 3, 5, 7]}
ridge_param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],  "solver": ["svd", "cholesky", "sparse_cg", "lsqr", "sag"]}
logit_param_grid = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10],  "solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],  "max_iter": [3000]}
MLP_param_grid = {"hidden_layer_sizes": [(15, 10, 5, 2), (15, 30, 15, 2), (15, 15, 15, 2), (20, 15, 10, 10, 2)], "solver": ["lbfgs", "sgd", "adam"],
                  "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1], "learning_rate": ["constant", "invscaling", "adaptive"], "max_iter": [3000]}
bag_param_grid = {"n_estimators": [10 * i for i in range(2, 41)], "max_samples": [i for i in range(1, 15, 2)],
                  "max_features": [i for i in range(1, 15, 2)]}
NB_param_grid = {}
XGB_param_grid = {"booster": ["gbtree", "dart", "gblinear"], "eta": [0.001, 0.01, 0.1, 0.3], "max_depth": [3, 6, 8, 16],
                  "num_parallel_tree": [1, 3, 5], "subsample": [0.3, 0.5, 1]}
grad_boost_param_grid = {"loss": ["deviance", "exponential"], "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.25],
                         "min_samples_split": [2, 4, 6, 8], "min_samples_leaf": [1, 3, 5, 7],
                         "max_depth": [3, 6, 8, 16], "n_estimators": [10 * i for i in range(5, 41)]}

def parse_name(name):
    """Given the name of each individual from the dataset, parses them into
     and returns their last name, title and other parts (inc. first names). """

    matches = re.match(r"(.*), (.*?)\. (.*)", name)
    last_name, title, other_name_part = matches.group(1, 2, 3)

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Dr']:
        title = 'Mr'
    elif title in ['the Countess', 'Dona', 'Mme', 'Lady']:
        title = 'Mrs'
    elif title in ['Mlle', 'Ms']:
        title = "Miss"
    else:
        pass

    return last_name, title, other_name_part

# remove_outliers is only applied on the training set.
def remove_outliers(X, features, z_score):
    """We get the index of outliers, given each feature.
    If an index appears multiple times, that means it's an outlier of multiple features.
    Removes the individuals corresponding to each index from the dataset inplace. """

    outlier_indices = []
    for feature in features:
        mean = X[feature].mean()
        std_dev = X[feature].std()
        outlier_list_col = X[(X[feature] < mean - z_score * std_dev) |
                             (X[feature] > mean + z_score * std_dev)].index
        outlier_indices.extend(outlier_list_col)

    # Remove duplicate indices, if an individual is an outlier of multiple features.
    outlier_indices = list(set(outlier_indices))
    X.drop(outlier_indices, axis=0, inplace=True)


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return

    def transform(self, X, y=None):
        mean_age = X["Age"].mean()
        std_dev = X["Age"].std()
        X["Age"].fillna(np.random.randint(mean_age - std_dev, mean_age + std_dev), inplace=True)
        X.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)

        last_name_list = []
        titles = []
        for name in X["Name"]:
            last_name, title, other_name_part = parse_name(name)
            last_name_list.append(last_name)
            titles.append(title)

        last_name_list = np.array(last_name_list)
        titles = np.array(titles)

        X.drop("Name", axis=1, inplace=True)
        X["Title"] = titles

        median_fare = X["Fare"].median()
        X["Fare"].fillna(median_fare, inplace=True)

        X.dropna(subset=["Embarked"], inplace=True)

        # Add new attributes:
        X["RelativesOnboard"] = X["SibSp"] + X["Parch"]
        X.drop("SibSp", axis=1, inplace=True)
        X.drop("Parch", axis=1, inplace=True)


initial_feature_transformer = FeaturesTransformer()

num_attributes = ["Age", "Fare", "Pclass", "RelativesOnboard"]
numeric_transformer = Pipeline(steps=[
    ('scale', StandardScaler())
])
cat_attributes = ["Title", "Embarked", "Sex"]
cat_transformer = Pipeline(steps=[
    ('encode', OneHotEncoder())
])

preprocessing_transformer = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_attributes),
    ('cat', cat_transformer, cat_attributes)
])

def main():
    import time
    training_set = pd.read_csv("datasets\\train.csv")
    # Drop outliers:
    remove_outliers(training_set, ["Age", "Fare"], z_score=3)
    # Manipulating features.
    initial_feature_transformer.transform(training_set)

    # Seperating the Survived column from the rest of the dataset.
    y_train = training_set[["Survived"]].to_numpy().reshape(-1, 1)
    x_train = training_set.drop("Survived", axis=1)

    # Scaling each numerical attribute. Encoding each categorical attribute.
    x_train = preprocessing_transformer.fit_transform(x_train)

    classifiers = [SV_classifier, KN_classifier, RF_classifier, ridge_classifier, logit_regression,
                    MLP_classifier, bag_classifier, NB_classifier,
                   XGB_classifier, gradient_boost_classifier]

    param_grids = [SVC_param_grid, KNN_param_grid, RFC_param_grid, ridge_param_grid, logit_param_grid,
                    MLP_param_grid, bag_param_grid, NB_param_grid,
                    XGB_param_grid, grad_boost_param_grid]

    # Use Grid search each classification model
    # Grid search returns the best model, with select hyperparameters.
    # Model with the best score is our final model.
    score = 0
    grid_search = 0
    no_of_classifiers = len(classifiers)
    times = []
    list_grid_searches = []
    for index, (classifier, param_grid) in enumerate(zip(classifiers, param_grids)):
        print("K-fold cross-validation of {}... ({}/{})" .format(classifier, index+1, no_of_classifiers))

        last_time = time.time()
        current_grid_search = GridSearchCV(classifier, param_grid, cv=4,
                                       return_train_score=True, n_jobs= -1, verbose=1,
                                       scoring=["accuracy", "precision", "recall"], refit="accuracy")
        current_grid_search.fit(x_train, y_train.ravel())
        times.append(time.time() - last_time)

        if current_grid_search.best_score_ > score:
            grid_search = current_grid_search
            score = current_grid_search.best_score_

        list_grid_searches.append(grid_search)

    print("Time taken for K-fold cross-validation for each classifier (seconds):")
    for classifier, time in zip(classifiers, times):
        print("{0}: {1}" .format(classifier, time))
    print("\n")

    print("Best scores for each classifier type:")
    for each_grid_search in list_grid_searches:
        print("{0}, Score: {1}" .format(grid_search.estimator, each_grid_search.best_score_))
    print("\n")
    print("Final classifier: ")
    print("Best estimator:", grid_search.best_estimator_)
    print("Best parameters:", grid_search.best_params_)
    print("Score:", grid_search.best_score_)
    print("Index:", grid_search.best_index_, "\n")
    print("\n")
    cvresults = pd.DataFrame(grid_search.cv_results_)
    print(cvresults.iloc[grid_search.best_index_])
    print("\n")
    print("Use the model on the test set with 'test.py'")

    # Saves the grid_search variable with joblib.
    dump(grid_search, "final_model.joblib")

if __name__ == '__main__':
    main()
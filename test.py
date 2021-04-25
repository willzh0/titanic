from train import initial_feature_transformer, preprocessing_transformer
from joblib import load
import pandas as pd

def main():
    test_set_initial = pd.read_csv("datasets\\test.csv")
    test_set = test_set_initial.copy()
    initial_feature_transformer.transform(test_set)
    test_set = preprocessing_transformer.fit_transform(test_set)

    chosen_grid_search = load("final_model.joblib")

    print("Using model based on {0}." .format(chosen_grid_search.best_estimator_))
    print("Predicting test set survivors...")
    y_predicted = chosen_grid_search.best_estimator_.predict(test_set)

    results = pd.DataFrame()
    results["PassengerId"] = test_set_initial["PassengerId"]
    results["Survived"] = y_predicted

    results.to_csv("datasets/predictions.csv", index=False)
    print("Predictions written to predictions.csv")

if __name__ == "__main__":
    main()
import os
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


class Model:
    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.feature_names = self.model.feature_names
    
    def predict(self, X):
        # X.reindex(columns=self.feature_names, fill_value=0)
        prediction = self.model.predict_proba(X)
        print(f"Prediction: {prediction}")
        return prediction[0][1]

    def train(self, grid_search=False):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(current_dir + "/model/data.csv")

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        params = None
        if grid_search:
            rf = RandomForestClassifier()

            param_grid = {
                'n_estimators': [100, 200, 300],  # The number of trees in the forest.
                'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split.
                'max_depth': [None, 10, 20, 30],  # The maximum depth of the tree.
                'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node.
                'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node.
                'max_features': ['auto', 'sqrt', 'log2', None],  # The number of features to consider when looking for the best split.
                'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees.
                'class_weight': ['balanced', 'balanced_subsample', None]  # Weights associated with classes. This could be useful if you have imbalance between classes.
            }

            print("Choosing best hyperparameters...")
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            params = best_params
            print("Best hyperparameters: ", best_params)

        if not params:
             params = {
                "bootstrap": True,
                "class_weight": "balanced_subsample",
                "criterion": "gini",
                "max_depth": None,
                "max_features": "sqrt",
                "min_samples_leaf": 4,
                "min_samples_split": 10,
                "n_estimators": 200
            }

        model = RandomForestClassifier(**params )

        print("Training model using Random Forest Classifier...")

        model.fit(X_train, y_train)

        calibrator = CalibratedClassifierCV(model, method="isotonic")
        calibrator.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy score: ", accuracy, "F1 score: ", f1)
        
        # --- Uncomment to see feature importances ---
        feature_names = X_train.columns
        feature_importances = model.feature_importances_

        print("Feature importances: ")
        for feature_name, feature_importance in zip(feature_names, feature_importances):
                print(f"Feature: {feature_name}, Importance: {feature_importance}")

        self.test_model(model)

        self.model = model
        self.model.feature_names = X.columns

        pickle.dump(self.model, open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model/model.pkl", "wb"))

    def test_model(self, model):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_df = pd.read_csv(current_dir + "/model/test_data.csv")

        test_X = test_df.drop("label", axis=1)
        test_y = test_df["label"]

        y_pred = model.predict(test_X)

        accuracy = accuracy_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)

        print("Test dataset accuracy score: ", accuracy, "Test dataset F1 score: ", f1)
        print("---------------------------------")

if __name__ == "__main__":
    model = Model()
    model.train()

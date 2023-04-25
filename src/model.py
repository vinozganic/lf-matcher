import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class Model:
    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.feature_names = self.model.feature_names
    
    def predict(self, X):
        X.reindex(columns=self.feature_names, fill_value=0)
        prediction = self.model.predict_proba(X)
        print(f"Prediction: {prediction}")
        return prediction[0][1]

    def train(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(current_dir + "/model/data.csv")

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        models = {
            # "logistic_regression": LogisticRegression(),
            # "knn": KNeighborsClassifier(),
            # "decision_tree": DecisionTreeClassifier(),
            # "svm": SVC(probability=True),
            "random_forest": RandomForestClassifier(n_estimators=1000),
        }

        best_model = None
        best_model_accuracy = 0
        
        for name, model in models.items():
            print("Training model: ", name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print("Accuracy score: ", accuracy, "F1 score: ", f1)
            
            # --- Uncomment to see feature importances ---
            # feature_names = X_train.columns
            # feature_importances = model.feature_importances_

            # for feature_name, feature_importance in zip(feature_names, feature_importances):
            #     print(f"Feature: {feature_name}, Importance: {feature_importance}")

            # self.test_model(model)

            if accuracy > best_model_accuracy:
                best_model = model
                best_model_accuracy = accuracy

        self.model = best_model
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

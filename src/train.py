# src/train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../model/knn_irish_model.pkl')


def train():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the trained model
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    print("housing_price_prediction_ml_implementation.py")


if __name__ == "__main__":
    train()

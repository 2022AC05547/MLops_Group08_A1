# src/test.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, '../model/logistic_regression_model.pkl')


def test_model():
    # Check if the model file exists

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the saved model
    model = joblib.load(model_path)

    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

    # Make predictions with the model
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    test_model()

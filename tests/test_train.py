import os
import sys
import pytest
from train import train

# Redirect stderr to avoid fileno issues
sys.stderr = open(os.devnull, 'w')

# Add src directory to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
)


def test_train():
    """Test the training function."""
    # Ensure the model directory does not exist before training
    model_path = "model/knn_irish_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Run the training function
    train()

    # Check that the model file is created
    assert os.path.exists(model_path), "Model file not created"


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_train.py"])

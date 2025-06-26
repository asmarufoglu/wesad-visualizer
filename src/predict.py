import pickle
import numpy as np
import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Maps numeric label values to human-readable emotion names
LABEL_MAP = {1: "Rest", 2: "Amusement", 3: "Stress"}

#  Load a pre-trained model from a .pkl file 
def load_model(model_path="model/trained_model.pkl"): # i create a model folder because avoid of overwrite to trained_model.pkl.
    """
    Loads a trained model from the given path using pickle.
    Returns the model object if successful.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        raise RuntimeError(f"❌ Model file not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

# Train and save a Decision Tree model 
def train_and_save_model(X, y, model_path="model/trained_model.pkl"):
    """
    Trains a simple Decision Tree classifier on the given data
    and saves the model to a .pkl file.
    """
    # Ensure inputs are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and test sets (80/20). suggested ratio, actually i dont know its details.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    #ensure for the model directory exists
    os.makedirs("model", exist_ok=True)

    # Save the trained model to disk
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully to:", model_path)

# Make a prediction for a single input sample
def predict_sample(model, feature_vector):
    """
    Predicts the class of a single input sample using the trained model.
    Returns both the numeric label and the mapped class name.
    """
    feature_vector = np.array(feature_vector)

    # Ensure input is a 2D array with shape (1, n_features)
    if feature_vector.ndim != 2 or feature_vector.shape[0] != 1:
        raise ValueError("Feature vector must be a 2D array with shape (1, n_features)")

    # Make prediction
    predicted_label = model.predict(feature_vector)[0]

    # Convert label to human-readable class name
    predicted_class = LABEL_MAP.get(predicted_label, "Unknown")

    return predicted_label, predicted_class

# END-TO-END-DATA-SCIENCE-PROJECT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from flask import Flask, request, jsonify
import os

# Step 1: Data Collection

def collect_data():
    """
    Simulate data collection by loading a dataset.
    Replace this with actual data collection logic if needed.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv("data.csv")  # Replace with actual data source

# Step 2: Data Preprocessing

def preprocess_data(data):
    """
    Preprocess the data by handling missing values and splitting into features and target.
    Args:
        data (pd.DataFrame): Raw data.
    Returns:
        tuple: Processed features (X) and target (y).
    """
    target_column = "target"  # Replace with your target column name
    features = data.drop(columns=[target_column])
    target = data[target_column]
    return features, target

# Step 3: Model Training

def train_model(X, y):
    """
    Train a Random Forest model.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Step 4: Model Evaluation

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.
    Returns:
        float: Model accuracy.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Step 5: Deployment with Flask

app = Flask(__name__)

# Load model and data columns
data_columns = []
model = None
if os.path.exists("model.joblib"):
    model = load("model.joblib")
    data_columns = load("columns.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict using the trained model.
    """
    if not model:
        return jsonify({"error": "Model not available."}), 500

    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data], columns=data_columns)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Data processing and training
    data = collect_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the model and data columns
    dump(model, "model.joblib")
    dump(X.columns.tolist(), "columns.joblib")

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "./models/model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Define the mapping for prediction labels
PREDICTION_MAPPING = {
    0: "normal",
    1: "severely stunted",
    2: "stunted",
    3: "high"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON
        data = request.json
        umur = data.get("umur")
        jenis_kelamin = data.get("jenis_kelamin")
        tinggi_badan = data.get("tinggi_badan")

        # Preprocess the input features
        jenis_kelamin_encoded = 0 if jenis_kelamin == "laki-laki" else 1
        features = np.array([[umur, jenis_kelamin_encoded, tinggi_badan]])

        # Make prediction
        numerical_prediction = model.predict(features)[0]  # Get the first prediction

        # Map numerical prediction to a label
        prediction_label = PREDICTION_MAPPING.get(int(numerical_prediction), "Unknown")

        # Return prediction as a JSON response
        return jsonify({
            "prediction": prediction_label,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        })

if __name__ == '__main__':
    app.run(debug=True)

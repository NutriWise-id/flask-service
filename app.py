from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained nutrition model
NUTRITION_MODEL_PATH = "./models/kalkulator_nutrisi/model.pkl"
with open(NUTRITION_MODEL_PATH, "rb") as file:
    nutrition_model = pickle.load(file)

# Load the food identification model
FOOD_MODEL_PATH = "./models/identifikasi_makanan/my_model.h5"
food_model = load_model(FOOD_MODEL_PATH)

# Load model architecture if needed (optional)
# with open("./models/identifikasi_makanan/model_architecture.json", "r") as json_file:
#     model_architecture = json_file.read()

# Define the mapping for prediction labels
NUTRITION_PREDICTION_MAPPING = {
    0: "normal",
    1: "severely stunted",
    2: "stunted",
    3: "high"
}

FOOD_PREDICTION_MAPPING = {
    0: "Bakso",
    1: "Bebek Betutu",
    2: "Gado-Gado",
    3: "Gudeg",
    4: "Nasi Goreng",
    5: "Pempek",
    6: "Rawon",
    7: "Rendang",
    8: "Sate",
    9: "Soto"
}

@app.route('/', methods=['GET'])
def hello():
    return "Hello World"

@app.route('/predict_stunting', methods=['POST'])
def predict_nutrition():
    try:
        data = request.json
        umur = data.get("umur")
        jenis_kelamin = data.get("jenis_kelamin")
        tinggi_badan = data.get("tinggi_badan")

        # Preprocess the input features
        jenis_kelamin_encoded = 0 if jenis_kelamin == "laki-laki" else 1
        features = np.array([[umur, jenis_kelamin_encoded, tinggi_badan]])

        # Make prediction
        numerical_prediction = nutrition_model.predict(features)[0]  # Get the first prediction

        # Map numerical prediction to a label
        prediction_label = NUTRITION_PREDICTION_MAPPING.get(int(numerical_prediction), "Unknown")

        return jsonify({
            "prediction": prediction_label,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        })

@app.route('/predict_food', methods=['POST'])
def predict_food():
    try:
        # Get the image file from the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided", "status": "failure"})

        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        image_file.save(image_path)

        # Preprocess the image
        img = load_img(image_path, target_size=(224, 224))  # Adjust to your model's input size
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = food_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Get the class index

        # Map the prediction to a label
        prediction_label = FOOD_PREDICTION_MAPPING.get(predicted_class, "Unknown")

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

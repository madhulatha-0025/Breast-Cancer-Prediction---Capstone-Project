from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "✅ BloomCare ML Backend Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    model_columns = model.feature_names_in_
    df = df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df)[0]
    result = "Benign" if prediction == 1 else "Malignant"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(port=8000, debug=True)

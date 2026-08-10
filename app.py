from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json(force=True)

        # Convert JSON data to NumPy array
        input_data = np.array(data)

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Return result
        return jsonify({
            'prediction': prediction.tolist(),
            'prediction_probabilities': prediction_proba.tolist()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the feature columns used during training
# Example:
# feature_columns = ['age', 'income', 'score']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Convert JSON data to DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({
                'error': 'Invalid input data format'
            }), 400

        # Ensure the order of columns matches training data
        global X_train

        if 'X_train' in globals():
            input_df = input_df[X_train.columns]
        else:
            return jsonify({
                'error': 'Model features not available.'
            }), 500

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Return prediction as JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'prediction_probabilities': prediction_proba.tolist()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

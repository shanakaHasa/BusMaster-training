from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsRegressor
import joblib

app = Flask(__name__)

# Load the KNN model
knn_model = joblib.load('time_pred_model.sav')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from Flutter app
        data = request.get_json(force=True)

        # Extract features for prediction
        input_data = [[
            data['school_encoded'],
            data['company_encoded'],
            data['traffic_encoded'],
            data['weather_encoded'],
            data['day'],
            data['hour']
        ]]

        # Make prediction using KNN model
        predicted_travel_time = knn_model.predict(input_data)

        # Return the prediction to the Flutter app
        response = {'predicted_travel_time': predicted_travel_time[0]}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)

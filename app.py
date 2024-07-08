from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)


model = load_model('weather_prediction_model.h5')
scaler = joblib.load('scaler.save')
label_encoder = joblib.load('label_encoder.save')


data = pd.read_csv('dataset/weather.csv', parse_dates=['date'], index_col='date')

# Convertir la colonne 'weather' en valeurs numériques
data['weather'] = label_encoder.transform(data['weather'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédiction météorologique basée sur la date fournie.
    ---
    parameters:
      - name: date
        in: body
        type: string
        required: true
        description: La date pour laquelle la prédiction est requise (format YYYY-MM-DD).
    responses:
      200:
        description: Prédiction réussie
        schema:
          type: object
          properties:
            predicted_precipitation:
              type: number
            predicted_temp_max:
              type: number
            predicted_temp_min:
              type: number
            predicted_wind:
              type: number
            predicted_weather:
              type: string
      400:
        description: Pas assez de données pour la prédiction
    """
    content = request.get_json()
    date_str = content['date']
    date = pd.to_datetime(date_str)

    past_data = data.loc[:date].tail(144)
    
    if len(past_data) < 144:
        available_start_date = data.index[143].strftime('%Y-%m-%d')
        return jsonify({'error': 'Pas assez de données pour la prédiction. ', 
                        'available_start_date': available_start_date}), 400
    
    input_data = past_data.values
    input_data_scaled = scaler.transform(input_data)

    input_sequence = input_data_scaled[-144:]  # Les dernières 144 observations
    input_sequence = np.expand_dims(input_sequence, axis=0)

    prediction = model.predict(input_sequence)
    prediction_unscaled = scaler.inverse_transform(np.concatenate((prediction, input_sequence[:, -1, 1:]), axis=-1))

    result = {
        'predicted_precipitation': prediction_unscaled[0, 0],
        'predicted_temp_max': prediction_unscaled[0, 1],
        'predicted_temp_min': prediction_unscaled[0, 2],
        'predicted_wind': prediction_unscaled[0, 3],
        'predicted_weather': label_encoder.inverse_transform([int(prediction_unscaled[0, 4])])[0]
    }

    return jsonify(result)

# Swagger  
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Weather Prediction API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run()

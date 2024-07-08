from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# 
df = pd.read_csv('dataset/weather.csv', parse_dates=True, index_col="date")

# Afficher les premières lignes du dataset pour inspection
print(df.head())

# Encoder les valeurs catégorielles de la colonne 'weather'
label_encoder = LabelEncoder()
df['weather_encoded'] = label_encoder.fit_transform(df['weather'])

# Sélection des colonnes pertinentes
data = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'weather_encoded']].copy()

# Normalisation des données
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

quit

X = sequences[:, :-1]
y = sequences[:, -1, 1]  # Supposez que nous prédisons 'temp_max'
 #----------------------------------
# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length-1, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Sauvegarder le modèle et le scaler
model.save('weather_prediction_model.h5')
joblib.dump(scaler, 'scaler.save')
joblib.dump(label_encoder, 'label_encoder.save')
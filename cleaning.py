import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Charger le dataset
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

seq_length = 144  # Par exemple, 10 jours de données horaires
sequences = create_sequences(data_scaled, seq_length)

X = sequences[:, :-1]
y = sequences[:, -1, 1]  # Supposez que nous prédisons 'temp_max'

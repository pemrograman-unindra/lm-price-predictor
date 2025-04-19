from preprocessing import load_data, normalize_data, create_sequences
from model import build_lstm_model
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess data
df = load_data('data/lm_price.csv')
df, scaler = normalize_data(df)
data = df["Price_Normalized"].values

# Create sequences
X, y = create_sequences(data, window_size=30)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)
model.save("model_lm.h5")
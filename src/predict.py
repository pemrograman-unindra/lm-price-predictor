import numpy as np
import pandas as pd
from keras.models import load_model
from preprocessing import normalize_data

# Load data
df = pd.read_csv("data/lm_price.csv", parse_dates=["dateTime"])
df.sort_values("dateTime", inplace=True)
df, scaler = normalize_data(df)

# Ambil data terakhir untuk prediksi (1 tahun terakhir)
window_size = 365
last_sequence = df["Price_Normalized"].values[-window_size:]
X_input = np.array(last_sequence).reshape((1, window_size, 1))

# Load model dan prediksi
model = load_model("model_lm.h5")
predicted_normalized = model.predict(X_input)

# Kembalikan ke skala asli
predicted_price = scaler.inverse_transform(predicted_normalized)
print(f"Next price prediction: Rp {predicted_price[0][0]:,.2f}")
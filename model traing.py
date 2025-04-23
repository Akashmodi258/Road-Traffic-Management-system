import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("preprocessed_traffic.csv")
print("Loaded preprocessed_traffic.csv | Shape:", df.shape)

# Drop rows with any NaNs
df.dropna(inplace=True)

# Target column
target_column = 'all_motor_vehicles'
if target_column not in df.columns:
    raise ValueError(f"'{target_column}' not found in dataset")

# Scaling
scaler = MinMaxScaler()
df['Scaled_Traffic'] = scaler.fit_transform(df[[target_column]])
joblib.dump(scaler, 'scaler.save')

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_len = 24
traffic_series = df['Scaled_Traffic'].values
X, y = create_sequences(traffic_series, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model (with Input layer to avoid warning)
model = Sequential([
    Input(shape=(seq_len, 1)),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

# Save model (modern format recommended)
model.save("traffic_lstm_model.keras")

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# Predict
y_pred = model.predict(X_test)

# Inverse scale predictions and ground truth
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Handle NaNs in prediction
if np.isnan(y_pred_inv).any() or np.isnan(y_test_inv).any():
    raise ValueError("NaN detected in predictions or ground truth")

# Evaluate
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot prediction vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Traffic")
plt.show()

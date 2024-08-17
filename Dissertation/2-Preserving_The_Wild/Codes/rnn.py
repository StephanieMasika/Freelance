import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Generate synthetic time series data
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) 
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) 
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis]

n_steps = 50
series = generate_time_series(10000, n_steps)
X_train, y_train = series[:7000, :], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :], series[7000:9000, -1]
X_test, y_test = series[9000:, :], series[9000:, -1]

# Define the LSTM model
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=[None, 1]),
    layers.LSTM(50, return_sequences=True),
    layers.LSTM(50),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, 
                    validation_data=(X_valid, y_valid))

# Plot training and validation loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

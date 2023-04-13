import json

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Input(shape=(11,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'],
)

model.summary()

# Load the data from the CSV file
data = pd.read_csv('heart_data.csv')

X = data.iloc[:, 2:13]
y = data.iloc[:, 13:14]
print(X.head())
print(y.head())

# 42 is a seed for random
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {test_acc:.3f}')

# Save the model in TensorFlow format
model.save("heart_model")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model("heart_model")
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("heart_model.tflite", "wb") as f:
    f.write(tflite_model)
with open('scaler_mean.json', 'w') as f:
    json.dump(scaler.mean_.tolist(), f)
with open('scaler_scale.json', 'w') as f:
    json.dump(scaler.scale_.tolist(), f)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Features (X) and Target (Y)
X = df[["Age", "BMI", "BloodPressure", "Glucose", "FamilyHistory"]].values
Y = df["Diabetes"].values

# Split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[5]),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # probability output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Predict for a new patient
new_patient = np.array([[45, 28, 120, 150, 1]])  # Example patient
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
print(f"Predicted diabetes chance: {prediction[0][0]*100:.2f}%")

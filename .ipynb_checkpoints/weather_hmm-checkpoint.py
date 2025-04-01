import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load & Preprocess Data
data = pd.read_csv("weather_data.csv")
weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
observations = data['Weather'].map(weather_mapping).values.reshape(-1, 1)

# Step 2: Train HMM
model = hmm.MultinomialHMM(n_components=3, n_iter=100)
model.fit(observations)

# Step 3: Predict Next Day's Weather
def predict_weather(today_weather):
    today_encoded = weather_mapping[today_weather]
    next_state = model.predict(np.array([[today_encoded]]))
    next_weather = list(weather_mapping.keys())[next_state[0]]
    return next_weather

# Example Prediction
today = "Sunny"
print(f"Today: {today} → Tomorrow: {predict_weather(today)}")

# Step 4: Evaluate Model
test_data = [...]  # Load test data
predictions = [predict_weather(w) for w in test_data[:-1]]
actual = test_data[1:]

accuracy = accuracy_score(actual, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(actual, predictions, labels=list(weather_mapping.keys()))
print("Confusion Matrix:")
print(cm)
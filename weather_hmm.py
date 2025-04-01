import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Step 1: Load and preprocess data
data = pd.DataFrame({
    'Day': range(1, 21),
    'Weather': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Cloudy', 
                'Sunny', 'Rainy', 'Rainy', 'Cloudy', 'Sunny',
                'Sunny', 'Cloudy', 'Cloudy', 'Rainy', 'Rainy',
                'Cloudy', 'Sunny', 'Sunny', 'Cloudy', 'Rainy']
})

# Encode weather states
encoder = LabelEncoder()
observations = encoder.fit_transform(data['Weather']).reshape(-1, 1)

# Step 2: Train HMM - Using CategoricalHMM (equivalent to old MultinomialHMM)
model = hmm.CategoricalHMM(n_components=3, random_state=42, n_iter=100)
model.fit(observations)

# Step 3: Predict next day's weather
def predict_weather(today_weather):
    today_encoded = encoder.transform([today_weather])
    next_state = model.predict(today_encoded.reshape(-1, 1))
    next_weather = encoder.inverse_transform(next_state)
    return next_weather[0]

# Example prediction
today = "Sunny"
prediction = predict_weather(today)
print(f"Today: {today} → Tomorrow: {prediction}")

# Step 4: Evaluate model
# Create test set (using same data for demonstration)
test_data = data['Weather'].values
predictions = []
for i in range(len(test_data)-1):
    predictions.append(predict_weather(test_data[i]))

actual = test_data[1:]

# Calculate accuracy
accuracy = accuracy_score(actual, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(actual, predictions, labels=encoder.classes_)
print("Confusion Matrix:")
print(cm)

# Print transition matrix
print("\nTransition Matrix between Hidden States:")
print(model.transmat_)

# Print emission probabilities
print("\nEmission Probabilities (Hidden State → Weather):")
for i, prob in enumerate(model.emissionprob_):
    print(f"State {i}: {dict(zip(encoder.classes_, prob))}")
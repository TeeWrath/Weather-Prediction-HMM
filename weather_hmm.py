import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load and preprocess the data from CSV file
def load_and_preprocess_data(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    
    # Read CSV file, skipping metadata rows until the header is found
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header row (assuming it starts with 'Year')
    header_row = 0
    for i, line in enumerate(lines):
        if line.startswith('Year'):
            header_row = i
            break
    
    # Read the CSV starting from the header row
    data = pd.read_csv(file_path, skiprows=header_row)
    
    # Convert numeric columns
    numeric_columns = ['Year', 'Month', 'Day', 'Hour', 'air_temp', 'windspeed', 'winddir', 'pressure', 'humidity', 'fog']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Filter out missing data
    data = data[data['fog'] != 128]  # Remove no data entries
    data = data[data['air_temp'] != -999]
    data = data[data['humidity'] != -999]
    data = data[data['pressure'] != -999]
    
    # Create datetime column
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour']], 
                                  errors='coerce', 
                                  format='%Y %m %d %H')
    
    return data

# Prepare observation sequence
def prepare_observation_sequence(data):
    # Use temperature, windspeed, pressure, and humidity as observations
    observations = data[['air_temp', 'windspeed', 'pressure', 'humidity']].values
    states = data['fog'].values.astype(int)
    
    # Standardize the observations
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)
    
    return observations_scaled, states, scaler

# Train HMM model
def train_hmm_model(observations, n_components=2):
    # n_components = 2 (fog/no fog)
    model = hmm.GaussianHMM(n_components=n_components, 
                           covariance_type="diag", 
                           n_iter=100,
                           random_state=42)
    
    # Fit the model
    model.fit(observations)
    return model

# Predict fog for new observations
def predict_fog(model, observations, scaler):
    # Scale the new observations
    observations_scaled = scaler.transform(observations)
    
    # Predict hidden states
    predicted_states = model.predict(observations_scaled)
    return predicted_states

# Evaluate model performance
def evaluate_model(actual_states, predicted_states):
    accuracy = np.mean(actual_states == predicted_states)
    return accuracy

# Map states to descriptive labels
def map_state(state, humidity):
    if state == 1:
        return "Foggy"
    elif state == 0 and humidity > 70:
        return "Humid"
    else:
        return "Clear"

# Main execution
def main():
    # Define the path to the CSV file (same level as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "weather_data.csv")
    
    # Load and preprocess data
    try:
        data = load_and_preprocess_data(csv_file_path)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare observation sequence
    observations, states, scaler = prepare_observation_sequence(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        observations, states, test_size=0.2, random_state=42
    )
    
    # Train HMM model
    model = train_hmm_model(X_train)
    
    # Make predictions on test set
    predicted_states = predict_fog(model, X_test, scaler)
    
    # Evaluate model
    accuracy = evaluate_model(y_test, predicted_states)
    print(f"Model Accuracy: {accuracy:.2%}")
    
    # Print model parameters
    print("\nModel Parameters:")
    print(f"Transition Matrix:\n{model.transmat_}")
    print(f"Means:\n{model.means_}")
    print(f"Variances:\n{model.covars_}")
    
    # Prepare test data with dates for tabular output
    test_data = data.iloc[X_test.shape[0]:X_test.shape[0] + X_test.shape[0]].copy()  # Align with test set
    test_data['Predicted_State'] = predicted_states
    test_data['Actual_State'] = y_test
    
    # Map states to descriptive labels
    test_data['Actual_State_Label'] = test_data.apply(
        lambda row: map_state(row['Actual_State'], row['humidity']), axis=1
    )
    test_data['Predicted_State_Label'] = test_data.apply(
        lambda row: map_state(row['Predicted_State'], row['humidity']), axis=1
    )
    
    # Select sample period (e.g., January 25-31, 2020)
    start_date = '2020-01-25'
    end_date = '2020-01-31'
    sample_data = test_data[
        (test_data['Date'] >= start_date) & (test_data['Date'] <= end_date)
    ].copy()
    
    # Format observed features
    sample_data['Observed_Features'] = sample_data.apply(
        lambda row: f"({row['air_temp']:.1f}, {row['windspeed']:.1f}, {row['pressure']:.1f}, {row['humidity']:.1f})", axis=1
    )
    
    # Create results table
    results = sample_data[['Date', 'Observed_Features', 'Actual_State_Label', 'Predicted_State_Label']]
    print("\nobservations. Sample results for January 25-31, 2020, are as follows:")
    print(results.to_string(index=False))
    
    # Example: Predict fog for new data
    new_observations = np.array([
        [15.6, 0.0, 925.0, 67.0],  # Matches 2020-01-25
        [14.8, 0.0, 925.0, 65.0],  # Matches 2020-01-26
        [15.9, 0.0, 925.0, 65.0],  # Matches 2020-01-27
        [16.2, 0.0, 925.0, 67.0],  # Matches 2020-01-28
        [15.8, 0.0, 925.0, 67.0],  # Matches 2020-01-29
        [16.1, 0.0, 925.0, 67.0],  # Matches 2020-01-30
        [14.2, 0.0, 925.0, 65.0]   # Matches 2020-01-31
    ])
    new_predictions = predict_fog(model, new_observations, scaler)
    print("\nNew Observations Predictions:")
    for i, pred in enumerate(new_predictions):
        date = datetime(2020, 1, 25 + i).strftime('%Y-%m-%d')
        state = map_state(pred, new_observations[i][3])
        print(f"Date: {date}, Observed Features: ({new_observations[i][0]:.1f}, {new_observations[i][1]:.1f}, {new_observations[i][2]:.1f}, {new_observations[i][3]:.1f}), Predicted State: {state}")

if __name__ == "__main__":
    main()
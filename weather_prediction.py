import pandas as pd
import numpy as np
from datetime import datetime, timedelta

STATES = ['Sunny', 'Cold', 'Rainy']
N_STATES = len(STATES)

class HMMWeatherPredictor:
    def __init__(self):
        self.transition_matrix = np.zeros((N_STATES, N_STATES))
        self.emission_probs = {
            'Sunny': {
                'temp_high': 0.8, 'temp_low': 0.1,
                'humid_high': 0.3, 'humid_low': 0.6,
                'wind_high': 0.4, 'wind_low': 0.6,
                'pressure_high': 0.7, 'pressure_low': 0.3,
                'rain_yes': 0.1, 'rain_no': 0.9,
                'fog_yes': 0.2, 'fog_no': 0.8,
                'aod_high': 0.3, 'aod_low': 0.7
            },
            'Cold': {
                'temp_high': 0.1, 'temp_low': 0.8,
                'humid_high': 0.5, 'humid_low': 0.5,
                'wind_high': 0.6, 'wind_low': 0.4,
                'pressure_high': 0.6, 'pressure_low': 0.4,
                'rain_yes': 0.3, 'rain_no': 0.7,
                'fog_yes': 0.4, 'fog_no': 0.6,
                'aod_high': 0.2, 'aod_low': 0.8
            },
            'Rainy': {
                'temp_high': 0.4, 'temp_low': 0.4,
                'humid_high': 0.8, 'humid_low': 0.2,
                'wind_high': 0.7, 'wind_low': 0.3,
                'pressure_high': 0.3, 'pressure_low': 0.7,
                'rain_yes': 0.9, 'rain_no': 0.1,
                'fog_yes': 0.6, 'fog_no': 0.4,
                'aod_high': 0.5, 'aod_low': 0.5
            }
        }
        self.initial_probs = np.array([0.4, 0.3, 0.3])

    def discretize_observations(self, row):
        obs = []
        obs.append('temp_high' if row['air_temp'] > 20 else 'temp_low')
        obs.append('humid_high' if row['humidity'] > 70 else 'humid_low')
        obs.append('wind_high' if row['windspeed'] > 0.5 else 'wind_low')
        obs.append('pressure_high' if row['pressure'] > 990 else 'pressure_low')
        obs.append('rain_yes' if row['rainfall'] > 0 else 'rain_no')
        obs.append('fog_yes' if row['fog'] > 0 else 'fog_no')
        obs.append('aod_high' if row['AOD'] > 0.5 and row['AOD'] != -999 else 'aod_low')
        return obs

    def classify_weather(self, row):
        if row['rainfall'] > 0 or (row['humidity'] > 80 and row['pressure'] < 990):
            return 'Rainy'
        elif row['air_temp'] < 15 or (row['air_temp'] < 20 and row['windspeed'] > 0.5):
            return 'Cold'
        else:
            return 'Sunny'

    def train(self, data):
        daily_data = data.groupby(['Year', 'Month', 'Day']).agg({
            'air_temp': 'mean', 'humidity': 'mean', 'windspeed': 'mean',
            'pressure': 'mean', 'rainfall': 'sum', 'fog': 'max', 'AOD': 'mean'
        }).reset_index()
        
        prev_state = None
        for _, row in daily_data.iterrows():
            curr_state = self.classify_weather(row)
            if prev_state is not None:
                self.transition_matrix[STATES.index(prev_state), STATES.index(curr_state)] += 1
            prev_state = curr_state
        
        for i in range(N_STATES):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
            else:
                self.transition_matrix[i] = np.ones(N_STATES) / N_STATES

    def predict_next_days(self, last_observations, days=10):
        predictions = []
        current_obs = last_observations[-1]
        last_state = self.classify_weather(last_observations[0])  # Use actual last day's data
        
        for _ in range(days):
            # Calculate next state probabilities using transition matrix from last state
            next_probs = np.zeros(N_STATES)
            last_state_idx = STATES.index(last_state)
            
            for i in range(N_STATES):
                emission_prob = 1.0
                for obs in current_obs:
                    emission_prob *= self.emission_probs[STATES[i]].get(obs, 0.33)
                next_probs[i] = self.transition_matrix[last_state_idx, i] * emission_prob
            
            # Normalize probabilities
            next_probs /= np.sum(next_probs)
            
            # Choose next state randomly based on probabilities
            next_state = np.random.choice(STATES, p=next_probs)
            predictions.append(next_state)
            
            # Generate randomized observations with wider ranges
            if next_state == 'Sunny':
                temp = np.random.uniform(18, 32)
                humid = np.random.uniform(30, 75)
                wind = np.random.uniform(0.1, 0.7)
                press = np.random.uniform(985, 1015)
                rain = np.random.choice([0, 1], p=[0.9, 0.1])
                fog = np.random.choice([0, 1], p=[0.9, 0.1])
                aod = np.random.uniform(0.2, 0.6)
            elif next_state == 'Cold':
                temp = np.random.uniform(0, 18)
                humid = np.random.uniform(40, 85)
                wind = np.random.uniform(0.2, 1.0)
                press = np.random.uniform(990, 1020)
                rain = np.random.choice([0, 1], p=[0.7, 0.3])
                fog = np.random.choice([0, 1], p=[0.7, 0.3])
                aod = np.random.uniform(0.1, 0.5)
            else:  # Rainy
                temp = np.random.uniform(10, 25)
                humid = np.random.uniform(60, 95)
                wind = np.random.uniform(0.3, 1.2)
                press = np.random.uniform(970, 1000)
                rain = np.random.choice([1, 0], p=[0.8, 0.2])
                fog = np.random.choice([0, 1], p=[0.5, 0.5])
                aod = np.random.uniform(0.3, 0.8)
                
            current_obs = self.discretize_observations({
                'air_temp': temp, 'humidity': humid, 'windspeed': wind,
                'pressure': press, 'rainfall': rain, 'fog': fog, 'AOD': aod
            })
            last_state = next_state
        
        return predictions

def main():
    try:
        data = pd.read_csv('weather_data.csv')
    except FileNotFoundError:
        print("Error: 'weather_data.csv' not found.")
        return

    data['date'] = pd.to_datetime(data[['Year', 'Month', 'Day']].astype(int))
    last_date = data['date'].max()
    print(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")

    hmm = HMMWeatherPredictor()
    hmm.train(data)
    
    last_day_data = data[data['date'] == last_date].agg({
        'air_temp': 'mean', 'humidity': 'mean', 'windspeed': 'mean',
        'pressure': 'mean', 'rainfall': 'sum', 'fog': 'max', 'AOD': 'mean'
    })
    last_observations = [last_day_data]  # Pass as dict for classify_weather
    last_state = hmm.classify_weather(last_day_data)
    print(f"Last day's state: {last_state}")
    
    start_date = last_date + timedelta(days=1)
    predictions = hmm.predict_next_days(last_observations, days=10)
    
    print("\nWeather Predictions for 10 days after last date:")
    for i, pred in enumerate(predictions):
        date = start_date + timedelta(days=i)
        print(f"{date.strftime('%Y-%m-%d')}: {pred}")
    
    print("\nLearned Transition Matrix:")
    print("   Sunny  Cold  Rainy")
    for i, state in enumerate(STATES):
        print(f"{state}: {hmm.transition_matrix[i]}")

if __name__ == "__main__":
    main()
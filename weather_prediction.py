import numpy as np
import pandas as pd

# Helper function for log-sum-exp trick
def logsumexp(arr, axis=None):
    arr_max = np.max(arr, axis=axis, keepdims=True)
    return arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis))

# Step 1: Data Preparation
def discretize_value(value, bins):
    return np.digitize(value, bins, right=True) - 1

def prepare_data(df, param, bins):
    values = df[param].values
    discretized = discretize_value(values, bins)
    return discretized, values

def get_bins(df, param, n_bins=3):
    values = df[param].values
    finite_values = values[np.isfinite(values)]
    return np.linspace(np.min(finite_values), np.max(finite_values), n_bins + 1)[1:-1]

# Step 2: HMM Implementation
class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self.eps = 1e-10
        
        # Initialize with uniform distributions
        self.transition_matrix = np.ones((n_states, n_states)) / n_states
        self.emission_matrix = np.ones((n_states, n_observations)) / n_observations
        self.initial_probs = np.ones(n_states) / n_states

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        alpha[0] = np.log(self.initial_probs + self.eps) + \
                  np.log(self.emission_matrix[:, observations[0]] + self.eps)
        
        for t in range(1, T):
            for j in range(self.n_states):
                log_sum = alpha[t-1] + np.log(self.transition_matrix[:, j] + self.eps)
                alpha[t, j] = logsumexp(log_sum) + \
                            np.log(self.emission_matrix[j, observations[t]] + self.eps)
        
        return alpha

    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        beta[-1] = 0.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_sum = np.log(self.transition_matrix[i, :] + self.eps) + \
                         np.log(self.emission_matrix[:, observations[t+1]] + self.eps) + \
                         beta[t+1]
                beta[t, i] = logsumexp(log_sum)
        
        return beta

    def baum_welch(self, observations, max_iter=100, tol=1e-2):
        T = len(observations)
        old_log_prob = -np.inf
        
        for iteration in range(max_iter):
            alpha = self.forward(observations)
            beta = self.backward(observations)
            
            log_prob = logsumexp(alpha[-1])
            if iteration > 0 and abs(log_prob - old_log_prob) < tol:
                print(f"Converged at iteration {iteration} for {self.param}")
                break
            old_log_prob = log_prob
            
            xi = np.zeros((T-1, self.n_states, self.n_states))
            gamma = np.zeros((T, self.n_states))
            
            for t in range(T-1):
                log_denominator = logsumexp(alpha[t] + beta[t])
                for i in range(self.n_states):
                    gamma[t, i] = np.exp(alpha[t, i] + beta[t, i] - log_denominator)
                    for j in range(self.n_states):
                        xi[t, i, j] = np.exp(alpha[t, i] + 
                                           np.log(self.transition_matrix[i, j] + self.eps) +
                                           np.log(self.emission_matrix[j, observations[t+1]] + self.eps) +
                                           beta[t+1, j] - log_denominator)
            
            gamma[-1] = np.exp(alpha[-1] - logsumexp(alpha[-1]))
            
            # M-step
            self.initial_probs = gamma[0] / (gamma[0].sum() + self.eps)
            
            for i in range(self.n_states):
                gamma_sum = gamma[:-1, i].sum() + self.eps
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = xi[:, i, j].sum() / gamma_sum
            
            for j in range(self.n_states):
                gamma_total = gamma[:, j].sum() + self.eps
                for k in range(self.n_observations):
                    self.emission_matrix[j, k] = gamma[observations == k, j].sum() / gamma_total
            
            self.transition_matrix /= (self.transition_matrix.sum(axis=1, keepdims=True) + self.eps)
            self.emission_matrix /= (self.emission_matrix.sum(axis=1, keepdims=True) + self.eps)

    def viterbi(self, observations, n_days=10):
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = np.log(self.initial_probs + self.eps) + \
                  np.log(self.emission_matrix[:, observations[0]] + self.eps)
        
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.transition_matrix[:, j] + self.eps)
                delta[t, j] = np.max(temp) + \
                            np.log(self.emission_matrix[j, observations[t]] + self.eps)
                psi[t, j] = np.argmax(temp)
        
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        # Predict next n_days
        pred_states = [states[-1]]
        for _ in range(n_days - 1):
            last_state = pred_states[-1]
            next_state = np.argmax(self.transition_matrix[last_state])
            pred_states.append(next_state)
        
        return pred_states[-n_days:]

    def fit(self, observations, param):
        self.param = param
        self.baum_welch(observations)

# Step 3: Prediction and Mapping
def map_state_to_value(state, bins, values):
    bin_idx = min(state, len(bins) - 1)
    if bin_idx == 0:
        return np.median(values[values <= bins[0]])
    elif bin_idx == len(bins):
        return np.median(values[values > bins[-1]])
    else:
        return np.median(values[(values > bins[bin_idx-1]) & (values <= bins[bin_idx])])

# Main execution
def main():
    # Load data
    df = pd.read_csv('weather_data.csv')
    
    # Convert float columns to integers for date creation
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    
    # Create date column with explicit format
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
    
    # Filter data up to 19/02/2020
    train_df = df[df['date'] <= '2020-02-19']
    
    # Parameters to model
    params = ['air_temp', 'windspeed', 'winddir', 'pressure', 'humidity', 'rainfall', 'fog', 'AOD']
    n_states = 3
    n_days = 10
    
    # Store HMMs and predictions
    models = {}
    predictions = {param: [] for param in params}
    
    for param in params:
        # Define bins dynamically
        bins = get_bins(train_df, param, n_bins=n_states)
        discretized_obs, raw_values = prepare_data(train_df, param, bins)
        n_observations = len(np.unique(discretized_obs))
        
        # Train HMM
        print(f"Training HMM for {param}...")
        model = HMM(n_states, n_observations)
        model.fit(discretized_obs, param)
        models[param] = (model, bins, raw_values)
        
        # Predict states for next 10 days
        pred_states = model.viterbi(discretized_obs, n_days=n_days)
        
        # Map states to values
        pred_values = [map_state_to_value(state, bins, raw_values) for state in pred_states]
        predictions[param] = pred_values
        
        # Print model parameters
        print(f"\n{param} Transition Matrix:")
        print(np.round(model.transition_matrix, 3))
        print(f"{param} Emission Matrix:")
        print(np.round(model.emission_matrix, 3))
        print(f"{param} Initial Probabilities:")
        print(np.round(model.initial_probs, 3))

    # Generate output for 20/02/2020 - 29/02/2020
    dates = pd.date_range(start='2020-02-20', end='2020-02-29', freq='D')
    print("\nPredictions for 20/02/2020 - 29/02/2020:")
    print("Date       | " + " | ".join(f"{param:<10}" for param in params))
    print("-" * 100)
    
    for i, date in enumerate(dates):
        row = [f"{date.strftime('%Y-%m-%d')}"]
        for param in params:
            value = predictions[param][i]
            row.append(f"{value:<10.2f}")
        print(" | ".join(row))

if __name__ == "__main__":
    # Save the CSV data to a file (replace with your full data)
#     with open('weather_data.csv', 'w') as f:
#         f.write("""Year,Month,Day,Hour,air_temp,windspeed,winddir,pressure,humidity,rainfall,fog,AOD
# 2009.00,11.0000,1.00000,0.000000,15.9600,0.0500000,358.990,987.020,75.9000,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,1.00000,16.1500,0.0500000,358.990,987.610,76.9800,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,2.00000,17.6700,0.0500000,358.990,988.590,76.9800,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,3.00000,21.2900,0.0500000,73.5600,989.270,64.9600,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,4.00000,24.1700,0.0500000,238.760,989.660,58.9900,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,5.00000,26.0800,0.240000,194.770,989.570,52.9300,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,6.00000,27.2000,0.440000,87.7300,988.980,44.9200,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,7.00000,28.2700,0.540000,93.6000,988.200,41.9800,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,8.00000,29.4000,0.340000,319.890,986.930,37.9800,502.000,128.000,-999.000
# 2009.00,11.0000,1.00000,9.00000,30.1800,0.0500000,358.990,986.440,38.9500,502.000,128.000,-999.000
# """)  # Replace with your full dataset
    
    main()
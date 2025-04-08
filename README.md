# Weather Prediction with Hidden Markov Model (HMM)

## Overview
This project uses a Hidden Markov Model (HMM) to predict fog occurrence based on weather data (temperature, wind speed, pressure, and humidity). The model is trained on historical weather data from a CSV file and provides predictions with descriptive labels ("Foggy," "Humid," "Clear").

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `hmmlearn`, `sklearn`, `matplotlib`

Install dependencies:
```bash
pip install pandas numpy hmmlearn scikit-learn matplotlib
```

## Usage
1. Place your `weather_data.csv` file in the same directory as the script.
2. Run the script:
```bash
python weather_hmm.py
```

### CSV Format
The CSV should contain:
- Columns: `Year`, `Month`, `Day`, `Hour`, `air_temp`, `windspeed`, `winddir`, `pressure`, `humidity`, `fog`
- `fog`: 0 (no fog), 1 (fog), 128 (missing)

## Functionality
- **Data Preprocessing**: Loads and cleans weather data, removing invalid entries.
- **HMM Training**: Trains a Gaussian HMM with 2 states (fog/no fog) on scaled observations.
- **Prediction**: Predicts fog states for test data and new observations.
- **Evaluation**: Calculates model accuracy and displays sample results for January 25-31, 2020.

## Output
- Model accuracy
- HMM parameters (transition matrix, means, variances)
- Sample predictions table
- Predictions for new example observations

## Notes
- Ensure the CSV file path is correct in the script.
- Adjust `n_components` in `train_hmm_model()` if more states are desired.
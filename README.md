# 🚲 Bike Sharing Demand Prediction

Predicting hourly bike rentals in a city bike-sharing system using time-based feature engineering and a Random Forest model. Accurate demand forecasts help operators keep bikes available where and when riders need them, improve customer satisfaction, and allocate resources across stations more efficiently.

## Highlights

- Hourly demand forecasting on 17,000+ records (10,886 train / 6,493 test)
- Time-based features: hour, day, month, year, day of week, and a rush-hour flag
- Log-transformed target to handle skewed rental counts
- Random Forest regressor reaching **R² ≈ 0.95** on the validation set
- Ready-to-use `submission.csv` with predictions for the test set

## Dataset

Hourly rental records with weather and calendar information, in the format of the Kaggle "Bike Sharing Demand" competition.

| Column | Description |
|--------|-------------|
| `datetime` | Hourly timestamp |
| `season` | 1 = spring, 2 = summer, 3 = fall, 4 = winter |
| `holiday`, `workingday` | Holiday and working-day flags |
| `weather` | 1 = clear, 2 = cloudy, 3 = rainy, 4 = heavy rain |
| `temp`, `atemp` | Temperature and "feels like" temperature (Celsius) |
| `humidity`, `windspeed` | Weather conditions |
| `count` | **Target:** total bikes rented in that hour |

## Approach

1. **Data preparation:** combined train and test sets so both get identical feature engineering.
2. **Feature engineering:**
   - Extracted `hour`, `day`, `month`, `year`, and `dayofweek` from the timestamp.
   - Added `is_peak_hour` for working-day rush hours (8:00, 17:00, 18:00).
   - One-hot encoded `season` and `weather`.
   - Removed `casual` and `registered`, since they add up to the target.
3. **Target transformation:** applied `log1p` to `count` and converted predictions back with `expm1`.
4. **Modeling:** `RandomForestRegressor` with 500 trees and a maximum depth of 15.
5. **Evaluation:** 80/20 train/validation split, scored on the original rental-count scale.
6. **Prediction:** generated hourly forecasts for the test set and exported them to `submission.csv`.

## Results

| Metric | Value |
|--------|-------|
| R² | 0.9476 |
| RMSE | 41.59 |
| MAE | 25.58 |

On average, predictions are off by about 26 bikes per hour, with the model explaining roughly 95% of the variance in hourly demand.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn matplotlib
```

Place `train.csv` and `test.csv` (from the Kaggle "Bike Sharing Demand" competition) in the project root, then run:

```bash
jupyter notebook BikeSharing.ipynb
```

The notebook generates `submission.csv` with the columns `datetime` and `count`.

## Project Structure

```
.
├── BikeSharing.ipynb   # Feature engineering, modeling, and prediction
├── train.csv           # Training data (not included)
├── test.csv            # Test data (not included)
├── submission.csv      # Generated predictions
└── README.md
```

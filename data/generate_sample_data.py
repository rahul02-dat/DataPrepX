import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_classification_dataset():
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(15000, 8000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['income'] = df['income'].clip(lower=20000)
    df['loan_amount'] = df['loan_amount'].clip(lower=1000, upper=40000)
    
    approval_score = (
        0.3 * (df['credit_score'] - 300) / 550 +
        0.2 * (df['income'] - 20000) / 80000 +
        0.2 * (df['employment_length']) / 40 +
        0.3 * (1 - (df['loan_amount'] - 1000) / 39000)
    )
    
    noise = np.random.normal(0, 0.1, n_samples)
    approval_score += noise
    
    df['loan_approved'] = (approval_score > 0.5).astype(int)
    
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'employment_length'] = np.nan
    
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    return df

def generate_regression_dataset():
    n_samples = 800
    
    data = {
        'square_feet': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'lot_size': np.random.normal(8000, 3000, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Waterfront'], n_samples),
        'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
    }
    
    df = pd.DataFrame(data)
    df['lot_size'] = df['lot_size'].clip(lower=2000, upper=20000)
    
    base_price = 100000
    price = (
        base_price +
        df['square_feet'] * 150 +
        df['bedrooms'] * 20000 +
        df['bathrooms'] * 15000 +
        (2024 - df['year_built']) * -500 +
        df['lot_size'] * 5 +
        df['garage_spaces'] * 10000
    )
    
    neighborhood_multiplier = df['neighborhood'].map({
        'Waterfront': 1.5,
        'Downtown': 1.3,
        'Suburbs': 1.0,
        'Rural': 0.8
    })
    
    condition_multiplier = df['condition'].map({
        'Excellent': 1.2,
        'Good': 1.0,
        'Fair': 0.9,
        'Poor': 0.75
    })
    
    price = price * neighborhood_multiplier * condition_multiplier
    
    noise = np.random.normal(0, 20000, n_samples)
    price += noise
    
    df['price'] = price.clip(lower=50000)
    
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.04), replace=False)
    df.loc[missing_indices, 'year_built'] = np.nan
    
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
    df.loc[missing_indices, 'lot_size'] = np.nan
    
    return df

def generate_timeseries_dataset():
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    trend = np.linspace(100, 200, n_samples)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)
    
    data = {
        'date': dates,
        'sales': trend + seasonality + noise,
        'marketing_spend': np.random.normal(5000, 1000, n_samples),
        'temperature': 50 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 5, n_samples),
        'day_of_week': dates.dayofweek,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'is_holiday': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    df['sales'] = df['sales'].clip(lower=50)
    
    return df

if __name__ == '__main__':
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("Generating sample datasets...")
    
    loan_df = generate_classification_dataset()
    loan_df.to_csv(data_dir / 'loan_approval.csv', index=False)
    print(f"✓ Generated loan_approval.csv ({len(loan_df)} rows)")
    
    housing_df = generate_regression_dataset()
    housing_df.to_csv(data_dir / 'housing_prices.csv', index=False)
    print(f"✓ Generated housing_prices.csv ({len(housing_df)} rows)")
    
    housing_df.to_excel(data_dir / 'housing_prices.xlsx', index=False)
    print(f"✓ Generated housing_prices.xlsx ({len(housing_df)} rows)")
    
    timeseries_df = generate_timeseries_dataset()
    timeseries_df.to_csv(data_dir / 'sales_timeseries.csv', index=False)
    print(f"✓ Generated sales_timeseries.csv ({len(timeseries_df)} rows)")
    
    print("\nSample data generated successfully!")
    print("\nDataset descriptions:")
    print("1. loan_approval.csv - Classification task (predict loan approval)")
    print("2. housing_prices.csv - Regression task (predict house prices)")
    print("3. housing_prices.xlsx - Same as #2 in Excel format")
    print("4. sales_timeseries.csv - Time series analysis")
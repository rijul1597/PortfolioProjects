import os
import pandas as pd
import boto3
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import argparse
import s3fs

# Define your S3 bucket and path
bucket_name = 'stockdatacsvfiles'
folder_path = 'output_folder/cleaned_data/'
output_path = 'data/output/'
# Create a boto3 S3 client
s3_client = boto3.client('s3')

# Get the list of files from the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

# Sort files by last modified date in descending order
files_sorted = sorted(response.get('Contents', []), key=lambda x: x['LastModified'], reverse=True)

# Select the most recent files, assuming they are uploaded daily or however often you expect
# Adjust the number based on your specific use case
latest_files = [obj['Key'] for obj in files_sorted][:6]  # Replace 6 with the number of files you expect each day

# Define a list to hold all the dataframes
dataframes = []

# Loop over the list of latest files and read each into a DataFrame
for file_key in latest_files:
    # Get the object from S3
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(obj['Body'])  # 'Body' is a byte-stream
    dataframes.append(df)

# Concatenate all DataFrames into one (if desired)
full_df = pd.concat(dataframes, ignore_index=True)

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

 
# Feature Engineering

full_df['daily_return'] = full_df.groupby('partition_0')['close'].pct_change()

full_df['high_low_diff'] = full_df['high'] - full_df['low']

full_df['avg_price'] = (full_df['high'] + full_df['low']) / 2

full_df['rolling_mean_7'] = full_df.groupby('partition_0')['close'].transform(lambda x: x.rolling(window=7).mean())

full_df['rolling_median_30'] = full_df.groupby('partition_0')['close'].transform(lambda x: x.rolling(window=30).median())
 
def add_lagged_features(full_df, group_by_column='partition_0', sort_by_column='timestamp'):
    """
    Add lagged features to the DataFrame after grouping by a specified column and sorting.
    Args:
    full_df (pd.DataFrame): The input DataFrame containing financial data.
    group_by_column (str): Column name to group by, typically a ticker or partition identifier.
    sort_by_column (str): Column name to sort by, typically a timestamp.
    Returns:
    pd.DataFrame: DataFrame with added lagged features.
    """
    # Sort DataFrame by the grouping column and then by timestamp
    full_df = full_df.sort_values(by=[group_by_column, sort_by_column])
 
    # List of features to create lags for
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                       'daily_return', 'high_low_diff', 'avg_price', 
                       'rolling_mean_7', 'rolling_median_30']

    # Generate lags for the specified number of periods
    for column in feature_columns:
        for lag in range(1, 4):  # Creating 1 to 3 lags
            full_df[f'{column}_lag{lag}'] = full_df.groupby(group_by_column)[column].shift(lag)
    return full_df
# Drop NA values
full_df['rolling_median_30'].fillna(value=full_df['rolling_median_30'].mean(), inplace=True)
full_df['rolling_mean_7'].fillna(value=full_df['rolling_mean_7'].mean(), inplace=True)
full_df['daily_return'].fillna(value=full_df['daily_return'].mean(), inplace=True)
nvda_df = full_df[full_df['partition_0'] == 'nvda']
aapl_df = full_df[full_df['partition_0'] == 'aapl']
amzn_df = full_df[full_df['partition_0'] == 'amzn']
meta_df = full_df[full_df['partition_0'] == 'meta']
nflx_df = full_df[full_df['partition_0'] == 'nflx']
googl_df = full_df[full_df['partition_0'] == 'googl']
def train_and_evaluate(full_df):
    # Define predictors and target
    X = full_df.drop(['timestamp', 'partition_0', 'close'], axis=1)
    y = full_df['close']

    # Temporal Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model (only Lasso)
    model = Lasso(alpha=0.1)

    # Train the model and make predictions
    model.fit(X_train, y_train)
    today_data = X_train[0].reshape(1, -1)
    predictions = model.predict(today_data)
    #rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #print(f'RMSE for {stock_name}: {rmse}')

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame({'Predicted': predictions})
    return predictions_df


# Define stocks dictionary with each DataFrame named
stocks = {
    'NVDA': nvda_df,
    'AAPL': aapl_df,
    'AMZN': amzn_df,
    'NFLX': nflx_df,
    'GOOGL': googl_df,
    'META': meta_df
}

# Now loop over your stocks dictionary
for stock_name, stock_df in stocks.items():
    predictions_df = train_and_evaluate(stock_df)
    prediction_value = predictions_df.iloc[0, 0]
    print(stock_name,prediction_value)

    fs = s3fs.S3FileSystem()

    # Define the complete S3 path for output
    job_output_path = f'{bucket_name}/{output_path}/{stock_name}_predictions.csv'

    # Use s3fs to write the DataFrame directly to S3
    with fs.open(job_output_path, 'w') as f:
        predictions_df.to_csv(f, index=False)

    print(f'Successfully saved predictions for {stock_name} to {job_output_path}')


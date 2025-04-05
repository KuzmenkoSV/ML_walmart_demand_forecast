import pandas as pd
import argparse
import os
import click
import boto3
import pandas as pd
from io import BytesIO


def feature_process_data():

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    response = s3.get_object(Bucket="ml-data", Key="features/df_sales_train_validation.csv")
    df_sales_train_validation = pd.read_csv(BytesIO(response["Body"].read()))
    df_sales_train_validation = df_sales_train_validation[df_sales_train_validation['item_id'] == 'FOODS_1_003']
    df_sales_train_validation['date'] = pd.to_datetime(df_sales_train_validation['date'])
  
    df_sales_train_validation['sales_last_week'] = df_sales_train_validation.groupby('id')['value'].shift(7)
    df_sales_train_validation['sales_last_month'] = df_sales_train_validation.groupby('id')['value'].shift(30)

    df_sales_train_validation['moving_avg_7'] = df_sales_train_validation.groupby('id')['value'].transform(lambda x: x.rolling(window = 7, min_periods = 1).mean())
    df_sales_train_validation['moving_avg_30'] = df_sales_train_validation.groupby('id')['value'].transform(lambda x: x.rolling(window = 30, min_periods = 1).mean())
    df_sales_train_validation.dropna(inplace=True)
    df_sales_train_validation.drop(columns= ['id', 'state_id'], inplace=True)
    df_sales_train_validation.drop(columns= ['d'], inplace=True)
    df_sales_train_validation.drop(columns= ['store_id'], inplace=True)

    # Create additional time-based features
    df_sales_train_validation['day_of_week'] = df_sales_train_validation['date'].dt.dayofweek
    df_sales_train_validation['month'] = df_sales_train_validation['date'].dt.month
    df_sales_train_validation['year'] = df_sales_train_validation['date'].dt.year

    train_start_date = '2011-01-29'
    train_end_date = '2016-04-18'
    val_start_date = '2016-04-18'
    val_end_date = '2016-04-24'

        # Разделение на train test
    train_df = df_sales_train_validation[
        (df_sales_train_validation['date'] >= train_start_date)
        & (df_sales_train_validation['date'] < train_end_date)
    ]
    val_df = df_sales_train_validation[
        (df_sales_train_validation['date'] >= val_start_date)
        & (df_sales_train_validation['date'] <= val_end_date)
    ]

    buffer = BytesIO()
    train_df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, "ml-data", "features/train_df.csv")
    buffer.close()

    buffer = BytesIO()
    val_df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, "ml-data", "features/val_df.csv")
    buffer.close()

if __name__ == "__main__":
    
    feature_process_data()

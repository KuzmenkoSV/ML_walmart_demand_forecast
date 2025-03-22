import pandas as pd
import argparse
import os

def process_data(prep_file, train_file, eval_file):
    df_sales_train_validation = pd.read_csv(prep_file)
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

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(eval_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Создаем парсер аргументов
    parser.add_argument("--input", type=str, required=True, help="Путь к подготовленному датасету данным")
    parser.add_argument("--output_train", type=str, required=True, help="Путь для train.csv")
    parser.add_argument("--output_test", type=str, required=True, help="Путь для test.csv")
    args = parser.parse_args()  # Считываем аргументы

    # Передаем аргументы в функцию
    process_data(args.input, args.output_train, args.output_test)

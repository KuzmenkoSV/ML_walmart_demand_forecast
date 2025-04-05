import pandas as pd
import argparse
import joblib
from catboost import CatBoostRegressor,  Pool
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
import click
import boto3
import pickle
import psycopg2
from io import BytesIO
from io import StringIO
'''
def predict_and_save(df):
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    obj = s3.get_object(Bucket="ml-data", Key="model.pkl")
    model = pickle.loads(obj["Body"].read())

    X = df.drop("target", axis=1)
    preds = model.predict(X)

    conn = psycopg2.connect(
        host="postgres",
        dbname="airflow",
        user="airflow",
        password="airflow"
    )

    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS ml_schema;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_schema.predictions (
                    id SERIAL PRIMARY KEY,
                    prediction INTEGER
                );
            """)
            for pred in preds:
                cur.execute("INSERT INTO ml_schema.predictions (prediction) VALUES (%s);", (int(pred),))


@click.command()
@click.option('--model', type=str, required=True, help="Путь для обученной модели")
@click.option('--test', type=str, required=True, help="Путь для теста")
'''
def test_data():

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    obj = s3.get_object(Bucket="ml-data", Key="models/model.pkl")
    model_prod = pickle.loads(obj["Body"].read())

    response = s3.get_object(Bucket="ml-data", Key="features/val_df.csv")
    val_df = pd.read_csv(BytesIO(response["Body"].read()))
    # Define features and target variable
    features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year', 'sales_last_week', 'sales_last_month', 'moving_avg_7', 'moving_avg_30']
    target = 'value'
    X_test = val_df[features]
    y_test = val_df[target]

    # Convert categorical features to categorical data type
    categorical_features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year']
    for feature in categorical_features:
        X_test[feature] = X_test[feature].astype('category')

    test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    # Train the model
    y_pred = model_prod.predict(X_test)

    val_df['predicted'] = y_pred
    val_df["store_id"] = 'CA_3'
    val_df['id'] = val_df["item_id"] + "_CA_3"

    val_df_for_roll_unique = val_df.drop_duplicates(subset=['id'])

    ######################### Метрики

    def smape_score(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    def wape_score(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    def evaluate(y_true, y_pred):
        metrics = dict()
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['smape'] = smape_score(y_true, y_pred)
        metrics['max_error'] = max_error(y_true, y_pred)
        metrics['wape'] = wape_score(y_true, y_pred)

        return metrics

    def aggregate_metrics(metrics_df):
        summary = {}
        for metric in metrics_df.columns:
            summary[metric] = {
                'mean': metrics_df[metric].mean(),
                '25%': metrics_df[metric].quantile(0.25),
                'median': metrics_df[metric].median(),
                '75%': metrics_df[metric].quantile(0.75)
            }
        return pd.DataFrame(summary)
    metrics_by_id = val_df.groupby('id').apply(lambda group: pd.Series(evaluate(group['value'], group['predicted'])))
    metrics_by_store = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['store_id']).apply(aggregate_metrics)
    metrics_by_cat = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['cat_id']).apply(aggregate_metrics)
    print(metrics_by_cat)

    print(val_df.info())

    conn = psycopg2.connect(
        host="postgres",
        dbname="airflow",
        user="airflow",
        password="airflow"
    )

    # Create a cursor object
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS ml_schema;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_schema.predictions (
            item_id TEXT,
            cat_id TEXT,
            dept_id TEXT,
            value INTEGER,
            date DATE,
            sales_last_week FLOAT,
            sales_last_month FLOAT,
            moving_avg_7 FLOAT,
            moving_avg_30 FLOAT,
            day_of_week INTEGER,
            month INTEGER,
            year INTEGER,
            predicted FLOAT,
            store_id TEXT,
            id TEXT
        );
    """)

    cur.execute("SET search_path TO ml_schema;")

    csv_buffer = StringIO()
    val_df.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    cur.copy_from(csv_buffer, "predictions", sep=",")
    conn.commit()
    conn.close()

    '''with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS ml_schema;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_schema.predictions (
                    id SERIAL PRIMARY KEY,
                    prediction INTEGER
                );
            """)
            for pred in preds:
                cur.execute("INSERT INTO ml_schema.predictions (prediction) VALUES (%s);", (int(pred),))

    #metrics_by_store.to_csv('C:\Walmart_project_Kuzmenko\\reports\metrics_by_store.csv')
    #metrics_by_cat.to_csv('C:\Walmart_project_Kuzmenko\\reports\metrics_by_cat.csv')

'''
if __name__ == "__main__":
   # parser = argparse.ArgumentParser()  # Создаем парсер аргументов
   # parser.add_argument("--test", type=str, required=True, help="Путь для теста")
   # parser.add_argument("--model", type=str, required=True, help="Путь для обученной модели")
   # args = parser.parse_args()  # Считываем аргументы

    # Передаем аргументы в функцию
    test_data()
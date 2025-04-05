from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Добавляем src в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from task_load_to_s3 import upload_csvs_to_s3
from task_process import process_data
from task_features import feature_process_data
from task_train import train_data
from task_test import test_data


with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    upload_data = PythonOperator(
        task_id="upload_raw_data", python_callable=upload_csvs_to_s3
    )

    preprocess = PythonOperator(
        task_id="preprocess", python_callable=process_data
    )

    feature_create = PythonOperator(
        task_id="feature_create", python_callable=feature_process_data
    )

    train = PythonOperator(
        task_id="train_model", python_callable=train_data
    )

    predict_and_load = PythonOperator(
        task_id="predict_and_load", python_callable=test_data
    )

    upload_data >> preprocess >> feature_create >> train >> predict_and_load

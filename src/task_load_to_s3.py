import os
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",  # внутри контейнера
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

BUCKET_NAME = "ml-data"
LOCAL_FOLDER = "/opt/airflow/data"

def upload_csvs_to_s3():
    for file_name in os.listdir(LOCAL_FOLDER):
        if file_name.endswith(".csv"):
            local_path = os.path.join(LOCAL_FOLDER, file_name)
            s3_key = f"raw/{file_name}"
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            print(f"Uploaded {file_name} to s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    upload_csvs_to_s3()

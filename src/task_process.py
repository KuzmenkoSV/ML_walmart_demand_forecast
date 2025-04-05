import boto3
import pandas as pd
from io import BytesIO



def process_data():
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    response = s3.get_object(Bucket="ml-data", Key="raw/sell_prices.csv")
    sell_prices = pd.read_csv(BytesIO(response["Body"].read()))

    response = s3.get_object(Bucket="ml-data", Key="raw/sample_submission.csv")
    sample_submission = pd.read_csv(BytesIO(response["Body"].read()))

    response = s3.get_object(Bucket="ml-data", Key="raw/calendar.csv")
    calendar = pd.read_csv(BytesIO(response["Body"].read()))

    response = s3.get_object(Bucket="ml-data", Key="raw/sales_train_validation.csv")
    sales_train_validation = pd.read_csv(BytesIO(response["Body"].read()))

    

    sales_train_validation = sales_train_validation[sales_train_validation['store_id'] == 'CA_3']
    data_cols = [col for col in sales_train_validation.columns if 'd_' in col]
    not_data_cols = list(set(sales_train_validation.columns) - set(data_cols))
    df_sales_train_validation = pd.melt(sales_train_validation, id_vars = not_data_cols, value_vars = data_cols)
    df_sales_train_validation = df_sales_train_validation.merge(calendar[['date', 'd']], left_on = ['variable'], right_on = ['d'])
    df_sales_train_validation = df_sales_train_validation.drop(columns = ['variable'])
    
    # Сохраняем обратно в S3
    buffer = BytesIO()
    df_sales_train_validation.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, "ml-data", "features/df_sales_train_validation.csv")
    buffer.close()
    print("Features saved to s3://ml-data/features/train_features.csv")

if __name__ == "__main__":
    process_data()

import pandas as pd
import argparse
import joblib
from catboost import CatBoostRegressor,  Pool
import click
from io import BytesIO
import boto3
import pickle

def train_data():    
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    response = s3.get_object(Bucket="ml-data", Key="features/train_df.csv")
    train_df = pd.read_csv(BytesIO(response["Body"].read()))
    # Define features and target variable
    features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year', 'sales_last_week', 'sales_last_month', 'moving_avg_7', 'moving_avg_30']
    target = 'value'

    # Split the data into training and testing sets
    X_train = train_df[features]
    y_train = train_df[target]

    # Convert categorical features to categorical data type
    categorical_features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year']
    for feature in categorical_features:
        X_train[feature] = X_train[feature].astype('category')



    best_catboost_params = {'cat_features': ('dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year'), 
                            'colsample_bylevel': 0.8500000000000001, 
                            'depth': 2, 'iterations': 500, 
                            'l2_leaf_reg': 0.0013949467397982418, 
                            'learning_rate': 0.02549101349261465, 
                            'random_state': 43, 
                            'verbose': False}
    
    print("Лучшие параметры CatBoost:", best_catboost_params)
    
    # Create Pool objects for training and validation
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    #test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    # Initialize CatBoostRegressor
    best_model = CatBoostRegressor(
        **best_catboost_params
    )

    # Train the model
    best_model.fit(train_pool, early_stopping_rounds=50)

    # Сохранение модели
    #joblib.dump(best_model)
    #print(f"Модель сохранена в {model}")

    #buffer = pickle.dumps(best_model)
    buffer = BytesIO()
    joblib.dump(best_model, buffer)
    #val_df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, "ml-data", "models/model.pkl")
    buffer.close()


    #s3.put_object(Bucket="ml-data", Key="models/model.pkl", Body=buffer)




if __name__ == "__main__":
  #  parser = argparse.ArgumentParser()  # Создаем парсер аргументов
   # parser.add_argument("--train", type=str, required=True, help="Путь для трэйна")
    #parser.add_argument("--model", type=str, required=True, help="Путь для обученной модели")
    #args = parser.parse_args()  # Считываем аргументы

    # Передаем аргументы в функцию
    train_data()

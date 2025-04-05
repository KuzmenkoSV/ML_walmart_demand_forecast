import pandas as pd
import argparse
import joblib
from catboost import CatBoostRegressor,  Pool
import click
  #  parser = argparse.ArgumentParser()  # Создаем парсер аргументов
   # parser.add_argument("--train", type=str, required=True, help="Путь для трэйна")
    #parser.add_argument("--model", type=str, required=True, help="Путь для обученной модели")



@click.command()
@click.option('--train', help="Путь для трейна")
@click.option('--model', help="Путь для модели")
def train_data(train, model):
    train_df = pd.read_csv(train)
    # Define features and target variable
    features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year', 'sales_last_week', 'sales_last_month', 'moving_avg_7', 'moving_avg_30']
    target = 'value'

    # Split the data into training and testing sets
    X_train = train_df[features]
    y_train = train_df[target]
    #X_test = val_df[features]
    #y_test = val_df[target]

    # Convert categorical features to categorical data type
    categorical_features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year']
    for feature in categorical_features:
        X_train[feature] = X_train[feature].astype('category')
        #X_test[feature] = X_test[feature].astype('category')



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
    joblib.dump(best_model, model)
    print(f"Модель сохранена в {model}")

if __name__ == "__main__":
  #  parser = argparse.ArgumentParser()  # Создаем парсер аргументов
   # parser.add_argument("--train", type=str, required=True, help="Путь для трэйна")
    #parser.add_argument("--model", type=str, required=True, help="Путь для обученной модели")
    #args = parser.parse_args()  # Считываем аргументы

    # Передаем аргументы в функцию
    train_data()

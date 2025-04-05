import pandas as pd

def load_and_process_data():

    SELL_PRICES_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sell_prices.csv'
    SAMPLE_SUBMISSION_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sample_submission.csv'
    CALENDAR_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\calendar.csv'
    TRAIN_VALIDATION_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sales_train_validation.csv'
    OUTPUT = 'C:\Walmart_project_Kuzmenko\data\processed\df_sales_train_validation.csv'
   
   
    sell_prices = pd.read_csv(SELL_PRICES_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    calendar = pd.read_csv(CALENDAR_PATH)
    sales_train_validation = pd.read_csv(TRAIN_VALIDATION_PATH)
    sales_train_validation = sales_train_validation[sales_train_validation['store_id'] == 'CA_3']

    data_cols = [col for col in sales_train_validation.columns if 'd_' in col]
    not_data_cols = list(set(sales_train_validation.columns) - set(data_cols))
    df_sales_train_validation = pd.melt(sales_train_validation, id_vars = not_data_cols, value_vars = data_cols)
    df_sales_train_validation = df_sales_train_validation.merge(calendar[['date', 'd']], left_on = ['variable'], right_on = ['d'])
    df_sales_train_validation = df_sales_train_validation.drop(columns = ['variable'])

    df_sales_train_validation.to_csv(OUTPUT, index=False)

if __name__ == "__main__":
    load_and_process_data()

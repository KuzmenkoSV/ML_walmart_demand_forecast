```python
(ML_venv_forecast)
```


```python
!python --version
```

## EDA

### Загрузка данных и первичное преобразование


```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
```


```python
# TODO Пути стоит выносить в отдельные переменные и объявлять их в начале ноутбука
# Так как это константы то будет выглядеть вот так :
SELL_PRICES_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sell_prices.csv'
SAMPLE_SUBMISSION_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sample_submission.csv'
CALENDAR_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\calendar.csv'
TRAIN_VALIDATION_PATH = 'C:\Walmart_project_Kuzmenko\data\external\m5-forecasting-accuracy\sales_train_validation.csv'
# sell_prices = pd.read_csv(SELL_PRICES_PATH)
sell_prices = pd.read_csv(SELL_PRICES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
calendar = pd.read_csv(CALENDAR_PATH)
sales_train_validation = pd.read_csv(TRAIN_VALIDATION_PATH)
```


```python
SELL_PRICES_PATH = 'sell_prices.csv'
SAMPLE_SUBMISSION_PATH =  'sample_submission.csv'
CALENDAR_PATH = 'calendar.csv'
TRAIN_VALIDATION_PATH = 'sales_train_validation.csv'

sell_prices = pd.read_csv(SELL_PRICES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
calendar = pd.read_csv(CALENDAR_PATH)
sales_train_validation = pd.read_csv(TRAIN_VALIDATION_PATH)
```

Выберем магаизн с максимальным оборотом CA_3


```python
sales_train_validation = sales_train_validation[sales_train_validation['store_id'] == 'CA_3']
```


```python
# Принтовать датафреймы не очень хорошая практика - так как исчезает красивая таблица которую создает пандас
# Стоит каждый фрейм отображать в отдельной ячейке с помощью df.head() или просто df
sell_prices.head()
```


```python
sample_submission.head()
```


```python
calendar.head()
```


```python
sales_train_validation.head()
```


```python
sales_train_validation.columns
```


```python
sales_train_validation['store_id'].value_counts()
```

Как видно у нас представлены продажи в 4-х магазинах калифорнии, 3-х магазинах Техаса и 3-х магазинах в Висконсине

> лучше это делать с помощью команды `sales_train_validation['store_id'].value_counts()`. Это покажет еще количество уникальных значений, чтобы было понимание относительно распределение данной категориальной фичи + можно визуализировать и построить barplot


```python
data_cols = [col for col in sales_train_validation.columns if 'd_' in col]
not_data_cols = list(set(sales_train_validation.columns) - set(data_cols))
```

Покажем графики продаж в разных штатах. Для этого сначала преобразуем таблицу с помощью pd.melt() и добавим дату из таблицы calendar


```python
df_sales_train_validation = pd.melt(sales_train_validation, id_vars = not_data_cols, value_vars = data_cols)
df_sales_train_validation = df_sales_train_validation.merge(calendar[['date', 'd']], left_on = ['variable'], right_on = ['d'])
df_sales_train_validation = df_sales_train_validation.drop(columns = ['variable'])
df_sales_train_validation.head(5)
```

Построение графиков c помощью plotly:


```python
df_agg = (
    df_sales_train_validation.groupby(["state_id", "store_id", "date"], as_index=False)["value"]
    .sum()
)
unique_states = df_agg['state_id'].unique()
fig = make_subplots(rows=len(unique_states), cols=1, subplot_titles=[f"State: {state}" for state in unique_states])


for i, state in enumerate(unique_states):
    state_data = df_agg[df_agg["state_id"] == state]
    store_ids = state_data["store_id"].unique()

    for store_id in store_ids:
        store_data = state_data[state_data["store_id"] == store_id]

        fig.add_trace(
            go.Scatter(
                x=store_data["date"],
                y=store_data["value"],
                mode="lines+markers",
                name=f"{store_id} (State: {state})"
            ),
            row=i + 1,
            col=1
        )

# Настройка макета
fig.update_layout(
    title="Time Series by Store and State",
    height=400 * len(unique_states),  # Зависит от числа графиков
    showlegend=True
)

# Отображение графика
fig.show()
```

### Детекция аномалий

#### Медианный фильтр


```python
df_agg['date'] = pd.to_datetime(df_agg['date'])
```


```python
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
```


```python
df_agg['anomaly_med'] = is_outlier(np.array(df_agg['value']))
```


```python
store_id = "CA_3"
state = "CA"
fig.add_trace(
    go.Scatter(
        x=df_agg["date"],
        y=df_agg["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x = df_agg[df_agg.anomaly_med]['date'],
        y = df_agg[df_agg.anomaly_med]['value'],
        mode="markers",
        marker=dict(color='Black', size=10)
    ),
    row=1,
    col=1,
)
# Настройка макета
fig.update_layout(
    title="Time Series by Store and State",
    height=400 * len(unique_states),  # Зависит от числа графиков
    showlegend=True
)

# Отображение графика
fig.show()
```

#### Изоляционный лес


```python
from sklearn.ensemble import IsolationForest
```


```python
df_agg = (
    df_sales_train_validation.groupby(["state_id", "store_id", "date"], as_index=False)["value"]
    .sum()
)
```


```python
df = df_agg.copy()
df['date'] = pd.to_datetime(df['date'])

# Столбец для хранения аномалий
df['iso_anomaly'] = False

for id in df['store_id'].unique():
    group = df[df['store_id'] == id]
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    group_values = group[['value']]
    group['iso_anomaly'] = iso_forest.fit_predict(group_values) == -1
    df.loc[group.index, 'iso_anomaly'] = group['iso_anomaly']
```


```python
unique_states = df['state_id'].unique()
fig = make_subplots(rows=len(unique_states), cols=1, subplot_titles=[f"State: {state}" for state in unique_states])


for i, state in enumerate(unique_states):
    state_data = df[df["state_id"] == state]
    store_ids = state_data["store_id"].unique()

    for store_id in store_ids:
        store_data = state_data[state_data["store_id"] == store_id]

        fig.add_trace(
            go.Scatter(
                x=store_data["date"],
                y=store_data["value"],
                mode="lines",
                name=f"{store_id} (State: {state})"
            ),
            row=i + 1,
            col=1
        ),
        fig.add_trace(
            go.Scatter(
                x = store_data['date'][store_data['iso_anomaly']],
                y = store_data['value'][store_data['iso_anomaly']],
                mode="markers",
                marker=dict(color='Black', size=10)
            ),
            row=i + 1,
            col=1,
        )

# Настройка макета
fig.update_layout(
    title="Time Series Anomaly by Store and State",
    height=400 * len(unique_states),  # Зависит от числа графиков
    showlegend=True
)

# Отображение графика
fig.show()
```

### Стационарность рядов


```python
!pip install statsmodels
```




```python
from statsmodels.tsa.stattools import adfuller
```


```python
df_agg.info()
```


```python
df_agg = df_agg.drop('anomaly_med', axis = 1)
```


```python
df_agg['date'] = pd.to_datetime(df_agg['date'])

adfuler_results_agg = []
ids_agg = []

for id in df_agg['store_id'].unique():
    ids_agg.append(id)
    group = df_agg[df_agg['store_id'] == store_id]
    result = adfuller(group['value'])
    adfuler_results_agg.append("Стационарный" if result[1] < 0.05 else "НЕ стационарный" )

df_adfuler_results_agg = pd.DataFrame({"id": ids_agg, "flag": adfuler_results_agg})
```


```python
df_adfuler_results_agg
```

## Рaзработка модели

### Base-line: Скользящее среднее


```python
df_sales_train_validation['date'] = pd.to_datetime(df_sales_train_validation['date'])
```


```python
print('минимальная дата:', df_sales_train_validation['date'].min())
print('максимальная дата:', df_sales_train_validation['date'].max())
```


```python
train_start_date = '2011-01-29'
train_end_date = '2016-04-18'
val_start_date = '2016-04-18'
val_end_date = '2016-04-24'
```


```python
# Разделение на train test
train_df = df_sales_train_validation[
    (df_sales_train_validation['date'] >= train_start_date)
    & (df_sales_train_validation['date'] < train_end_date)
]
val_df = df_sales_train_validation[
    (df_sales_train_validation['date'] >= val_start_date)
    & (df_sales_train_validation['date'] <= val_end_date)
]
```


```python
#train_df_test = train_df[train_df['cat_id'] == 'HOBBIES']
#val_df_test = val_df[val_df['cat_id'] == 'HOBBIES']
```


```python
from tqdm import tqdm
```


```python
import pandas as pd
import numpy as np

class MovingAverageModel:
    def __init__(self, window=7):
        """
        Инициализация модели.
        :param window: Размер окна для вычисления скользящего среднего
        """
        self.window = window
        self.history = {}

    def fit(self, data):
        """
        Обучение модели (сохранение истории продаж).
        :param data: DataFrame с колонками ['item_id', 'date', 'value']
        """
        data['date'] = pd.to_datetime(data['date'])
        for item in tqdm(data['id'].unique()):
            self.history[item] = (
                data[data['id'] == item]
                .sort_values('date')
                ['value']
                .tolist()
            )

    def predict(self, id, horizon=7):
        """
        Прогнозирование продаж.
        :param item_id: ID товара, для которого строится прогноз
        :param horizon: Число дней для предсказания
        :return: Список прогнозных значений
        """
        if id not in self.history:
            raise ValueError(f"Товар {id} не найден в обучающем наборе данных")

        sales = self.history[id]
        if len(sales) < self.window:
            raise ValueError("Недостаточно данных для расчета скользящего среднего")

        predictions = []
        temp_sales = sales.copy()

        for _ in range(horizon):
            moving_avg = np.mean(temp_sales[-self.window:])
            predictions.append(moving_avg)
            temp_sales.append(moving_avg)

        return predictions
```


```python
mov_ang_model = MovingAverageModel()
mov_ang_model.fit(train_df)
```


```python
for id in tqdm(train_df['id'].unique()):
    res = mov_ang_model.predict(id, horizon=7)
    val_df.loc[val_df['id'] == id, 'predicted'] = np.array(res)
```


```python
#Выберем id для демонстрации предсказаний временного ряда
my_index = 'FOODS_2_001_CA_3_validation'
val_df_for_roll = val_df[val_df['date'] >= pd.to_datetime('2016-03-26')]

fig = make_subplots(rows=1, cols=1)

train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["id"] == my_index)
        ]

val_df_for_plot = val_df_for_roll[val_df_for_roll["id"] == my_index]

store_id = 'CA_3'
state = 'CA'


fig.add_trace(
    go.Scatter(
        x=train_df_for_plot["date"],
        y=train_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["predicted"],
        mode="lines+markers",
        name="Predicted"
    ),
    row=1,
    col=1
)

fig.update_layout(
    title=f"Time Series baseline for {my_index}",
    height=400,
    showlegend=True
)

fig.show()
```


```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error

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
```


```python
metrics_by_id = val_df.groupby('id').apply(lambda group: pd.Series(evaluate(group['value'], group['predicted'])))
```


```python
val_df_for_roll_unique = val_df.drop_duplicates(subset=['id'])

metrics_by_store = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['store_id']).apply(aggregate_metrics)
metrics_by_cat = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['cat_id']).apply(aggregate_metrics)

```


```python
metrics_by_store
```


```python
metrics_by_cat
```

### Попробуем SARIMAX

Эта модель должно учитывать и отсутсвие стационарности и сезонность. Так что мы не будем предобрабатывать ряд


```python
import warnings
warnings.simplefilter(action = 'ignore', category = Warning)

from statsmodels.tsa.statespace.sarimax import SARIMAX
```


```python
from tqdm import tqdm
```


```python
for id in tqdm(train_df['id'].unique()):
    group = train_df[train_df['id'] == id]
    model = SARIMAX(group['value'],
                order = (3, 0, 0),
                seasonal_order = (0, 1, 1, 7)).fit()
    res = model.get_forecast(steps = 7)
    val_df.loc[val_df['id'] == id, 'predicted'] = np.array(res.predicted_mean)

```


```python
fig = make_subplots(rows=1, cols=1)

my_index = "FOODS_1_002_CA_3_validation"
train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["id"] == my_index)
        ]

val_df_for_plot = val_df[val_df["id"] == my_index]
store_id = 'CA_3'
state = 'CA'
fig.add_trace(
    go.Scatter(
        x=train_df_for_plot["date"],
        y=train_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["predicted"],
        mode="lines+markers",
        name="Predicted"
    ),
    row=1,
    col=1
)

fig.update_layout(
    title=f"Time Series baseline for {my_index}",
    height=400,
    showlegend=True
)

fig.show()
```


```python
metrics_by_id = val_df.groupby('id').apply(lambda group: pd.Series(evaluate(group['value'], group['predicted'])))
```


```python
val_df_for_roll_unique = val_df.drop_duplicates(subset=['id'])

metrics_by_store = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['store_id']).apply(aggregate_metrics)
metrics_by_cat = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['cat_id']).apply(aggregate_metrics)

```


```python
metrics_by_store
```


```python
metrics_by_cat
```


```python

```

### Prophet


```python
!pip install prophet
```


```python
val_df.to_csv('val_df_arima_pred', sep=',', index=False, encoding='utf-8')

```


```python
from prophet import Prophet
```


```python
from tqdm import tqdm
```


```python
for id in tqdm(train_df['id'].unique()):
    group = train_df[train_df['id'] == id][['date', 'value']]
    group.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True, mcmc_samples=0)
    model.fit(group)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    val_df.loc[val_df['id'] == id, 'predicted'] = np.array(forecast[forecast['ds'] >= val_start_date]['yhat'])
```


```python
fig = make_subplots(rows=1, cols=1)

my_index = "FOODS_1_002_CA_3_validation"
train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["id"] == my_index)
        ]

val_df_for_plot = val_df[val_df["id"] == my_index]
store_id = 'CA_3'
state = 'CA'
fig.add_trace(
    go.Scatter(
        x=train_df_for_plot["date"],
        y=train_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["predicted"],
        mode="lines+markers",
        name="Predicted"
    ),
    row=1,
    col=1
)

fig.update_layout(
    title=f"Time Series baseline for {my_index}",
    height=400,
    showlegend=True
)

fig.show()
```


```python
metrics_by_id = val_df.groupby('id').apply(lambda group: pd.Series(evaluate(group['value'], group['predicted'])))
```


```python
val_df_for_roll_unique = val_df.drop_duplicates(subset=['id'])

metrics_by_store = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['store_id']).apply(aggregate_metrics)
metrics_by_cat = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['cat_id']).apply(aggregate_metrics)

```


```python
metrics_by_store
```


```python
metrics_by_cat
```


```python
val_df.to_csv('val_df_prophet_pred', sep=',', index=False, encoding='utf-8')
```

### Градиентный бустинг CatBoost


```python
df_sales_train_validation['date'] = pd.to_datetime(df_sales_train_validation['date'])
```


```python
df_sales_train_validation.info()
```


```python
df_sales_train_validation['sales_last_week'] = df_sales_train_validation.groupby('id')['value'].shift(7)
df_sales_train_validation['sales_last_month'] = df_sales_train_validation.groupby('id')['value'].shift(30)
```


```python
df_sales_train_validation['moving_avg_7'] = df_sales_train_validation.groupby('id')['value'].transform(lambda x: x.rolling(window = 7, min_periods = 1).mean())
df_sales_train_validation['moving_avg_30'] = df_sales_train_validation.groupby('id')['value'].transform(lambda x: x.rolling(window = 30, min_periods = 1).mean())
```


```python
df_sales_train_validation.dropna(inplace=True)
```


```python
df_sales_train_validation.head(40)
```


```python
df_sales_train_validation.info()
```


```python
df_sales_train_validation.drop(columns= ['id', 'state_id'], inplace=True)
```


```python
df_sales_train_validation.drop(columns= ['d'], inplace=True)
```


```python
df_sales_train_validation.drop(columns= ['store_id'], inplace=True)
```


```python

```


```python
df_sales_train_validation.info()
```


```python
!pip install catboost
```


```python
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
```


```python
# Create additional time-based features
df_sales_train_validation['day_of_week'] = df_sales_train_validation['date'].dt.dayofweek
df_sales_train_validation['month'] = df_sales_train_validation['date'].dt.month
df_sales_train_validation['year'] = df_sales_train_validation['date'].dt.year
df_sales_train_validation.info()
```


```python
train_start_date = '2011-01-29'
train_end_date = '2016-04-18'
val_start_date = '2016-04-18'
val_end_date = '2016-04-24'
```


```python
# Разделение на train test
train_df = df_sales_train_validation[
    (df_sales_train_validation['date'] >= train_start_date)
    & (df_sales_train_validation['date'] < train_end_date)
]
val_df = df_sales_train_validation[
    (df_sales_train_validation['date'] >= val_start_date)
    & (df_sales_train_validation['date'] <= val_end_date)
]
```


```python
from tqdm import tqdm
```


```python
# Define features and target variable
features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year', 'sales_last_week', 'sales_last_month', 'moving_avg_7', 'moving_avg_30']
target = 'value'

```


```python
# Split the data into training and testing sets
X_train = train_df[features]
y_train = train_df[target]
X_test = val_df[features]
y_test = val_df[target]
```


```python
# Convert categorical features to categorical data type
categorical_features = ['dept_id', 'item_id', 'cat_id', 'day_of_week', 'month', 'year']
for feature in categorical_features:
    X_train[feature] = X_train[feature].astype('category')
    X_test[feature] = X_test[feature].astype('category')

```


```python
# Create Pool objects for training and validation
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize CatBoostRegressor
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.3,
    depth=8,
    loss_function='RMSE',
    eval_metric='MAE',
    verbose=100
)

# Train the model
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

```


```python
# Make predictions
y_pred = model.predict(X_test)
```


```python
val_df['predicted'] = y_pred
```


```python
val_df.info()
```


```python
val_df["store_id"] = 'CA_3'
train_df["store_id"] = 'CA_3'
```


```python
fig = make_subplots(rows=1, cols=1)

#my_index = "FOODS_1_002_CA_3_validation"
item_id = 'FOODS_1_003'
store_id = 'CA_3'
state = 'CA'
train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["store_id"] == store_id) &
            (train_df["item_id"] == item_id)

        ]

val_df_for_plot = val_df[(val_df["store_id"] == store_id) &
            (val_df["item_id"] == item_id)]

fig.add_trace(
    go.Scatter(
        x=train_df_for_plot["date"],
        y=train_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["value"],
        mode="lines+markers",
        name=f"{store_id} (State: {state})"
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=val_df_for_plot["date"],
        y=val_df_for_plot["predicted"],
        mode="lines+markers",
        name="Predicted"
    ),
    row=1,
    col=1
)

fig.update_layout(
    title=f"Time Series baseline for {item_id}",
    height=400,
    showlegend=True
)

fig.show()
```


```python
val_df['id'] = val_df["item_id"] + "_CA_3"
```


```python
val_df.head()
```


```python
metrics_by_id = val_df.groupby('id').apply(lambda group: pd.Series(evaluate(group['value'], group['predicted'])))
```


```python
val_df_for_roll_unique = val_df.drop_duplicates(subset=['id'])

metrics_by_store = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['store_id']).apply(aggregate_metrics)
metrics_by_cat = metrics_by_id.groupby(val_df_for_roll_unique.set_index('id')['cat_id']).apply(aggregate_metrics)

```


```python
metrics_by_store
```


```python
metrics_by_cat
```

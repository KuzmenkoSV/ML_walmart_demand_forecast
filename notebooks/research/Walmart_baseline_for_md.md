```python
(ML_venv_forecast)
```


```python
!python --version
```

## Исследовательский анализ данных


```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
```

# Комментарии

1. В рамках EDA стоит отделять с помощью хэдингов каждый набор данных и отдельно его рассматривать, то есть в данном случае 
Была бы такая структура:

> Это больше относится к реальным проектам, когда у нас данные не такие идеальные как на кагле
--------
# EDA

## sell_prices
Тут необходимо дать краткое описание этих данных и затем проверить ряд базовых вещей (делаем это все в разных ячейках не через print, чтобы красиво отображалось):
1. df.shape()
1. df.head()
2. df.info()
3. df.describe()

Дальше можно:
1. посмотреть число пропущенных значений, число дубликатов (например вывести variance по каждой колонке)
2. посмотреть аномальные значения (поможет визуализаия распределения а также правилоа 3х сигм)
3. сделать визуализаии по необходимости - проверить распределения фичей, а также еще какие-нибудь графики которые помогут нам понять данные (если таргет дан в явном виде, то всякие pairplot и матрицы корреляций будут полезны)
4. по результатм предыдущих постараться придумать несколько гипотез относительно данных и затем их проверить. 

## calendar
Тут аналогично
## sales_train_validation
Тут аналогично

# Объединение данных
тут кажется все датасеты можно объединить в один датафрейм сделав соответственный маппинг. 
-----
После того как были исследованы данные, можно переходить к созданию датасета (фичей и таргета)

# Create dataset (X, y)
Начинать стоит с создания таргета - в данной задаче он находится в явном виде


```python
sell_prices = ...
sell_prices.head()
```


```python
sell_prices.info()
```


```python
sell_prices.describe()
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

> Отличные графики. Дополнительно стоит рассмотреть:
1. Аномальные кейсы (например когда у вас значение резко становятся нулевые):
    > Сгладить данные (взять для примера ноутбук который я раньше скинул)
    > Попробовать сдетектировать аномалии. В [этом](https://www.kaggle.com/code/joshuaswords/time-series-anomaly-detection) ноутбуке подробно описаны методы
2. Для State: WI в середине 2012 года виден интересный тренд - зеленые продажи (WI_1) идет вниз, в то время как розовые идут (WI_2) вверх. А оранжевые W_3 вроде как особенно не меняются. 
    > Рассмотреть отдельно этот период и попытаться объяснить такое поведение используя только данные из прошлого. Предположить почему такое произрошло в формате гипотезы и проверить её
3. Проверить стационарность временных рядов, например использовать тест Дики-Фуллера (нулевая гипотеза состоит в том, что ряд не стационарен) или тест KPSS (нулевая гипотеза состоит в том, что ряд стационарен)
4. Корреляции между рядами в рамках одного штата (кажется что для State WI корреляции могут быть). Использовать для этого кросс-коррелицию. Почитать с примерами [тут](https://www.geeksforgeeks.org/cross-correlation-analysis-in-python/)

В [этом](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python#11.-How-to-test-for-stationarity?-) ноутбуке можно почитать про анализ временных рядов и в частности про стационарность и разные стат тесты на это

### Детекция аномалий


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
df_agg
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

### Поведение State WI

Честно говоря не знаю, что тут исследовать. Просто скажу, что характер ряда поменялся. резкий скачок тренда. Будем брать данные с Ноября 2012
Причем я сделаю фильтр для всех товаров и магазинов, а не только для штата WI, чтобы не заморачиваться


```python
df_sales_train_validation_recent = df_sales_train_validation[df_sales_train_validation["date"] >= '2012-10-01']
df_sales_train_validation_recent["date"].min()
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


```python
df_sales_train_validation_recent['id'].nunique()
```

По хорошему у нас временной ряд идет для каждого id (он сочитает в себе товар и магазин). То есть у нас 30490 временных рядов. Мне кажется было бы логичным попробовать провести тест на стацонарность для каждого из них


```python
df_sales_train_validation_recent['date'] = pd.to_datetime(df_sales_train_validation_recent['date'])

adfuler_results = []
ids = []

for id in df_sales_train_validation_recent['id'].unique():
    ids.append(id)
    group = df_sales_train_validation_recent[df_sales_train_validation_recent['id'] == id]
    result = adfuller(group['value'])
    adfuler_results.append("Стационарный" if result[1] < 0.05 else "НЕ стационарный" )

df_adfuler_results = pd.DataFrame({"id": ids, "flag": adfuler_results})
```

Я пробовал сделать тест дики-фулера для каждого id. Но бросил это идею, так как он должен считаться больше суток....


## Разработка модели

> Для проведения эксперимента по созданию и оценки качества модели необходимо:
1. Явно выделить что является таргетом а что фичами и создать соответсвующие переменные `features`, `target`, в которых будут лежать соответствующие колонки. По-хорошему это делать на предыдущих этапах.
2. Сделать train_test_split. Параметры сплита вынести в отдельные переменные 
3. Сделать модель. Сначала должна идти краткая аннотация к ней. 
3. Написать функцию evaluate в которой будут все необходимые метрики и на выход будет выдавать словарь `{metric: value, ...}`

Все это разбить с помощью хэдингов

# Разработка модели 

## Features and target
```python
features = [...]
target = "value"
```


## Train test split

```python
train_start_date = ...
train_end_date = ...
val_end_date = ...
test_end_date ... 


train_df = ...
val_df = ...
test_df ... 


X = train_df[features]
y = train_df[target]

```

## Train your model 
В идеаеле писать класс у которого есть метод fit, predict
```python
class SlidingWindowModel(object):
    def __init__(self, window_size):
        self.window_size = window_size
    
    def fit(self, X, y):
        return self

    def predict(self, X, y):
        ...
```
Таким образом будет легко заменить твою модель на любую другую и пайплайн не придется менять.

## Evaluate 

```python
def evaluate(y_true, y_pred):
    metrics = dict()

    metrics['mae'] = mae(y_true, y_pred)
    metrics['mape'] = mape(y_true, y_pred)
    metrics['smape'] = smape(y_true, y_pred)

    ...

    return metrics
```

### Base-line: Скользящее среднее


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, max_error

def smape_score(y_true, y_pred):
    n = len(y_true)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = (y_true + y_pred)

    return  1 / n * np.sum(numerator / denominator)

def wape_score(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    return numerator / denominator

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
```


```python
df_sales_train_validation_recent['date'] = pd.to_datetime(df_sales_train_validation_recent['date'])
```


```python
print('минимальная дата:', df_sales_train_validation_recent['date'].min())
print('максимальная дата:', df_sales_train_validation_recent['date'].max())
```


```python
df_sales_train_validation_recent = df_sales_train_validation_recent.sort_values(by = ['id', 'date'])
```


```python
train_start_date = '2011-01-29'
train_end_date = '2016-03-26'
val_start_date = '2016-03-26'
val_end_date = '2016-04-24'
```


```python
# Разделение на train test
train_df = df_sales_train_validation_recent[
    (df_sales_train_validation_recent['date'] >= train_start_date)
    & (df_sales_train_validation_recent['date'] < train_end_date)
]
val_df = df_sales_train_validation_recent[
    (df_sales_train_validation_recent['date'] >= val_start_date)
    & (df_sales_train_validation_recent['date'] < val_end_date)
]
```


```python
train_df.columns
```


```python
val_df_0 = df_sales_train_validation[
    (df_sales_train_validation['date'] >= pd.to_datetime('2016-03-26') - timedelta(days = 30))
    & (df_sales_train_validation['date'] < '2016-03-26')
]

val_df_for_roll = pd.concat([val_df_0, val_df], axis = 0)
val_df_for_roll.head(10)
```


```python
val_df_for_roll['pred'] = val_df_for_roll.groupby('id')['value'].transform(
    lambda x: x.rolling(window = 30, min_periods = 1).mean())
```


```python
val_df_for_roll = val_df_for_roll[val_df_for_roll['date'] >= pd.to_datetime('2016-03-26')]
```


```python
#Выберем id для демонстрации предсказаний временного ряда
my_index = 'FOODS_1_001_CA_1_validation'
```


```python
fig = make_subplots(rows=1, cols=1)

train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["id"] == my_index)
        ]

val_df_for_plot = val_df_for_roll[val_df_for_roll["id"] == my_index]

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
        y=val_df_for_plot["pred"],
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

> Для зеленого ряда стоит изменить название на predicted 


```python
metrics_by_store = val_df_for_roll.groupby('store_id').apply(lambda group: evaluate(group['value'], group['pred']))
metrics_by_cat = val_df_for_roll.groupby('cat_id').apply(lambda group: evaluate(group['value'], group['pred']))

df_metrics_by_store = pd.DataFrame(metrics_by_store.tolist(), index=metrics_by_store.index)
df_metrics_by_cat = pd.DataFrame(metrics_by_cat.tolist(), index=metrics_by_cat.index)
```


```python
df_metrics_by_store
```


```python
df_metrics_by_cat
```

Теперь мы представляем, какое качество даёт наивный подход и отталкиваясь от него можем двигаться дальше

> Стоит добавить чуть большие метрик:
1. MAPE 
1. SMAPE. Для TS она является более предпочтительной 
2. WMAPE. Когда для нас критичнее либо недопрогноз либо перепрогноз
2. Max Error и Absolute Max Error(чтобы понимать насколько мы максимально можем ошибиться)

Хорошая [статья](https://medium.com/@vinitkothari.24/time-series-evaluation-metrics-mape-vs-wmape-vs-smape-which-one-to-use-why-and-when-part1-32d3852b4779) про разницу MAPE, SMAPE, WMAPE

> И посмотреть метрик в разрезе разных штатов/магазинов/товаров

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
    res = model.get_forecast(steps = 29)
    val_df.loc[val_df
    ['id'] == id, 'predicted'] = np.array(res.predicted_mean)
    
```

Честно говоря, показывает, что по всем id считать более 30 часов. Поэтому остановил. Просчитал где-то сотню idшников. Выведу метрики по ним


```python
Id_list_for_arima = np.random.choice(train_df['id'].unique(), size=100)
len(Id_list_for_arima)
```


```python
val_df_100 = val_df[val_df['id'].isin(Id_list_for_arima)]
```


```python

for id in tqdm(Id_list_for_arima):
    group = train_df[train_df['id'] == id]
    model = SARIMAX(group['value'], 
                order = (3, 0, 0), 
                seasonal_order = (0, 1, 1, 7)).fit()
    res = model.get_forecast(steps = 29)

    val_df_100.loc[val_df_100['id'] == id, 'predicted'] = np.array(res.predicted_mean)
```


```python
fig = make_subplots(rows=1, cols=1)

my_index = "FOODS_1_002_CA_1_validation"
train_df_for_plot = train_df[
            (train_df["date"] >= pd.to_datetime('2016-03-26') - timedelta(days = 100)) &
            (train_df["id"] == my_index)
        ]

val_df_for_plot = val_df_100[val_df_100["id"] == my_index]

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
metrics_by_store = val_df_100.groupby('store_id').apply(lambda group: evaluate(group['value'], group['predicted']))
metrics_by_cat = val_df_100.groupby('cat_id').apply(lambda group: evaluate(group['value'], group['predicted']))

df_metrics_by_store = pd.DataFrame(metrics_by_store.tolist(), index=metrics_by_store.index)
df_metrics_by_cat = pd.DataFrame(metrics_by_cat.tolist(), index=metrics_by_cat.index)
```


```python
df_metrics_by_store
```


```python
df_metrics_by_cat
```


```python
!pip install prophet
```

### Prophet


```python
from prophet import Prophet
```


```python
from tqdm import tqdm
```


```python
for id in tqdm(Id_list_for_arima):
    group = train_df[train_df['id'] == id][['date', 'value']]
    group.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True, mcmc_samples=0) 
    model.fit(group)
    future = model.make_future_dataframe(periods=29)
    forecast = model.predict(future)
    val_df_100.loc[val_df_100['id'] == id, 'predicted'] = np.array(forecast[forecast['ds'] >= val_start_date]['yhat'])
```

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# импортируем необходимые функции из библиотеки
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
sns.set()

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from scipy.interpolate import interp1d
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

#Метрики качества R^2, MSE, MAE, MAPE и т.д
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# 1. Скачиваем данные по загруженности метро вот отсюда https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volum

# In[2]:


data2 = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')


# In[3]:


data2.head()


# In[4]:


columns = ['traffic_volume', 'date_time', 'holiday', 'temp']
df2 = pd.DataFrame(data2, columns=columns)
df2.head()


# In[5]:


#Поищем дубликаты 
df2[df2.duplicated(keep=False)].head()


# In[6]:


#Удалим дубликаты
df2 = df2.drop_duplicates(keep='last')


# 2. Проводим базовый EDA, вам понадобятся только 4 столбца датасета - traffic_volume (наша целевая переменная), date_time, holiday (является ли день некоторым праздником) и temp (температура воздуха)

# In[7]:


df2.info()


# In[8]:


#Поищем дубликаты 
df2[df2.duplicated(keep=False)].head()


# In[9]:


df2.count()


# In[10]:


df2.set_index('date_time')


# In[11]:


#Посмотрим пустые значения
df2.isnull().sum()


# 3. По результатам EDA убеждаемся, что в этом временном ряду во-первых, есть дубликаты, а во-вторых, нарушена равномерность временных интервалов, т.е. не все значения отстоят друг от друга на 1 час - дубликаты удаляем, а временные интервалы выравниваем и заполняем пропуски при помощи линейной интерполяции (подсказка - в этом вам помогут функции pd.date_range, и interpolate, пример можно найти здесь - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html)

# In[12]:


plt.figure(figsize=(12, 6))
plt.plot(df2.traffic_volume)
plt.title('temp watched (hourly data)')
plt.grid(True)
plt.show()


# In[13]:


df2.head()


# In[14]:


df2.set_index('date_time',inplace=True)


# In[15]:


df2.head()


# In[16]:


df2[df2.index.duplicated()]


# In[17]:


df4 = df2[~df2.index.duplicated()]


# In[18]:


df4.head()


# In[19]:


df4.index = pd.to_datetime(df4.index)


# In[20]:


df4.index


# In[21]:


plt.figure(figsize=(12, 6))
plt.plot(df4.traffic_volume)
plt.title('temp watched (hourly data)')
plt.grid(True)
plt.show()


# In[22]:


index = pd.date_range(start=df4.index.min(), end=df4.index.max(), freq='1H')


# In[23]:


len(index)


# In[24]:


print(index)


# In[25]:


df_reindexed = df4.reindex(index)


# In[26]:


df_reindexed.count()


# In[27]:


df_reindexed.head()


# In[28]:


plt.figure(figsize=(12, 6))
plt.plot(df4.traffic_volume)
plt.title('temp watched (hourly data)')
plt.grid(True)
plt.show()


# In[29]:


df_reindexed['traffic_volume'] = df_reindexed['traffic_volume'].interpolate(method='linear')
df_reindexed['holiday'] = df_reindexed['holiday'].interpolate(method='pad')
df_reindexed['temp'] = df_reindexed['temp'].interpolate(method='linear')


# In[30]:


df_reindexed.isnull().sum()


# In[31]:


plt.figure(figsize=(12, 6))
plt.plot(df_reindexed.traffic_volume)
plt.title('temp watched (hourly data)')
plt.grid(True)
plt.show()


# In[32]:


df_reindexed = df_reindexed[~df_reindexed.index.duplicated()]


# In[33]:


df_reindexed.count()


# In[34]:


df_reindexed.isnull().sum()


# In[35]:


#Распределение трафика и температуры
num_vars = ['traffic_volume', 'temp']
from pandas.plotting import scatter_matrix
scatter_matrix(df_reindexed[num_vars],figsize=(10,8))
plt.show()


# In[36]:


#график зависимости температуры от интенсивности движения
plt.figure(figsize=(10,8))
sns.set_style('darkgrid')
sns.jointplot(y='traffic_volume', x='temp', data = df_reindexed.loc[df_reindexed.temp>-50])
plt.show()


# In[37]:


#корреляция для числовых переменных
sns.heatmap(df_reindexed.corr(), annot=True)
plt.show()


# Feature engineering and Data cleaning

# In[38]:


df_traffic_features = df_reindexed.copy()


# In[39]:


#Уберем выброс
df_traffic_features = df_traffic_features.loc[df_traffic_features.temp>-250]


# In[40]:


df_traffic_features.head()


# In[41]:


df_traffic_features['holiday'].unique()


# In[42]:


#работа с категориальными переменными (dummy)
def modify_holiday(x):
    if x == 'None':
        return 0
    else:
        return 1
df_traffic_features['holiday'] = df_traffic_features['holiday'].map(modify_holiday)


# In[43]:


df_traffic_features.head()


# In[44]:


#корреляция для числовых переменных
sns.heatmap(df_traffic_features.corr(), annot=True)
plt.show()


# In[45]:


df_traffic_features.index.max()


# In[46]:


df_traffic_features.index.min()


# 3. Генерируем дополнительные признаки из индекса, особенно нужен день недели и час дня

# In[47]:


#Для 24*7 расчитать не смогла(компьютер слабый), пришлось взять по дням
#df_traffic_features_days = df_traffic_features.copy()
#df_traffic_features_days = df_traffic_features_days.resample('D').mean()


# In[48]:


df_traffic_features.head()


# 1. Отложите последние две недели в датасете для тестирования вашей модели - на этих данных вы будете проверять финальное качество всех моделек

# Моделирование
# Нашей целью будет построить модель, которая способна прогнозировать загрузку метро на ближайшую неделю (т.е, так как данные у нас дневные, модель должна предсказывать на 24*7 точек вперёд). 

# Вопрос: Вот тут мне не понятно зачем мы для теста откладываем две недели, если должны предсказывать одну? Поэтому я отложу одну

# In[49]:


y_test = pd.DataFrame(df_traffic_features['2018-09-24 00:00:00':].traffic_volume.copy())
X_test = pd.DataFrame(df_traffic_features['2018-09-24 00:00:00':].copy()).drop(['traffic_volume'], axis=1)


# In[50]:


y_test.count()


# In[51]:


plt.figure(figsize=(12, 6))
plt.plot(y_test.traffic_volume)
plt.title('temp watched (hourly data)')
plt.grid(True)
plt.show()


# 2. Теперь у вас осталось еще много-много наблюдений во временном ряду, исходя из графиков, трендов и т.д., попробуйте предположить, какие исторические данные действительно будут релевантны для прогнозов текущих значений, возможно, предыдущие три года уже не так сильно влияют на следующую неделю и можно значительную часть данных просто выкинуть

# In[52]:


y_train = pd.DataFrame(df_traffic_features['2018-07-24 00:00:00': '2018-09-24 00:00:00'].traffic_volume.copy())
X_train = pd.DataFrame(df_traffic_features['2018-07-24 00:00:00': '2018-09-24 00:00:00'].copy()).drop(['traffic_volume'], axis=1)


# In[53]:


X_train['weekday'] = X_train.index.weekday
X_train['hour'] = X_train.index.hour


# In[54]:


plt.figure(figsize=(12, 6))
plt.plot(y_train.traffic_volume)
plt.title('temp watched (data)')
plt.grid(True)
plt.show()


# In[55]:


X_train.isna().sum()


# In[56]:


X_train.index.min()


# 4. Строим baseline прогноз - по тем данным, которые вы решили оставить для обучения модели, посчитайте средние значения по часам и по дням (вам поможет data.groupby(["weekday", "hour"])["traffic_volume"].mean() и используйте эти значения в качестве прогноза на отложенную выборку - посчитайте метрики качества, которые вы посчитаете нужными

# In[57]:


df_baseline = pd.DataFrame(y_train["traffic_volume"].copy())
df_baseline['weekday'] = y_train.index.weekday
df_baseline['hour'] = y_train.index.hour


# In[58]:


baseline = df_baseline.groupby(["weekday", "hour"])["traffic_volume"].mean()


# In[59]:


baseline.count()


# In[60]:


print(baseline)


# Посчитайте метрики качества, которые вы посчитаете нужными

# In[61]:


# подгружаем метрики
#метрики качества (можно взять несколько, R2, MAE, RMSE)
#Средняя абсолютная ошибка (MAE) представляет собой разницу между исходными и прогнозируемыми значениями, извлеченными посредством усредненной абсолютной разницы по набору данных.
#Среднеквадратическая ошибка (RMSE) это частота ошибок, вычисляемая квадратным корнем из MSE.
#Коэффициент детерминации (R2) - описявает качество модели. Чем выше значение, тем лучше модель (от 0 до 1, 1 идеальный)
#Средняя абсолютнаяпроцентная ошибка (MAPE) - процент кол-ва ошибок
#Среднеквадратическая ошибка (MSE) - представляет собой разницу между исходным и прогнозируемым значениями, извлеченными путем возведения в квадрат средней разницы по набору данных.
from sklearn import metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def dataframe_metrics(y_test,y_pred):
    stats = [
       metrics.mean_absolute_error(y_test, y_pred),
       np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
       metrics.r2_score(y_test, y_pred),
       mean_absolute_percentage_error(y_test, y_pred)
    ]
    return stats
measured_metrics = pd.DataFrame({"error_type":["MAE", "RMSE", "R2", "MAPE"]})
measured_metrics.set_index("error_type")


# In[62]:


y_pred_naive = np.ones(len(y_test)) * baseline  # спрогнозировали им цену всех квартир в тестовой выборке
measured_metrics["y_pred_naive"] = dataframe_metrics(y_test, y_pred_naive)
measured_metrics


# 5. А теперь свободное творчество - попробуйте построить разные модели, которые мы с вами разбирали, и побить качество базового прогноза.

# ### SARIMA

# In[63]:


import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


# In[64]:


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, method="ols")
        plt.tight_layout()


# In[65]:


tsplot(y_train.traffic_volume, lags=168)


# ряд по критерию Дикки-Фуллера стационарен, по его графику тоже видно, что тренд, как таковой, отсутствует, т.е., матожидание постоянно, разброс вокруг среднего тоже примерно одинаковый, значит, постоянство дисперсии также наблюдается

# Косинусойда - сезонность, с которой нужно разобраться перед построением модели. Для этого выполним преобразование под хитрым названием "сезонное дифференцирование", под которым скрывается простое вычитание из ряда самого себя с лагом равным периоду сезонности. В нашем случает 24*7=168 часа

# In[66]:


season = 168


# In[67]:


y_train.count()


# In[68]:


y_train_season = y_train.traffic_volume - y_train.traffic_volume.shift(season)
tsplot(y_train_season[season:], lags=season)


# Уже лучше, от видимой сезонности избавились, в автокорреляционной функции по-прежнему осталось много значимых лагов, попробуем от них избавиться, взяв еще и первые разности - то есть вычитание ряда самого из себя с лагом в один шаг

# In[72]:


y_train_shift = y_train_season - y_train_season.shift(1)
tsplot(y_train_shift[season+1:], lags=season*2)


# In[70]:


tsplot(y_train_shift[season+1:], lags=60)


# теперь ряд выглядит как непонятно что, колеблющееся вокруг нуля, по критерию Дикки-Фуллера он стационарен, а в автокорреляционной функции пропали многочисленные значимые пики. Можно приступать к моделированию!

# 𝐴𝑅(𝑝)  - модель авторегрессии (Смотрим значимые лаги в Partial Autocorrelation)
# 𝑀𝐴(𝑞)  - модель скользящего среднего (Смотрим значимые лаги в Autocorrelation)
# 𝐴𝑅(𝑝)+𝑀𝐴(𝑞)=𝐴𝑅𝑀𝐴(𝑝,𝑞)
# 𝐼(𝑑)  - порядок интегрированности временного ряда (Кол-во не сезонных разностей)
# 𝑆(𝑠)  - эта буква отвечает за сезонность и равна длине периода сезонности во временном ряде
# 𝑃  - порядок модели авторегрессии для сезонной компоненты, определяется по PACF, смотреть нужно на число значимых лагов (Смотрим кол-во значемых лагов кратные сезонности в Partial Autocorrelation)
# 𝑄  - аналогично, но для модели скользящего среднего по сезонной компоненте, определяется по ACF (Смотрим кол-во значемых лагов кратные сезонности в Autocorrelation)
# 𝐷  - порядок сезонной интегрированности временного ряда. (Кол-во сезонных диференцирований)

# Подбор параметров модель будет лучше

# In[75]:


# setting initial values and some bounds for them
ps = [8, 13]
d=1 
qs = [6]
Ps = [2,3]
D=1 
Qs = [1]
s = 24*7

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[76]:


def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(y_train.traffic_volume, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


# In[ ]:


# На моем компьютере за ночь не смогло расчитаться 
#%time
#result_table = optimizeSARIMA(parameters_list, d, D, s)


# In[78]:


# set the parameters that give the lowest AIC
#p, q, P, Q = result_table.parameters[0]
#p, q, P, Q = [8,6,2,1]
#best_model=sm.tsa.statespace.SARIMAX(y_train.traffic_volume, order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
#print(best_model.summary())


# Посмотрим на остатки модели:

# In[ ]:


#tsplot(best_model.resid[24*7+1:], lags=168)


# Что ж, остатки стационарны, явных автокорреляций нет, построим прогноз по получившейся модели

# In[ ]:


def plotSARIMA(series, model, n_steps, s=24*7, d=1, plot_intervals=True, alpha=0.2):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = best_model.get_forecast(steps=n_steps)
    model_predictions = data.arima_model.append(forecast.predicted_mean)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))    
    
    plt.plot(model_predictions, color='r', label="model")
    plt.plot(data.actual, label="actual")
    
    if plot_intervals:
        intervals = forecast.conf_int(alpha=alpha)
        intervals.columns=['lower', 'upper']
        plt.plot(intervals['lower'], "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(intervals['upper'], "r--", alpha=0.5)
        plt.fill_between(x=intervals.index, y1=intervals['upper'], 
                         y2=intervals['lower'], alpha=0.2, color = "grey")    
    
    
    plt.legend()
    plt.grid(True);


# In[ ]:


#plotSARIMA(y_test, best_model, 168, alpha=0.5)


# В финале получаем очень хороший прогноз, заказчику средняя абсолютная ошибка в 6 процента также наверняка понравится, однако суммарные затраты на подготовку данных, приведение к стационарности, определение и перебор параметров могут такой точности и не стоить.

# ### Лаги временного ряда

# Сдвигая ряд на  𝑛  шагов мы получаем столбец-признак, в котором текущему значению ряда в момент  𝑡  будет соответствовать его значение в момент времени  𝑡−𝑛 . Таким образом, если сделать отступ в 1 шаг, то модель, обученная на таком признаке, будет способна давать предсказание на 1 шаг вперед, зная текущее состояние ряда. Увеличивая сдвиг, например, до 14 лага позволит модели делать предсказания на 6 шагов вперёд, однако опираться она будет на данные, которые видела 14 временных периодов назад и если за это время что-то кардинально поменялось, модель сразу не уловит изменений и выдаст прогноз с большой ошибкой. Поэтому при выборе начального лага приходиться балансировать между желанием получить предсказания на бОльшее число периодов вперёд и приемлимым качеством предсказания

# In[188]:


data_copy = pd.DataFrame(df_traffic_features['2018-07-24 00:00:00': '2018-09-30 23:00:00'].copy())
data_copy.drop(["holiday", "temp"], axis=1, inplace=True)


# In[189]:


data_copy.columns


# In[190]:


data_copy.head()


# In[191]:


data_copy.tail()


# In[192]:


# Adding the lag of the target variable from 6 steps back up to 24
for i in range(6, 25):
    data_copy["lag_{}".format(i)] = data_copy.traffic_volume.shift(i)


# In[193]:


# take a look at the new dataframe 
data_copy.head(25)


# In[194]:


data_copy = data_copy.dropna()


# In[195]:


# take a look at the new dataframe 
data_copy.head(25)


# In[196]:


data_copy.shape


# In[197]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)


# In[238]:


# Нашей целью будет построить модель, которая способна прогнозировать загрузку метро на ближайшую неделю.
# Отложим неделю на тест
def timeseries_train_test_split(X, y, test_index):
    """
        Perform train-test split with respect to time series structure
    """
    X_train = X.iloc[:test_index-1]
    y_train = y.iloc[:test_index-1]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    
    return X_train, X_test, y_train, y_test


# In[199]:


y = data_copy.dropna().traffic_volume
X = data_copy.dropna().drop(['traffic_volume'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, -24*7)


# In[200]:


# machine learning in two lines
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[201]:


def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');


# In[202]:


plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)


# Большая ошибка, добавим признаков и прошкалируем

# In[203]:


def code_mean(data, cat_feature, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())


# In[229]:


def cycle_data_encoding(df, cycle, name):
    df['sin_' + name] = np.sin(2*np.pi*df[name]/cycle)
    df['cos_' + name] = np.cos(2*np.pi*df[name]/cycle)

    return df


# In[240]:


def prepareData(series, lag_start, lag_end, test_index, target_encoding=False, circle_encoding=False):
    """
        series: pd.DataFrame
            dataframe with timeseries

        lag_start: int
            initial step back in time to slice target variable 
            example - lag_start = 1 means that the model 
                      will see yesterday's values to predict today

        lag_end: int
            final step back in time to slice target variable
            example - lag_end = 4 means that the model 
                      will see up to 4 days back in time to predict today

        test_size: float
            size of the test dataset after train/test split as percentage of dataset

        target_encoding: boolean
            if True - add target averages to the dataset
            
        get_dummies: boolean
            if True - encode categorical into dummies
        
    """
    
    # copy of the initial dataset
    data = pd.DataFrame(series.copy())
    data.columns = ["traffic_volume"]
    
    # lags of series
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.traffic_volume.shift(i)
    
    # datetime features
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5,6])*1
    if target_encoding:
        # calculate averages on train set only
        data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "traffic_volume").get, data.weekday))
        data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "traffic_volume").get, data.hour))
    if circle_encoding:
        data = cycle_data_encoding(data, 24, 'hour')
        data = cycle_data_encoding(data, 7, 'weekday')
        # drop encoded variables 
    if circle_encoding or target_encoding:
        data.drop(["hour", "weekday"], axis=1, inplace=True)
    
    # train-test split
    y = data.dropna().traffic_volume
    X = data.dropna().drop(['traffic_volume'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_index)

    return X_train, X_test, y_train, y_test


# In[221]:


data_copy.head(10)


# In[231]:


X_train, X_test, y_train, y_test =prepareData(data_copy.traffic_volume, lag_start=6, lag_end=25, target_encoding=False, circle_encoding=True)

column_to_scale = [col for col in X_train.columns if 'lag' in col]
X_train_scaled = scaler.fit_transform(X_train[column_to_scale])
X_test_scaled = scaler.transform(X_test[column_to_scale])

X_train[column_to_scale] = X_train_scaled
X_test[column_to_scale] = X_test_scaled


# In[232]:


X_train.head()


# In[233]:


lr = LinearRegression()
lr.fit(X_train, y_train)

plotModelResults(lr, X_train=X_train, X_test=X_test, plot_intervals=True)
plotCoefficients(lr)


# кодировать часы и недели c помощию синусов и косинусов

# ### Регуляризация и отбор признаков

# Для начала убедимся, что нам есть, что отбрасывать и в данных действительно очень много скоррелированных признаков

# In[234]:


plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr());


# In[241]:


X_train, X_test, y_train, y_test =prepareData(data_copy.traffic_volume, lag_start=6, lag_end=25, test_index=-24*7, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lr)


# Переобучение, переменные average настолько понравились нашей модели на тренировочном датасете, что по ней, в основном, модель и стала прогнозировать. 

# ### Регуляризация и отбор признаков

# In[252]:


plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr());


# In[253]:


from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv, alphas=np.linspace(5, 10))
ridge.fit(X_train, y_train)

plotModelResults(ridge, 
                 X_train=X_train, 
                 X_test=X_test, 
                 plot_intervals=True, plot_anomalies=True)
plotCoefficients(ridge)


# In[254]:


lasso = LassoCV(cv=tscv, eps=0.01)
lasso.fit(X_train, y_train)

plotModelResults(lasso, 
                 X_train=X_train, 
                 X_test=X_test, 
                 plot_intervals=True, plot_anomalies=True)
plotCoefficients(lasso)


# ridge регрессия оказалась более удачной в отборе.

# ### Boosting

# In[255]:


from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

gbr = AdaBoostRegressor(n_estimators=100)
gbr.fit(X_train_scaled, y_train)

plotModelResults(gbr, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)


# In[258]:


from catboost import CatBoostRegressor

X_train, X_test, y_train, y_test =prepareData(data_copy.traffic_volume, lag_start=6, lag_end=25, test_index=-24*7, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


catboost = CatBoostRegressor(iterations=100, learning_rate=0.7,
                          loss_function='RMSE', verbose=0)

catboost.fit(X_train_scaled, y_train, verbose=0)

plotModelResults(catboost, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)


# В идеале - предварительно очистить данные от тренда, спрогнозировать тренд отдельно при помощи линейной модели, и отдельно - очищенный ряд, а потом сложить вместе предсказания.

# ### Facebook Prophet - automating routines

# https://habrahabr.ru/company/ods/blog/323730/

# In[259]:


from fbprophet import Prophet


# In[261]:


# number of future predictions
predictions = 168


# In[267]:


df = df_traffic_features['2018-07-24':]
df = df.reset_index()
for i in ['holiday', 'temp']:
    df = df.drop([i], axis=1)
df.columns = ['ds', 'y']


# In[268]:


df.head()


# In[269]:


# reserve some data for testing
train_df = df[:-predictions-1]


# In[270]:


# declaring the model, it's necessary to pay attention to seasonality type we want to use
model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
model.fit(train_df)

# creating one more specific dataset for predictions
# we also need to set data frequency here (by default - days)
future = model.make_future_dataframe(periods=predictions, freq='H')
future.tail()


# In[271]:


# finally, making predictions
forecast = model.predict(future)
forecast.tail()


# In[272]:


# pictures!
error = mean_absolute_percentage_error(df[-predictions:]['y'], forecast[-predictions:]['yhat'])
print("Mean absolute percentage error {0:.2f}%".format(error))
_ = model.plot(forecast)


# In[273]:


_ = model.plot_components(forecast)


# Полезна для прототипирования или маркетинговых иследований, для понимания взлетит , не взлетит. Точность ниже, чем у остальных

# Amazon GluonTS - new big player on the market

# Попробуем построить модель. GluonTS предоставляет верхнеуровневую абстрацию Dataset, которая переводит разнородные форматы данных в один, удобный для последующей работы моделей. В частности, ListDataset переводит данные в список словарей, где отдельно записаны значения ряда и таймстэмпы. Для создания такого датасета мы передаём наш исходный временной ряд, указываем его частоту (в данном случае у нас почасовые данные, поэтому частота "H"), а также точку, до которой наш ряд будет отнесён к тренировочной выборке:

# In[274]:


df = df_traffic_features['2018-07-24':].copy()
for i in ['holiday', 'temp']:
    df = df.drop([i], axis=1)


# In[275]:


from gluonts.dataset.common import ListDataset
training_data = ListDataset(
    [{"start": df.index[0], "target": df.traffic_volume[:"2018-09-15"]}],
    freq = "H"
)


# Посмотрим, в какой формат преобразовались данные:

# In[276]:


training_data.list_data


# Обучаем модель

# In[277]:


from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

estimator = DeepAREstimator(
    freq="H", 
    prediction_length=168, 
    trainer=Trainer(epochs=10)
)
predictor = estimator.train(training_data=training_data)


# In[278]:


df.index[0]


# In[279]:


from gluonts.dataset.util import to_pandas

df = df_traffic_features['2018-09-16':].copy()
for i in ['holiday', 'temp']:
    df = df.drop([i], axis=1)

test_data = ListDataset(
    [{"start": df.index[0], "target": df.traffic_volume["2018-09-16":]}],
    freq = "H"
)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry).plot(linewidth=2, figsize=(15, 7), label="historical values")
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0], label="forecast")
    
plt.legend(loc='upper left')
plt.grid(axis='x')


# В результате мы получили очень правдоподобный прогноз, который учитывает и недельную сезонность и 30-дневную цикличность. Хорошо видно, что доверительные интервалы прогноза расширяются в момент пика, где исторические значения были наименее стабильными, и сужаются в обычные дни, где дисперсия исторических данных была не такой большой.
# 
# GluonTS - очень удобный инструмент, который позволяет максимально быстро и верхнеуровнево получить вероятностую модель временного ряда, используя глубокое обучение под капотом. Помимо хороших результатов, которые получаются прямо из коробки, GluonTS можно тонко настраивать под любые нужды

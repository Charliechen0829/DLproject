import torch
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import seaborn as sns

if torch.cuda.is_available():
    device = 'cuda'
    print("GPU")
else:
    device = 'cpu'
    print("CPU")

data = pd.read_csv('./data/Metro_Interstate_Traffic_Volume.csv', parse_dates=['date_time'])
# 取2016年以后的数据作为数据集
data = data[data['date_time'] >= '2016-01-01']
data = data.set_index('date_time').reset_index()


# 数据预处理
def preprocessing():
    global data

    # 是否存在缺失值
    # print(data.isnull().sum())

    # 根据时间轴去除重复数据
    data.drop_duplicates(subset='date_time', inplace=True)
    data = data.set_index('date_time').reset_index()

    # 观察数据集可以看到snow_1h在一个异常值后经过较短的时间内降雪量一直为0,此数据可以丢弃 6年出现几小时大降雪后不再出现可能是数据搜集问题
    data.drop(columns='snow_1h', inplace=True)

    # 独热编码
    data = pd.get_dummies(data, columns=['holiday', 'weather_main', 'weather_description'], drop_first=True)

    # 异常值处理
    outlier_handling()


# 拆分数据集 并 数据归一化
def split_dataset():
    global data

    # 使用(70 %, 20 %, 10 %)拆分出训练集、验证集和测试集
    n = len(data)
    train_dataset = data[0:int(n * 0.7)]
    val_dataset = data[int(n * 0.7):int(n * 0.9)]
    test_dataset = data[int(n * 0.9):]

    # 数据归一化 Z - Score Normalization
    train_mean, train_std = train_dataset.mean(), train_dataset.std()
    data_mean, data_std = data.mean(), data.std()

    col_name = ['temp', 'rain_1h', 'clouds_all', 'traffic_volume']

    data_copy = data.copy()
    for col in col_name:
        data_copy[col] = (data_copy[col] - data_mean[col]) / data_std[col]
        train_dataset[col] = (train_dataset[col] - train_mean[col]) / train_std[col]
        val_dataset[col] = (val_dataset[col] - train_mean[col]) / train_std[col]
        test_dataset[col] = (test_dataset[col] - train_mean[col]) / train_std[col]

    # 保存为csv文件
    train_dataset.to_csv('./data/train_dataset.csv', index=False)
    val_dataset.to_csv('./data/val_dataset.csv', index=False)
    test_dataset.to_csv('./data/test_dataset.csv', index=False)

    data_copy = data_copy[col_name]
    data_copy = data_copy.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=data_copy)
    _ = ax.set_xticklabels(col_name)
    plt.show()


# 异常值的处理
# Z-score检测出异常值同时用移动平均的思想将异常值进行了替换
def outlier_handling():
    global data

    col_name = ['temp', 'rain_1h', 'clouds_all', 'traffic_volume']

    for name in col_name:
        data['moving_avg'] = data[name].rolling(window=5, min_periods=1, closed='left').mean()
        # 计算 Z-Score
        data['z_score'] = (data[name] - data[name].mean()) / data[name].std()
        # 将异常值替换为移动平均值
        data.loc[data['z_score'].abs() > 3, name] = data['moving_avg']

    data.drop(columns=['moving_avg', 'z_score'], inplace=True)


# 绘制变量的时间序列图
def visualize():
    global data

    col_name = ['temp', 'rain_1h', 'clouds_all', 'traffic_volume']
    plot_ylabel = ['Average temp in kelvin',
                   'Hourly rainfall',
                   'Percentage of cloud cover',
                   'traffic volume']

    vis_nums = 3000
    fig, axs = plt.subplots(len(col_name), 1)
    for i in range(len(col_name)):
        axs[i].plot(data.index[:vis_nums], data[col_name[i]][:vis_nums])
        axs[i].set_ylabel(plot_ylabel[i], rotation=90)
        axs[i].set_xlabel('Date Time')
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()


# 数据分析
def data_analysis():
    # 趋势分析
    decomposition = seasonal_decompose(data['traffic_volume'][:1000], model='additive', period=100)
    trend = decomposition.trend
    trend.plot()
    plt.title('trend')
    plt.show()

    # 季节性分析
    seasonal = decomposition.seasonal
    seasonal.plot()
    plt.title('season')
    plt.show()

    # 周期性分析（残差）
    residual = decomposition.resid
    residual.plot()
    plt.title('Periodicity (residual)')
    plt.show()


# 平稳性分析
def adf_test():
    global data

    # p值小于0.05，表示拒绝原假设，即时间序列是平稳的
    # 意味着时间序列数据中很可能不存在随时间变化的趋势或季节性成分
    # ADF Statistic: -19.293283303079704
    # n_lags: 0.0
    # p - value: 0.0

    # 使用ADF Test
    result = adfuller(data['traffic_volume'], autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[1]}')
    print(f'p-value: {result[1]}')


# 寻找自相关
def acf_and_pacf():
    # 自相关和偏自相关图
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    sm.graphics.tsa.plot_acf(data['traffic_volume'].dropna(), lags=100, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data['traffic_volume'].dropna(), lags=100, ax=ax[1])
    plt.show()


if __name__ == '__main__':
    # 数据集预处理
    preprocessing()

    # 拆分数据集 并 数据归一化
    split_dataset()

    # 查看数据集
    print(data.describe().transpose())

    # 绘制变量的时间序列图
    visualize()

    # 数据分析
    data_analysis()

    # 平稳性分析
    adf_test()

    # 寻找自相关
    acf_and_pacf()
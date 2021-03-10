# -*- coding = utf-8 -*-
# @time:2021/3/9 22:27
# Author:ljc
# @File:数据分析1-温度分析.py
# @Software:PyCharm

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 设置简黑字体
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内


city_names = ["asti", "bologna", "cesena", "faenza", "ferrara",
        "mantova", "milano", "piacenza", "ravenna", "torino"]  # 数据城市名称

asti, bologna, cesena, faenza, ferrara, \
mantova, milano, piacenza, ravenna, torino = \
    [pd.read_csv("WeatherData/%s_270615.csv" % city_names[i], index_col=0) for i in range(len(city_names))]   # 以城市名称表示对应城市csv数据

city_data = [asti, bologna, cesena, faenza, ferrara,
mantova, milano, piacenza, ravenna, torino]


# 1.做出asti阿斯蒂市一天温度变化曲线图
asti_temp = asti["temp"]
asti_time = asti["day"]

asti_time = [parser.parse(x) for x in asti_time]  # 把日期转化为datetime格式
hours = mdates.DateFormatter("%H:%M")

plt.plot(asti_time, asti_temp, "r")
plt.scatter(asti_time[np.argmax(asti_temp)], np.max(asti_temp),
            label="最高温时间{0},温度{1}".format(asti_time[np.argmax(asti_temp)].strftime("%H:%M"), np.round(np.max(asti_temp), 2)))
plt.scatter(asti_time[np.argmin(asti_temp)], np.min(asti_temp),
            label="最低温时间{0},温度{1}".format(asti_time[np.argmin(asti_temp)].strftime("%H:%M"), np.round(np.min(asti_temp), 2)))

plt.gca().xaxis.set_major_formatter(hours)
# plt.locator_params("x", nbins=5)        # 自动设置大约多少个刻度
plt.gcf().autofmt_xdate()                 # 自动旋转日期标记
plt.legend()
plt.show()


# 2.分别做出3个距离海洋最近与最远的城市温度时间曲线图
"""
最近：ravena, faenza, cesena
最远: milano, asti, torino
"""
ravena_temp, faenza_temp, cesena_temp, milano_temp, asti_temp, torino_temp = \
    ravenna["temp"], faenza["temp"], cesena["temp"], milano["temp"], asti["temp"], torino["temp"]

ravena_time, faenza_time, cesena_time, milano_time, asti_time, torino_time = \
    ravenna["day"], faenza["day"], cesena["day"], milano["day"], asti["day"], torino["day"]

ravena_time, faenza_time, cesena_time, milano_time, asti_time, torino_time\
    = [parser.parse(x) for x in ravena_time], [parser.parse(x) for x in faenza_time], \
      [parser.parse(x) for x in cesena_time], [parser.parse(x) for x in milano_time], \
      [parser.parse(x) for x in asti_time], [parser.parse(x) for x in torino_time]  # 把日期转化为datetime格式

hours = mdates.DateFormatter("%H:%M")

plt.plot(ravena_time, ravena_temp, "r")
plt.plot(faenza_time, faenza_temp, "r")
plt.plot(cesena_time, cesena_temp, "r")
plt.plot(milano_time, milano_temp, "g")
plt.plot(asti_time, asti_temp, "g")
plt.plot(torino_time, torino_temp, "g")
plt.gca().xaxis.set_major_formatter(hours)
# plt.locator_params("x", nbins=5)        # 自动设置大约多少个刻度
plt.gcf().autofmt_xdate()                 # 自动旋转日期标记
plt.show()


# 3. 距海洋距离与城市最高温,最低温,平均温度的变化关系
city_dist = [city_data[i]["dist"][0] for i in range(len(city_names))]
city_max_temp = [city_data[i]["temp"].max() for i in range(len(city_names))]
city_min_temp = [city_data[i]["temp"].min() for i in range(len(city_names))]
city_mean_temp = [city_data[i]["temp"].mean() for i in range(len(city_names))]
print(len(city_dist), len(city_max_temp), len(city_mean_temp), len(city_min_temp))

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.scatter(city_dist, city_min_temp, c="r")
plt.ylim([np.min(city_min_temp) - 1, np.max(city_min_temp) + 1])
plt.ylabel("city_min_temp", fontsize=20)
plt.subplot(3, 1, 2)
plt.scatter(city_dist, city_mean_temp, c="g")
plt.ylim([np.min(city_mean_temp) - 1, np.max(city_mean_temp) + 1])
plt.ylabel("city_mean_temp", fontsize=20)
plt.subplot(3, 1, 3)
plt.scatter(city_dist, city_max_temp, c="y")
plt.ylim([np.min(city_max_temp) - 1, np.max(city_max_temp) + 1])
plt.xlabel("city_sea_dist", fontsize=20)
plt.ylabel("city_max_temp", fontsize=20)
plt.show()


# 4.城市距离海洋最佳距离，海洋影响较弱距离（考虑温度）
from sklearn.svm import SVR
from scipy.optimize import fsolve

city_dist_new = np.sort(city_dist).reshape(-1, 1)
city_min_temp_new = np.array(city_min_temp)[np.argsort(city_dist)]
city_mean_temp_new = np.array(city_mean_temp)[np.argsort(city_dist)]
city_max_temp_new = np.array(city_max_temp)[np.argsort(city_dist)]

svr_max_1 = SVR(kernel="linear")
svr_max_1.fit(city_dist_new[:5], city_max_temp_new[:5])
svr_max_2 = SVR(kernel="linear")
svr_max_2.fit(city_dist_new[5:], city_max_temp_new[5:])
x1 = np.arange(0, 100, 10).reshape(10, 1)
x2 = np.arange(50, 400, 50).reshape(7, 1)
y1 = svr_max_1.predict(x1)
y2 = svr_max_2.predict(x2)

def cross_point(x):
    """
    解方程，求交点
    """
    y1_max_temp = svr_max_1.coef_[0][0] * x + svr_max_1.intercept_[0]
    y2_max_temp = svr_max_2.coef_[0][0] * x + svr_max_2.intercept_[0]
    return y1_max_temp - y2_max_temp
best_dist = np.round(fsolve(cross_point, x0=40)[0], 3)
best_max_temp = np.round(svr_max_1.coef_[0][0] * best_dist +
                         svr_max_1.intercept_[0], 3)

plt.plot(x1, y1, c="r", label="Light sea effect")
plt.plot(x2, y2, c="y", label="Strong sea effect")
plt.scatter(best_dist, best_max_temp,
            label="best_dist:{0},best_max_temp:{1}"
            .format(best_dist, best_max_temp))
plt.scatter(city_dist, city_max_temp)
plt.legend()
plt.show()
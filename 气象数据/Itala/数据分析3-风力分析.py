# -*- coding = utf-8 -*-
# @time:2021/3/10 22:06
# Author:ljc
# @File:数据分析3-风力分析.py
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


# 1.做出asti阿斯蒂市一天风向变化曲线图与风速变化曲线
asti_wind_deg = asti["wind_deg"]
asti_wind_speed = asti["wind_speed"]
asti_time = asti["day"]

asti_time = [parser.parse(x) for x in asti_time]  # 把日期转化为datetime格式
hours = mdates.DateFormatter("%H:%M")

plt.plot(asti_time, asti_wind_deg, "r")
plt.scatter(asti_time[np.argmax(asti_wind_deg)], np.max(asti_wind_deg),
            label="最大风向时间{0},风向{1}".format(asti_time[np.argmax(asti_wind_deg)].strftime("%H:%M"), np.round(np.max(asti_wind_deg), 2)))
plt.scatter(asti_time[np.argmin(asti_wind_deg)], np.min(asti_wind_deg),
            label="最低风向时间{0},风向{1}".format(asti_time[np.argmin(asti_wind_deg)].strftime("%H:%M"), np.round(np.min(asti_wind_deg), 2)))

plt.gca().xaxis.set_major_formatter(hours)
# plt.locator_params("x", nbins=5)        # 自动设置大约多少个刻度
plt.gcf().autofmt_xdate()                 # 自动旋转日期标记
plt.legend()
plt.show()

plt.plot(asti_time, asti_wind_speed, "r")
plt.scatter(asti_time[np.argmax(asti_wind_speed)], np.max(asti_wind_speed),
            label="最大风速时间{0},风速{1}".format(asti_time[np.argmax(asti_wind_speed)].strftime("%H:%M"), np.round(np.max(asti_wind_speed), 2)))
plt.scatter(asti_time[np.argmin(asti_wind_speed)], np.min(asti_wind_speed),
            label="最低风速时间{0},风速{1}".format(asti_time[np.argmin(asti_wind_speed)].strftime("%H:%M"), np.round(np.min(asti_wind_speed), 2)))

plt.gca().xaxis.set_major_formatter(hours)
# plt.locator_params("x", nbins=5)        # 自动设置大约多少个刻度
plt.gcf().autofmt_xdate()                 # 自动旋转日期标记
plt.legend()
plt.show()


# 2.使用极区图可视化风向
def showRoseWind(values, city_name, max_value):
    N = 8
    theta = np.arange(0., 2 * np.pi, 2 * np.pi / N)
    radii = np.array(values)
    plt.figure(figsize=(10, 8))
    # 绘制极区图的坐标系
    plt.axes(polar=True)
    # 列表中包含的是每一个扇区的 rgb 值，x越大，对应的color越接近蓝色
    colors = [(1-x/max_value, 1-x/max_value, 0.75) for x in radii]
    # 画出每个扇区
    plt.bar(theta, radii, width=(2*np.pi/N), bottom=0.0, color=colors)
    # 设置极区图的标题
    plt.title(city_name, x=0.2, fontsize=20)

hist, bins = np.histogram(asti['wind_deg'], 8, [0, 360])
showRoseWind(hist, 'Ravenna', max(hist))
plt.show()


# 计算风速均值的分布情况
def RoseWind_Speed(data_city):
    # degs = [45, 90, ..., 360]
    degs = np.arange(45, 361, 45)
    tmp = []
    for deg in degs:
        # 获取 wind_deg 在指定范围的风速平均值数据
        tmp.append(data_city[(data_city['wind_deg'] > (deg-46)) & (data_city['wind_deg'] < deg)]
        ['wind_speed'].mean())
    return np.array(tmp)

showRoseWind(RoseWind_Speed(asti),'Ravenna',max(hist))
plt.show()

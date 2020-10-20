# -*- coding: UTF-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = './train/'
TRAIN_FILE = 'train.csv'
CLEAN_FILE = 'clean.csv'

TEST_DATA_PATH = './test/'
TEST_FILE = 'test.csv'


def get_data():
    # 读文件
    train_data = pd.read_csv(TRAIN_DATA_PATH + TRAIN_FILE)
    test_data = pd.read_csv(TEST_DATA_PATH + TEST_FILE)

    # 性别转换为离散数值
    train_data['Sex'] = pd.Categorical(train_data['Sex'])
    train_data['Sex'] = train_data.Sex.cat.codes
    test_data['Sex'] = pd.Categorical(test_data['Sex'])
    test_data['Sex'] = test_data.Sex.cat.codes

    # 年龄空缺补0
    train_data['Age'] = np.where(train_data['Age'].isnull(), 0, train_data['Age'])
    test_data['Age'] = np.where(test_data['Age'].isnull(), 0, test_data['Age'])

    # 登船港口转换为离散数值
    train_data['Embarked'] = np.where(train_data['Embarked'].isnull(), 0, train_data['Embarked'])
    test_data['Embarked'] = np.where(test_data['Embarked'].isnull(), 0, test_data['Embarked'])
    train_data['Embarked'] = pd.Categorical(train_data['Embarked'])
    train_data['Embarked'] = train_data.Embarked.cat.codes
    test_data['Embarked'] = pd.Categorical(test_data['Embarked'])
    test_data['Embarked'] = test_data.Embarked.cat.codes

    # 票价空缺补0并归一化票价
    train_data['Fare'] = np.where(train_data['Fare'].isnull(), 0, train_data['Fare'])
    test_data['Fare'] = np.where(test_data['Fare'].isnull(), 0, test_data['Fare'])

    avg = train_data['Fare'].mean()
    train_data['Fare'] -= avg
    det = train_data['Fare'].var() / len(train_data['Fare'])
    train_data['Fare'] /= det

    avg = test_data['Fare'].mean()
    test_data['Fare'] -= avg
    det = test_data['Fare'].var() / len(test_data['Fare'])
    test_data['Fare'] /= det

    # 删除姓名
    train_data = train_data.drop(columns=['Name'])
    test_data = test_data.drop(columns=['Name'])

    # 删除船舱号
    train_data = train_data.drop(columns=['Cabin'])
    test_data = test_data.drop(columns=['Cabin'])

    # 删除票号
    train_data = train_data.drop(columns=['Ticket'])
    test_data = test_data.drop(columns=['Ticket'])

    # 删除乘客ID
    train_data = train_data.drop(columns=['PassengerId'])
    test_data = test_data.drop(columns=['PassengerId'])

    # 保存到CSV
    train_data.to_csv(TRAIN_DATA_PATH + CLEAN_FILE, index=False)
    test_data.to_csv(TEST_DATA_PATH + CLEAN_FILE, index=False)

    # 分开x与y
    x = train_data.iloc[:, 1:]
    y = train_data.iloc[:, 0]
    x_predict = test_data.iloc[:, 0:]

    # 六四开随机返回训练与开发集
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_dev, y_train, y_dev, x_predict


'''
    # 数据可视化
    # 支持中文
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.hist(x_train['Sex'], bins=40,  facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("频数/频率")
    # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()
'''

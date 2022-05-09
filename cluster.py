import os
import re
import time
from itertools import combinations
from sklearn.cluster import DBSCAN
import psycopg2
import numpy as np
import pandas as pd


def get_rd(attr_list, data_list):
    """
    计算冗余度
    :param attr_list:该组包括的属性名
    :param data_list: 数据列表，包含各个属性的数据
    :return: list[attr_list(list),data_list(df), rd_list(list)]
    """
    if len(attr_list) == 1:
        a = pd.value_counts(data_list) / len(data_list)
        entropy = sum(np.log2(a) * a * (-1))
        rd = np.log2(len(data_list)) - entropy
        if rd < 0.0005:
            rd = 0
        return rd
    else:
        rd_sum = 0
        for attribute in attr_list:
            # 冗余度计算：log2（n）-entropy
            a = pd.value_counts(data_list[attribute]) / len(data_list[attribute])
            entropy = sum(np.log2(a) * a * (-1))
            rd = np.log2(len(data_list[attribute])) - entropy
            if rd < 0.0005:
                rd = 0
            rd_sum = rd_sum + rd
        rd = rd_sum / len(attr_list)
        return rd


def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）为找key服务'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


def get_attribute(database, user, password, port, table_name):
    """
    get data from database
    attribute: 要分解的表名
    :return: list[attr_list(list),data_list(df)]
    """
    connect = psycopg2.connect(database=database,
                               user=user,
                               password=password,
                               port=port
                               )
    # 创建一个cursor来执行数据库的操作
    cur = connect.cursor()
    all_list = []
    sql_data = "SELECT * FROM " + table_name
    sql_attributes = "SELECT a.attname FROM pg_attribute a, pg_class c where c.relname='" + table_name + "' and " \
                                                                                                         "a.attrelid" \
                                                                                                         "=c.oid " \
                                                                                                         "and " \
                                                                                                         "a.attnum>0 "
    data = pd.read_sql(sql_data, con=connect)
    attributes = pd.read_sql(sql_attributes, con=connect)
    all_list.append(attributes['attname'].tolist())
    all_list.append(data)
    cur.close()
    return all_list


def merge_attr(left_group, right_group, origin_data):
    """merge when merge successfully"""
    merge_attr = left_group.attr_list + right_group.attr_list
    merge_data = origin_data[merge_attr].drop_duplicates()
    merge = AttrGroup(merge_attr, merge_data)
    return merge.rd

def merge(x, y, group_list, origin_data):
    merge_attr = list(set(group_list[x].attr_list).union(set(group_list[y].attr_list)))
    merge_data = origin_data[merge_attr].drop_duplicates()
    merge = AttrGroup(merge_attr, merge_data)
    return merge.rd


def distance(x, y, group, origin_data):
    x = x.tolist()
    y = y.tolist()
    rd_x = group[int(x[0])].rd
    rd_y = group[int(y[0])].rd
    i = int(x[0])
    j = int(y[0])
    if not i == j:
        rd_xy = merge(i, j, group, origin_data)
    else:
        rd_xy = group[i].rd
    return (rd_x + rd_y - rd_xy) / ((rd_x + rd_y)/2)


class AttrGroup:
    """
    不同分组类，每个attrgroup中的属性被组合到一个表中
    可以访问到该表的冗余度，属性列，数据分布和主键（作为其他表的外键）
    """
    attr_list = []
    key = []
    rd = 0.0
    name = ""

    def __init__(self, attr_list, data_list):
        """
        初始化group
        :param attr_list: 列表，包含的所有属性
        :param data_list: dataframe
        """
        self.attr_list = attr_list
        self.data_list = data_list
        self.set_rd()

    def set_rd(self):
        """
        计算冗余度并设置rd
        :return: void
        """
        self.rd = get_rd(attr_list=self.attr_list, data_list=self.data_list)
        # if len(self.attr_list) == 1:
        #     a = pd.value_counts(self.data_list) / len(self.data_list)
        #     entropy = sum(np.log2(a) * a * (-1))
        #     rd = np.log2(len(self.data_list)) - entropy
        #     if rd < 0.0005:
        #         rd = 0
        #     self.rd = rd
        # else:
        #     rd_sum = 0
        #     for attribute in self.attr_list:
        #         # 冗余度计算：log2（n）-entropy
        #         a = pd.value_counts(self.data_list[attribute]) / len(self.data_list[attribute])
        #         entropy = sum(np.log2(a) * a * (-1))
        #         rd = np.log2(len(self.data_list[attribute])) - entropy
        #         if rd < 0.0005:
        #             rd = 0
        #         rd_sum = rd_sum + rd
        #         self.rd = rd_sum / len(self.attr_list)

    def set_attr_list(self, attr_list, data_list):
        """
        设置该组的属性列并存储数据
        :param attr_list: 属性列表
        :param data_list: 数据列表
        :return: void
        """
        self.attr_list = attr_list
        self.data_list = data_list
        pass

    def set_key(self):
        """
        确定该组主键
        :return: boolean
        """
        if len(self.attr_list) == 1:
            self.key = self.attr_list
            return True
        key_prob = []
        for i in range(len(self.attr_list) - 1):
            key_prob.extend(combine(self.attr_list, i + 1))
        for attrs in key_prob:
            if len(self.data_list[list(attrs)].drop_duplicates()) == len(self.data_list):
                self.key = list(attrs)
                return True
        return False


# config:
table_name = "history"
user = 'Sevent'
database = 'test'
port = '5432'
password = ''

# test:

group_list = []
data = get_attribute(database=database, user=user, password=password, port=port, table_name=table_name)
for i in range(len(data[0])):
    group_list.append(AttrGroup([data[0][i]], data[1][data[0][i]]))

length = len(group_list)
list1 = []
for i in range(length):
    list1.append([i])
X = np.array(list1)


clustering = DBSCAN(eps=1, min_samples=2, metric=lambda x, y: distance(x, y, group_list, data[1])).fit(X)
print(clustering.labels_)

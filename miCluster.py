import os
import re
import time
from itertools import combinations
from typing import List

from sklearn.cluster import DBSCAN
import psycopg2
import numpy as np
import pandas as pd
import pdb


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


def get_score(group_list, origin_data):
    sum_entropy = 0.0
    sum_group = set()
    attr_list = []
    for group in group_list:
        attr_list.append(group.attr_list)
        sum_entropy = sum_entropy + group.entropy
        sum_group = sum_group.union(set(group.attr_list))
    n = len(group_list)
    if sum_entropy == 0:
        mi = 0
    else:
        integrate_entropy = get_entropy(origin_data[list(sum_group)])
        mi = (sum_entropy - integrate_entropy) / (((n - 1) / n) * sum_entropy)
    num1 = len(origin_data[attr_list[0]].drop_duplicates())
    num2 = len(origin_data[attr_list[1]].drop_duplicates())
    row_est = (max(num1, num2) - num1 * num2) * mi + num1 * num2
    return abs(min(row_est, len(origin_data) - max(num1, num2))) / len(origin_data)


def get_mi_respectively(group_list, origin_data):
    """计算互信息填充表格"""
    sum_entropy = 0.0
    sum_group = set()
    for group in group_list:
        sum_entropy = sum_entropy + group.entropy
        sum_group = sum_group.union(set(group.attr_list))
    n = len(group_list)
    if sum_entropy == 0:
        return 0
    integrate_entropy = get_entropy(origin_data[list(sum_group)])
    mi = (sum_entropy - integrate_entropy) / (((n - 1) / n) * sum_entropy)
    if abs(mi) < 0.0005:
        mi = 0.0
    return mi


def get_mi(group_list, origin_data):
    """计算互信息填充表格"""
    sum_entropy = 0.0
    sum_group = set()
    for group in group_list:
        sum_group = sum_group.union(set(group.attr_list))
    sum_group = list(sum_group)
    n = len(sum_group)
    for attr in sum_group:
        sum_entropy = sum_entropy + get_entropy(origin_data[attr])
    if sum_entropy == 0.0:
        return 0.0
    integrate_entropy = get_entropy(origin_data[sum_group])
    mi = (sum_entropy - integrate_entropy) / (((n - 1) / n) * sum_entropy)
    if abs(mi) < 0.0000005:
        mi = 0.0
    return mi


def get_mi_average(group_list, mi_matrix, dic):
    """平均信息度"""
    left_num = len(group_list[0].attr_list)
    right_num = len(group_list[1].attr_list)
    sum_mi = 0.0
    for i in group_list[0].attr_list:
        for j in group_list[1].attr_list:
            sum_mi = sum_mi + mi_matrix[dic[i], dic[j]]
    return sum_mi / (left_num * right_num)


def get_mi_matrix(group_list, origin_data):
    matrix_size = len(group_list)
    if matrix_size > 1:
        mi_matrix = np.zeros(matrix_size * matrix_size).reshape(matrix_size, matrix_size)
        for k in range(matrix_size):
            for j in range(k):
                mi_matrix[k][j] = get_mi([group_list[k], group_list[j]], origin_data)

    return mi_matrix


def get_max_mi(group_list, mi_matrix, dic):
    """最大信息度"""
    max_mi = 0.0
    for i, left_attr in enumerate(group_list[0].attr_list):
        for j, right_attr in enumerate(group_list[1].attr_list):
            if mi_matrix[dic[left_attr], dic[right_attr]] > max_mi:
                max_mi = mi_matrix[dic[left_attr], dic[right_attr]]
    return max_mi
    pass


def get_total_mi(group_list, origin_data):
    total = get_entropy(origin_data[group_list])
    sum = 0.0
    for i in group_list:
        sum = sum + get_entropy(origin_data[i])
    return (sum - total)/(len(group_list) - 1)


def get_entropy(data_list):
    a = data_list.value_counts() / len(data_list)
    entropy = sum(np.log2(a) * a * (-1))
    return entropy


def combine(temp_list, n):
    """根据n获得列表中的所有可能组合（n个元素为一组）为找key服务"""
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


def auto_cluster(ans_list, group_list, origin_data):
    n = len(group_list)
    if n > 2:
        max_mi = 0
        max_k = 0
        max_j = 0
        mi_matrix = np.zeros(n * n).reshape(n, n)
        for k in range(n):
            for j in range(k):
                mi_matrix[k][j] = get_mi([group_list[k], group_list[j]], origin_data)
                if mi_matrix[k][j] > max_mi:
                    max_mi = mi_matrix[k][j]
                    max_k = k
                    max_j = j
        print(mi_matrix)
        # result = merge_group(group_list[max_k], group_list[max_j], origin_data)
        # if result[0]:
        #     group_list.pop(max_k)
        #     group_list.pop(max_j)
        #     group_list.insert(0, result[1])
        #     aim_merge(ans_list, group_list, origin_data)
        #     auto_cluster(ans_list, group_list, origin_data)


def insert_by_mi(ans_list, group_list, origin_data):
    for attr in group_list:
        max_mi = 0.0
        max_i = 0
        for i in range(len(ans_list)):
            if get_mi([attr, ans_list[i]], origin_data) > max_mi:
                max_i = i
        merge_attr = attr.attr_list + ans_list[i].attr_list
        ans_list[i] = AttrGroup(merge_attr, origin_data)


def another_auto_cluster_score(group_list, origin_data):
    """根据score进行聚类"""
    matrix_size = len(group_list)
    if matrix_size > 1:
        mi_matrix = np.ones(matrix_size * matrix_size).reshape(matrix_size, matrix_size)
        max_mi = 1000
        max_row = 0
        max_col = 0
        for k in range(matrix_size):
            for j in range(k):
                mi_matrix[k][j] = get_score([group_list[k], group_list[j]], origin_data)
                if mi_matrix[k][j] < max_mi:
                    max_mi = mi_matrix[k][j]
                    max_row = k
                    max_col = j
        # if max_mi > 0.70:
        print(group_list[max_row].attr_list, group_list[max_col].attr_list)
        print(max_mi)
        print("-----------------------------------------------------------------")

        merge_attr = group_list[max_row].attr_list + group_list[max_col].attr_list
        group_list.append(AttrGroup(merge_attr, origin_data))
        group_list.pop(max_row)
        group_list.pop(max_col)
        another_auto_cluster_score(group_list, origin_data)
        # else:
        #     ans_list.append(group_list[max_row])
        #     ans_list.append(group_list[max_col])
        #     group_list.pop(max_row)
        #     group_list.pop(max_col)
        #     another_auto_cluster(ans_list, group_list, origin_data)


def another_auto_cluster_mi(group_list, origin_data, matrix, dic):
    """根据最大互信息进行聚类"""
    matrix_size = len(group_list)
    if matrix_size > 1:
        mi_matrix = np.zeros(matrix_size * matrix_size).reshape(matrix_size, matrix_size)
        max_mi = 0
        max_row = 1
        max_col = 0
        for k in range(matrix_size):
            for j in range(k):
                mi_matrix[k][j] = get_mi_average([group_list[k], group_list[j]], matrix, dic)
                if mi_matrix[k][j] > max_mi:
                    max_mi = mi_matrix[k][j]
                    max_row = k
                    max_col = j
        # if max_mi > 0.70:
        print(group_list[max_row].attr_list, group_list[max_col].attr_list)
        print(max_mi)
        print("-----------------------------------------------------------------")

        merge_attr = group_list[max_row].attr_list + group_list[max_col].attr_list
        group_list.append(AttrGroup(merge_attr, origin_data))
        group_list.pop(max_row)
        group_list.pop(max_col)
        another_auto_cluster_mi(group_list, origin_data, matrix, dic)


class AttrGroup:
    """
    不同分组类，每个attrgroup中的属性被组合到一个表中
    可以访问到该表的冗余度，属性列，数据分布和主键（作为其他表的外键）
    """
    attr_list = []
    key = []
    entropy = 0.0
    name = ""

    def __init__(self, attr_list, origin_data):
        """
        初始化group
        :param attr_list: 列表，包含的所有属性
        :param data_list: dataframe
        """
        self.attr_list = attr_list
        self.data_list = origin_data[attr_list]
        self.set_entropy()

    def set_entropy(self):
        """
        计算冗余度并设置rd
        :return: void
        """
        self.entropy = get_entropy(self.data_list)

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

#
# class Mytest:
#     """根据分解结果构建数据表并进行相关测试等"""
#     table_list = []
#     table_name_list = []
#     key_list = []
#     flag = True
#
#     def __init__(self, config, tablelist):
#         self.table_list = tablelist
#         self.create_test_table(config)
#         key_str = ""
#         for table in self.table_list:
#             table_key_str = table.key[0]
#             for attr in table.key[1:]:
#                 key_str = key_str + ", " + attr
#             self.key_list.append(key_str)
#
#     def create_test_table(self, config):
#         connect = psycopg2.connect(database=config.database,
#                                    user=config.user,
#                                    password=config.password,
#                                    port=config.port
#                                    )
#         # 创建一个cursor来执行数据库的操作
#         cur = connect.cursor()
#         for table in self.table_list:
#             table_attr_str = table.attr_list[0]
#             for attr in table.attr_list[1:]:
#                 table_attr_str = table_attr_str + ", " + attr
#             new_table = table_attr_str.replace(", ", "_")
#             table.name = new_table
#             self.table_name_list.append(new_table)
#             table.name = new_table
#             sql_create = "CREATE TABLE IF NOT EXISTS " + new_table + " AS SELECT DISTINCT " + table_attr_str + " from " + config.tablename
#             cur.execute(sql_create)
#             connect.commit()
#         cur.close()
#
#     def test_sql(self, config, query_attrs=[], filter_attrs=[], clause=[]):
#         """找需要参与查询的分解后的表并改写sql语句。
#         依次访问各组group找其与sql涉及的属性的交集并保留，与之前的交集做对比，
#         如果都包含在之前的交集中则该group代表的表不需要参与查询，直到交集与sql涉及属性相同
#         根据key调整连接顺序"""
#         all_attrs = set(query_attrs + filter_attrs)
#         linked_attrs = set()
#         linked_tables = []
#         for attrs in self.table_list:
#             if linked_attrs == all_attrs:
#                 break
#             n = set(attrs.attr_list)
#             inter_list = all_attrs.intersection(n)
#             if not inter_list <= linked_attrs:
#                 linked_tables.append(attrs)
#                 linked_attrs = linked_attrs.union(inter_list)
#         for i in range(len(query_attrs)):
#             for j in linked_tables:
#                 if query_attrs[i] in j.attr_list:
#                     query_attrs[i] = j.name + "." + query_attrs[i]
#                     print(query_attrs[i])
#                     continue
#         for i in range(len(filter_attrs)):
#             for j in linked_tables:
#                 if filter_attrs[i] in j.attr_list:
#                     filter_attrs[i] = j.name + "." + filter_attrs[i]
#                     print(filter_attrs[i])
#         print(query_attrs, filter_attrs)
#
#         query_str = query_attrs[0]
#         for i in query_attrs[1:]:
#             query_str = query_str + "," + i
#         filter_str = filter_attrs[0]
#         for i in filter_attrs[1:]:
#             filter_str = filter_str + "," + i
#         table_str = linked_tables[0].name
#         for i in linked_tables[1:]:
#             table_str = table_str + "," + i.name
#         sql_sel = "EXPLAIN SELECT " + query_str + " FROM " + table_str + " WHERE " + filter_str + ">1"
#         print(sql_sel)
#         connect = psycopg2.connect(database=config.database,
#                                    user=config.user,
#                                    password=config.password,
#                                    port=config.port
#                                    )
#         # 创建一个cursor来执行数据库的操作
#         cur = connect.cursor()
#         cardinality = pd.read_sql(sql_sel, con=connect)
#         print(cardinality)
#         # sql = "SELECT "+query_str + from


# config:
table_name = "ol_s_item"
user = 'Sevent'
database = 'test'
port = '5432'
password = ''

# test:
group_list = []
data = get_attribute(database=database, user=user, password=password, port=port, table_name=table_name)
for i in range(len(data[0])):
    group_list.append(AttrGroup([data[0][i]], data[1]))

attrs = []
for i in group_list:
    attrs.append(i.attr_list[0])

print(attrs)
# comb = list(combinations(attrs, 2))
# comb2 = list(combinations(attrs, 3))
# max_total_mi = 0.0
# max_attrs = []
# for i in comb2:
#     temp = get_total_mi(list(i), data[1])
#     if temp > max_total_mi:
#         max_total_mi = temp
#         max_attrs = list(i)
#
# print(max_attrs)
# print(max_total_mi)
# max_total_mi = 0.0
# max_attrs = []
# for i in comb:
#     temp = get_total_mi(list(i), data[1])
#     if temp > max_total_mi:
#         max_total_mi = temp
#         max_attrs = list(i)
#
# print(max_attrs)
# print(max_total_mi)
# print(get_total_mi(['c_id', 'c_last', 'c_since'], data[1]))

# ans = [['c_d_id'], ['c_w_id'], ['c_payment_cnt'], ['c_delivery_cnt'], ['c_credit'], ['c_middle'], ['c_credit_lim']]
# for i in group_list:
#     print(len(data[1][i.attr_list].drop_duplicates()))
#     if i.attr_list in ans:
#         group_list.remove(i)
# group_list.pop(7)
# group_list.pop(0)
# an = []
# for i in ans:
# #     an.append(AttrGroup(i, data[1]))

dic = {}
for num, attr in enumerate(attrs):
    dic[attr] = num
print(dic)
t1 = time.time()
matrix = get_mi_matrix(group_list, data[1])
t3 = time.time()
print((t3-t1)*1000)
t3 = time.time()
another_auto_cluster_mi(group_list, data[1], matrix, dic)
t2 = time.time()
print("聚合用时:{}".format((t2-t3)*1000))
# print('**********************************************************************************')
# another_auto_cluster_mi(an, data[1])
# for i in ans:
#     print(i.attr_list)
#
#
# print("-------------------------------------------------------------------------------------------------------------")
#
# for group in group_list:
#     print(group.attr_list)
# print(get_total_mi(attrs, data[1]))
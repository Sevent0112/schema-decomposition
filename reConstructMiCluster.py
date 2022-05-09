from itertools import combinations
from typing import List

from sklearn.cluster import DBSCAN
import psycopg2
import numpy as np
import pandas as pd


class Information:
    def __init__(self, database: str, user: str, password: str, port: str, table_name: str):
        connect = psycopg2.connect(database=database,
                                   user=user,
                                   password=password,
                                   port=port
                                   )
        # 创建一个cursor来执行数据库的操作
        cur = connect.cursor()
        sql_data = "SELECT * FROM " + table_name
        sql_attributes = "SELECT a.attname FROM pg_attribute a, pg_class c where c.relname='" + table_name + "' and " \
                                                                                                             "a.attrelid" \
                                                                                                             "=c.oid " \
                                                                                                             "and " \
                                                                                                             "a.attnum>0 "
        self.data = pd.read_sql(sql_data, con=connect)
        attributes = pd.read_sql(sql_attributes, con=connect)
        self.attrs = set(attributes['attname'].tolist())
        cur.close()

    def get_entropy(self, attr: List[str]) -> float:
        a = self.data[attr].value_counts() / len(self.data[attr])
        entropy = sum(np.log2(a) * a * (-1))
        return entropy

    def get_mi(self, left_attrs: List[str], right_attrs: List[str]) -> float:
        sum_entropy = 0.0
        sum_group = set(left_attrs).union(set(right_attrs))
        sum_group = list(sum_group)
        n = len(sum_group)
        for attr in sum_group:
            sum_entropy = sum_entropy + self.get_entropy([attr])
        if sum_entropy == 0.0:
            return 0.0
        integrate_entropy = self.get_entropy(sum_group)
        mi = (sum_entropy - integrate_entropy) / (((n - 1) / n) * sum_entropy)
        if abs(mi) < 0.0000005:
            mi = 0.0
        return mi

    def get_average_mi(self, left_attrs: List[str], right_attrs: List[str]) -> float:
        num = len(left_attrs) + len(right_attrs)






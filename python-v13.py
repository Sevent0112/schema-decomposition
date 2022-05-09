from math import ceil
import os
import pandas as pd
import numpy as np
import scipy.stats as sc
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
import math
import pdb

error_count = 0


def entropy(prob_vector):
    """
    Computes the Shannon entropy of a probability distribution corresponding to
    a random variable"""
    return sc.entropy(prob_vector, base=2);


def to_numpy_if_not(X):
    """ Returns the numpy representation if dataframe"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return X


def size_and_counts_of_contingency_table(X, Y, return_joint_counts=False, with_cross_tab=False, contingency_table=None):
    """
    Returns the size, and the marginal counts of X, Y, and XY (optionally)"""

    if contingency_table != None:
        contingency_table = to_numpy_if_not(contingency_table)
        size = contingency_table[-1, -1]
        marginal_counts_Y = contingency_table[-1, :-1]
        marginal_counts_X = contingency_table[:-1, -1]
        if return_joint_counts:
            joint_counts = contingency_table[:-1, :-1].flatten()
    else:
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()

        X = merge_columns(X)
        Y = merge_columns(Y)

        if with_cross_tab == True:
            contingency_table = pd.crosstab(X, Y, margins=True)
            contingency_table = to_numpy_if_not(contingency_table)
            size = contingency_table[-1, -1]
            marginal_counts_Y = contingency_table[-1, :-1]
            marginal_counts_X = contingency_table[:-1, -1]
            if return_joint_counts:
                joint_counts = contingency_table[:-1, :-1].flatten()
        else:
            size = np.size(X, 0)
            marginal_counts_X = np.unique(X, return_counts=True, axis=0)[1]
            marginal_counts_Y = np.unique(Y, return_counts=True, axis=0)[1]
            if return_joint_counts:
                XY = append_two_arrays(X, Y)
                joint_counts = np.unique(XY, return_counts=True, axis=0)[1]

    if return_joint_counts:
        return size, marginal_counts_X, marginal_counts_Y, joint_counts
    else:
        return size, marginal_counts_X, marginal_counts_Y


def mutual_information_from_cross_tab(X, Y, contingency_table=None, return_marginal_entropies=False):
    """
    Computes mutual information using cross_tab from pandas. A precomputed 
    contingency table can be provided if it is available """
    size, marginal_counts_X, marginal_counts_Y, joint_counts = size_and_counts_of_contingency_table(X, Y,
                                                                                                    return_joint_counts=True,
                                                                                                    with_cross_tab=True,
                                                                                                    contingency_table=contingency_table)

    entropy_X = entropy(marginal_counts_X / size)
    entropy_Y = entropy(marginal_counts_Y / size)
    entropy_XY = entropy(joint_counts / size)
    mi = entropy_X + entropy_Y - entropy_XY

    if return_marginal_entropies:
        return mi / ((entropy_X + entropy_Y) / 2), entropy_X, entropy_Y
    else:
        return mi / ((entropy_X + entropy_Y) / 2)


def append_two_arrays(X, Z):
    """ Appends X and Z horizontally """
    if Z is None:
        return X

    if X is None:
        return Z

    if X is None and Z is None:
        raise ValueError('Both arrays cannot be None')

    return np.column_stack((X, Z))


def empirical_distribution_from_counts(counts, size=None):
    """
    Computes the empirical distribution of an attribute
    given the counts of its domain values (a.k.a distinct values)
    """
    if size == None:
        size = np.sum(counts);

    empirical_distribution = counts / size;
    assert np.isclose(np.sum(empirical_distribution), 1, rtol=1e-05, atol=1e-08,
                      equal_nan=False), "Sum of empirical distibution should be 1";
    return empirical_distribution;


def merge_columns(X):
    """ Combines multiple columns into one with resulting domain the distinct JOINT values of the input columns"""
    if isinstance(X, pd.DataFrame):
        num_columns = X.shape[1]
        if num_columns > 1:
            return X[X.columns].astype('str').agg('-'.join, axis=1)
        else:
            return X
    elif isinstance(X, np.ndarray):
        num_dim = X.ndim
        if num_dim == 2:
            return np.unique(X, return_inverse=True, axis=0)[1]
        elif num_dim == 1:
            return X
    elif isinstance(X, pd.Series):
        return X


def size_and_counts_of_attribute(X):
    """
    Returns the size, and the value counts of X"""
    X = merge_columns(X)
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        counts = X.value_counts()
        length = len(X.index)
    elif isinstance(X, np.ndarray):
        counts = np.unique(X, return_counts=True, axis=0)[1]
        length = np.size(X, 0)

    return length, counts


def empirical_statistics(X):
    """
    Returns the empirical distribution (a.k.a relative frequencies), counts,
    and size of an attribute
    """
    length, counts = size_and_counts_of_attribute(X)
    empirical_distribution = empirical_distribution_from_counts(counts)
    return empirical_distribution, len(counts), length;


def entropy_plugin(X, return_statistics=False):
    """
    The plugin estimator for Shannon entropy H(X) of an attribute X. Can optionally 
    return the domain size and length of X"""
    empiricalDistribution, domainSize, length = empirical_statistics(X);
    if return_statistics == True:
        return entropy(empiricalDistribution), domainSize, length
    else:
        return entropy(empiricalDistribution)


def mutual_information_plugin(X, Y, with_cross_tab=False, contingency_table=None, return_marginal_entropies=False):
    """
    The plugin estimator for mutual information I(X;Y) between two attribute sets X 
    and Y. It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table can be provided if it is available"""
    # pdb.set_trace()
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy();

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy();

    if with_cross_tab == True or contingency_table != None:
        return mutual_information_from_cross_tab(X, Y, contingency_table=contingency_table,
                                                 return_marginal_entropies=return_marginal_entropies);
    else:
        entropy_X = entropy_plugin(X)
        entropy_Y = entropy_plugin(Y)
        # pdb.set_trace()
        data_XY = append_two_arrays(X, Y)
        entropy_XY = entropy_plugin(data_XY)
        mi = entropy_X + entropy_Y - entropy_XY
        if ((entropy_X + entropy_Y) == 0):
            return 0

        if return_marginal_entropies:
            return mi / ((entropy_X + entropy_Y) / 2), entropy_X, entropy_Y
        else:
            return mi / ((entropy_X + entropy_Y) / 2)


# 定义计算信息熵的函数：计算Infor(D)
# 计算列
def row(x_data, y_data, x, y, table_row):
    # pdb.set_trace()
    ixy = mutual_information_plugin(x_data, y_data)
    if ixy == 0:
        return x * y
    elif ixy == 1:
        return max(x, y)
    else:
        print(ixy)
        if (ixy < 0):
            print((math.floor(y - x * y) * ixy + x * y))
        return min((math.floor(y - x * y) * ixy + x * y), table_row)


def changedata(X, Y):
    X = [str(a) + str(b) for a, b in zip(X, Y)]
    X[:] = labelencoder.fit_transform(X[:])
    X = pd.DataFrame(X)[0]
    return X


def baseline(X, Y, table):
    str = ""
    for i in X:
        str = str + " " + i[0] + ","
    str = str + Y[0]
    conn = psycopg2.connect(database="test", user="autonf_ding", password="123456", host="127.0.0.1", port="5432")
    sql = "SELECT distinct " + str + " FROM " + table
    return len(pd.read_sql(sql, con=conn))


# 分解表
def suanfa1(n, X_left, X_right, lists, tablename, table_row):
    # pdb.set_trace()
    X_left.append(lists[0])
    X_left_data = data[lists[0][0]]
    X_right.append(lists[n - 1])
    X_right_data = data[lists[n - 1][0]]
    row_X_left = lists[0][1]
    row_X_right = lists[n - 1][1]
    len_X_left = lists[0][2]
    len_X_right = lists[n - 1][2]
    if n == 2:
        distinctname = []
        if X_left[0][3] < X_right[0][3]:
            distinctname = X_left[0]
        else:
            distinctname = X_right[0]
        return X_left, X_right, ceil(row_X_left), ceil(row_X_right), len_X_left, len_X_right, distinctname
    # pdb.set_trace()
    lists = lists[1:n - 1]
    while lists:
        # pdb.set_trace()
        X = lists[0]
        X_data = data[X[0]]
        X_left_row = row(X_left_data[:], X_data[:], row_X_left, X[1], table_row)

        data_size_X_left_x = X_left_row * (len_X_left + X[2])
        X_right_row = row(X_data[:], X_right_data[:], X[1], row_X_right, table_row)
        # X_right_row = baseline(X_right,X,tablename)
        data_size_X_right_x = X_right_row * (len_X_right + X[2])
        delta_1 = data_size_X_left_x - row_X_left * len_X_left
        delta_2 = data_size_X_right_x - row_X_right * len_X_right
        if delta_1 < delta_2:
            X_left.append(X)
            X_left_data = changedata(X_left_data, X_data)
            row_X_left = X_left_row
            len_X_left = len_X_left + X[2]
        else:
            X_right.append(X)
            X_right_data = changedata(X_right_data, X_data)
            row_X_right = X_right_row
            len_X_right = len_X_right + X[2]
        lists = lists[1:len(lists)]
    distinctname = []
    num = len(X_data)
    if len(X_left) < len(X_right):
        for x in X_left:
            if x[3] < num:
                num = x[3]
                distinctname = x
    else:
        for x in X_right:
            if x[3] < num:
                num = x[3]
                distinctname = x
    return X_left, X_right, ceil(row_X_left), ceil(row_X_right), len_X_left, len_X_right, distinctname


class TreeNode():
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


# data格式[表名，页数，行数，外键,[含有的列]]
class BinTree():
    def __init__(self):
        self.root = None
        self.ls = []

    # 后序遍历得到join后的表
    def hxbl(self, node, lie, lie1):
        if (node == None):
            return None, None, None
        str1, din1, list1 = self.hxbl(node.left, lie, lie1)
        str2, din2, list2 = self.hxbl(node.right, lie, lie1)
        if (node.left == None and node.right == None):
            filePath = 'E:\postgresql-14.0-cc\data_sta.txt'
            with open(filePath, mode='a') as filename:
                filename.write(str(node.data[1]))
                filename.write('\n')  # 换行
                filename.write(str(node.data[2]))
                filename.write('\n')  # 换行
            return node.data[0], node.data[3], [[node.data[0], node.data[4]]]
        # print (din1,list1,list2)
        st = ''
        for x in range(len(list1)):
            if din1 in list1[x][1]:
                st = '(' + str1 + ' inner join ' + str2 + ' on ' + list1[x][0] + '.' + din1
                break
        for x in range(len(list2)):
            if din2 in list2[x][1]:
                st = st + ' = ' + list2[x][0] + '.' + din2 + ')' + " "
                break
        list1 = list1 + list2
        # print(st)
        return st, node.data[3], list1


import time


def zhixing(txt, cur, tablename, lie, lie1, lei, cost, totalX):
    start = time.time()
    i = 1
    l = 0
    # 用二叉树来存分解后的表，根据后序遍历得到join
    bintree = BinTree()
    bintree.root = TreeNode(['table0', 1, 60000, None, lie])
    bintree.ls.append(bintree.root)
    jieguo = []
    while (totalX):
        X_current = totalX.pop(0)
        X_data = X_current[0]
        table_row = X_current[1]
        if len(X_data) == 1:
            l = l + 1
            bintree.ls[l - 1].left = None
            bintree.ls[l - 1].right = None
            jieguo.append(X_current)
            continue;
        X_left = []
        X_right = []
        # print(X_data)
        X_data.sort(key=lambda x: x[1])
        start1 = time.time()
        X_left, X_right, row_X_left, row_X_right, len_X_left, len_X_right, dintinctname = suanfa1(len(X_data), X_left,
                                                                                                  X_right, X_data,
                                                                                                  tablename, table_row)
        end1 = time.time()
        print("suanfa1 用时：", end1 - start1)
        if (dintinctname not in X_left):
            X_left.append(dintinctname)
        # 有问提，一个属性不足以代表整个集合,因此现在的分解是有损的
        if (dintinctname not in X_right):
            X_right.append(dintinctname)
        '''if(len(X_left) < len(X_right)):
            X_left,X_right = X_right,X_left'''
        # if l == 0:
        lie1_copy = lie1[:]
        '''for x in range(len(lie)):
            for y in X_left:
                if(lie[x] == y[0]):
                    lie1[x] = i
            for y in X_right:
                if(lie[x] == y[0]):
                    lie1[x] = i + 1'''
        # print(lie1)
        # print(X_left,X_right,row_X_left,row_X_right,len_X_left,len_X_right,sep = '    ')
        # 获得页数
        page_X_left = ceil(row_X_left * (len_X_left + 23) / 8192)
        page_X_right = ceil(row_X_right * (len_X_right + 23) / 8192)
        filePath = 'E:\postgresql-14.0-cc\data_sta.txt'
        f = open(filePath, 'w')
        f.truncate()
        f.close()
        # 建表
        s = ""
        s1 = ""
        x_left = []
        x_right = []
        for x in X_left:
            s = s + x[0] + " " + lei[x[0]] + ','
            x_left.append(x[0])
        s = s[0:len(s) - 1]
        for x in X_right:
            s1 = s1 + x[0] + " " + lei[x[0]] + ','
            x_right.append(x[0])
        s1 = s1[0:len(s1) - 1]
        cur.execute("create table table" + str(i) + '(' + s + ')')
        # print(page_X_left,row_X_left,page_X_right,row_X_right)
        node1 = TreeNode(['table' + str(i), page_X_left, row_X_left, dintinctname[0], x_left])
        i = i + 1
        cur.execute("create table table" + str(i) + '(' + s1 + ')')
        node2 = TreeNode(['table' + str(i), page_X_right, row_X_right, dintinctname[0], x_right])
        i = i + 1
        bintree.ls[l].left = node1
        bintree.ls[l].right = node2
        l = l + 1
        bintree.ls.append(node1)
        bintree.ls.append(node2)
        start2 = time.time()
        st, n, list0 = bintree.hxbl(bintree.ls[0], lie, lie1)
        print(list0)
        for x in list0:
            for y in range(len(lie)):
                if (lie[y] in x[1]):
                    lie1[y] = x[0][5:len(x[0])]
        # ['1', '1', '2', '2', '2', '2', '2', '2', '2', '1', '2', '2', '2', '2', '2', '2', '2', '1', '2', '1', '1']
        end2 = time.time()
        print("sql语言生成用时：", end2 - start2)
        # print(st)
        # print("SELECT '学历', '婚否', '是否有车'  FROM " + st + "WHERE 学历 = '2'    AND 婚否 = '2'    AND 是否有车 = '1085'")
        '''出现column是ambiguous的  这种报错'''
        txt1 = txt.split("FROM")
        txt1[1] = txt1[1].split("WHERE")[1]
        for x in range(len(lie)):
            txt1[1] = txt1[1].replace("customer." + lie[x], "table" + str(lie1[x]) + "." + lie[x])
            txt1[0] = txt1[0].replace("customer." + lie[x], "table" + str(lie1[x]) + "." + lie[x])
        print(txt1[0] + " FROM " + st + " WHERE " + txt1[1])
        cur.execute(txt1[0] + " FROM " + st + " WHERE " + txt1[1])
        # 获得cost_current
        filePath = 'E:\postgresql-14.0-cc\data\pg_log'
        filePath = filePath + '\\' + os.listdir(filePath)[-2]
        with open(filePath) as csvfile:
            mLines = csvfile.readlines()
        cost_current = float(mLines[-1].strip('\n'))
        # print("cost_current = ",cost_current)
        if cost_current > cost:
            jieguo.append(X_current)
            i = i - 1
            cur.execute("drop table table" + str(i))
            i = i - 1
            cur.execute("drop table table" + str(i))
            bintree.ls[l - 1].left = None
            bintree.ls[l - 1].right = None
            bintree.ls.pop()
            bintree.ls.pop()
            lie1 = lie1_copy[:]
            continue
        else:
            totalX.append([X_left, row_X_left])
            totalX.append([X_right, row_X_right])
            cost = cost_current
    for x in range(1, i):
        cur.execute("drop table table" + str(x))
    end = time.time()
    print("一条sql总用时：", end - start)
    return jieguo, cost


import psycopg2

conn = psycopg2.connect(database="test", user="Sevent", password="", host="127.0.0.1", port="5432")
error_count = 0
# 格式[[列名，不重复数据的num，类型长度，distinct大小排名]]
sql = "SELECT  * FROM customer"
cur = conn.cursor()
cur.execute("set enable_bitmapscan = off;")
cur.execute("set enable_hashagg = off;")
cur.execute("set enable_hashjoin = off;")
cur.execute("set enable_indexscan = off;")
cur.execute("set enable_indexonlyscan = off;")
cur.execute("set enable_material = off;")
cur.execute("set enable_mergejoin = off;")
cur.execute("set enable_nestloop = off;")
cur.execute("set enable_parallel_append = off;")
cur.execute("set enable_seqscan = on;")
cur.execute("set enable_sort = off;")
cur.execute("set enable_tidscan = off;")
cur.execute("set enable_partitionwise_join = off;")
cur.execute("set enable_partitionwise_aggregate = off;")
cur.execute("set enable_parallel_hash = off;")
cur.execute("set enable_partition_pruning = off;")
data = pd.read_sql(sql, con=conn)
tablename = "customer"
X = [['c_w_id', 2, 4, 18], ['c_d_id', 10, 4, 16], ['c_id', 3000, 4, 11], ['c_discount', 4991, 5, 10],
     ['c_credit', 2, 3, 19], ['c_last', 1000, 13, 13], ['c_first', 60000, 11, 1], ['c_credit_lim', 1, 5, 20],
     ['c_balance', 11157, 6, 7], ['c_ytd_payment', 4664, 8, 9], ['c_payment_cnt', 12, 4, 15],
     ['c_delivery_cnt', 4, 4, 17], ['c_street_1', 60000, 15, 2], ['c_street_2', 60000, 15, 3], ['c_city', 60000, 14, 4],
     ['c_state', 676, 3, 14], ['c_zip', 9778, 10, 8], ['c_phone', 60000, 17, 5], ['c_since', 1037, 8, 12],
     ['c_middle', 1, 3, 21], ['c_data', 60000, 403, 6]]
table_row = 60000
totalX = [[X, table_row]]
txt = [
    "SELECT customer.c_discount, customer.c_last, customer.c_credit  FROM customer WHERE customer.c_w_id = 2    AND customer.c_d_id = 2    AND customer.c_id = 1085",
    "SELECT customer.c_first, customer.c_middle, customer.c_id, customer.c_street_1, customer.c_street_2, customer.c_city,  customer.c_state, customer.c_zip, customer.c_phone, customer.c_credit, customer.c_credit_lim, customer.c_discount, customer.c_balance, customer.c_ytd_payment, customer.c_payment_cnt, customer.c_since   FROM customer WHERE customer.c_w_id = 2    AND customer.c_d_id = 3    AND customer.c_last = 'OUGHTESEATION'",
    "SELECT customer.c_data   FROM customer WHERE customer.c_w_id = 2    AND customer.c_d_id = 7    AND customer.c_id = 735"
    ]
lie = ['c_w_id', 'c_d_id', 'c_id', 'c_discount', 'c_last', 'c_first', 'c_balance', 'c_ytd_payment', 'c_payment_cnt',
       'c_delivery_cnt', 'c_street_1', 'c_street_2', 'c_city', 'c_state', 'c_zip', 'c_phone', 'c_since', 'c_middle',
       'c_data', 'c_credit_lim', 'c_credit']
lie1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lei = {'c_w_id': 'INT NOT NULL', 'c_d_id': 'INT NOT NULL', 'c_id': 'INT NOT NULL',
       'c_discount': 'DECIMAL(4,4) NOT NULL', 'c_credit': 'CHAR(2) NOT NULL', 'c_last': 'VARCHAR(16) NOT NULL',
       'c_first': 'VARCHAR(16) NOT NULL', 'c_credit_lim': 'DECIMAL(12,2) NOT NULL',
       'c_balance': 'DECIMAL(12,2) NOT NULL', 'c_ytd_payment': 'FLOAT NOT NULL', 'c_payment_cnt': 'INT NOT NULL',
       'c_delivery_cnt': 'INT NOT NULL', 'c_street_1': 'VARCHAR(20) NOT NULL', 'c_street_2': 'VARCHAR(20) NOT NULL',
       'c_city': 'VARCHAR(20) NOT NULL', 'c_state': 'CHAR(2) NOT NULL', 'c_zip': 'CHAR(9) NOT NULL',
       'c_phone': 'CHAR(16) NOT NULL', 'c_since': 'TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP',
       'c_middle': 'CHAR(2) NOT NULL', 'c_data': 'VARCHAR(500) NOT NULL'}
# 取得cost
for x in lie:
    data[x][:] = labelencoder.fit_transform(data[x][:])
    if (isinstance(data[x][0], int)):
        data[x] = data[x].astype("int64")
cost = float("inf")
for t in txt:
    totalx, costx = zhixing(t, cur, tablename, lie[:], lie1[:], lei, cost, totalX[:])
    print("分解出的表：", totalx)
    print("时间：", costx)
conn.close()

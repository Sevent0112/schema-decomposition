from math import ceil
import os
import pandas as pd
import numpy as np
import scipy.stats as sc
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
import math
import pdb
import psycopg2
conn = psycopg2.connect(database="test", user="autonf_ding", password="123456", host="127.0.0.1", port="5432")

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
        size = np.sum(counts)

    empirical_distribution = counts / size
    assert np.isclose(np.sum(empirical_distribution), 1, rtol=1e-05, atol=1e-08,
                      equal_nan=False), "Sum of empirical distibution should be 1"
    return empirical_distribution


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
    return empirical_distribution, len(counts), length


def entropy_plugin(X, return_statistics=False):
    """
    The plugin estimator for Shannon entropy H(X) of an attribute X. Can optionally
    return the domain size and length of X"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    empiricalDistribution, domainSize, length = empirical_statistics(X)
    if return_statistics == True:
        return entropy(empiricalDistribution), domainSize, length
    else:
        return entropy(empiricalDistribution)


def mutual_information_plugin(X, Y, with_cross_tab=False, contingency_table=None, return_marginal_entropies=False):
    """
    The plugin estimator for mutual information I(X;Y) between two attribute sets X
    and Y. It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table can be provided if it is available"""
    pdb.set_trace()
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if with_cross_tab == True or contingency_table != None:
        return mutual_information_from_cross_tab(X, Y, contingency_table=contingency_table,
                                                 return_marginal_entropies=return_marginal_entropies)
    else:
        entropy_X = entropy_plugin(X)
        entropy_Y = entropy_plugin(Y)
        data_XY = append_two_arrays(X, Y)
        entropy_XY = entropy_plugin(data_XY)
        mi = entropy_X + entropy_Y - entropy_XY

        if return_marginal_entropies:
            return mi / ((entropy_X + entropy_Y) / 2), entropy_X, entropy_Y
        else:
            return mi / ((entropy_X + entropy_Y) / 2)
#pdb.set_trace()
sql="select * from customer"
#sql = "SELECT  * FROM (select distinct c_state, c_last, c_since, c_id, c_discount, c_ytd_payment, c_zip, c_balance, c_first, c_street_1, c_street_2, c_city, c_phone, c_data from customer) as fool"
#sql = "SELECT  * FROM (select distinct c_state, c_last, c_since, c_id, c_discount, c_ytd_payment, c_zip, c_balance from customer) as fool"
#sql = "SELECT  * FROM (select distinct c_d_id, c_payment_cnt from customer) as fool"
sql_sta = "select attname,n_distinct,avg_width from pg_stats where tablename ='customer'"

data = pd.read_sql(sql, con=conn)
data_sta=pd.read_sql(sql_sta,con=conn)
tablename = "customer"
#X = [['c_w_id',2,4,18],['c_d_id',10,4,16], ['c_id',3000,4,11],['c_discount',5000,5,10],['c_credit',2,3,19],['c_last',1000,13,13], ['c_first',60000,11,1],['c_credit_lim',1,5,20],['c_balance',14738,5,7],['c_ytd_payment',6738,8,9],['c_payment_cnt',11,4,15],['c_delivery_cnt',3,4,17],['c_street_1',60000,15,2], ['c_street_2',60000,15,3],['c_city',60000,14,4],['c_state',676,3,14],['c_zip',9983,10,8],['c_phone',60000,17,5],['c_since',1049,8,12],['c_middle',1,3,21], ['c_data',60000,403,6]]
table_row = 60000
#pdb.set_trace()
for i in range(len(data_sta)):
    #print(data_sta['n_distinct'][i])
    if(data_sta.loc[i,'n_distinct']<0):
        data_sta.loc[i,'n_distinct']=abs(data_sta.loc[i,'n_distinct']*table_row)
#pdb.set_trace()
data_sta=data_sta.sort_values(by='n_distinct')
data_sta.reset_index(drop=True, inplace=True)
#X.sort(key = lambda x: x[1])
#lie=[x[0] for x in X]
#len=[x[2] for x in X]
#s=sum(len)
for x in data_sta['attname']:
    data[x][:] = labelencoder.fit_transform(data[x][:])
    if(isinstance(data[x][0],int)):
        data[x] = data[x].astype("int64")

pdb.set_trace()
data_xy=data[data_sta.loc[0,'attname']]
for i in range(20):
    entropy_x = entropy_plugin(data_xy)
    redundancy = np.log2(table_row) - 1/(i+1)*entropy_x
    print(redundancy)
    data_xy=append_two_arrays(data_xy, data[data_sta.loc[i+1,'attname']])


pdb.set_trace()
entropy_single=[]
redundancy_single=[]
for i in range(21):
    #print(data_sta.loc[i,'attname'])
    entropy_x=entropy_plugin(data[data_sta.loc[i,'attname']])
    redundancy = np.log2(table_row) -entropy_x
    #print(redundancy)
    entropy_single.append(entropy_x)
    redundancy_single.append(redundancy)
pdb.set_trace()
print("loop 1")
data_XY = append_two_arrays(data[data_sta.loc[0,'attname']], data[data_sta.loc[1,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[2,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/3*entropy_xy
#print(redundancy)
print("loop 2")
data_XY = append_two_arrays(data[data_sta.loc[0, 'attname']], data[data_sta.loc[2, 'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/2*entropy_xy
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[3,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/3*entropy_xy
print(redundancy)

print("loop 3")
data_XY = append_two_arrays(data[data_sta.loc[2,'attname']], data[data_sta.loc[3,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/2*entropy_xy
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/3*entropy_xy
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[5,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/4*entropy_xy
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[6,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/5*entropy_xy
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[7,'attname']])
entropy_xy=entropy_plugin(data_XY)
redundancy = np.log2(table_row) -1/6*entropy_xy
print(redundancy)

pdb.set_trace()

entropy_x=entropy_plugin(data[data_sta.loc[0,'attname']])
redundancy=np.log2(table_row)-1/1*entropy_x
print("0:"+str(redundancy))

data_XY = append_two_arrays(data[data_sta.loc[0,'attname']], data[data_sta.loc[1,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/2*entropy_XY
print("0,1:"+str(redundancy))

print("loop 2")
entropy_x=entropy_plugin(data[data_sta.loc[2,'attname']])
redundancy=np.log2(table_row)-1/1*entropy_x
print("2:"+str(redundancy))

data_XY = append_two_arrays(data[data_sta.loc[2,'attname']], data[data_sta.loc[3,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/2*entropy_XY
print("2,3:"+str(redundancy))

data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/3*entropy_XY
print("2,3,4:"+str(redundancy))


data_XY = append_two_arrays(data_XY, data[data_sta.loc[2,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[3,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[5,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)
pdb.set_trace()
data_XY = append_two_arrays(data_XY, data[data_sta.loc[6,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[7,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[8,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[9,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[10,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[11,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[12,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data_XY, data[data_sta.loc[13,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/table_row*entropy_XY
print(redundancy)

pdb.set_trace()
data_XY = append_two_arrays(data[data_sta.loc[2,'attname']], data[data_sta.loc[3,'attname']])

redundancy=np.log2(table_row)-1/2*entropy_XY
print(redundancy)
data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/3*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data[data_sta.loc[0,'attname']], data[data_sta.loc[2,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[3,'attname']])


data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[5,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[6,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[7,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[8,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[9,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[10,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[11,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[12,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[13,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[14,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[15,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[16,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[17,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[18,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[19,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[20,'attname']])
pdb.set_trace()
entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/20*entropy_XY
print(redundancy)

data_XY = append_two_arrays(data[data_sta.loc[0,'attname']], data[data_sta.loc[2,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[3,'attname']])
data_XY = append_two_arrays(data_XY, data[data_sta.loc[4,'attname']])

entropy_XY = entropy_plugin(data_XY)
redundancy=np.log2(table_row)-1/5*entropy_XY
#print(redundancy)
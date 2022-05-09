import re
import psycopg2
import pandas as pd
from typing import List, Set


class Table:
    def __init__(self, name: str, attrs: Set[str], pks: Set[str]):
        self.name = name
        self.attrs = attrs
        self.pks = pks


class Sql:
    def __init__(self, sql: str, tables: List[Table]):
        self.sql = sql
        self.tables = tables
        self.query_attrs, self.filters, self.filter_attrs, self.order_attr = self.get_needed_attrs()
        self.attrs_so_far = set()
        self.ordered_tables = self.get_ordered_tables()

    def get_ordered_tables(self) -> List[Table]:
        """Find a join path """
        unordered_tables = self.tables
        ordered_tables = []
        join_condition = ''
        for i, t_i in enumerate(unordered_tables):
            for j, t_j in enumerate(unordered_tables):
                if i != j and t_j.pks.issubset(t_i.attrs):
                    ordered_tables.append(t_i)
                    self.attrs_so_far = self.attrs_so_far.union(t_i.attrs)
                    ordered_tables.append(t_j)
                    self.attrs_so_far = self.attrs_so_far.union(t_j.attrs)
                    unordered_tables.pop(max(i, j))
                    unordered_tables.pop(min(i, j))
                    break
            if len(ordered_tables) > 0:
                break
        while len(unordered_tables):
            for i, unordered_table in enumerate(unordered_tables):
                if unordered_table.pks.issubset(self.attrs_so_far):
                    ordered_tables.append(unordered_table)
                    self.attrs_so_far = self.attrs_so_far.union(unordered_table.attrs)
                    unordered_tables.pop(i)
        return ordered_tables

    def get_join_condition(self) -> str:
        """Get the join condition of the needed tables."""
        if len(self.ordered_tables) == 1:
            return ""
        init_pk = self.ordered_tables[1].pks.pop()
        join_condition = "{}.{}={}.{}".format(self.ordered_tables[0].name, init_pk, self.ordered_tables[1].name,
                                              init_pk)
        for i, t in enumerate(self.ordered_tables):
            for pk in t.pks:
                for pre_t in self.ordered_tables[:i]:
                    if pk in pre_t.attrs:
                        join_condition += " AND {}.{}={}.{}".format(pre_t.name, pk, t.name, pk)

        return join_condition

    def get_linked_tables(self, query_attrs: List[Table], filter_attrs: List[Table], order_attr: str) -> List[Table]:
        """Get the needed tables"""
        all_attrs = set(query_attrs + filter_attrs + [order_attr])
        linked_attrs = set()
        linked_tables = []
        for table in self.tables:
            if linked_attrs == all_attrs:
                break
            n = table.attrs
            inter_list = all_attrs.intersection(n)
            if not inter_list <= linked_attrs:
                linked_tables.append(table)
                linked_attrs = linked_attrs.union(inter_list)
        return linked_tables

    def get_from(self):
        from_ = "{}".format(self.ordered_tables[0].name)
        for i in self.ordered_tables[1:]:
            from_ += ",{}".format(i.name)
        return from_

    def get_needed_attrs(self):
        """Get some needed information from the input sql"""
        '^ (SELECT\s+(. +?)(\s+)(FROM\s +)(.+?)(\s +)(WHERE\s+)(. +?)(\s+ORDER BY\s+.+?; |;)$'
        sql_reg = re.compile(r'^(SELECT\s+)(.*)(\s+FROM\s+)(.*)(\s+)(WHERE\s+)(.*)(\s+ORDER BY\s+.+?;|;)$')
        attrs_pre = re.match(sql_reg, self.sql).groups()
        query_attrs = re.split(r'[\s,]+', attrs_pre[1])
        filters = []
        filter_attrs = []
        filters_pre = re.split(r'\s+AND\s+', attrs_pre[6])
        filter_reg = re.compile(r'^(.+?)(\s*[>|>=|<|<=|=|!=]\s*)(.+)$')
        for filter_pre in filters_pre:
            filters.append(
                [re.match(filter_reg, filter_pre).groups()[0], re.match(filter_reg, filter_pre).groups()[1],
                 re.match(filter_reg, filter_pre).groups()[2]])
            filter_attrs.append(re.match(filter_reg, filter_pre).groups()[0].strip())
            filter_attrs.append(re.match(filter_reg, filter_pre).groups()[2].strip())
        if attrs_pre[7] != ';':
            order_attr = re.compile(r'(\s+ORDER BY\s+)(.+?)(;)').match(attrs_pre[7]).groups()[1]
        else:
            order_attr = None
        return query_attrs, filters, filter_attrs, order_attr

    def rewrite_query_attrs(self) -> str:
        re_query_attrs = []
        for query_attr in self.query_attrs:
            for table in self.ordered_tables:
                if query_attr in table.attrs:
                    re_query_attrs.append("{}.{}".format(table.name, query_attr))
                    break
        select = re_query_attrs[0]
        for attr in re_query_attrs[1:]:
            select += ",{}".format(attr)

        return select

    def rewrite_filter_condition(self) -> str:
        if len(self.filter_attrs) == 0:
            return ''
        idx = {}  # attr(int)-Table键值对
        # 构建attr-table索引
        for attr in self.filter_attrs:
            table = idx.get(attr)
            if not table:
                for t in self.ordered_tables:
                    if attr in t.attrs:
                        idx[attr] = t
                        break
        condition = ""
        if len(self.ordered_tables) > 1:
            for i in range(len(self.filters)):
                condition += " AND {}.{}{}{}.{}".format(idx[self.filters[i][0]].name, self.filters[i][0],
                                                        self.filters[i][1],
                                                        idx[self.filters[i][2]].name, self.filters[i][2])
        else:
            condition = "{}.{}{}{}.{}".format(idx[self.filters[0][0]].name, self.filters[0][0], self.filters[0][1],
                                              idx[self.filters[0][2]].name,
                                              self.filters[0][2])
            for i in range(1, len(self.filters)):
                condition += " and {}.{}{}{}.{}".format(idx[self.filters[i][0]].name, self.filters[i][0],
                                                        self.filters[i][1],
                                                        idx[self.filters[i][2]].name, self.filters[i][2])
        return condition

    def get_order(self):
        if self.order_attr:
            for t in self.ordered_tables:
                if self.order_attr in t.attrs:
                    return "{}.{}".format(t.name, self.order_attr)
        return ''

    def get_rewrite_sql(self) -> str:
        select = self.rewrite_query_attrs()
        from_ = self.get_from()
        filters = self.rewrite_filter_condition()
        join_condition = self.get_join_condition()
        rewrite_sql = "EXPLAIN SELECT " + select + " FROM " + from_ + " WHERE " + join_condition + filters
        if self.order_attr is not None:
            rewrite_sql = rewrite_sql + " ORDER BY " + self.order_attr
        rewrite_sql = rewrite_sql + ";"
        return rewrite_sql


def test():
    names = ["table1", "table2", "table3", "table4", "table5"]
    tables = []
    attrs = [['c_credit', 'c_balance', 'c_last', 'c_state', 'c_since', 'c_id', 'c_discount', 'c_zip', 'c_city',
              'c_data', 'c_first', 'c_street_1', 'c_phone', 'c_street_2'],
             ['c_credit', 'c_w_id', 'c_balance', 'c_d_id', 'c_last', 'c_state', 'c_since'],
             ['c_credit', 'c_payment_cnt', 'c_w_id', 'c_balance', 'c_d_id'],
             ['c_credit', 'c_payment_cnt', 'c_delivery_cnt', 'c_w_id', 'c_ytd_payment', 'c_balance'],
             ['c_credit_lim', 'c_middle', 'c_credit', 'c_payment_cnt', 'c_delivery_cnt', 'c_w_id']]
    pks = [['c_city'],
           ['c_credit', 'c_balance', 'c_last', 'c_state', 'c_since'],
           ['c_credit', 'c_w_id', 'c_balance', 'c_d_id'],
           ['c_credit', 'c_payment_cnt', 'c_w_id', 'c_balance'],
           ['c_credit', 'c_payment_cnt', 'c_delivery_cnt', 'c_w_id']]
    for i in range(5):
        tables.append(Table(names[i], set(attrs[i]), set(pks[i])))
    for i in tables:
        print(i.name)
        print(i.attrs)
        print(i.pks)
    my_sql = "SELECT c_credit,c_balance FROM customer WHERE c_payment_cnt=c_since;"
    sql = Sql(my_sql, tables)
    print(sql.get_rewrite_sql())


test()

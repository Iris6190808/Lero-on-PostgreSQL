class CardPicker():
    def __init__(self, rows_arr, table_arr,
                 swing_factor_lower_bound=0.1**2, swing_factor_upper_bound=10**2, swing_factor_step=10) -> None:
        self.rows_arr = rows_arr
        self.table_arr = table_arr

        # Mapping the number of table in sub-query to the index of its cardinality in rows_arr
        #table_arr索引等同rows_arr索引
        self.table_num_2_card_idx_dict = {}
        for i, tables in enumerate(self.table_arr):
            table_num = len(tables)
            if table_num not in self.table_num_2_card_idx_dict:
                self.table_num_2_card_idx_dict[table_num] = []
            self.table_num_2_card_idx_dict[table_num].append(i)
        # 获取子查询中表数最多的table_arr索引
        self.max_table_num = max(self.table_num_2_card_idx_dict.keys())

        # Each time we will adjust all the cardinalities of sub-queries in the same group (group by the number of table here)
        # And adjust according to the number of tables from more to less, 
        # because the more complex the execution plan is, the more likely it is to produce wrong estimates on cardinality
        # 每次我们都会调整同一组子查询的所有基数(按表的数量分组)，并根据表的数量从多到少进行调整，因为执行计划越复杂，就越有可能对基数产生错误的估计
        self.cur_sub_query_table_num = self.max_table_num
        self.cur_sub_query_related_card_idx_list = self.table_num_2_card_idx_dict[self.cur_sub_query_table_num]

        # create the swing factor list
        #摆动因子
        assert swing_factor_lower_bound < swing_factor_upper_bound
        self.swing_factor_lower_bound = swing_factor_lower_bound
        self.swing_factor_upper_bound = swing_factor_upper_bound
        self.step = swing_factor_step
        self.sub_query_swing_factor_index = 0

        self.swing_factors = set()
        cur_swing_factor = 1
        while cur_swing_factor <= self.swing_factor_upper_bound:
            self.swing_factors.add(cur_swing_factor)
            cur_swing_factor *= self.step
        self.swing_factors.add(self.swing_factor_upper_bound)

        cur_swing_factor = 1
        while cur_swing_factor >= self.swing_factor_lower_bound:
            self.swing_factors.add(cur_swing_factor)
            cur_swing_factor /= self.step
        self.swing_factors.add(self.swing_factor_lower_bound)
        self.swing_factors = list(self.swing_factors)

        # indicates whether all combinations have been tried
        self.finish = False

    def get_card_list(self):
        #如果当前子查询相关的基数索引列表为空，则从字典中获取对应数量表的基数索引。
        if len(self.cur_sub_query_related_card_idx_list) == 0:
            self.cur_sub_query_related_card_idx_list = self.table_num_2_card_idx_dict[self.cur_sub_query_table_num]
        #根据当前的 swing_factor，计算新的行数数组 new_rows_arr，对每个相关索引的基数进行调整。
        cur_swing_factor = self.swing_factors[self.sub_query_swing_factor_index]

        new_rows_arr = [float(item) for item in self.rows_arr]
        for join_idx in self.cur_sub_query_related_card_idx_list:
            new_rows_arr[join_idx] = float(int(new_rows_arr[join_idx] * cur_swing_factor))

        return new_rows_arr

    def next(self):
        #增加 sub_query_swing_factor_index，表示下一个基数调整因子的索引。
        self.sub_query_swing_factor_index += 1
        #如果到达因子列表的末尾，重置索引并减少当前子查询的表数量。
        if self.sub_query_swing_factor_index == len(self.swing_factors):
            self.sub_query_swing_factor_index = 0
            self.cur_sub_query_table_num -= 1
            self.cur_sub_query_related_card_idx_list = []
        #如果当前表数量小于或等于 1，则重置为最大表数量并设置 finish 为 True，表示所有组合都已经尝试过。
        if self.cur_sub_query_table_num <= 1:
            self.cur_sub_query_table_num = self.max_table_num
            self.finish = True

        return self.finish
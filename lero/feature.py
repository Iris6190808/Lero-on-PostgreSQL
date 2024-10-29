import json
#创建抽象基类
from abc import ABCMeta, abstractmethod
#数值计算
import numpy as np

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


def json_str_to_json_obj(json_data):
    json_obj = json.loads(json_data)
    if type(json_obj) == list:
        assert len(json_obj) == 1
        json_obj = json_obj[0]
        assert type(json_obj) == dict
    return json_obj


class FeatureGenerator():

    def __init__(self) -> None:
        self.normalizer = None
        self.feature_parser = None

    def fit(self, trees):
        exec_times = []
        startup_costs = []
        total_costs = []
        rows = []
        input_relations = set()
        rel_type = set()
        #recurse 是个递归函数，提取计划节点中的特征并将其加入列表或集合中。递归处理嵌套查询计划结构。
        def recurse(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            rel_type.add(n["Node Type"])
            if "Relation Name" in n:
                # base table
                input_relations.add(n["Relation Name"])

            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)
        #遍历每个查询计划 tree，将 Execution Time 加入 exec_times，并对 Plan 调用 recurse 函数提取特征。
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            recurse(json_obj["Plan"])
        #将特征值转换为 numpy 数组并取对数，确保特征值范围较小、数据分布较均匀。
        startup_costs = np.array(startup_costs)
        total_costs = np.array(total_costs)
        rows = np.array(rows)

        startup_costs = np.log(startup_costs + 1)
        total_costs = np.log(total_costs + 1)
        rows = np.log(rows + 1)
        #计算各特征的最小值和最大值，用于归一化，并打印关系类型。
        startup_costs_min = np.min(startup_costs)
        startup_costs_max = np.max(startup_costs)
        total_costs_min = np.min(total_costs)
        total_costs_max = np.max(total_costs)
        rows_min = np.min(rows)
        rows_max = np.max(rows)

        print("RelType : ", rel_type)
        #如果有执行时间数据，则归一化执行时间。否则只归一化启动成本、总成本和行数。
        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Execution Time": exec_times_max, "Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        else:
            self.normalizer = Normalizer(
                {"Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        #初始化 AnalyzeJsonParser，用于后续特征解析。
        self.feature_parser = AnalyzeJsonParser(self.normalizer, list(input_relations))

    def transform(self, trees):
        #特征
        local_features = []
        #标签
        y = []
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if type(json_obj["Plan"]) != dict:
                json_obj["Plan"] = json.loads(json_obj["Plan"])
            local_feature = self.feature_parser.extract_feature(
                json_obj["Plan"])
            local_features.append(local_feature)

            if "Execution Time" in json_obj:
                label = float(json_obj["Execution Time"])
                if self.normalizer.contains("Execution Time"):
                    label = self.normalizer.norm(label, "Execution Time")
                y.append(label)
            else:
                y.append(None)
        return local_features, y

#提供了对执行计划节点的全面表示和操作接口，支持树形结构的处理。
class SampleEntity():
    def __init__(self, node_type: np.ndarray, startup_cost: float, total_cost: float,
                 rows: float, width: int,
                 left, right,
                 startup_time: float, total_time: float,
                 input_tables: list, encoded_input_tables: list) -> None:
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.rows = rows
        self.width = width
        self.left = left
        self.right = right
        self.startup_time = startup_time
        self.total_time = total_time
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables

    def __str__(self):
        return "{%s, %s, %s, %s, %s, [%s], [%s], %s, %s, [%s], [%s]}" % (self.node_type,
                                                                        self.startup_cost, self.total_cost, self.rows,
                                                                        self.width, self.left, self.right,
                                                                        self.startup_time, self.total_time,
                                                                        self.input_tables, self.encoded_input_tables)

    def get_feature(self):
        # return np.hstack((self.node_type, np.array([self.width, self.rows])))
        return np.hstack((self.node_type, np.array(self.encoded_input_tables), np.array([self.width, self.rows])))

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def subtrees(self):
        trees = []
        trees.append(self)
        if self.left is not None:
            trees += self.left.subtrees()
        if self.right is not None:
            trees += self.right.subtrees()
        return trees


class Normalizer():
    def __init__(self, mins: dict, maxs: dict) -> None:
        self._mins = mins
        self._maxs = maxs

    def norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to normalize " + name)

        return (np.log(x + 1) - self._mins[name]) / (self._maxs[name] - self._mins[name])

    def inverse_norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to inversely normalize " + name)

        return np.exp((x * (self._maxs[name] - self._mins[name])) + self._mins[name]) - 1

    def contains(self, name):
        return name in self._mins and name in self._maxs

#方法：extract_feature：需要在子类中实现的方法，用于从输入数据中提取特征。
#抽象基类（ABC）：ABC 是一种特殊的类，它不能被实例化，只能被继承。它通常包含一个或多个抽象方法，这些方法在子类中必须实现。
#元类（metaclass）：元类是用于创建类的类。通过指定元类，你可以改变类的创建过程和行为。
#ABCMeta 是 abc 模块中提供的一个元类，允许你创建可以包含抽象方法的类。
#使用 ABCMeta 的好处
#强制实现：通过在抽象基类中定义抽象方法，子类必须实现这些方法，否则会引发错误。这可以确保所有子类都遵循同一接口。
#设计规范：使用 ABC 可以帮助开发者遵循良好的设计原则，明确哪些方法是必须实现的，增强代码的可读性和可维护性。
class FeatureParser(metaclass=ABCMeta):

    @abstractmethod
    def extract_feature(self, json_data) -> SampleEntity:
        pass


# the json file is created by "EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING, SUMMARY, FORMAT JSON) ..."
class AnalyzeJsonParser(FeatureParser):

    def __init__(self, normalizer: Normalizer, input_relations: list) -> None:
        self.normalizer = normalizer
        self.input_relations = input_relations
    #：从 JSON 格式的执行计划中提取特征并创建一个 SampleEntity 实例。
    def extract_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]))

        node_type = op_to_one_hot(json_rel['Node Type'])
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        return SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations))
    #将输入表名列表进行编码，生成一个独热编码数组。
    def encode_relation_names(self, l):
        #最后一位用于表示未知关系
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in l:
            if name not in self.input_relations:
                # -1 means UNKNOWN
                encode_arr[-1] += 1
            else:
                encode_arr[list(self.input_relations).index(name)] += 1
        return encode_arr


def op_to_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    if op_name not in OP_TYPES:
        arr[OP_TYPES.index(UNKNOWN_OP_TYPE)] = 1
    else:
        arr[OP_TYPES.index(op_name)] = 1
    return arr

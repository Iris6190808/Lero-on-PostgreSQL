import configparser
import os

from feature import JOIN_TYPES, SCAN_TYPES


def read_config():
    config = configparser.ConfigParser()
    config.read("server.conf")

    if "lero" not in config:
        print("server.conf does not have a [lero] section.")
        exit(-1)

    config = config["lero"]
    return config

def print_log(s, log_path, print_to_std_out=False):
    os.system("echo \"" + str(s) + "\" >> " + log_path)
    # 使用os.system()运行命令行指令可能会有安全隐患，特别是当s的内容来源于用户输入时，可能导致命令注入攻击。
    # 推荐使用Python内置的文件操作方法来写入日志文件，避免潜在的安全问题。例如，可以用with open(log_path, "a") as log_file: 来打开文件并写入内容。
    # with open(log_path, "a") as log_file:
    #     log_file.write(str(s) + "\n")
    if print_to_std_out:
        print(s)

# Lero guides the optimizer to generate different plans by changing cardinalities,
# but row count will be used as the input feature when predicting the plan score.
# So we need to restore all the row counts to the original values before feeding the model.
class PlanCardReplacer():
    #-> None 表示该方法的返回类型是 None，即这个方法的意图是执行初始化操作，而不需要返回任何结果。
    #在 Python 中，构造函数 __init__ 的默认行为就是不返回任何值，通常返回 None。使用 -> None 只是为了增强代码的可读性，让读者明确知道该方法不会有返回值。
    #__init__ 是一个特殊方法，用于类的构造函数。当你创建一个类的实例时，Python 会自动调用 __init__ 方法来初始化对象的属性。
    def __init__(self, table_array, rows_array) -> None:
        self.table_array = table_array
        # table_array = [
        #     ["table1", "table2"],  # 子查询1
        #     ["table3"],            # 子查询2
        #     ["table1", "table3"],  # 子查询3
        #     ["table2"]             # 子查询4
        # ]
        self.rows_array = rows_array
        # rows_array = [
        #     100,  # 子查询1
        #     50,   # 子查询2
        #     150,  # 子查询3
        #     30    # 子查询4
        # ]
        self.SCAN_TYPES = SCAN_TYPES
        self.JOIN_TYPES = JOIN_TYPES
        self.SAME_CARD_TYPES = ["Hash", "Materialize",
                                "Sort", "Incremental Sort", "Limit"]
        self.OP_TYPES = ["Aggregate", "Bitmap Index Scan"] + \
            self.SCAN_TYPES + self.JOIN_TYPES + self.SAME_CARD_TYPES
        self.table_idx_map = {}
        for arr in table_array:
            for t in arr:
                if t not in self.table_idx_map:
                    self.table_idx_map[t] = len(self.table_idx_map)
        # {
        #     "table1": 0,
        #     "table2": 1,
        #     "table3": 2
        # }
        self.table_num = len(self.table_idx_map)
        self.table_card_map = {}
        for i in range(len(table_array)):
            arr = table_array[i]
            card = rows_array[i]
            code = self.encode_input_tables(arr)
            if code in self.table_card_map:
                pass
            else:
                self.table_card_map[code] = card
        #         {
        #     "code1": 100,  # 对应表组合 ["table1", "table2"]
        #     "code2": 50,   # 对应表组合 ["table3"]
        #     "code3": 150,  # 对应表组合 ["table1", "table3"]
        #     "code4": 30    # 对应表组合 ["table2"]
        # }
    #遍历查询计划的节点，根据不同的节点类型更新其基数 (Plan Rows)。
    #它通过递归处理子计划，确保每个节点都能获得正确的基数，从而为后续的查询优化提供支持。
    def replace(self, plan):
        input_card = None
        input_tables = []
        output_card = None

        if "Plans" in plan:
            children = plan['Plans']
            child_input_tables = None
            if len(children) == 1:
                child_input_card, child_input_tables = self.replace(children[0])
                #当前节点的输入基数为子节点基数
                input_card = child_input_card
                input_tables += child_input_tables
            else:
                for child in children:
                    #元组解法，下划线_是一个常见的约定，表示这个值会被忽略。
                    _, child_input_tables = self.replace(child)
                    input_tables += child_input_tables
        #每种节点类型在处理基数时也有不同的策略
        node_type = plan['Node Type']
        #对于连接操作，输出的基数通常取决于输入表的基数和连接条件。
        if node_type in self.JOIN_TYPES:
            tag = self.encode_input_tables(input_tables)
            if tag not in self.table_card_map:
                print(input_tables)
                raise Exception("Unknown tag " + str(tag))
            #为子查询的基数如50，不等于单个子查询的基数如100和200
            card = self.table_card_map[tag]
            plan['Plan Rows'] = card
            output_card = card
        #同基数类型的节点在计划中的基数不会更改
        elif node_type in self.SAME_CARD_TYPES:
            if input_card is not None:
                plan['Plan Rows'] = input_card
                output_card = input_card
        elif node_type in self.SCAN_TYPES:
            input_tables.append(plan['Relation Name'])
        elif node_type not in self.OP_TYPES:
            raise Exception("Unknown node type " + node_type)
        #没有复杂的基数计算逻辑导致 output_card 为 None。
        return output_card, input_tables

    def encode_input_tables(self, input_table_list):
        l = [0 for _ in range(self.table_num)]
        for t in input_table_list:
            l[self.table_idx_map[t]] += 1

        code = 0
        for i in range(len(l)):
            code += l[i] * (10**i)
        return code

def get_tree_signature(json_tree):
    signature = {}
    if "Plans" in json_tree:
        children = json_tree['Plans']
        if len(children) == 1:
            signature['L'] = get_tree_signature(children[0])
        else:
            assert len(children) == 2
            signature['L'] = get_tree_signature(children[0])
            signature['R'] = get_tree_signature(children[1])

    node_type = json_tree['Node Type']
    if node_type in SCAN_TYPES:
        signature["T"] = json_tree['Relation Name']
    elif node_type in JOIN_TYPES:
        signature["J"] = node_type[0]
    return signature
    # json_tree
    # {
    # "Node Type": "Join",
    # "Plans": [
    #     {
    #         "Node Type": "Seq Scan",
    #         "Relation Name": "table1"
    #     },
    #     {
    #         "Node Type": "Seq Scan",
    #         "Relation Name": "table2"
    #     }
    # ]
    # }
#      signature
#      {
#     "L": {"T": "table1"},
#     "R": {"T": "table2"},
#     "J": "J"  # 连接类型的标识
#       }
class OptState:
    def __init__(self, card_picker, plan_card_replacer, dump_card = False) -> None:
        self.card_picker = card_picker
        self.plan_card_replacer = plan_card_replacer
        if dump_card:
            self.card_list_with_score = []
            self.visited_trees = set()
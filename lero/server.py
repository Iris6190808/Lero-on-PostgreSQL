import json
import socketserver

from card_picker import CardPicker
from model import LeroModel
from test_script.config import LERO_DUMP_CARD_FILE
from utils import (OptState, PlanCardReplacer, get_tree_signature, print_log,
                   read_config)

#定义了一个名为 LeroJSONHandler 的类，它继承自 socketserver.BaseRequestHandler，用于处理 TCP 请求中的 JSON 消息。
#子类可以通过 super() 调用父类的 __init__ 方法，确保父类属性被正确初始化。
#子类不一定需要定义 __init__ 方法，默认情况下会继承父类的 __init__ 方法。
#在类中并不一定要有 __init__，只有在需要时才定义。
class LeroJSONHandler(socketserver.BaseRequestHandler):
    #这个方法在每个请求处理之前被调用。当前实现为空，但可以用于初始化操作。
    #setup 方法是 BaseRequestHandler 类的一部分，它在每个请求处理之前自动调用。这个方法的主要目的是进行一些请求处理前的初始化操作。
    def setup(self):
        pass
    #接收数据并解析 JSON 消息。
    #引用当前实例: self 允许你在类的方法中引用调用该方法的对象。通过 self，你可以访问实例的属性和方法。
    #类的方法必须包含 self 参数: 所有实例方法（非静态方法）都需要包含 self 作为第一个参数，以便在方法内部访问实例的属性和其他方法。
    def handle(self):
        str_buf = ""
        while True:
            #self.request.recv(81960).decode("UTF-8") 从请求中接收最多 81960 字节的数据，并解码为字符串。
            str_buf += self.request.recv(81960).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            #检查数据是否包含结束标志 *LERO_END*，如果找到，将其之前的内容作为 JSON 消息进行处理。
            if (null_loc := str_buf.find("*LERO_END*")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + len("*LERO_END*"):]
                if json_msg:
                    try:
                        #处理 JSON 消息时，调用 handle_msg 方法，并处理可能出现的 JSON 解码错误。
                        self.handle_msg(json_msg)
                        break
                    except json.decoder.JSONDecodeError as e:
                        print(str(e))
                        print_log(
                            "Error decoding JSON:" + json_msg.replace("\"", "\'"), "./server.log", True)
                        break

    def handle_msg(self, json_msg):
        #解析传入的 JSON 消息，提取消息类型 msg_type 并准备响应消息 reply_msg。
        json_obj = json.loads(json_msg)
        msg_type = json_obj['msg_type']
        reply_msg = {}
        # 打印传入的json对象
        formatted_data = json.dumps(json_obj, indent=4)
        print(formatted_data)
        try:
            if msg_type == "init":
                self._init(json_obj, reply_msg)
            elif msg_type == "guided_optimization":
                self._guided_optimization(json_obj, reply_msg)
            elif msg_type == "predict":
                self._predict(json_msg, reply_msg)
            elif msg_type == "join_card":
                reply_msg['msg_type'] = "succ"
                new_card_list = self.server.opt_state_dict[json_obj['query_id']].card_picker.get_card_list()
                reply_msg['join_card'] = new_card_list
            elif msg_type == "load":
                self._load(json_obj, reply_msg)
            elif msg_type == "reset":
                self._reset(reply_msg)
            elif msg_type == "remove_state":
                self._remove_state(json_obj, reply_msg)
            else:
                print("Unknown msg type: " + msg_type)
                reply_msg['msg_type'] = "error"
        except Exception as e:
            reply_msg['msg_type'] = "error"
            reply_msg['error'] = str(e)
            print(e)
        #发送响应消息
        self.request.sendall(bytes(json.dumps(reply_msg), "utf-8"))
        self.request.close()
    #单下划线 _:用于表示“受保护的”属性或方法。这是一种约定，意味着这些属性或方法不应该在类外部直接访问。
    #用于名称修饰（name mangling），以避免与子类中的同名属性或方法冲突。Python 会将 __my_variable 转换为 _ClassName__my_variable，使其在外部不可访问。在这种情况下，__private_var 只能通过类内部的方法访问，而不能直接通过实例访问。
    def _init(self, json_obj, reply_msg):
        qid = json_obj['query_id']
        print("init query", qid)
        # card_picker: 用于选择基数的对象。
        card_picker = CardPicker(json_obj['rows_array'], json_obj['table_array'],
                                self.server.swing_factor_lower_bound, self.server.swing_factor_upper_bound, self.server.swing_factor_step)
        print(json_obj['table_array'], json_obj['rows_array'])
        # plan_card_replacer: 用于替换计划和基数的对象。
        plan_card_replacer = PlanCardReplacer(json_obj['table_array'], json_obj['rows_array'])
        #通过 dump_card 参数，可以选择是否记录基数列表及其评分，并跟踪已访问的查询计划树。
        opt_state = OptState(card_picker, plan_card_replacer, self.server.dump_card)
        self.server.opt_state_dict[qid] = opt_state
        reply_msg['msg_type'] = "succ"
    # Lero 通过更改基数来指导优化器生成不同的计划，但在预测计划分数时，行数将用作输入特征。因此，我们需要在向模型提供之前将所有行数恢复为原始值。
    def _guided_optimization(self, json_obj, reply_msg):
        qid = json_obj['query_id']
        opt_state = self.server.opt_state_dict[qid]
        plan_card_replacer = opt_state.plan_card_replacer
        #处理查询计划（plan）并将所有行数恢复为原始值
        plan_card_replacer.replace(json_obj['Plan'])
        new_json_msg = json.dumps(json_obj)
        #在 Python 中，所有的参数传递实际上是通过“对象引用”实现的。这意味着当你将一个可变对象（如列表、字典或自定义对象）作为参数传递给函数时，传递的是对象的引用，而不是对象的副本。
        #java
        #值传递：指的是参数传递的是值的副本。对于基本数据类型，传递的是其值；对于对象，传递的是对象引用的值。
        #集合类的修改：当你修改一个集合类的内容（例如添加或删除元素）时，实际上是修改了这个对象本身，因为你操作的是同一个对象的引用。因此，对象的内容会发生变化，而引用本身并没有变化。
        #在预测计划分数时，行数将用作输入特征。
        self._predict(new_json_msg, reply_msg)

        if self.server.dump_card:
            signature = str(get_tree_signature(json_obj['Plan']['Plans'][0]))
            if signature not in opt_state.visited_trees:
                # 通过更改基数来指导优化器生成不同的计划

                card_list = opt_state.card_picker.get_card_list()
                opt_state.card_list_with_score.append(([str(card) for card in card_list], reply_msg['latency']))
                opt_state.visited_trees.add(signature)

        finish = opt_state.card_picker.next()
        reply_msg['finish'] = 1 if finish else 0

    # 调用模型进行预测，并将结果存储在响应中。
    def _predict(self, json_msg, reply_msg):
        if self.server.model is not None:
            local_features, _ = self.server.feature_generator.transform([json_msg])
            y = self.server.model.predict(local_features)
            #返回一个单一的预测值。
            assert y.shape == (1, 1)
            y = y[0][0]
        #模型未被加载或初始化
        else:
            y = 1

        reply_msg['msg_type'] = "succ"
        reply_msg['latency'] = y
    #加载新的 Lero 模型，并更新服务器的模型和特征生成器。
    def _load(self, json_obj, reply_msg):
        print("load new Lero model")
        model_path = json_obj['model_path']
        lero_model = LeroModel(None)
        lero_model.load(model_path)
        self.server.model = lero_model
        self.server.feature_generator = lero_model._feature_generator
        reply_msg['msg_type'] = "succ"
    #清空服务器的模型和特征生成器。
    def _reset(self, reply_msg):
        print("reset")
        self.server.model = None
        self.server.feature_generator = None
        reply_msg['msg_type'] = "succ"
    #删除特定查询的优化器状态，如果需要转储基数打分列表。
    def _remove_state(self, json_obj, reply_msg):
        qid = json_obj['query_id']
        if self.server.dump_card:
            print("dump cardinalities and plan scores of query:", qid)
            self._dump_card_with_score(self.server.opt_state_dict[qid].card_list_with_score)

        del self.server.opt_state_dict[qid]
        reply_msg['msg_type'] = "succ"
        print("remove state: qid =", qid)
    #将基数及其评分写入指定文件。
    def _dump_card_with_score(self, card_list_with_score):
        with open(self.server.dump_card_with_score_path, "w") as f:
            w_str = [" ".join(cards) + ";" + str(score)
                     for (cards, score) in card_list_with_score]
            w_str = "\n".join(w_str)
            f.write(w_str)


def start_server(listen_on, port, model: LeroModel):
    #with 语句用于确保服务器在使用完毕后正确关闭，避免资源泄漏。
    #socketserver.TCPServer 是一个简单的 TCP 服务器类，接收客户端的连接请求并调用指定的处理器来处理请求。
    #server 是通过 socketserver.TCPServer 创建的 TCP 服务器实例，它负责管理网络连接和请求。
    #当有客户端连接时，server 会根据定义的处理器（在这个例子中是 LeroJSONHandler）创建一个 LeroJSONHandler 的实例来处理该请求。
    #在这个过程中，所有的服务器属性设置在调用 LeroJSONHandler 之前完成。
    #LeroJSONHandler 是在有客户端连接时被创建的处理器，而属性设置是为了确保当处理请求时，server 已经具有必要的配置和状态。这种顺序确保了处理请求时可以访问到正确的服务器上下文和状态。
    with socketserver.TCPServer((listen_on, port), LeroJSONHandler) as server:
        server.model = model
        server.feature_generator = model._feature_generator if model is not None else None
        server.opt_state_dict = {}

        server.best_plan = None
        server.best_score = None

        server.swing_factor_lower_bound = 0.1**2
        server.swing_factor_upper_bound = 10**2
        server.swing_factor_step = 10
        print("swing_factor_lower_bound", server.swing_factor_lower_bound)
        print("swing_factor_upper_bound", server.swing_factor_upper_bound)
        print("swing_factor_step", server.swing_factor_step)

        # dump card
        server.dump_card = True
        server.dump_card_with_score_path = LERO_DUMP_CARD_FILE
        #调用 serve_forever() 方法，使服务器进入一个无限循环，持续监听并处理来自客户端的请求。这意味着服务器将在此调用后一直运行，直到外部因素（例如手动停止或错误）导致它退出。
        server.serve_forever()


if __name__ == "__main__":
    config = read_config()
    port = int(config["Port"])
    listen_on = config["ListenOn"]
    print_log(f"Listening on {listen_on} port {port}", "./server.log", True)

    lero_model = None
    if "ModelPath" in config:
        lero_model = LeroModel(None)
        lero_model.load(config["ModelPath"])
        print("Load model", config["ModelPath"])

    print("start server process...")
    start_server(listen_on, port, lero_model)

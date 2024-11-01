import argparse

from utils import *
import os
import socket
from config import *
from multiprocessing import Pool

#代表一个策略实体，具有分数属性。
class PolicyEntity:
    def __init__(self, score) -> None:
        self.score = score

    def get_score(self):
        return self.score

#继承自 PolicyEntity，增加了 card_str 属性，用于存储基数信息。
class CardinalityGuidedEntity(PolicyEntity):
    def __init__(self, score, card_str) -> None:
        super().__init__(score)
        self.card_str = card_str

#处理 PostgreSQL 查询的执行
class PgHelper():
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file
    #1. GIL（全局解释器锁）
    #在 Python 中，特别是 CPython 实现，存在一个全局解释器锁（GIL），它限制了同一时间只能有一个线程执行 Python 字节码。这意味着多线程在 CPU 密集型任务中并不会有效利用多核 CPU 的性能。
    #进程池通过创建多个进程，每个进程都有自己的 Python 解释器和内存空间，因此可以真正并行地执行任务，绕过 GIL 的限制。
    #2.任务类型
    #如果任务是 CPU 密集型（例如计算密集型的算法），使用进程池会更有效，因为可以充分利用多核 CPU 的计算能力。
    #对于 I/O 密集型任务（例如网络请求、文件读写），线程池可能更合适，因为它们通常会等待外部资源，而进程池在这类场景下可能导致不必要的开销。
    #多核 CPU 可以同时运行多个进程和线程，通过操作系统的调度机制实现高效的并行处理。一个进程不一定只使用一个核心；它的线程可以在多个核心上运行，进而提高计算效率。
    def start(self, pool_num):
        #创建一个进程池，Pool 是 Python 的 multiprocessing 模块中的类。pool_num 指定了进程池中可以同时运行的进程数量。使用进程池可以有效管理和复用进程，避免频繁地创建和销毁进程带来的开销。
        pool = Pool(pool_num)
        #这行代码开始遍历 self.queries，假设 self.queries 是一个包含查询信息的列表，每个元素是一个元组 (fp, q)，其中 fp 可能是文件路径或某种标识，而 q 是实际要执行的查询。
        for fp, q in self.queries:
            #使用 apply_async 方法异步地将 do_run_query 函数加入进程池进行执行。apply_async 不会阻塞主线程，允许主线程继续执行。
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, True, None, None))
        print('Waiting for all subprocesses done...')
        pool.close()
        #阻塞主线程，直到进程池中的所有进程都完成工作。这确保了在所有子进程结束之前，主线程不会继续执行后续的代码。
        pool.join()

# 处理 Lero 查询优化的类。
class LeroHelper():
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file, 
                test_queries, model_prefix, topK) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)
    #将输入列表 lst 切分成大小为 n 的块，逐块返回。
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def start(self, pool_num):
        lero_chunks = list(self.chunks(self.queries, self.query_num_per_chunk))

        run_args = self.get_run_args()
        for c_idx, chunk in enumerate(lero_chunks):
            #将查询分块并创建一个进程池，使用 Pool 来并行处理。
            pool = Pool(pool_num)
            for fp, q in chunk:
                #遍历每个块并调用 run_pairwise 方法处理查询。
                self.run_pairwise(q, fp, run_args, self.output_query_latency_file, self.output_query_latency_file + "_exploratory", pool)
            print('Waiting for all subprocesses done...')
            #等待所有子进程完成，关闭进程池。
            pool.close()
            #重新训练模型并进行基准测试。
            pool.join()

            model_name = self.model_prefix + "_" + str(c_idx)
            self.retrain(model_name)
            self.test_benchmark(self.output_query_latency_file + "_" + model_name)

    def retrain(self, model_name):
        #创建训练数据文件，调用训练脚本重新训练模型。
        training_data_file = self.output_query_latency_file + ".training"
        create_training_file(training_data_file, self.output_query_latency_file, self.output_query_latency_file + "_exploratory")
        print("retrain Lero model:", model_name, "with file", training_data_file)
        
        cmd_str = "cd " + self.lero_server_path + " && python3.8 train.py" \
                                                + " --training_data " + os.path.abspath(training_data_file) \
                                                + " --model_name " + model_name \
                                                + " --training_type 1"
        print("run cmd:", cmd_str)
        os.system(cmd_str)
        #运行训练命令并加载训练后的模型。
        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        #通过 TCP 连接到 Lero 服务器并加载指定模型。
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type":"load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)
        #创建 socket 连接，发送加载模型的请求，接收服务器的响应。
        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        print(reply_json)
        os.system("sync")
    #使用测试查询进行基准测试，记录查询延迟。
    def test_benchmark(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
            do_run_query(q, fp, run_args, output_file, True, None, None)
    #返回设置好的运行参数列表，在执行查询时使用。
    def get_run_args(self):
        run_args = []
        run_args.append("SET enable_lero TO True")
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args
    # 从基数文件中读取策略实体，按得分排序，生成基数文件，然后异步执行与这些基数相关的查询，以便评估其性能。
    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool):
        #执行查询并解释。
        explain_query(q, run_args)
        policy_entities = []
        with open(self.lero_card_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split(";") for line in lines]
            for line in lines:
                policy_entities.append(CardinalityGuidedEntity(float(line[1]), line[0]))
        #从基数文件中读取策略实体，按分数排序，选择前 K 个。
        policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
        policy_entities = policy_entities[:self.topK]
        #为每个实体创建基数文件，并使用进程池异步运行查询。
        i = 0
        for entity in policy_entities:
            if isinstance(entity, CardinalityGuidedEntity):
                card_str = "\n".join(entity.card_str.strip().split(" "))
                # ensure that the cardinality file will not be changed during planning
                card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                with open(card_file_path, "w") as card_file:
                    card_file.write(card_str)
                #执行 SQL 查询
                output_file = output_query_latency_file if i == 0 else exploratory_query_latency_file
                #添加基数信息重新查询
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name), output_file, True, None, None))
                i += 1
    #向 Lero 服务器发送查询计划并接收预测结果。
    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type":"predict", "Plan":plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']

# 侧重于模型应用中的查询执行和性能评估。
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--test_query_path",
                        metavar="PATH",
                        help="Load the test queries")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--query_num_per_chunk", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int)
    parser.add_argument("--topK", type=int)
    args = parser.parse_args()
    #解析命令行参数，读取训练查询和测试查询文件。
    query_path = args.query_path
    print("Load queries from ", query_path)
    queries = []
    with open(query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(SEP)
            queries.append((arr[0], arr[1]))
    print("Read", len(queries), "training queries.")

    output_query_latency_file = args.output_query_latency_file
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 10
    if args.pool_num:
        pool_num = args.pool_num
    print("pool_num:", pool_num)

    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if args.algo:
        assert args.algo.lower() in ALGO_LIST
        algo = args.algo.lower()
    print("algo:", algo)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    #根据选择的算法（PostgreSQL 或 Lero）实例化相应的助手类并执行查询处理。
    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        test_queries = []
        if args.test_query_path is not None:
            with open(args.test_query_path, 'r') as f:
                for line in f.readlines():
                    arr = line.strip().split(SEP)
                    test_queries.append((arr[0], arr[1]))
        print("Read", len(test_queries), "test queries.")

        query_num_per_chunk = args.query_num_per_chunk
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if args.model_prefix:
            model_prefix = args.model_prefix
        print("model_prefix:", model_prefix)

        topK = 5
        if args.topK is not None:
            topK = args.topK
        print("topK", topK)
        
        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK)
        helper.start(pool_num)

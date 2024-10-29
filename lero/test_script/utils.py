import hashlib
import json
import os
from time import time
from config import *
import fcntl
import psycopg2

def encode_str(s):
    #将输入字符串 s 编码为 MD5 哈希值。
    #通常用于生成查询或计划的唯一标识，以便后续查找和比较。
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()

#执行 SQL 查询，并返回执行时间和结果。
def run_query(q, run_args):
    #建立与数据库的连接。
    #设置客户端编码为 UTF-8。
    #执行可选的运行参数 run_args（例如设置某些配置）。
    #设置查询超时时间。
    #执行传入的查询 q，并获取结果。
    #最后关闭连接并返回执行时间和结果。
    start = time()
    conn = psycopg2.connect(CONNECTION_STR)
    conn.set_client_encoding('UTF8')
    result = None
    try:
        cur = conn.cursor()
        if run_args is not None and len(run_args) > 0:
            for arg in run_args:
                cur.execute(arg)
        cur.execute("SET statement_timeout TO " + str(TIMEOUT))
        print(run_args)
        print(q)
        cur.execute(q)
        result = cur.fetchall()
    finally:
        
        conn.close()
    # except Exception as e:
    #     conn.close()
    #     raise e
    
    stop = time()
    return stop - start, result
#获取历史查询的执行计划。
def get_history(encoded_q_str, plan_str, encoded_plan_str):
    history_path = os.path.join(LOG_PATH, encoded_q_str, encoded_plan_str)
    if not os.path.exists(history_path):
        return None
    
    print("visit histroy path: ", history_path)
    with open(os.path.join(history_path, "check_plan"), "r") as f:
        history_plan_str = f.read().strip()
        if plan_str != history_plan_str:
            print("there is a hash conflict between two plans:", history_path)
            print("given", plan_str)
            print("wanted", history_plan_str)
            return None
    
    print("get the history file:", history_path)
    with open(os.path.join(history_path, "plan"), "r") as f:
        return f.read().strip()
#保存当前查询及其执行计划到历史记录。
def save_history(q, encoded_q_str, plan_str, encoded_plan_str, latency_str):
    history_q_path = os.path.join(LOG_PATH, encoded_q_str)
    if not os.path.exists(history_q_path):
        os.makedirs(history_q_path)
        with open(os.path.join(history_q_path, "query"), "w") as f:
            f.write(q)
    else:
        with open(os.path.join(history_q_path, "query"), "r") as f:
            history_q = f.read()
            if q != history_q:
                print("there is a hash conflict between two queries:", history_q_path)
                print("given", q)
                print("wanted", history_q)
                return
    
    history_plan_path = os.path.join(history_q_path, encoded_plan_str)
    if os.path.exists(history_plan_path):
        print("the plan has been saved by other processes:", history_plan_path)
        return
    else:
        os.makedirs(history_plan_path)
        
    with open(os.path.join(history_plan_path, "check_plan"), "w") as f:
        f.write(plan_str)
    with open(os.path.join(history_plan_path, "plan"), "w") as f:
        f.write(latency_str)
    print("save history:", history_plan_path)
#获取查询的执行计划。
def explain_query(q, run_args, contains_cost = False):
    q = "EXPLAIN (COSTS " + ("" if contains_cost else "False") + ", FORMAT JSON, SUMMARY) " + (q.strip().replace("\n", " ").replace("\t", " "))
    _, plan_json = run_query(q, run_args)
    plan_json = plan_json[0][0]
    if len(plan_json) == 2:
        # remove bao's prediction
        plan_json = [plan_json[1]]
    return plan_json
#创建训练数据文件，将多个延迟文件的数据合并。
def create_training_file(training_data_file, *latency_files):
    lines = []
    for file in latency_files:
        with open(file, 'r') as f:
            lines += f.readlines()

    pair_dict = {}

    for line in lines:
        arr = line.strip().split(SEP)
        if arr[0] not in pair_dict:
            pair_dict[arr[0]] = []
        pair_dict[arr[0]].append(arr[1])
    #读取多个延迟文件的内容，按查询分组并保存每个查询的延迟信息。
    pair_str = []
    for k in pair_dict:
        if len(pair_dict[k]) > 1:
            candidate_list = pair_dict[k]
            pair_str.append(SEP.join(candidate_list))
    str = "\n".join(pair_str)
    #将结果写入指定的训练数据文件。
    with open(training_data_file, 'w') as f2:
        f2.write(str)
#执行 SQL 查询，并记录其执行计划和延迟时间，同时处理可能的历史记录和冲突。
def do_run_query(sql, query_name, run_args, latency_file, write_latency_file = True, manager_dict = None, manager_lock = None):
    #对输入的 SQL 查询进行清理，去除多余的空白和换行，以确保其在日志中保持整洁。
    sql = sql.strip().replace("\n", " ").replace("\t", " ")

    # 1. run query with pg hint
    #通过 EXPLAIN 语句获取查询的执行计划，COSTS FALSE 表示不计算成本。
    _, plan_json = run_query("EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) " + sql, run_args)
    plan_json = plan_json[0][0]
    if len(plan_json) == 2:
        # remove bao's prediction
        plan_json = [plan_json[1]]
    #从执行计划中提取规划时间和当前计划的字符串表示形式。
    planning_time = plan_json[0]['Planning Time']
    cur_plan_str = json.dumps(plan_json[0]['Plan'])
    try:
        # 2. get previous running result
        #使用 get_history 函数检查是否已有该查询的历史记录。如果有，则直接使用历史延迟数据。
        latency_json = None
        encoded_plan_str = encode_str(cur_plan_str)
        encoded_q_str = encode_str(sql)
        previous_result = get_history(encoded_q_str, cur_plan_str, encoded_plan_str)
        if previous_result is not None:
            latency_json = json.loads(previous_result)
        else:
            #如果没有历史记录，使用锁机制（manager_lock）确保只有一个进程可以执行当前计划，避免重复执行。
            if manager_dict is not None and manager_lock is not None:
                manager_lock.acquire()
                if cur_plan_str in manager_dict:
                    manager_lock.release()
                    print("another process will run this plan:", cur_plan_str)
                    return
                else:
                    manager_dict[cur_plan_str] = 1
                    manager_lock.release()

            # 3. run current query 
            run_start = time()
            try:
                #运行查询，并使用 ANALYZE 选项获取实际的执行时间和详细信息。
                _, latency_json = run_query("EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + sql, run_args)
                latency_json = latency_json[0][0]
                if len(latency_json) == 2:
                    # remove bao's prediction
                    latency_json = [latency_json[1]]
            except Exception as e:
                #如果查询超时，则执行 EXPLAIN 以获取查询计划和信息。
                if  time() - run_start > (TIMEOUT / 1000 * 0.9):
                    # Execution timeout
                    _, latency_json = run_query("EXPLAIN (VERBOSE, COSTS, FORMAT JSON, SUMMARY) " + sql, run_args)
                    latency_json = latency_json[0][0]
                    if len(latency_json) == 2:
                        # remove bao's prediction
                        latency_json = [latency_json[1]]
                    latency_json[0]["Execution Time"] = TIMEOUT
                else:
                    raise e

            latency_str = json.dumps(latency_json)
            save_history(sql, encoded_q_str, cur_plan_str, encoded_plan_str, latency_str)

        # 4. save latency
        #如果 write_latency_file 为 True，则将查询的延迟数据写入指定的延迟文件中，并使用文件锁来避免并发写入问题。
        latency_json[0]['Planning Time'] = planning_time
        if write_latency_file:
            with open(latency_file, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(query_name + SEP + json.dumps(latency_json) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)

        exec_time = latency_json[0]["Execution Time"]
        print(time(), query_name, exec_time, flush=True)
    except Exception as e:
        with open(latency_file + "_error", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(query_name + "\n")
            f.write(str(e).strip() + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
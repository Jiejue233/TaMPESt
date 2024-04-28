import copy
import random
import sys
import numpy as np
from deprecated import deprecated
import Data.Synthesiser as syn
import GenaticPlanning as gp
import plotly.figure_factory as ff
import datetime
import pickle
from flask import Flask, jsonify, request, render_template
import os
import json
import plotly
import Task
import re

app = Flask(__name__)
port = 5600

PROJ_JSON_FOLDER = 'File/json_files/project'
TASK_JSON_FOLDER = 'File/json_files/task'
PROJ_DATA_FOLDER = 'File/data_files/project'
TASK_DATA_FOLDER = 'File/data_files/task'
AJMT_DATA_FOLDER = 'File/ajmt_files'


def load_plan(target,planset):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        raise AttributeError("no such target")
    with open('File/regis.json', "r") as file:
        seq: list[int] = json.load(file)[planset]

    with open(os.path.join(data_addr, planset + ".pkl"), "rb") as file:
        data: list[Task.Task] = pickle.load(file)

    num_tasks = len(data)
    unchanged = []
    unplanned = data
    pivot = 0
    for i in range(num_tasks):
        if data[seq[i]].start_date <= datetime.date.today() and (
                data[seq[i]].finish_date() > datetime.date.today() or data[seq[i]].due_date > datetime.date.today()):
            pivot = i + 1
            break

    for i in seq[:pivot]:
        unchanged.append(data[i])
        unplanned.remove(data[i])
    return (seq, pivot), unchanged, unplanned


def fetch_data(data_list: list[dict], start_date=datetime.date.today()):
    tasks = []
    for data in data_list:
        t = Task.Task(data["name"], data["priority"], data["due_date"], data['required_time'])
        t.start_date = start_date
        tasks.append(t)
    return tasks

def fetch_adjacency_matrix(name, edge_list, size):
    adjacency_matrix = np.zeros((size, size), dtype=np.int8)
    for edge in edge_list:
        ind, outd = edge
        adjacency_matrix[ind, outd] = 1
    with open(os.path.join(AJMT_DATA_FOLDER, name + '.npy'), "wb") as file:
        np.save(file, adjacency_matrix)
    return adjacency_matrix


def generate_plan(data: list[Task.Task], adjacency_matrix: np.ndarray = None):
    planning = gp.Incubator(len(data))
    if adjacency_matrix is not None:
        evaluator = gp.TaskEvaluator(data, adjacency_matrix)
        planning.evolution_mode("task", (evaluator.customCrossover, evaluator.customMutation))
        planning.set_evolution_goal(evaluator.reliance_evaluate)
    else:
        evaluator = gp.TaskEvaluator(data)
        planning.evolution_mode("project")
        planning.set_evolution_goal(evaluator.simple_evaluate)
    result = planning.evolve()
    # print(result)
    # print(evaluator.simple_evaluate(result))

    return result


def plan_task(start_date, data: list[Task.Task], order:list[int], adjacency_matrix = None):
    num_tasks = len(data)
    for i in range(num_tasks):
        task_id = order[i]
        # task_id = reordered_seq[i]
        if adjacency_matrix is not None:
            in_task = adjacency_matrix[:, i]
            # print(in_task)
            if in_task.sum() == 0:
                data[task_id].start_date = start_date
            else:
                indices = np.where(in_task > 0)
                # print(indices[0])
                latest = data[indices[0][0]].finish_date()
                for ti in indices[0]:
                    tempd = data[ti].finish_date()
                    if tempd > latest:
                        latest = tempd
                # print(latest)
                # print(task_id)
                data[task_id].start_date = latest
        else:
            if i == 0:
                data[task_id].start_date = start_date
            else:
                prev_id = order[i - 1]
                # prev_id = reordered_seq[i - 1]
                data[task_id].start_date = data[prev_id].finish_date()
    return data


def gen_gantt(target, name, data: list[Task.Task], order: list[int]):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        return None

    num_tasks = len(data)

    df = []

    for i in range(num_tasks):
        if target == "project":
            df.append(dict(Task=data[i].name, Start=data[i].start_date,
                           Finish=data[i].start_date + data[i].required_time,
                           Resource=f"Priority {data[i].priority}"))
        elif target == "task":
            df.append(dict(Task=data[i].name, Start=data[i].start_date,
                           Finish=data[i].start_date + data[i].required_time,))
    if target == "project":
        fig = ff.create_gantt(df, index_col='Resource', title='Project Schedule', show_colorbar=True, group_tasks=True)
    else:
        fig = ff.create_gantt(df, title='task Schedule', group_tasks=True)
    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
        )
    )
    if target == "project":
        for i in range(num_tasks):
            y_pos = num_tasks - i - 1
            print(f"task {i}: {data[i]}")
            fig.add_shape(type="rect",
                          x0=data[i].start_date, y0=y_pos - 0.3, x1=data[i].due_date, y1=y_pos + 0.3,
                          line=dict(width=0),
                          fillcolor="black",
                          opacity=0.3)

        fig.add_shape(type="rect",
                      x0=data[order[0]].start_date, y0=-1000, x1=datetime.date.today(), y1=1000,
                      line=dict(width=0),
                      fillcolor="#1f1e33",
                      opacity=0.1)
    # fig.show()

    fig.add_vline(x=datetime.datetime.now(), line_width=2, line_dash="solid", line_color="red")
    with open(os.path.join(data_addr, name + ".pkl"), "wb") as file:
        pickle.dump(data, file)
    with open(os.path.join(json_addr, name + ".json"), "w") as file:
        json.dump(fig, file, cls=plotly.utils.PlotlyJSONEncoder, indent=4)


def target_route(target):
    if target == 'project':
        data_addr = PROJ_DATA_FOLDER
        json_addr = PROJ_JSON_FOLDER
    elif target == 'task':
        data_addr = TASK_DATA_FOLDER
        json_addr = TASK_JSON_FOLDER
    else:
        return jsonify({'error': 'Invalid target'}), 404
    return json_addr, data_addr


def regexMatch(regex, d):
    name_term = re.compile(regex)
    matching_keys = [key for key in d if name_term.match(key)]
    return matching_keys

def find_longest(adjacency_matrix:np.ndarray, task_list:list[Task.Task], order):
    starts = []
    trip = {}
    for index in order:
        candidate = task_list[index]
        trip.update({index: candidate.required_time.days})
        if adjacency_matrix[:, index].sum() == 0:
            starts.append(index)

    V = len(adjacency_matrix)
    dist = [-sys.maxsize] * V
    path = [-1] * V
    for start in starts:
        dist[start] = trip[start]

    for i in order:
        if dist[i] != -sys.maxsize:
            for j in range(V):
                if adjacency_matrix[i][j] != 0 and dist[j] < dist[i] + trip[j]:
                    dist[j] = dist[i] + trip[j]
                    path[j] = i

    max_distance = max(dist)
    end_vertex = dist.index(max_distance)

    longest_path = []
    current_vertex = end_vertex
    while current_vertex != -1:
        longest_path.append(task_list[current_vertex].name)
        current_vertex = path[current_vertex]

    longest_path.reverse()

    return max_distance, longest_path


@app.route('/')
def base():
    return render_template("index.html")


@app.route('/<target>/')
def project_home(target):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        return json_addr, data_addr
    files = os.listdir(json_addr)
    json_files = [f[:-5] for f in files if f.endswith('.json')]
    return render_template(f"{target}.html", json_files=json_files)


@app.route('/project/get-json/<filename>')
def get_json(filename):
    file_path = os.path.join(PROJ_JSON_FOLDER, filename + ".json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return jsonify(json.load(file))
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/project/get-data/<filename>')
def get_data(filename):
    file_path = os.path.join(PROJ_DATA_FOLDER, filename + ".pkl")
    if os.path.exists(file_path):
        with open('File/regis.json', 'r') as file:
            order_list: dict = json.load(file)
            first_item = order_list[filename][0]
        (seq, pivot), _, _ = load_plan("project", filename)
        with open(file_path, 'rb') as file:
            task_list: list[Task.Task] = pickle.load(file)
            start_date = task_list[first_item].start_date.strftime("%Y-%m-%d")
            json_task = []
            for i, task in enumerate(task_list):
                if i in seq[:pivot]:
                    status = 2
                else:
                    status = 0
                if i == seq[pivot]:
                    status = 1

                json_task.append({"name": task.name,
                                  "priority": task.priority,
                                  "due_date": task.due_date.strftime("%Y-%m-%d"),
                                  "required_time": task.required_time.days,
                                  "status": status})

            json_entity = {"plan_name": filename, "start_date": start_date, "form": json_task}
            return jsonify(json_entity)
    else:
        return jsonify({'error': 'File not found'}), 404


@deprecated(reason= "this function is no longer meaningful, everything that task need is integrated into "
                    "get-everything function")
@app.route('/task/get-matx/<filename>')
def get_matx(filename):
    file_path = os.path.join(AJMT_DATA_FOLDER, filename + ".npy")
    json_mat = []
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            matrix = np.load(file)
        indices = np.where(matrix > 0)
        for col, row in zip(indices[0],indices[1]):
            json_mat.append(f"{col},{row}")
        return jsonify(json_mat)
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route("/task/get-everything/<filename>")
def get_everything(filename):
    json_chunk = {}
    file_path = os.path.join(TASK_JSON_FOLDER, filename + ".json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            json_chunk.update({"gantt": json.load(file)})
    else:
        json_chunk.update({"gantt": None})
    file_path = os.path.join(TASK_DATA_FOLDER, filename + ".pkl")
    if os.path.exists(file_path):
        with open('File/regis.json', 'r') as file:
            order_list: dict = json.load(file)
            first_item = order_list[filename][0]
        (seq, pivot), _, _ = load_plan("task", filename)
        with open(file_path, 'rb') as file:
            task_list: list[Task.Task] = pickle.load(file)
            start_date = task_list[first_item].start_date.strftime("%Y-%m-%d")
            json_task = []
            for i, task in enumerate(task_list):
                if task.start_date <= datetime.date.today():
                    if task.finish_date() >= datetime.date.today():
                        status = 1
                    else:
                        status = 2
                else:
                    status = 0


                json_task.append({"name": task.name,
                                  "start_date": task.start_date.strftime("%Y-%m-%d"),
                                  "finish_date": (task.start_date + task.required_time).strftime("%Y-%m-%d"),
                                  "required_time": task.required_time.days,
                                  "status": status})

            json_entity = {"plan_name": filename, "start_date": start_date, "form": json_task}
            json_chunk.update({"data": json_entity})
    else:
        json_chunk.update({"data": None})
    file_path = os.path.join(AJMT_DATA_FOLDER, filename + ".npy")
    json_mat = []
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            matrix = np.load(file)
        indices = np.where(matrix > 0)
        for col, row in zip(indices[0], indices[1]):
            json_mat.append(f"{col},{row}")
        json_chunk.update({"mat": json_mat})
    else:
        json_chunk.update({"mat": None})
    dist, lpath = find_longest(matrix, task_list, order_list[filename])
    json_chunk.update({"path": f"the longest path: {lpath} with {dist} days"})
    return jsonify(json_chunk)



@app.route('/<target>/refresh-json-files')
def refresh_json_files(target):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        return json_addr, data_addr
    files = os.listdir(json_addr)
    json_files = [f[:-5] for f in files if f.endswith('.json')]
    return jsonify(json_files)


@app.route('/<target>/del-plan/<plan_name>')
def del_plan(target, plan_name):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        return json_addr, data_addr

    with open('File/regis.json', 'r') as file:
        order_list: dict = json.load(file)
        order_list.pop(plan_name)
    os.remove(os.path.join(json_addr, plan_name + '.json'))
    os.remove(os.path.join(data_addr, plan_name + '.pkl'))
    if target == "task":
        os.remove(os.path.join(AJMT_DATA_FOLDER, plan_name + ".npy"))
    with open('File/regis.json', 'w') as file:
        json.dump(order_list, file, indent=4)
    return refresh_json_files(target)


@app.route('/<target>/generate', methods=['POST'])
def generate(target):
    json_addr, data_addr = target_route(target)
    if data_addr == 404:
        return json_addr, data_addr
    table_content = request.get_json()
    data_dict = []
    edge_list = []
    name = table_content['plan_name']

    name_term = regexMatch(r"name\|row\d+", table_content)
    for key in name_term:
        value = table_content[key]
        data_dict.append({"name": value})

    priority_term = regexMatch(r"priority\|row\d+", table_content)
    for key in priority_term:
        value = table_content[key]
        item = key[key.index("|")+1+3:]
        data_dict[int(item)].update({"priority": value})

    dt_term = regexMatch(r"due_date\|row\d+", table_content)
    for key in dt_term:
        temp = table_content[key].split("-")
        value = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
        #     data_dict[int(item[1][3:])].update({item[0]: value})
        item = key[key.index("|")+1+3:]
        data_dict[int(item)].update({"due_date": value})

    rt_term = regexMatch(r"required_time\|row\d+", table_content)
    for key in rt_term:
        value = int(table_content[key])
        item = key[key.index("|")+1+3:]
        data_dict[int(item)].update({"required_time": value})

    task_term = regexMatch(r"finish_date\|row\d+", table_content)
    for key in task_term:
        temp = datetime.date.today()
        value = datetime.date(year=temp.year + 20, month=temp.month, day=temp.day)
        item = key[key.index("|")+1+3:]
        data_dict[int(item)].update({"due_date": value})
        data_dict[int(item)].update({"priority": 1})

    mat = regexMatch(r"\(\d+,\d+\)\|mat", table_content)
    for key in mat:
        item = key[:key.index("|")]
        temp = item.split(",")
        if table_content[key] == 1:
            edge_list.append((int(temp[0][1:]), int(temp[1][:-1])))

    # print(edge_list)
    # print(data_dict)

    file_path = os.path.join(data_addr, name + ".pkl")
    with open('File/regis.json', 'r') as file:
        order_list: dict = json.load(file)
    if os.path.exists(file_path):
        (seq, pivot), _, unplanned = load_plan(target, name)
        # print(pivot)
        temp = table_content['start_date'].split("-")
        start_date = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
        task_list = fetch_data(data_dict, start_date)
        reordered_seq = seq[:pivot]
        trimmed_matx = None
        ajce_matx = None
        if len(edge_list) != 0:
            ajce_matx = fetch_adjacency_matrix(name, edge_list, len(task_list))
            trimmed_matx = np.delete(ajce_matx, reordered_seq, axis=0)
            trimmed_matx = np.delete(trimmed_matx, reordered_seq, axis=1)

        result = generate_plan(unplanned, trimmed_matx)

        unplan_list = np.delete(np.arange(len(task_list)), reordered_seq).tolist()
        for i in result:
            reordered_seq.append(unplan_list[i])
        order_list.update({name: reordered_seq})
        graph_data = plan_task(start_date, task_list, reordered_seq, ajce_matx)
        gen_gantt(target, name, graph_data, reordered_seq)
    else:
        temp = table_content['start_date'].split("-")
        start_date = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
        task_list = fetch_data(data_dict, start_date)
        ajce_matx = None
        if len(edge_list) != 0:
            ajce_matx = fetch_adjacency_matrix(name, edge_list, len(task_list))
        result = generate_plan(task_list, ajce_matx)
        order_list.update({name: result})
        graph_data = plan_task(start_date, task_list, result, ajce_matx)
        gen_gantt(target, name, graph_data, result)

    with open('File/regis.json', 'w') as file:
        json.dump(order_list, file, indent=4)
    # print(table_content)
    # print(ajce_matx)
    if target == "task":
        return get_everything(name)
    elif target == "project":
        return get_json(target, name)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=port, debug=True)

#   if __name__ == '__main__':
#     syner = syn.RandomSynthesiser(5, datetime.date(year=2024, month=12, day=31), 30)
#     priority_density = syner.generate_density()
#     amount = 12
#     print(f"synthesizing {amount} data with priority density: {priority_density}")
#     data = syner.sample(amount=amount, weight_density=priority_density)
#     with open("File/latest/tasks.pkl", "wb") as file:
#         pickle.dump(data, file)
#     # folder = os.listdir("File/latest")
#     # for file in folder:
#     #     if ".json" in file:
#     #         seq: list[int] = json.loads(file[:-5])
#     #
#     # with open("File/latest" + "/tasks.pkl", "rb") as file:
#     #     data: list[Task.Task] = pickle.load(file)
#
#     # for task in data:
#     #     print(task)
#     num_tasks = len(data)
#     unchanged = []
#     unplanned = data
#     pivot = 0
#     # for i in range(num_tasks):
#     #     if data[seq[i]].start_date <= datetime.date.today() and (
#     #             data[seq[i]].finish_date() > datetime.date.today() or data[seq[i]].due_date > datetime.date.today()):
#     #         pivot = i + 1
#     #         break
#     #
#     # for i in seq[:pivot]:
#     #     unchanged.append(data[i])
#     #     unplanned.remove(data[i])
#
#     # planning = gp.Incubator(len(unplanned))
#
#     planning = gp.Incubator(12)
#     evaluator = gp.TaskEvaluator(unplanned)
#     planning.set_evolution_goal(evaluator.evaluate)
#     result = planning.evolve()
#     print(result)
#     print(evaluator.evaluate(result))
#
#     # reordered_seq = seq[:pivot]
#     # for i in result:
#     #     reordered_seq.append(seq[pivot:][i])
#
#     graph_data = copy.deepcopy(data)
#
#     df = []
#     for i in range(num_tasks):
#         task_id = result[i]
#         # task_id = reordered_seq[i]
#         if i == 0:
#             graph_data[task_id].start_date = datetime.date.today()
#         else:
#             prev_id = result[i - 1]
#             # prev_id = reordered_seq[i - 1]
#             graph_data[task_id].start_date = graph_data[prev_id].finish_date()
#
#     for i in range(num_tasks):
#         df.append(dict(Task=f"Task {i + 1}", Start=graph_data[i].start_date,
#                        Finish=graph_data[i].start_date + graph_data[i].required_time,
#                        Resource=f"Priority {graph_data[i].priority}"))
#
#     fig = ff.create_gantt(df, index_col='Resource', title='Project Schedule', show_colorbar=True, group_tasks=True)
#     fig.update_layout(
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='lightgray',
#             gridwidth=1,
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='lightgray',
#             gridwidth=1,
#         )
#     )
#
#     for i in range(num_tasks):
#         y_pos = num_tasks - i - 1
#         print(f"task {i}: {graph_data[i]}")
#         fig.add_shape(type="rect",
#                       x0=graph_data[i].start_date, y0=y_pos - 0.3, x1=graph_data[i].due_date, y1=y_pos + 0.3,
#                       line=dict(width=0),
#                       fillcolor="black",
#                       opacity=0.3)
#
#     fig.add_shape(type="rect",
#                   x0=graph_data[result[0]].start_date, y0=-1000, x1=datetime.date.today(), y1=1000,
#                   # x0=graph_data[reordered_seq[0]].start_date, y0=-1000, x1=datetime.date.today(), y1=1000,
#                   line=dict(width=0),
#                   fillcolor="#1f1e33",
#                   opacity=0.1)
#     fig.show()
#
#     with open("File/latest/"+json.dumps(result)+".json", "w") as file:
#         json.dump(fig, file, cls=plotly.utils.PlotlyJSONEncoder)

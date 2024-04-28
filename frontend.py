import datetime
import pickle

from flask import Flask, jsonify, request, render_template
import os
import json
import plotly
import main
import Task

app = Flask(__name__)

PROJ_JSON_FOLDER = main.PROJ_JSON_FOLDER
PROJ_DATA_FOLDER = main.PROJ_DATA_FOLDER
TASK_JSON_FOLDER = main.TASK_JSON_FOLDER
TASK_DATA_FOLDER = main.TASK_DATA_FOLDER
AJMT_DATA_FOLDER = main.AJMT_DATA_FOLDER
port = 5600


@app.route('/')
def base():
    return render_template("index.html")


@app.route('/<target>/')
def project_home(target):
    json_addr, _ = target_route(target)
    if _ == 404:
        return json_addr, _
    files = os.listdir(json_addr)
    json_files = [f[:-5] for f in files if f.endswith('.json')]
    return render_template(f"{target}.html", json_files=json_files)


@app.route('/<target>/get-json/<filename>')
def get_json(target, filename):
    json_addr, _ = target_route(target)
    if _ == 404:
        return json_addr, _
    file_path = os.path.join(json_addr, filename + ".json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return jsonify(json.load(file))
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/<target>/get-data/<filename>')
def get_data(target, filename):
    _, data_addr = target_route(target)
    if data_addr == 404:
        return _, data_addr
    file_path = os.path.join(data_addr, filename + ".pkl")
    if os.path.exists(file_path):
        with open('File/regis.json', 'r') as file:
            order_list: dict = json.load(file)
            first_item = order_list[filename][0]
        (seq, pivot), _, _ = main.load_plan(filename)
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


@app.route('/<target>/refresh-json-files')
def refresh_json_files(target):
    json_addr, _ = target_route(target)
    if _ == 404:
        return json_addr, _
    files = os.listdir(json_addr)
    json_files = [f[:-5] for f in files if f.endswith('.json')]
    return jsonify(json_files)


@app.route('/project/del-plan/<plan_name>')
def del_plan(plan_name):
    with open('File/regis.json', 'r') as file:
        order_list: dict = json.load(file)
        order_list.pop(plan_name)
    os.remove(os.path.join(PROJ_JSON_FOLDER, plan_name + '.json'))
    os.remove(os.path.join(PROJ_DATA_FOLDER, plan_name + '.json'))
    with open('File/regis.json', 'w') as file:
        json.dump(order_list, file, indent=4)


@app.route('/project/generate', methods=['POST'])
def generate():
    table_content = request.get_json()
    data_dict = []
    name = table_content['plan_name']
    for key in table_content.keys():
        item = key.split("\\")
        if len(item) == 2 and "row" in item[1]:
            # data["name"], data["priority"], data["due_date"], data['required_time']
            match item[0]:
                case "name":
                    value = table_content[key]
                    data_dict.append({item[0]: value})
                case "priority":
                    value = int(table_content[key])
                    data_dict[int(item[1][3:])].update({item[0]: value})
                case "due_date":
                    temp = table_content[key].split("-")
                    value = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
                    data_dict[int(item[1][3:])].update({item[0]: value})
                case "required_time":
                    value = int(table_content[key])
                    data_dict[int(item[1][3:])].update({item[0]: value})

    file_path = os.path.join(PROJ_DATA_FOLDER, name + ".pkl")
    with open('File/regis.json', 'r') as file:
        order_list: dict = json.load(file)
    if os.path.exists(file_path):
        (seq, pivot), _, _ = main.load_plan(name)
        temp = table_content['start_date'].split("-")
        start_date = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
        task_list = main.fetch_data(name, data_dict, start_date)
        result = main.generate_plan(task_list)
        reordered_seq = seq[:pivot]
        for i in result:
            reordered_seq.append(seq[pivot:][i])
        order_list.update({name: reordered_seq})
        main.gen_gantt(name, task_list, reordered_seq, start_date=start_date)
    else:
        temp = table_content['start_date'].split("-")
        start_date = datetime.date(year=int(temp[0]), month=int(temp[1]), day=int(temp[2]))
        task_list = main.fetch_data(name, data_dict, start_date)
        result = main.generate_plan(task_list)
        order_list.update({name: result})
        main.gen_gantt(name, task_list, result, start_date= start_date)
    with open('File/regis.json', 'w') as file:
        json.dump(order_list, file, indent=4)
    print(table_content)
    return get_json(name)


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


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=port, debug=True)

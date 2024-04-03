import copy
import datetime
import json
import os
import random

import plotly

import Data.Synthesiser as syn
import GenaticPlanning as gp
import plotly.figure_factory as ff
import pickle

import Task

if __name__ == '__main__':
    # syner = syn.RandomSynthesiser(5, datetime.date(year=2024, month=12, day=31), 30)
    # priority_density = syner.generate_density()
    # data = syner.sample(amount=12, weight_density=priority_density)
    folder = os.listdir("File/latest")
    for file in folder:
        if ".json" in file:
            seq: list[int] = json.loads(file[:-5])

    with open("File/latest"+"/tasks.pkl", "rb") as file:
        data: list[Task.Task] = pickle.load(file)

    # for task in data:
    #     print(task)
    num_tasks = len(data)
    unchanged = []
    unplanned = data
    pivot = -1
    for i in range(num_tasks):
        if data[seq[i]].start_date <= datetime.date.today() and (data[seq[i]].finish_date() > datetime.date.today() or data[seq[i]].due_date > datetime.date.today()):
            pivot = i
            break
    if pivot!=-1:
        unplanned = data[seq[i:]]

    planning = gp.Incubator(len(unplanned))
    evaluator = gp.TaskEvaluator(unplanned)
    planning.set_evolution_goal(evaluator.evaluate)
    result = planning.evolve()
    print(result)
    print(evaluator.evaluate(result))

    unchanged.extend(result)

    graph_data = copy.deepcopy(data)

    df = []
    for i in range(num_tasks):
        task_id = result[i]
        if i == 0:
            graph_data[task_id].start_date = datetime.date.today()
        else:
            prev_id = result[i - 1]
            graph_data[task_id].start_date = graph_data[prev_id].finish_date()

    for i in range(num_tasks):
        df.append(dict(Task=f"Task {i + 1}", Start=graph_data[i].start_date, Finish=graph_data[i].start_date+graph_data[i].required_time,
                       Resource=f"Priority {graph_data[i].priority}"))

    fig = ff.create_gantt(df, index_col='Resource', title='Project Schedule', show_colorbar=True, group_tasks=True)
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

    for i in range(num_tasks):
        y_pos = num_tasks - i - 1
        print(f"task {i}: {graph_data[i]}")
        fig.add_shape(type="rect",
                      x0=graph_data[i].start_date, y0=y_pos - 0.3, x1=graph_data[i].due_date, y1=y_pos + 0.3,
                      line=dict(width=0),
                      fillcolor="black",
                      opacity=0.3)

    fig.add_shape(type="rect",
                  x0=graph_data[result[0]].start_date, y0= -1000, x1=datetime.date.today(), y1=1000,
                  line=dict(width=0),
                  fillcolor="#1f1e33",
                  opacity=0.1)
    fig.show()

    # graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # with open("File/latest/"+json.dumps(result)+".json", "w") as file:
    #     json.dump(fig, file, cls=plotly.utils.PlotlyJSONEncoder)
    # 12, 11



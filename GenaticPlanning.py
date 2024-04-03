import datetime
import random
from typing import Callable

from deap import base, creator, tools, algorithms

import Task

class TaskEvaluator():
    def __init__(self, tasks: list[Task.Task]):
        self.tasks = tasks
    def evaluate(self, individual):
        totalTime = 0
        penalty = 0
        priority = 0
        exceeds_time = 0
        emergency = 0
        weighted_emergency = 0
        for index in range(len(individual)):
            task_id = individual[index]
            if index == 0:
                self.tasks[task_id].start_date = datetime.date.today()
            else:
                prev_id = individual[index - 1]
                self.tasks[task_id].start_date = self.tasks[prev_id].finish_date()
                priority += self.tasks[prev_id].priority if self.tasks[task_id].priority <= self.tasks[prev_id].priority else - self.tasks[task_id].priority

            exceeds_time += 1 if self.tasks[task_id].exceed_or_idle_time().days > 0 else 0  # ?
            emergency += self.tasks[task_id].critical_ratio()
            weighted_emergency += self.tasks[task_id].critical_ratio() * self.tasks[task_id].priority

        return priority, exceeds_time, weighted_emergency

class Incubator():
    def __init__(self, NUM_TASKS, weight=(1.0, -1.0, -2.0)):
        # first, second, third
        creator.create("FitnessArg", base.Fitness, weights=weight)
        creator.create("Individual", list, fitness=creator.FitnessArg)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.sample, range(NUM_TASKS), NUM_TASKS)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

        # self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)

    def set_evolution_goal(self, func: Callable):
        self.toolbox.register("evaluate", func)

    def evolve(self, cxpb=0.5, mutpb=0.3, ngen=50, verbose=False):
        population = self.toolbox.population(n=50)
        algorithms.eaSimple(population, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=verbose)

        return tools.selBest(population, 1)[0]
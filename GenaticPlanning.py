import datetime
import random
from typing import Callable
import numpy as np
from deap import base, creator, tools, algorithms

import Task

class TaskEvaluator():
    def __init__(self, tasks: list[Task.Task], adjacency_matrix: np.ndarray= None):
        self.adjacency_matrix = adjacency_matrix
        self.tasks = tasks
    def simple_evaluate(self, individual):
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

    def reliance_evaluate(self, individual):
        exceeds_time = 0
        emergency = 0
        weighted_emergency = 0
        rely = 1
        for index in range(len(individual)):
            task_id = individual[index]
            if index == 0:
                self.tasks[task_id].start_date = datetime.date.today()
            else:
                prev_id = individual[index - 1]
                self.tasks[task_id].start_date = self.tasks[prev_id].finish_date()
                if rely !=0:
                    if self.adjacency_matrix[prev_id, index] == 1:

                        rely += 200
                    elif self.adjacency_matrix[index, prev_id] == 1:
                        rely = 0

            exceeds_time += 1 if self.tasks[task_id].exceed_or_idle_time().days > 0 else 0  # ?
            emergency += self.tasks[task_id].critical_ratio()
            weighted_emergency += self.tasks[task_id].critical_ratio() * self.tasks[task_id].priority

        return rely, exceeds_time, weighted_emergency

    def customCrossover(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        p1, p2 = [0] * size, [0] * size

        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[ind1[i]] = i
            p2[ind2[i]] = i
        # Choose crossover points
        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Apply crossover between cx points
        for i in range(cxpoint1, cxpoint2):
            # Keep track of the selected values
            temp1 = ind1[i]
            temp2 = ind2[i]
            # Swap the matched value
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        for i in range(1, size):
            for j in range(i):
                if self.adjacency_matrix[ind1[i]][ind1[j]] == 1:
                    ind1.insert(i, ind1.pop(j))
                if self.adjacency_matrix[ind2[i]][ind2[j]] == 1:
                    ind2.insert(i, ind2.pop(j))

        return ind1, ind2

    def customMutation(self,individual, indpb):
        size = len(individual)
        for i in range(size):
            if random.random() < indpb:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                individual[i], individual[swap_indx] = \
                    individual[swap_indx], individual[i]
        # print(f"pre: {individual}")
        for i in range(1, size):
            for j in range(i):
                if self.adjacency_matrix[individual[i], individual[j]] == 1:
                    individual.insert(i, individual.pop(j))
        # print(f"post: {individual}")
        return individual,


class Incubator():
    def __init__(self, NUM_TASKS, weight=(1.0, -1.0, -4.0)):
        # first, second, third
        creator.create("FitnessArg", base.Fitness, weights=weight)
        creator.create("Individual", list, fitness=creator.FitnessArg)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.sample, range(NUM_TASKS), NUM_TASKS)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)

    def evolution_mode(self, mode: str = "project", task_func: tuple[Callable,Callable] = None):
        if mode == 'project':
            self.toolbox.register("mate", tools.cxPartialyMatched)
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        elif mode == 'task':
            func1, func2 = task_func
            self.toolbox.register("mate", func1)
            self.toolbox.register("mutate", func2, indpb=0.05)
        else:
            raise AttributeError("mode must be either 'project' or 'task'")


    def set_evolution_goal(self, func: Callable):
        self.toolbox.register("evaluate", func)

    def evolve(self, cxpb=0.5, mutpb=0.3, ngen=10, verbose=False):
        population = self.toolbox.population(n=10)
        algorithms.eaSimple(population, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=verbose)

        return tools.selBest(population, 1)[0]
import random
import datetime
import numpy as np
import Task

class RandomSynthesiser():
    def __init__(self, max_priority: int, max_end_time: datetime.date, max_dilation: int | tuple[int, int]):
        self.max_priority = max_priority
        self.start_date = datetime.date.today()
        self.max_end_time = max_end_time
        self.max_dilation = max_dilation

    def generate_density(self):
        temp_density: list[float] = []
        for i in range(self.max_priority):
            temp_density.append(random.random())

    def sample(self, amount=1, *, weight_density):
        tasks = []
        for i in range(amount):
            max_delta = (self.max_end_time - self.start_date).days
            end_delta = random.randint(0, int(max_delta))
            random_end_date = self.start_date + datetime.timedelta(
                days=end_delta
            )
            if type(self.max_dilation) is int:
                low_bound = 1
                high_bound = self.max_dilation
            else:
                surplus, exceed = self.max_dilation
                low_bound = end_delta - surplus
                high_bound = end_delta + exceed

            random_delta = datetime.timedelta(days=random.randint(max(1, low_bound), high_bound))
            priorities = np.arange(1, self.max_priority+1).tolist()
            random_priority = random.choices(priorities, weights=weight_density, k=1)[0]
            tasks.append(Task.Task(random_priority, random_end_date, random_delta))
        return tasks

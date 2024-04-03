import numpy as np
import datetime

class Task():
    def __init__(self, task_prior: int, due_date: datetime.date,
                 required_time=datetime.timedelta()):
        self.required_time = required_time
        self.priority = task_prior
        self.accepted_date = datetime.date.today()
        self.start_date = datetime.date.today()
        self.due_date = due_date
        self.last_update = datetime.date.today()


    def update(self):
        self.required_time -= (datetime.date.today() - self.last_update)
        self.last_update = datetime.date.today()

    def exceed_or_idle_time(self):
        """
        calculate the exceeded or the idling time
        :return: a dateTime object, positive means exceed, negative means idle
        """
        return self.required_time - self.remaining_time()

    def remaining_time(self):
        return self.due_date - self.start_date

    def demanded_time(self):
        return (self.due_date - self.accepted_date)

    def critical_ratio(self):
        return 1.0 - (self.remaining_time() / self.demanded_time())
    def schedule_ratio(self):
        return 1.0 - ((self.due_date - datetime.date.today()) / self.demanded_time())

    def finish_date(self):
        return self.start_date + self.required_time

    def __str__(self):
        return f"task started at {self.start_date}, due at {self.due_date}, and still require {self.required_time} to finish"



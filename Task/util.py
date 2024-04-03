import math


def sigmoid(num):
    return 1.0 / (1 + math.e ** (-num))

# Stochastic gradient decent + momentum + acceleration is the algorithm ill be recreating
import torch
from config import momentum, learning_rate

class SGDWithMomentum:
    def __init__(self, params):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.parameters = params
        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            # update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + self.learning_rate * p.grad
            # update parameter
            p.data -= self.velocities[i]

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()

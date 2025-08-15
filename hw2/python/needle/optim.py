"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            # assert w.grad.data.dtype == "float32"
            grad = ndl.Tensor(w.grad.data, dtype="float32") + (self.weight_decay) * w.data
            # grad = w.grad.data + (self.weight_decay) * w.data
            if w in self.u:
                self.u[w] = self.momentum * self.u[w] + (1-self.momentum) * grad
            else:
                self.u[w] = (1-self.momentum) * grad
            
            w.data = w.data - self.lr * self.u[w]
        
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            grad = ndl.Tensor(p.grad.data, dtype="float32") + self.weight_decay * p.data
            if p in self.m:
                self.m[p] = self.beta1*self.m[p] + (1-self.beta1)*grad
            else:
                self.m[p] = (1-self.beta1)*grad
            # self.m[p] = u_new
            if p in self.v:
                self.v[p] = self.beta2*self.v[p] + (1-self.beta2)*(grad**2)
            else:
                self.v[p] = (1-self.beta2)*(grad**2)
            # self.v[p] = v_new
            
            u_hat = self.m[p]/(1-self.beta1**self.t)
            v_hat = self.v[p]/(1-self.beta2**self.t)
            
            p.data = p.data - self.lr * ndl.divide(u_hat, (v_hat**0.5 + self.eps))
            
        ### END YOUR SOLUTION

import numpy as np
from typing import *
from functools import *

'''
    TODO: implement autograd DL library backbone, to be used
    throughout all my implementations.

    what about arrays of numbers? 
'''
def pprint(x, name):
    print('-'*10,name,'-'*10)
    print(x)

class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.inputs_need_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.inputs_need_grad) else False #else (None if any(x is None for x in self.inputs_need_grad) else False)

    @classmethod
    def apply(cls, *x, **kwargs):
        # initialize custom fn. ctx is object of custom function!
        ctx = cls(*x)
        # do forward pass on custom fn, return new val
        ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs), 
            requires_grad=ctx.requires_grad)
        # it is needed for backprop and the mode is not inference
        if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx
        return ret
    
from mygrad import mlops
# a wrapper around np that adds autograd capabilities, that's it
class Tensor:
    # it may take in a list
    debug = False
    no_grad = False
    default_type = np.float32
    def __init__(self, data, requires_grad=False, requires_update=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data 
        self.shape = data.shape
        self.grad = None
        # for inference 
        self.requires_grad = requires_grad
        self.requires_update = requires_update
        self._ctx = None
    
    def __repr__(self):
        return f'{self.data}'

    def __len__(self): return len(self.data) 

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __add__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Add.apply(self,x)
    def __sub__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Sub.apply(self,x)
    def __mul__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Mul.apply(self,x)
    def __pow__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Pow.apply(self,x)
    def __truediv__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Div.apply(self,x)
    def __matmul__(self, x): 
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return mlops.Matmul.apply(self, x)

    def sum(self, axis=1, keepdims=True): return Tensor(self.data.sum(axis=axis, keepdims=keepdims))
    def log(self): return Tensor(np.log(self.data))
    def exp(self): return Tensor(np.exp(self.data))
    
    def relu(self):
        out = mlops.ReLU.apply(self)
        return out

    def sigmoid(self): 
        out = mlops.Sigmoid.apply(self)
        return out
    def softmax(self): 
        out = mlops.Softmax.apply(self)
        return out

    def cross_entropy_loss(self, x): 
        out = mlops.CrossEntropyLoss.apply(self,x)
        return out

    def argmax(self,x): return Tensor(self.data.argmax(x))
    def repeat(self, dim, axis=0): 
        return Tensor(
            self.data.repeat(dim, axis=axis),
            requires_grad=self.requires_grad,
            requires_update=self.requires_update
            )
    # TODO: movement operation should be done 
    def add_bias(self): return mlops.Bias.apply(self)

    def backward(self):
        self.grad = Tensor(np.ones(self.data.shape), requires_grad=False)
        self.nodes = self.deepwalk()[::-1]
        if self.debug:
            print('back')
            print('num nodes',len(self.nodes))
        # there should be 5 total nodes
        for i,node in enumerate(self.nodes):
            if not node._ctx: continue
            #else: pprint(node._ctx, 'node')
            grads = node._ctx.backward(node.grad.data)
            grads = [Tensor(g, requires_grad=False) if g is not None else None \
                for g in ([grads] if len(node._ctx.parents)==1 else grads if isinstance(grads, tuple) else [grads])]
            if self.debug:
                print(f'node {i}: {node.shape}')
                print(f'node {i} parents: {[p.shape for p in node._ctx.parents]}')
                print(f'node grads: {[g.shape if g else None for g in grads]}')
            for n, g in zip(node._ctx.parents, grads):
                if g is not None and n.requires_grad:
                    n.grad = g if n.grad is None else (n.grad+g)

    def deepwalk(self):
        def _deepwalk(node, visited, graph):
            visited.add(node)
            if node._ctx:
                for n in node._ctx.parents:
                    if n not in visited: _deepwalk(n,visited,graph)
            graph.append(node)
            return graph
        return _deepwalk(self, set(), [])

    def getWeights(self):
        if not self.nodes: raise Exception('call backward() first')
        ws = []
        for node in self.nodes:
            if node.requires_update: ws.append(node)
        return ws

class MiniBatchOptimizer(object):
    def __init__(self, lr, reg_strength, decay_rate, decay_step, batch_size):
        self.lr = Tensor(lr)
        self.reg_strength = Tensor(reg_strength)
        self.decay_rate = Tensor(decay_rate)
        self.decay_step = Tensor(decay_step)
        self.batch_size = Tensor(batch_size)
      # inv time decay 
    def decay(self, current_iteration):
        out = self.lr / (Tensor(1) + self.decay_rate * Tensor(current_iteration) / self.decay_step)
        return out

    def step(self, loss: Tensor, iteration: int):
        lr = self.decay(iteration)
        loss.backward()
        weights = loss.getWeights()
        for t in weights:
            t.data -= lr*t.grad/self.batch_size  # Update weights based on gradients
            t.data -= lr * self.reg_strength * t.data  # L2 regularization
        return lr

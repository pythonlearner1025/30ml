import numpy as np
from typing import *
from functools import *

'''
    TODO: implement autograd DL library backbone, to be used
    throughout all my implementations.

    what about arrays of numbers? 
'''

class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.inputs_need_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.inputs_need_grad) else (None if any(x is None for x in self.inputs_need_grad) else False)

    @classmethod
    def apply(fn, *x, **kwargs):
        # initialize custom fn. ctx is object of custom function!
        ctx = fn(*x)
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
    no_grad = False
    default_type = np.float32
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            raise NotImplementedError('yo only np arrays')
        self.data = data 
        self.grad = None
        # for inference 
        self.requires_grad = requires_grad
        # backward context, type Function
        self._ctx = None
    
    def __repr__(self):
        return f'({self.data},{self.grad})'

    def matmul(self, x):
        return mlops.matmul(self.data, x)
    
    def __add__(self, x): return mlops.Add.apply(self,x)
    def __sub__(self, x): return mlops.Sub.apply(self,x)
    def __mul__(self, x): return mlops.Mul.apply(self,x)
    def __pow__(self, x): return mlops.Pow.apply(self,x)
    def __truediv__(self, x): return mlops.Div.apply(self,x)
    def __matmul__(self, x): return mlops.Matmul.apply(self, x)

    # seconda pass impl
    def backward(self):
        self.grad = Tensor(np.ones(self.data.shape), requires_grad=False)
        self.nodes = reversed(self.deepwalk())
        for node in self.nodes:
            grads = node._ctx.backward(node.grad.data)
            grads = [Tensor(g, requires_grad=False) if g is not None else None \
                for g in ([grads] if len(node._ctx.parents)==1 else grads)]
            for n, g in zip(node._ctx.parents, grads):
                if g is not None and n.requires_grad:
                    n.grad = g if n.grad is None else (n.grad + g)
        return self.nodes

    # second pass impl
    def deepwalk(self):
        def _deepwalk(node, visited, graph):
            visited.add(node)
            if node._ctx:
                for n in node._ctx.parents:
                    if n not in visited: _deepwalk(n,visited,graph)
                graph.append(node)
            return graph
        return _deepwalk(self, set(), [])

    def _step(self,lr):
        self.data -= lr*self.grad

    # for now do naive, until optimizer impl
    def step(self, lr):
        if not self.nodes: raise Exception('call backward() first')
        for node in self.nodes:
            if node.requires_grad: node._step(lr)










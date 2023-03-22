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
        # initialize custom fn
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

    # toy example, let's assume computational graph is 
    # z1 = x1 @ w1 
    # z2 = x2 @ w2
    # z3 = z1 + z2 
    # and we call zw.backward()
    def backward(self, debug=False):
        # implicit
        grads = [np.ones(self.data.shape)]
        nodes = self.deepwalk()
        # grad descent
        i = 0
        logger = []
        while i < len(nodes):
            logger.append('*'*50)
            logger.append(f'iter: {i}')
            logger.append(f'grads: {grads}')
            n_grads = len(grads)
            if not n_grads: break 
            children = nodes[i:i+n_grads]
            logger.append(f'children: {children}')
            newgrads = []
            for child,grad in zip(children,grads):
                logger.append(f'child: {child}, grad: {grad}')
                if child._ctx:
                    newgrad = [*child._ctx.backward(grad)]
                    newgrads.extend(newgrad)
                    logger.append(f'newgrad: {newgrad}')
            grads = newgrads
            i+=n_grads

        if debug:
            for log in logger:
                print(log)

        self.nodes = nodes
        return self.nodes

    # return all nodes... without order 
    def deepwalk(self):
        nodes = [self]
        checked = set()
        for node in nodes:
            if node in checked:
                continue
            checked.add(node)
            if node._ctx:
                parents = node._ctx.parents
                for parent in parents:
                    nodes.append(parent)
        return nodes

    def _step(self,lr):
        self.data -= lr*self.grad

    # for now do naive, until optimizer impl
    def step(self, lr):
        if not self.nodes: raise Exception('call backward() first')
        for node in self.nodes:
            if node.requires_grad: node._step(lr)










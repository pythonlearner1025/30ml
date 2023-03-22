from mygrad.tensor import Function
import numpy as np

# why do we return grad for both x and y? how would both be relevant in
# backprop process? 

# assume z = Matmul.apply(x,w)
# assume we have dLdz 
# find z.backward() would return dLdx, dLdw
# now, Tensor for x does not require grad for it is input. Thus, it would be
# marked with requires_grad = False. Then, dLdx would be None  
# and this makes sense because we don't have a use for dLdx (no need to step)
# this will all make more sense once impl backprop
class Matmul(Function):
    def forward(self,x,y):
        self.x,self.y = x,y
        return x@y
    def backward(self,grad):
        return grad@self.y.T if self.inputs_need_grad[0] else None,\
             self.x.T@grad if self.inputs_need_grad[1] else None
    
class Mul(Function):
    def forward(self,x,y):
        self.x, self.y
        return x*y
    def backward(self,grad):
        return grad*self.x if self.inputs_need_grad[0] else None,\
            grad*self.y if self.inputs_need_grad[1] else None

class Add(Function):
    def forward(self,x,y):
        return x+y
    def backward(self,grad):
        return grad if self.inputs_need_grad[0] else None,\
            grad if self.inputs_need_grad[1] else None
        
class Sub(Function):
    def forward(self,x,y):
        return x-y
    def backward(self,grad):
        return grad if self.inputs_need_grad[0] else None,\
            -grad if self.inputs_need_grad[1] else None

class Div(Function):
    def forward(self,x,y):
        self.x, self.y = x,y
        return x/y
    def backward(self,grad):
        return grad/self.y if self.inputs_need_grad[0] else None,\
            grad * -self.x/self.y**2 if self.inputs_need_grad[1] else None

class Pow(Function):
    def forward(self,x,y):
        self.x,self.y, self.ret = x,y, x**y
        return self.ret
    def backward(self,grad):
        # dx = y*x**(y-1) = y*x**y/x
        return grad*self.y*self.ret/self.x if self.inputs_need_grad[0] else None, \
            grad*self.x*np.log(self.y) if self.inputs_need_grad[1] else None


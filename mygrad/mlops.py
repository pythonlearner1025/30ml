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
def pprint(x, name):
    print('-'*10,name,'-'*10)
    print(x)
DEBUG = False

class Matmul(Function):
    def forward(self,x,y):
        self.x,self.y = x,y
        out = x@y
        return out
    def backward(self,grad):
        return grad@self.y.T if self.inputs_need_grad[0] else None,\
             self.x.T@grad if self.inputs_need_grad[1] else None
class Mul(Function):
    def forward(self,x,y):
        self.x, self.y = x,y
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

class ReLU(Function):
    def forward(self, x):
        self.ret = np.maximum(0, x)
        return self.ret

    def backward(self, grad):
        return grad * (self.ret > 0)

class Sigmoid(Function):
    def forward(self, x):
        #print(x)
        #input()
        self.ret = 1.0/(1.0+np.exp(-x))
        return self.ret
    def backward(self, grad):
        return grad*self.ret*(1.0-self.ret)

class Softmax(Function):
    def forward(self,x):
        self.ret = np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)
        return self.ret
    
    def backward(self, grad):
        jacobian = np.einsum('ij,ik->ijk', self.ret, -self.ret) # shape (32,10,10)
        diag = np.zeros(jacobian.shape)
        for i in range(diag.shape[0]):
            np.fill_diagonal(diag[i], self.ret[i])
        jacobian += diag
        grad_input = np.einsum('ij,ijk->ik', grad, jacobian)
        return grad_input

class CrossEntropyLoss(Function):
    def forward(self,x,y):
        self.x,self.y = x,y
        loss = (y*np.log(x)).sum(keepdims=False)/len(y)
        return loss
    
    def backward(self, grad):
        out = grad*-self.y/self.x
        return out

class Bias(Function):
    def forward(self,x):
        self.x = x
        b = np.ones((x.shape[0], 1))
        return np.hstack([x, b])
    
    def backward(self, grad):
        return grad[:,:-1]

class ZeroPad(Function):
    def forward(self, x, p):
        self.p = p
        #print(x.shape)
        B,W,H = x.shape
        z = np.zeros((B,W+2*p,H+2*p))
        z[:,p:-p, p:-p] = x
        return z
    def backward(self, grad):
        p = self.p
        return grad[:, p:-p, p:-p]

# this is fixed under stride = 1
class Convolve(Function):
    # x = input, y = kernels 
    def forward(self,x,y):
        self.x, self.y = x,y
        D,K,K = y.shape
        B,W,H = x.shape
        assert(W==H)
        spread = np.zeros((B,D,K,K))
        for i in range(W-K):
            for j in range(H-K):
                spread[:,i*(W-K)+j] = x[:,i:i+K,j:j+K]
        self.spread = spread
        self.ret = (spread * y).sum(axis=(-2,-1))
        assert(self.ret.shape == (B,D))
        return self.ret

    def backward(self,grad):
        '''
         DL w.r.t to x, w.r.t to y
         for dLdx it would be grad*self.y, but not in shape B,D,K,K
         but in shape B,W,H. 
         this is because there are many overlapping input neurons
         that contribute to a single activation neurons, and thus 
         their contribution must be "aggregated" by overlaying
         the kernels over zero array of size B,W,H again but with += this time.
        '''
        D,K,K = self.y.shape
        B,W,H = self.x.shape
        mat = np.zeros(self.x.shape)
        for i in range(W-K):
            for j in range(H-K):
                mat[:,i:i+K, j:j+K] += self.y[i*(W-K)+j]
        return grad*mat if self.inputs_need_grad[0] else None,\
            grad*self.spread if self.inputs_need_grad[1] else None

class MaxPool(Function):
    def forward(self,x):
        self.x = x
        B,W,H = x.shape
        mat = np.zeros((B,W//2,H//2))
        for i in range(W//2):
            for j in range(H//2):
                mat[:,i,j] = x[:,i*2:i*2+2,j*2:j*2+2].sum(axis=(-1,-2))
        return mat

    def backward(self,grad):
        mat = np.zeros(self.x.shape)
        B,W,H = grad.shape
        for i in range(W):
            for j in range(H):
                mat[:,i*2:i*2+2,j*2:j*2+2]+=np.ones((2,2))*grad[:,i,j] 
        return grad * mat

# TODO: i need reshape operator for gradients in between 
# conv and linear

class Flatten(Function):
    def forward(self,x):
        B,W,H = x.shape
        self.prev_shape = B,W,H
        return x.reshape(B,W*H)
    
    def backward(self,grad):
        B,W,H = self.prev_shape
        return grad.reshape(B,W,H)
        
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
        B,C,W,H = x.shape
        z = np.zeros((B,C,W+2*p,H+2*p))
        z[:,:,p:-p, p:-p] = x
        return z
    def backward(self, grad):
        p = self.p
        return grad[:,:,p:-p, p:-p]

class Convolve(Function):
    # x = input, y = kernels 
    def forward(self,img,kernel,S):
        self.img, self.kernel, self.S = img, kernel, S
        F,C,K,K = self.kernel.shape
        B,C,W,H, = self.img.shape
        out_size = int((W-K)/S+1)
        out = np.zeros((B,F,out_size,out_size))
        def conv(x,y):
            #print(x.shape)
            #print(y.shape)
            B,W,H = x.shape
            K,K = y.shape
            out = np.zeros((B,out_size,out_size))
            for i in range(out_size):
                for j in range(out_size):
                    out[:,i,j] = (x[:,S*i:S*i+K,S*j:S*j+K]*y).sum(axis=(-2,-1))
            return out

        for i in range(F):
            for j in range(C):
                out[:,i] += conv(img[:,j], kernel[i,j])
        self.out = out
        return out 

    def backward(self,grad):
        '''
         grad shape = B,F,O,O
         1. DLdx B,C,W,H
         grad * self.kernel 
         2. DLdy F,C,K,K
         grad * self.img
        '''
        # grad shape = B,F,O,O
        
        S = self.S
        F,C,K,K = self.kernel.shape
        B,C,W,H = self.img.shape
        mat = np.zeros((B,C,W,H))
        mat2 = np.zeros((F,C,K,K))
        out_size = int(((W-K)/S)+1)
        assert(out_size == grad.shape[-1])
        for i in range(F):
            for j in range(C):
                for k in range(out_size):
                    for l in range(out_size):
                        # TODO: fix                       # 32,1,1,1          1,5,5 
                        mat[:,j,S*k:S*k+K,S*l:S*l+K] += grad[:,i,j,k].reshape(B,1,1,1)*self.kernel[i,j].reshape(1,K,K)
                        mat2[i,j] += grad[:,i,j,k]*self.img[:,j,S*k:S*k+K,S*l:S*l+K]

        return mat if self.inputs_need_grad[0] else None,\
             mat2 if self.inputs_need_grad[1] else None

class MaxPool(Function):
    def forward(self,x):
        self.x = x
        B,C,W,H = x.shape
        mat = np.zeros((B,C,W//2,H//2))
        for j in range(W//2):
            for k in range(H//2):
                mat[:,:,j,k] = x[:,:,j*2:j*2+2,k*2:k*2+2].sum(axis=(-1,-2))
        return mat

    def backward(self,grad):
        B,C,W,H = grad.shape
        mat = np.zeros(self.x.shape)
        for j in range(W):
            for k in range(H):
                mat[:,:,j*2:j*2+2,k*2:k*2+2] = np.ones((B,C,2,2))*grad[:,:,j,k].reshape(B,C,1,1) 
        return mat

# TODO: i need reshape operator for gradients in between 
# conv and linear

class Flatten(Function):
    def forward(self,x):
        B,C,W,H = x.shape
        self.x = B,C,W,H
        return x.reshape(B,C*W*H)
    
    def backward(self,grad):
        B,C,W,H = self.x
        return grad.reshape(B,C,W,H)
        
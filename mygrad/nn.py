from mygrad.tensor import Tensor
import numpy as np
import math

class Sigmoid:
    def __init__(self):
        self.a = None
    
    def __call__(self,x): return self.forward(x)
    
    def forward(self, x):
        self.a = 1.0 / (1.0+np.exp(-x))
        return self.a
    
    def backward(self, dz):
        return dz*self.a*(1.0-self.a) 

def softmax(x):
    s = np.exp(x) / (np.exp(x)).sum(axis=1, keepdims=True)
    return s

class OldLayer:
    def __init__(self, input_size, output_size, lr):
        self.w = np.random.uniform(-1.0, 1.0, (input_size, output_size)).astype(np.float32)
        self.w = np.vstack([self.w, np.ones((1, output_size))])
        self.dLdw = None
        self.x = None
        self.lr = lr

    def __call__(self, x): return self.forward(x)

    def forward(self, x):
        x_biases = np.ones((x.shape[0], 1))
        x = np.hstack([x, x_biases])
        self.x = x
        z = x @ self.w
        return z

    def backward(self, dLdz, debug=False):
        dzdx = self.w[:-1,:].T
        dLdx = dLdz @ dzdx
        dzdw = self.x.T
        self.dLdw = dzdw @ dLdz 
        if debug:
            print(f'dLdz shape: {dLdz.shape}')
            print(f'dLdx shape: {dLdx.shape}')
            print(f'dzdw shape: {dzdw.shape}')
            print(f'dw shape: {self.dLdw.shape}')
        return dLdx
    
    def step(self):
        b_size = self.x.shape[0]
        self.w -= self.lr * self.dLdw / b_size

# TODO: automate this 
class OldMLP:
    def __init__(self, input_size, h1_size, h2_size, h3_size, output_size, lr):
        self.input_layer = Layer(input_size, h1_size, lr)
        self.h1 = Layer(h1_size, h2_size, lr)
        self.h2 = Layer(h2_size, h3_size, lr)
        self.h3 = Layer(h3_size, output_size, lr)
        self.activation1 = Sigmoid()
        self.activation2 = Sigmoid()
        self.activation3 = Sigmoid()
        # i want logits for outputs
    
    def __call__(self,x): return self.forward(x)

    def update_learning_rate(self,lr):
        self.h1.lr = lr
        self.h2.lr = lr
        self.h3.lr = lr
    
    def forward(self, x):
        z1 = self.input_layer(x)
        h1 = self.activation1(z1)
        z2 = self.h1(h1)
        h2 = self.activation2(z2)
        z3 = self.h2(h2)
        h3 = self.activation3(z3)
        z4 = self.h3(h3)
        s = softmax(z4)
        return s

    # dLdz = softmax loss = s - y
    def backward(self, y, y_pred):
        dLdz4 = y_pred-y
        dz4dh3 = self.h3.backward(dLdz4)
        dLdz3 = self.activation3.backward(dz4dh3)
        dz3dh2 = self.h2.backward(dLdz3)
        dh2dz2 = self.activation2.backward(dz3dh2)
        dz2dh1 = self.h1.backward(dh2dz2)
        dh1dz1 = self.activation1.backward(dz2dh1)
        dz1dx = self.input_layer.backward(dh1dz1)

    # call after 
    def step(self, n):
        self.h3.step(n)
        self.h2.step(n)
        self.h1.step(n)
        self.input_layer.step(n)

'''
class Layer:
    def __init__(self, input_size, output_size):
        w = np.random.uniform(-0.03, 0.03, (input_size, output_size)).astype(np.float32)
        w = np.vstack([w, np.ones((1, output_size))])
        self.w = Tensor(w, requires_grad=True, requires_update=True)

    def __call__(self, x: Tensor): return self.forward(x)

    def forward(self, x: Tensor)-> Tensor:
        x = x.add_bias()
        #print('forward')
        #print(self.w.shape, self.w.requires_update)
        z = x @ self.w
        #print('z')
        #print(z.shape)
        #print(z._ctx)
        #print(z.requires_grad)
      #  if (z._ctx): print([p.shape for p in z._ctx.parents])
        return z
'''
class Layer:
    def __init__(self, input_size, output_size):
        # xavier activation#
        W = np.random.normal(0, np.sqrt(2 / (input_size + output_size)), size=(input_size, output_size))
        self.w = Tensor(W, requires_grad=True, requires_update=True)
        self.b = Tensor(np.ones((1, output_size), dtype=np.float32),
                        requires_grad=True, requires_update=True)

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        #print('x',x.shape,'w', self.w.shape, 'b', self.b.shape)
        z = (x @ self.w)
        z += self.b.repeat(z.shape[0], axis=0)
        return z

# TODO: automate this 
class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(len(sizes)) if i != len(sizes)-1]
        print('layers count: ', len(self.layers))
    
    def __call__(self,x): return self.forward(x)
    
    def forward(self, x: Tensor):
        for i in range(len(self.layers)):
            z = self.layers[i](x)
            if i != len(self.layers)-1:
                x = z.relu()
            else:
                x = z.softmax()
        return x
    


        

import numpy as np
from mygrad.tensor import Tensor

def pprint(x, name):
    print('-'*10,name,'-'*10)
    print(x)

def one_hot(hot, size):
    res = [0] * size
    res[int(hot)] = 1
    return res

def loss(y,p):
    return (y*np.log(p)).sum()/len(y)

def highest(L):
    res = []
    for i in range(len(L)):
        res.append(L[i].index(max(L[i])))
    return res 

def to_labels(b):
    idx = np.where(b==1)[1]
    return idx

def minmax(x, scale=255.0):
    return x/scale


def test_accuracy(X: Tensor, y: Tensor, nn) -> tuple[np.ndarray, float]:
    if not isinstance(X, Tensor):
        X = Tensor(X)
    if not isinstance(y, Tensor):
        y = Tensor(y)
    def matches(y, y_pred):
        c = 0
        for i in range(len(y)):
            for j in range(len(y[0])):
                if y[i][j] == 1:
                    if y_pred[i][j] == 1: 
                        c+=1
                    break
        return c
    y_pred = nn(minmax(X).reshape(len(X),28,28))
    #print('test')
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    #print(b)
    #print(y)
    corr = matches(b, y)
    return to_labels(b), (corr / len(y))
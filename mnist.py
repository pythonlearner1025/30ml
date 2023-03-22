import csv
import random
import numpy as np
from mygrad.nn import *
from itertools import islice

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

def test_accuracy(X: np.ndarray, y: np.ndarray, nn) -> tuple[np.ndarray, float]:
    def matches(y, y_pred):
        c = 0
        for i in range(len(y)):
            for j in range(len(y[0])):
                if y[i][j] == 1:
                    if y_pred[i][j] == 1: 
                        c+=1
                    break
        return c
    y_pred = nn(X)
    #print('test')
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    #print(b)
    #print(y)
    corr = matches(b, y)
    return to_labels(b), (corr / len(y))


if __name__ == '__main__':
    train = '/Users/minjunes/30ml/mnist/data/train.csv'
    with open(train, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # skip header row
        X, Y = [], []
        for row in islice(csv_reader, 0, 40000):
            X.append(row[1:])
            Y.append(one_hot(row[0], 10))

    train_split = int(0.8*len(X))
    X_train,Y_train = np.array(X).astype(np.float32)[:train_split], np.array(Y).astype(np.float32)[:train_split]
    X_test,Y_test = np.array(X).astype(np.float32)[train_split:], np.array(Y).astype(np.float32)[train_split:]

    epoch_count = 500000
    batch_n = 32 
    initial_learning_rate = 0.15
    decay_rate = 0.001
    # what was SGD? 
    # basically avg across gradients of L w.r.t to W for N,
    # and use this approximte gradient to update.

    print(X_train.shape[1])
    mlp = MLP(X_train.shape[1], 512, 256, 128, 10, initial_learning_rate)
    #mlp = MLP2(X.shape[1], 128, 10, 0.0001)
    for i in range(epoch_count):
        learning_rate = initial_learning_rate / (1 + decay_rate * i)
        mlp.update_learning_rate(learning_rate)
        


        idxs = np.random.permutation(len(X_train))[:batch_n]
        x,y = X_train[idxs, :], Y_train[idxs, :]
        #print(x.shape, y.shape)
        y_preds = mlp(x)
        #print(highest(y.tolist()))
        #print(highest(y_preds.tolist()))
        L = loss(y, y_preds)

        mlp.backward(y, y_preds)
        mlp.step(batch_n)

        if i % 1000 == 0:
            acc = test_accuracy(X_test,Y_test,mlp)
            print(f'{i} loss: {L}, lr: {learning_rate}, acc: {acc[1]}')
            
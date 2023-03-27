from mygrad.tensor import Tensor, MiniBatchOptimizer
from mygrad.nn import CNN
from mygrad.utils import *
from itertools import islice
import numpy as np
import csv
import cv2 

def show(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

# TODO: more ops in Tensor, like shape, ones, rand, and mlops
# autograd works, at least naively without real mlops
if __name__ == '__main__':
    train = '/Users/minjunes/30ml/data/mnist/train.csv'
    with open(train, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # skip header row
        X, Y = [], []
        for row in islice(csv_reader, 1, 10000):
            X.append(row[1:])
            Y.append(one_hot(row[0], 10))

    train_split = int(0.8*len(X))
    X_train,Y_train = np.array(X).astype(np.float32)[:train_split], np.array(Y).astype(np.float32)[:train_split]
    X_test,Y_test = np.array(X).astype(np.float32)[train_split:], np.array(Y).astype(np.float32)[train_split:]

    epoch_count = 10000
    batch_n = 32
    initial_learning_rate = 1e-4
    decay_rate = 1*3
    decay_step = epoch_count // 2
    reg_strength = 1e-4

    # sample
    '''
    x_s = X_train[0]
    x_s = x_s.reshape(28,28).astype(np.uint8)
    show('one', x_s)
    '''
    cnn = CNN(28,5,1,2)
    optim = MiniBatchOptimizer(
        initial_learning_rate,
        reg_strength,
        decay_rate,
        decay_step,
        batch_n
        )
    
    
    
    for i in range(epoch_count):
        idxs = np.random.permutation(len(X_train))[:batch_n]
        x,y = Tensor(X_train[idxs, :]), Tensor(Y_train[idxs, :])
        y_preds = cnn(minmax(x).reshape(len(x),28,28))
        loss = y_preds.cross_entropy_loss(y)
        lr = optim.step(loss, i)

        if i % 100 == 0 or i == epoch_count-1:
            acc = test_accuracy(X_test,Y_test,cnn)
            print(f'{i} loss: {loss}, lr: {lr}, acc: {acc[1]}')



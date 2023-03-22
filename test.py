from mygrad.tensor import Tensor
import numpy as np

# TODO: more ops in Tensor, like shape, ones, rand, and mlops
# autograd works, at least naively without real mlops
if __name__ == '__main__':
    a, b, c, d = np.array([[10]]), np.array([[1]]), np.array([[1]]), np.array([[10]])
    z1, z2 = Tensor(a, requires_grad=True) @ Tensor(b, requires_grad=True), \
         Tensor(c, requires_grad=True) @ Tensor(d, requires_grad=True)
    z = z1 @ z2
    print(type(z))
    print(z)

    z.backward(debug=True)
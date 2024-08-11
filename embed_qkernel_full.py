

# source: https://pennylane.ai/qml/demos/tutorial_kernels_module/
# evaluate kernels, use them for classification and train them with gradient-based optimization
# only for training full data on PRETRAINED parameters (for full code, including training parameters go to embed_qkernel_orig)

from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# LOAD DATA 
X = np.loadtxt("trainX.txt")
Y = np.loadtxt("trainY.txt")

import pennylane as qml

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0][j], wires=[wire])
        # qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])
    # qml.broadcast(unitary=qml.CRZ, pattern="chain", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

dev = qml.device("default.qubit", wires=4, shots=None)
wires = dev.wires.tolist()

@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0] # kernel value = probabality of |000> (all-zero state)


# In[20]:


def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


# In[21]:


import time
start = time.time()

init_params = [[[ 4.678508  ,  4.14885934,  3.155578  ,  2.65471382],
  [ 5.20051584,  3.86308722,  2.19494955,  5.26733587]],

 [[ 0.39929889,  5.99720972,  3.50568351,  4.50757321],
  [ 2.41379447,  2.45148959,  3.43955678,  4.54858641]],

 [[ 1.86522407,  5.09149427,  4.95372749,  5.40004187],
  [ 5.62381085,  2.82733993,  0.64518751,  4.7338107 ]],

 [[ 4.56582763,  2.69783208,  1.1902086 ,  6.24481321],
  [-0.03951283,  1.21197015,  5.99104953,  2.57392452]],

 [[ 5.05880251, -0.07648325,  4.85126563,  0.25246614],
  [ 2.97848995,  1.01900462,  0.57240624,  1.52812446]],

 [[ 0.15334048,  5.05816494,  4.85652837,  3.15806128],
  [ 5.70443867,  6.30528542,  3.13471566,  5.59195699]]]

init_kernel = lambda x1, x2: kernel(x1, x2, init_params)

svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)

accuracy_init = accuracy(svm, X, Y)
print(f"The training accuracy of the kernel with pre-trained parameters on the full data is {accuracy_init:.3f}")

end = time.time()
print("runtime in min: ", (end-start)/60)

# PRINT MATRIX
# K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
# with np.printoptions(precision=3, suppress=True):
#     print(K_init)

# TEST DATA ACCURACY
Xtest = np.loadtxt("testX.txt")
Ytest = np.loadtxt("testY.txt")
accuracy_test = accuracy(svm, Xtest, Ytest)
print(f"The testing accuracy of a kernel with trained parameters is {accuracy_test:.3f}")




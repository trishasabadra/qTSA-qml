

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

dev = qml.device("default.qubit", wires=5, shots=None)
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

init_params = [[[4.34915284, 1.90521043, 0.89487594, 2.89041899, 0.02271626],
  [4.72476861, 1.92149705, 1.57909534, 3.5183381, 4.93010125]],

 [[3.8192236, 5.42214887, 5.87692541, 0.66209829, 2.10644943],
  [4.33952621, 5.8125166, 1.21813832, 5.7992338, 1.12446473]],

 [[4.30806575, 3.37587106, 1.34349696, 1.90229268, 4.83968543],
  [2.46908815, 4.91264202, 2.37852693, 4.66077644, 1.83491211]],

 [[0.62740497, 6.96062317, 2.30073218, 0.14672272, 3.80292147],
  [5.16528554, 1.68618512, 1.50066882, 5.84425022, 0.07635773]],

 [[5.28079018, 3.47458228, 4.6359661, 5.11342561, 3.16168067],
  [4.89732651, 1.31611988, 5.82376933, 1.44690304, 0.23477567]],

 [[0.56361251, 3.66743937, 3.17546828, 3.879045, 1.5368378],
  [4.81617029, 4.00503433, 0.08591562, 0.20144711, 2.81264479]],

 [[1.19648398, 4.52677161, 0.32882232, 4.00199811, 5.20513432],
  [4.12162333, 1.76792419, 3.85560848, 0.61976434, 2.87908385]]]

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




from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
start = time.time()

# LOAD DATA 
X_full = np.loadtxt("trainX.txt")
Y_full = np.loadtxt("trainY.txt")
# Y_train = np.where(Y == 0, -1.0, 1.0)
X = X_full[:400]
Y = Y_full[:400]

# Defining a Quantum Embedding Kernel
# ===================================
# 
# PennyLane\'s [kernels
# module](https://pennylane.readthedocs.io/en/latest/code/qml_kernels.html)
# allows for a particularly simple implementation of Quantum Embedding
# Kernels. The first ingredient we need for this is an *ansatz*, which we
# will construct by repeating a layer as building block. Let\'s start by
# defining this layer:

import pennylane as qml

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


# To construct the ansatz, this layer is repeated multiple times, reusing
# the datapoint `x` but feeding different variational parameters `params`
# into each of them. Together, the datapoint and the variational
# parameters fully determine the embedding ansatz $U(\boldsymbol{x})$. In
# order to construct the full kernel circuit, we also require its adjoint
# $U(\boldsymbol{x})^\dagger$, which we can obtain via `qml.adjoint`.
# 

# In[6]:


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)


# Together with the ansatz we only need a device to run the quantum
# circuit on. For the purpose of this tutorial we will use PennyLane\'s
# `default.qubit` device with 5 wires in analytic mode.
# 

# In[7]:


dev = qml.device("default.qubit", wires=7, shots=None)
wires = dev.wires.tolist()


# Let us now define the quantum circuit that realizes the kernel. We will
# compute the overlap of the quantum states by first applying the
# embedding of the first datapoint and then the adjoint of the embedding
# of the second datapoint. We finally extract the probabilities of
# observing each basis state.
# 

# In[8]:


@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)


# The kernel function itself is now obtained by looking at the probability
# of observing the all-zero state at the end of the kernel circuit --
# because of the ordering in `qml.probs`, this is the first entry:
# 

# In[9]:


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]

# In[10]:

init_params = random_params(num_wires=7, num_layers=6)

init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)

with np.printoptions(precision=3, suppress=True):
    print(K_init)


# In[17]:


def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


# In[18]:


# svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)
# accuracy_init = accuracy(svm, X, Y)
# print(f"The accuracy of the kernel with random parameters is {accuracy_init:.3f}")

kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)

print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")



# In[21]:


from pennylane.optimize import AdamOptimizer

def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))

    # Check if norm is zero to avoid division by zero
    if norm == 0:
        return inner_product
    
    inner_product = inner_product / norm

    return inner_product


params = init_params
opt = qml.AdamOptimizer(0.01)

for i in range(250): # originally 500
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X))), 4)
    # Define the cost function for optimization
    cost = lambda _params: -target_alignment(
        X[subset],
        Y[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 25 == 0:
        current_alignment = target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")


# We want to assess the impact of training the parameters of the quantum
# kernel. Thus, let\'s build a second support vector classifier with the
# trained kernel:
# 

# In[22]:

print("TRAINED parameters for 5 qubits, not full data: ", params)
# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

from sklearn.svm import SVC
# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y)
# In[23]:


accuracy_trained = accuracy(svm_trained, X, Y)
print(f"The accuracy of a kernel with trained parameters and subset of data is {accuracy_trained:.3f}")


accuracy_trained = accuracy(svm_trained, X_full, Y_full)
print(f"The accuracy of a kernel with trained parameters and FULL data is {accuracy_trained:.3f}")

svm2 = SVC(kernel=trained_kernel_matrix).fit(X_full, Y_full)
acc_full = accuracy(svm2, X_full, Y_full)
print(f"The accuracy of a kernel with SVM trained on FULL data is {acc_full:.3f}")

end = time.time()
print("runtime in min: ", (end-start)/60)

# TEST DATA
X_test = np.loadtxt("testX.txt")
Y_test = np.loadtxt("testY.txt")
accuracy_test = accuracy(svm_trained, X_test, Y_test)
print(f"The accuracy of a kernel on the test data is: {accuracy_test:.3f}")
# trained_plot_data = plot_decision_boundaries(svm_trained, plt.gca())


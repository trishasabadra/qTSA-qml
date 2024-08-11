# In[13]:


from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.svm import SVC
import time

start = time.time()

# LOAD DATA
X_full = np.loadtxt("trainX.txt")
Y_full = np.loadtxt("trainY.txt")

X = X_full[:400]
Y = Y_full[:400]

# Create a PennyLane device
# dev = qml.device('default.mixed', wires=4)
# wires = dev.wires.tolist()

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    p = 0.01 # probability of noise
    i = i0
    for j, wire in enumerate(wires):
        # qml.BitFlip(p, wires=[wire]) # adds noise  
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0][j], wires=[wire])
        # qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

# NOISY CODE: from Yifan
import qiskit
import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error

def setNoise(p_reset, p_meas, p_gate1):
    # Example error probabilities
    p_reset = 0.03   # reset error
    p_meas = 0.1     # measurement error
    p_gate1 = 0.001  # 1-qubit gate
    p_gate2 = 0.01   # 2-qubit gate
    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x90", "u1", "u2", "u3", "U", "id", "x", "y", "z", "h", "s", "sdg",
        "t", "tdg", "r", "rx", "ry", "rz", "p"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx", "cy", "cz", "swap", "rxx", "ryy", "rzz",
                                "rzx", "cu1", "cu2", "cu3", "cp"])

    return noise_model

# create noise model
error_Set = [10**i for i in np.arange(-4.5, -0.9, 0.1)]  # a set of gate errors
my_noise_model = setNoise(0.03, 0.04413, error_Set[0])
dev = qml.device('qiskit.aer', wires=4, noise_model = my_noise_model)
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


# In[ ]:


from pennylane.optimize import AdamOptimizer

def target_alignment( X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True,):
    """Kernel-target alignment between kernel and labels."""

    # calculate kernel matrix K
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

    # calculate target matrix T
    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))

    # Check if norm is zero to avoid division by zero
    if norm == 0:
        return inner_product

    inner_product = inner_product / norm

    return inner_product

params = random_params(num_wires=4, num_layers=6)
opt = qml.AdamOptimizer(0.01)
batch_size = 4

for i in range(200):
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X))), batch_size)
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
    if (i + 1) % 50  == 0:
        current_alignment = target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")


# In[ ]:


print("Optimized paramters: ", params)

# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.
svm = SVC(kernel=trained_kernel_matrix).fit(X, Y)

accuracy_init = accuracy(svm, X, Y)
print(f"The training accuracy of the kernel with optimized parameters on the subset data is {accuracy_init:.3f}")

accuracy_init = accuracy(svm, X_full, Y_full)
print(f"The accuracy on the FULL data is {accuracy_init:.3f}", flush=True)

svm2 = SVC(kernel=trained_kernel_matrix).fit(X_full, Y_full)
acc_full = accuracy(svm2, X_full, Y_full)
print(f"The accuracy of a kernel with SVM trained on FULL data is {acc_full:.3f}")

end = time.time()
print("runtime in min: ", (end-start)/60)

# TEST DATA ACCURACY
Xtest = np.loadtxt("testX.txt")
Ytest = np.loadtxt("testY.txt")
accuracy_test = accuracy(svm, Xtest, Ytest)
print(f"The testing accuracy of a kernel with trained parameters is {accuracy_test:.3f}")

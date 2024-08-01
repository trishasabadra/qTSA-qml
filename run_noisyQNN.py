# __author__ = "Yifan Zhou"
# __copyright__ = "Copyright 2021, DOE Project"
# __email__ = "yifan.zhou.1@stonybrook.edu"
import pennylane as qml
from setnoise import setNoise
#from pennylane import numpy as np
import numpy as np

# this is the ideal device you previously used
n_qubits = 5
dev_ideal = qml.device("default.qubit", wires=n_qubits) # set up an ideal quantum device for baseline performance evaluation.
# dev_ideal = qml.device("default.qubit", analytic=True, wires=n_qubits) - says analytic parameter doesn't exist

# add gate error to see its impact (you can definitely change the noise level)
# error_Set = [10**i for i in np.arange(-4.5, n-0.9, 0.1)]  # a set of gate errors - n is not defined
error_Set = [10**i for i in np.arange(-4.5, -0.9, 0.1)]  # a set of gate errors
# test QNN under a set of gate errors

for i_Set in range(len(error_Set)-1,-1,-1):
    # set up noise level
    gate_error = error_Set[i_Set]
    noise_model = setNoise(0.03, 0.04413, gate_error)
    # set up noisy device
    dev_noisy = qml.device("qiskit.aer", wires=n_qubits, backend='qasm_simulator', backend_options={"basis_gates": noise_model.basis_gates, "noise_model": noise_model})

    # initialize your qnn and run it using dev_noisy instead of dev_ideal
    #<*test_your_qnn*>

        

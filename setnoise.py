# __author__ = "Yifan Zhou"
# __copyright__ = "Copyright 2021, DOE Project"
# __email__ = "yifan.zhou.1@stonybrook.edu"


from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error
#from qiskit.providers.aer.noise import noise
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise import QuantumError, ReadoutError
# from qiskit.providers.aer.noise import pauli_error
# from qiskit.providers.aer.noise import depolarizing_error
# from qiskit.providers.aer.noise import thermal_relaxation_error


def setNoise(p_reset, p_meas, p_gate1):
#     # Example error probabilities
#     p_reset = 0.03   # reset error
#     p_meas = 0.1     # measurement error
#     p_gate1 = 0.001  # 1-qubit gate
#     p_gate2 = 0.01   # 2-qubit gate
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

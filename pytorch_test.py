import pennylane as qml
import torch

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, interface='torch')
def circuit(phi, theta):
    qml.RX(phi[0], wires=0)
    qml.RY(phi[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(theta, wires=0)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))
from qiskit_algorithms.utils import algorithm_globals
import time

algorithm_globals.random_seed = 12345

import numpy as np
adhoc_dimension = 2
train_features = np.loadtxt("trainX.txt") # size 1600
train_labels = np.loadtxt("trainY.txt")
test_features = np.loadtxt("testX.txt") # size 256
test_labels = np.loadtxt("testY.txt")
# print(len(train_features))
adhoc_total = 1856

# size 20 -> accuracy 0.58
# size 100 -> 0.65
# size 200 -> 0.755
# size 256 -> 0.789; time = 6.09 min
# size 500 -> 0.83; time = 16.55
# size 700 -> 0.879; time = 34 min

train_features = train_features[:700]
train_labels = train_labels[:700]
test_features = test_features[:256]
test_labels = test_labels[:256]
adhoc_total = 1856

# ZZFeatureMap: A quantum feature map used to encode data into quantum states.
# FidelityQuantumKernel: A quantum kernel using the fidelity (overlap) between quantum states as the kernel value.

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

from sklearn.svm import SVC
start = time.time()
print(1)
adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
print(2)
adhoc_svc.fit(train_features, train_labels)
print(3)
adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)

print(f"Callable kernel classification test score: {adhoc_score_callable_function}")

end = time.time()

print("runtime in min: ", (end-start)/60)

# RBF KERNEL
# start = time.time()
# print(1)
# # Create an SVC classifier with the RBF kernel
# rbf_svc = SVC(kernel='rbf', gamma='scale')
# print(2)
# # Fit the classifier to your training data
# rbf_svc.fit(train_features, train_labels)
# print(3)
# # Evaluate the classifier on your test data
# rbf_score = rbf_svc.score(test_features, test_labels)
# print(f'RBF Kernel Accuracy: {rbf_score}')
#
# end = time.time()
# print("runtime in min: ", (end-start)/60)
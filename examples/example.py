from pyflann import *
import numpy as np
import pyflann
print pyflann.__file__

dataset = np.random.rand(5000, 128)
dataset2 = np.random.rand(5000, 128)
testset = np.random.rand(1000, 128)

flann = FLANN()

index = flann.build_index(dataset, algorithm="kmeans")
flann.add_points(dataset2)
result, dists = flann.nn_index(testset, 2)

print result, dists

"""result, dists = flann.nn(
            dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16
        )
print result
print dists

dataset = np.random.rand(10000, 128)
testset = np.random.rand(1000, 128)
flann = FLANN()
result, dists = flann.nn(
            dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16
        )
print result
print dists"""

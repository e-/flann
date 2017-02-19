from pyflann import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

all_data = np.loadtxt("data/glove.shuffled.txt")
testset = np.loadtxt("data/glove.query.txt")

repeat = 10
num_rows = 5000
test_size = 1000
dim = all_data.shape[1]
nn = 20

dataset = all_data[:num_rows]
testset = np.random.rand(test_size, dim)

flann = FLANN()
#index = flann.build_index(dataset, algorithm="kdtree", trees = 8, checks = 4000)
index = flann.build_index(dataset, algorithm="kdtree_balanced", trees = 8, checks = 4000, rebuild_imbalance_threshold=1.05, rebuild_size_threshold=1.0002)

for r in xrange(repeat):
    print "Iter%d: " % r, 
    inc = all_data[num_rows*(r+1):num_rows*(r+2)]
    dataset = all_data[:num_rows*(r+2)]
    
    build_start = time.clock()
    flann.add_points(inc)
    build_end = time.clock()

    query_start = time.clock()
    ann_result, dists = flann.nn_index(testset, nn)
    query_end = time.clock()

    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(dataset)
    distances, knn_result = nbrs.kneighbors(testset)
    
    correct = 0
    for ann, knn in zip(ann_result, knn_result):
        correct = correct + len(np.intersect1d(ann, knn))
    
    acc = float(correct) / nn / test_size
    
    print "build_time = %.3f, QPS = %.1f, acc = %.3f" % (build_end - build_start, test_size / (query_end - query_start), acc)

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

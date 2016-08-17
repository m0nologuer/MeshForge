#Create feature vector


import mesh_processing
import mesh_library
import remesh_voxel
import numpy

import scipy.cluster
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


remesher = remesh_voxel.RemeshVoxel()
processor = mesh_processing.MeshProcessor()

humans = processor.load("../Datasets/Labelled_Segmented_Models/Human/")


import csv
import numpy

import scipy.cluster
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

hashcsv = []
with open('quiz.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		hashcsv.append(row.values())

feature_vectors = [[float(x) for x in (row[:8]+row[9:])] for row in hashcsv]

pca = PCA()
pca.fit(numpy.array(feature_vectors))
reduced_feature_vectors = pca.transform(feature_vectors)

distance_mat = pairwise_distances(reduced_feature_vectors, metric="cosine")
linkage = scipy.cluster.hierarchy.centroid(distance_mat)

clusters = scipy.cluster.hierarchy.fcluster(linkage, 4, criterion="maxclust")

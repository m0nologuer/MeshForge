import numpy
import scipy
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
        
def ReLU(x):
    return x * (x > 0)

def cluster(feature_vectors, cluster_count=None):
	pca = PCA()
	pca.fit(numpy.array(feature_vectors))
	reduced_feature_vectors = pca.transform(feature_vectors)

	distance_mat = pairwise_distances(reduced_feature_vectors, metric="cosine")
	linkage = scipy.cluster.hierarchy.centroid(distance_mat)

	clusters = None
	if (cluster_count==None):
		clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.9)
	else:
		clusters = scipy.cluster.hierarchy.fcluster(linkage, cluster_count, criterion="maxclust")

	return clusters

def graph_cycle_search(graph, current_node, nodes, visited):

	if visited[current_node.label]:
		return True

	visited[current_node.label] = True

	# Real node children
	for node in current_node.children:
		self.graph_cycle_search(graph, node, nodes, visited)

	# "Artificial" children 
	for edge in graph:
		#Edge[0] and edge[1] are opposite sides of an edge
		if edge[0] == current_node.label:
			next_node = [node for node in nodes if node.label == edge[1]][0]
			if self.graph_cycle_search(graph, next_node, nodes, visited):
				return True
				
			return False

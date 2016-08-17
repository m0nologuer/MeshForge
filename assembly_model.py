import tensorflow as tf
import numpy_helpers
from tensorflow.python.framework import ops
import numpy
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import scipy.stats as stats
rng = numpy.random
import random
from random import shuffle
import math

import mesh_processing

class ProbabilityNode(object):
	"""Node in the probabilistic model that generates meshes"""
	def __init__(self, label, distribution, continuous=False):
		self.label = label
		self.distribution = distribution
		self.is_continuous = continuous

		#Links to other nodes
		self.children = []
		self.parents = []

		self.graph_initialized = False

	def add_parent(self, parent_node):
		self.parents.append(parent_node)
		parent_node.children.append(self)

	def build_node_tf_graph(self, batch_size):

		if (self.graph_initialized):
			return

		if (self.is_continuous):
			self.n = len(self.distribution["Mean"])
			self.output_size = self.n + self.n*(self.n+1)/2 + self.n #Size of vector + 1/2 covariance matrix
			self.feature_vectors = [tf.placeholder(tf.float32, shape= [self.n] ) for _ in range(batch_size)]
		else:
			self.output_size = len(self.distribution)
			self.feature_vectors = [tf.placeholder(tf.float32, shape= [len(self.distribution)] ) for _ in range(batch_size)]


		#If this is the root node
		if (len(self.parents) == 0):
			self.weights= [tf.get_variable("w_root", 
				[1, self.output_size], initializer=tf.random_normal_initializer())]
			self.biases = [tf.get_variable("b_root", 
				[self.output_size], initializer=tf.constant_initializer(0.0))]
			layer_in = numpy.reshape(1.0, (1,1)).astype("float32")
			self.layer_output = tf.matmul(layer_in, self.weights[0])+self.biases[0]
			self.node_outputs = [self.layer_output]
		else:
			self.node_outputs = []
			self.weights = []
			self.biases = []

			outputs = 0

			for parent in self.parents:
				#Initialize nodes for each parent
				input_size = len(parent.distribution)
				w = tf.get_variable("w_{}_{}".format(parent.label, self.label), 
					[input_size, self.output_size], initializer=tf.random_normal_initializer())
				b = tf.get_variable("b_{}_{}".format(parent.label, self.label), 
					[self.output_size], initializer=tf.constant_initializer(0.0))

				if (parent.graph_initialized == False):
					parent.build_node_tf_graph()

				# nn operators
				node_output = tf.matmul(parent.layer_output,w)+b
				if (outputs == 0):
					outputs = node_output
				else:
					outputs = outputs + node_output
				self.node_outputs.append(node_output)
				self.weights.append(w)
				self.biases.append(b)

			self.layer_output = outputs


		if self.is_continuous:
			#First n elements give you the mean vector
			error = tf.reduce_sum(tf.abs(self.feature_vector-self.layer_output[0,:self.n]))

			#The rest of the layer gives half the covariance matrix
			matrix_index = self.n
			for i in range(self.n):
				for j in range(i+1):
					matrix_index = matrix_index + 1
					E_i = tf.reshape(self.layer_output[0, i]- self.feature_vector[i], [1])
					E_j = tf.reshape(self.layer_output[0, j]- self.feature_vector[j], [1])

					cov_ij_error = tf.log(tf.abs(self.layer_output[0,matrix_index] - E_i*E_j))

					error = tf.add(error, cov_ij_error)
			self.error_function = tf.reshape(error, [1])

		else:
			#Minimize difference between probability functions of training set vs output from each discrete node
			output = tf.reshape(tf.nn.relu(self.layer_output),[1, len(self.distribution)])
			features = tf.reshape(self.feature_vector,[1, len(self.distribution)])
			entropy = tf.nn.softmax_cross_entropy_with_logits(output, features)
			self.error_function = tf.reshape(tf.reduce_mean(entropy),[1])


		self.graph_initialized = True

		for child in self.children:
			child.build_node_tf_graph()

	def copy_variables_out(self, sess):

		#Copy out neuron parameters
		input_nodes = len(self.parents)
		input_size = sum([len(parent.distribution) for parent in self.parents])
		
		if (input_size ==0):
			input_size = 1
		if (input_nodes ==0):
			input_nodes = 1

		self.neurons_w = []
		self.neurons_b = []

		for _ in range(input_nodes):
			self.neurons_w.append(sess.run(self.weights[_]))
			self.neurons_b.append(sess.run(self.biases[_]))

		self.neurons_w = numpy.reshape(self.neurons_w, (input_size, self.output_size))
		self.neurons_b = numpy.reshape(self.neurons_b, (self.output_size))


		#Copy out outputs
		if (self.is_continuous):
			output_layer = sess.run(self.layer_output)[0]
			self.distribution = self.distribution_from_output(output_layer)

			
		else:
			layer_probs = tf.nn.softmax(tf.nn.relu(self.layer_output))
			output_layer = sess.run(layer_probs)[0] #normalization

			for _ in range(len(self.distribution.keys())):
				key = self.distribution.keys()[_]
				self.distribution[key] = output_layer[_]

		print self.label
		print "dist", self.distribution
		print "w", self.neurons_w
		print "b", self.neurons_b

	def distribution_from_output(self, output_layer):
		distribution = {}
		distribution["Mean"] = numpy.reshape(output_layer[:self.n],(self.n))

		distribution["Cov"] = numpy.zeros((self.n,self.n))
		#The rest of the layer gives half the covariance matrix
		matrix_index = self.n
		for i in range(self.n):
			for j in range(i+1):
				matrix_index = matrix_index + 1
				distribution["Cov"][i][j] = output_layer[matrix_index]
				distribution["Cov"][j][i] = output_layer[matrix_index]

			distribution["Cov"][i][i] = abs(distribution["Cov"][i][i])

		return distribution

	def probability(self,outcome):
		if (self.is_continuous):
			#Reverse normal distribution
			min_vector = outcome[0] #Find probability of being a box set by 2 vectors
			max_vector = outcome[1] 

			normalized_min = numpy.linalg.inv(self.distribution["Cov"])*(min_vector - self.distribution["Mean"])
			normalized_max = numpy.linalg.inv(self.distribution["Cov"])*(max_vector - self.distribution["Mean"])

			#Inverse norm cdf
			probabilities = abs(stats.norm.cdf(normalized_max) - stats.norm.cdf(normalized_min))
			probability = numpy.multiply.reduce(probabilities)
			return probability

		else:
			return self.distribution[outcome]

	#Probability of node, conditioned on other nodes
	def joint_probability(self,outcomes):

		#If not the root node
		if len(self.parents) > 0:

			#Sum up the inputs to each neuron from all parents
			input_vector = []
			for _i in range(len(self.parents)):
				#Create vector of activated neurons for discrete parent
				input_vector.append(outcomes[self.parents[_i].label])

			input_vector = numpy.reshape(input_vector, (1,-1))
			output = numpy.reshape((numpy.dot(input_vector,self.neurons_w))+self.neurons_b, (-1))
			
			#Now compute conditional probabilties
			if (self.is_continuous):
				#Reverse normal distribution
				vector = outcomes[self.label] #Find probability of being a box set by 2 vectors
				dist = self.distribution_from_output(output)


				normalized_vector = numpy.dot(numpy.linalg.inv(dist["Cov"]),vector - dist["Mean"])

				#Inverse norm cdf
				probabilities = abs(stats.norm.pdf(normalized_vector))
				probability = numpy.multiply.reduce(probabilities)
				

				#print "continuous", vector, dist, probabilities
				return probability
			else:
				#Lookup outcome in distribution
				output = numpy_helpers.softmax(numpy.array([numpy_helpers.ReLU(x) for x in output]))
				outcome_index = numpy.where(outcomes[self.label]==1.0)[0][0]

				#print "discrete", output[outcome_index]

				return output[outcome_index]
		else:
			desired_node_outcome = numpy.where(outcomes[self.label]==1.0)[0][0]
			#print "root", self.probability(desired_node_outcome)

			return self.probability(desired_node_outcome)


class AssemblyModel(object):

	def build_node_tree(self, R, S, N, labels):

		def equal_dictionary (n): 
			equal_prob = 1.0/float(n)
			equal_dict = {}
			for _ in range(n):
				equal_dict[_] = equal_prob
			return equal_dict

		#To later feed to the optimizer
		nodes = []

		#Set all R style clusters to be equally likely
		root = ProbabilityNode("R", equal_dictionary(R))
		nodes.append(root)

		for _ in range(len(labels)):
			number_node = ProbabilityNode("N_{}".format(labels[_]), equal_dictionary(N[_]))
			style_node = ProbabilityNode("S_{}".format(labels[_]), equal_dictionary(S[_]))
			
			number_node.add_parent(root)
			style_node.add_parent(root)
			
			nodes.append(style_node)
			nodes.append(number_node)

			#For discrete & continuous feature vectors
			#discrete_node = ProbabilityNode("D_{}".format(label))
			continous_node = ProbabilityNode("C_{}".format(labels[_]), 
				{"Mean":[0]*N[_], "Cov":[[0]*N[_] for _ in range(N[_])]}, True)

			#discrete_node.add_parent(style_node)
			continous_node.add_parent(style_node)
			
			#nodes.append(discrete_node)
			nodes.append(continous_node)


		root.build_node_tf_graph()

		node_errors = [node.error_function for node in nodes ]#if node.is_continuous]
		var_list = [w for node in nodes for w in node.weights]
		var_list.extend([b for node in nodes for b in node.biases])

		batch=tf.Variable(0)
		obj=tf.add_n(node_errors)
		opt=tf.train.GradientDescentOptimizer(0.1).minimize(obj,global_step=batch,var_list=var_list)

		return (opt, obj, nodes)

	def train(self, feature_vectors, opt, error, nodes, iterations):

		with tf.Session() as sess:
			tf.initialize_all_variables().run()

			for _ in range(iterations):
				best_measurement_error = 0

				#Set variables
				feed_dict = {}
				for node in nodes:
					for _ in range(len(feature_vectors))
						if node.is_continuous:
							feed_dict[node.feature_vectors[_]] = numpy.reshape(feature_vectors[_][node.label], (node.n)) 
						else:
							feed_dict[node.feature_vector[_]] = numpy.reshape(feature_vectors[_][node.label], (len(node.distribution)))
						
				for node in [node for node in nodes if node.label == "C_type0"]:
					print _, node.label, sess.run(node.error_function, feed_dict), sess.run(node.layer_output[0,:3])

				sess.run(opt, feed_dict)

			#Update variables
			for node in nodes:
				node.copy_variables_out(sess)

		return best_measurement_error/len(feature_vectors)

	def create_feature_vectors(self,poly_list):
		#Fragment list
		processor = mesh_processing.MeshProcessor()
		fragments = [processor.segmentation(poly) for poly in poly_list]

		#Retrieve label for each fragment 
		### REWRITE THIS
		###(for now cluster, usually we would retrieve from library)
		###
		cluster_polys = [fragment_poly for sublist in fragments for fragment_poly in sublist] #Flatten list
		poly_features = [processor.feature_vectors(poly) for poly in cluster_polys]
		clustering = self.cluster(poly_features)
		labels = ["type{}".format(x) for x in range(max(clustering))] #generate labels
		fragment_mapping = {}
		for _ in range(len(cluster_polys)):
			fragment_mapping[cluster_polys[_]] = labels[clustering[_]-1]
		###

		#Create feature vectors 
		feature_vectors = [{} for poly in poly_list]
		number_length = max([len([fragment for fragment in cluster_polys if fragment_mapping[fragment] == label]) for label in labels]) + 1
		for _ in range(len(poly_list)):
			for label in labels:
				#Select this model's fragments under a certain label
				label_fragments = [fragment for fragment in fragments[_] if fragment_mapping[fragment] == label]
				feature_vectors[_]["N_{}".format(label)]  = numpy.zeros((number_length))
				feature_vectors[_]["N_{}".format(label)][len(label_fragments)] = 1

				#Mean vector
				continuous_feature_vec = [processor.feature_vectors(poly) for poly in label_fragments]
				feature_vectors[_]["C_{}".format(label)] = numpy.reshape(numpy.array(continuous_feature_vec),(-1))


		#PCA on the continuous feature means
		max_feature_length = max([len(vec["C_{}".format(label)]) for vec in feature_vectors for label in labels])
		for label in labels:
			#Copy out vecs
			label_vectors = [vec["C_{}".format(label)] for vec in feature_vectors]
			label_vectors = [numpy.pad(vec, (0,max_feature_length -len(vec)), "constant") for vec in label_vectors]
			
			#Reduce
			pca = PCA()
			pca.fit(numpy.array(label_vectors))
			reduced_label_vectors = pca.transform(label_vectors)

			#Copy them back
			scale_factor = numpy.amax(numpy.array(reduced_label_vectors))
			for _ in range(len(feature_vectors)):
				feature_vectors[_]["C_{}".format(label)] = reduced_label_vectors[_]/scale_factor


		return feature_vectors, labels 

	def cluster(self, feature_vectors, cluster_count=None):
		pca = PCA()
		pca.fit(numpy.array(feature_vectors))
		reduced_feature_vectors = pca.transform(feature_vectors)

		distance_mat = pairwise_distances(reduced_feature_vectors, metric="cosine")
		linkage = scipy.cluster.hierarchy.centroid(distance_mat)

		clusters = None
		if (cluster_count==None):
			clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.5)
		else:
			clusters = scipy.cluster.hierarchy.fcluster(linkage, cluster_count, criterion="maxclust")

		return clusters

	def probability(self, root, outcomes, nodes_counted):
		#For each node, we have joint probability directly
		#Multiply together probability of (childen/node joint probability)
		#If a node has multiple parents, only count it once

		probability = root.joint_probability(outcomes)
		nodes_counted[root] = True

		conditional_probs = 1
		for node in root.children:
			chance, nodes_counted = self.probability(node, outcomes, nodes_counted)
			conditional_probs =  conditional_probs * (chance/probability)

		probability = probability * conditional_probs

		return probability, nodes_counted



	def score(self,feature_vectors, R, S, labels):

		#Create feature vectors from geometric description
		#Geometric description = list of components with bounding box & signature

		#Cluster component styles into S[_] categories
		for _ in range(len(labels)):
			label_vectors = [vec["C_{}".format(labels[_])] for vec in feature_vectors]
			clustering = self.cluster(label_vectors, S[_])

			#Assign component style
			for x in range(len(feature_vectors)):
				feature_vectors[x]["S_{}".format(labels[_])] = numpy.zeros((S[_]))
				feature_vectors[x]["S_{}".format(labels[_])][clustering[x]-1] = 1

		#Cluster shape styles into R categories
		def shape_signature(feature_vec):
			signature = numpy.array([])
			for label in labels:
				signature = numpy.append(signature,feature_vec["S_{}".format(label)])
			for label in labels:
				signature = numpy.append(signature,feature_vec["N_{}".format(label)])
			return numpy.reshape(signature,(-1))
		shape_vectors = [shape_signature(vec) for vec in feature_vectors]
		clustering = self.cluster(shape_vectors, R)

		#Assign component style vector
		for _ in range(len(feature_vectors)):
			feature_vectors[_]["R"] = numpy.zeros((R))
			feature_vectors[_]["R"][clustering[_]-1] = 1

		#Max occurence for discrete distribition prior
		N = [ len(feature_vectors[0]["N_{}".format(label)]) for label in labels]

		#Train
		opt, error, nodes = self.build_node_tree(R,S, N, labels)
		self.train(feature_vectors, opt, error, nodes, 1000)
		ops.reset_default_graph() 

		#Calculate average joint probability
		sample_vectors = feature_vectors
		shuffle(sample_vectors)
		sample_vectors = sample_vectors[:5]
		joint_probability = sum([self.probability(nodes[0], vec, {})[0] for vec in sample_vectors])
		joint_probability = joint_probability/len(feature_vectors)

		print R, S, joint_probability

		return joint_probability

	def build_component_style_model(self, feature_vectors, labels):
		score_max = 0
		iteration_improvement = 0

		#Domain size variables
		R = 1
		S = [1 for _ in range(len(labels))] #component style array, start with only 1 style sub-class
		O = feature_vectors

		while (iteration_improvement < 10):
			R = R + 1 #increase possible number of shape styles
			S_mod = S 
			new_score = self.score(O,R,S_mod, labels)

			for _ in range(len(labels)): #increase possible number of component styles
				S_mod_mod = S_mod
				S_mod_mod[_] = S_mod_mod[_] + 1

				new_label_score = self.score(O,R,S_mod_mod,labels)

 				while (new_label_score > new_score and S_mod_mod[_] < R + 1):
					S_mod_mod[_] = S_mod_mod[_] + 1
					S_mod[_] = S_mod[_] + 1
					new_score = new_label_score
					new_label_score = self.score(O,R,S_mod_mod, labels)

			if (new_score > score_max):
				score_max = new_score
				iteration_improvement = 0
				S = S_mod
			else:
				iteration_improvement = iteration_improvement+1

		return [R,S]

	#def build_model_edges(self, feature_vectors, R,S):
		#Search for lateral edges between components


	#Get boudning box for poly
	#Get open edges
	#match eve placement ry component (energy minimizer)
	#Energy minimize totalof components
	#for non-matched components: create hole
	#Stich using triangles

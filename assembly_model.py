import tensorflow as tf
import numpy_helpers
from tensorflow.python.framework import ops
import numpy
import scipy.stats as stats
rng = numpy.random
import random
from random import shuffle
import math
import collections

import mesh
import mesh_processing
import probability_node
from probability_node import ProbabilityNode
import feature_vectors
from feature_vectors import MeshFeatureVectorProcessor

class AssemblyModel(object):

	def build_node_tree(self, R, S, N, C, D, labels, batch_size, graph=[]):

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
			discrete_node = ProbabilityNode("D_{}".format(labels[_]), equal_dictionary(D[_]))
			continous_node = ProbabilityNode("C_{}".format(labels[_]), 
				{"Mean":[0]*C[_], "Cov":[[0]*C[_] for _ in range(len(labels))]}, True)

			discrete_node.add_parent(style_node)
			continous_node.add_parent(style_node)
			
			nodes.append(discrete_node)
			nodes.append(continous_node)

		#Build lateral edges
		for edge in graph:
			#Find the nodes with the right labels
			parent_node = [node for node in nodes if node.label == edge[0]][0]
			child_node = [node for node in nodes if node.label == edge[1]][0]

			#Make the connection
			child_node.add_parent(parent_node)

		root.build_node_tf_graph(batch_size)

		node_errors = [node.error_function for node in nodes]
		var_list = [w for node in nodes for w in node.weights]
		var_list.extend([b for node in nodes for b in node.biases])

		batch=tf.Variable(0)
		obj=tf.add_n(node_errors)
		opt=tf.train.GradientDescentOptimizer(0.001).minimize(obj,global_step=batch,var_list=var_list)

		return (opt, obj, nodes)

	def train(self, feature_vectors, opt, error, nodes, iterations):

		with tf.Session() as sess:
			tf.initialize_all_variables().run()

			for _ in range(iterations):
				best_measurement_error = 0

				#Set variables
				feed_dict = {}
				for node in nodes:
					for x in range(len(feature_vectors)):
						if node.is_continuous:
							print feature_vectors[x][node.label], node.n
							feed_dict[node.feature_vectors[x]] = numpy.reshape(feature_vectors[x][node.label], (node.n)) 
						else:
							feed_dict[node.feature_vectors[x]] = numpy.reshape(feature_vectors[x][node.label], (len(node.distribution)))
						
				for node in [node for node in nodes if node.label == "C_type0"]:
					print _, node.label, sess.run(node.error_function, feed_dict), sess.run(node.layer_output[0,:3])

				sess.run(opt, feed_dict)

			#Update variables
			for node in nodes:
				node.copy_variables_out(sess)

		return best_measurement_error/len(feature_vectors)


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

	def score(self,feature_vectors, R, S, labels, graph=[]):

		#Raw feature vectors = list of components with bounding box & signature
		fv_processor = MeshFeatureVectorProcessor()
		O = fv_processor.modify_feature_vectors(feature_vectors, R, S, labels, graph)

		#Max occurence for discrete distribition prior
		#Max length of continuous features
		N = [ len(O[0]["N_{}".format(label)]) for label in labels]
		C = [ len(O[0]["C_{}".format(label)]) for label in labels]
		D = [ len(O[0]["D_{}".format(label)]) for label in labels]

		#Train
		opt, error, nodes = self.build_node_tree(R,S,N, C, D, labels, len(O), graph)
		self.train(O, opt, error, nodes, 100)
		ops.reset_default_graph() 

		#Calculate average joint probability
		sample_vectors = feature_vectors
		shuffle(sample_vectors)
		sample_vectors = sample_vectors[:5]
		joint_probability = sum([self.probability(nodes[0], vec, {})[0] for vec in sample_vectors])
		joint_probability = joint_probability/len(feature_vectors)

		print R, S, joint_probability

		return joint_probability, nodes

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
			new_score, nodes = self.score(O,R,S_mod, labels)

			for _ in range(len(labels)): #increase possible number of component styles
				S_mod_mod = S_mod
				S_mod_mod[_] = S_mod_mod[_] + 1

				new_label_score, nodes = self.score(O,R,S_mod_mod,labels)

 				while (new_label_score > new_score and S_mod_mod[_] < R + 1):
					S_mod_mod[_] = S_mod_mod[_] + 1
					S_mod[_] = S_mod[_] + 1
					new_score = new_label_score
					new_label_score, nodes = self.score(O,R,S_mod_mod, labels)

			if (new_score > score_max):
				score_max = new_score
				iteration_improvement = 0
				S = S_mod
			else:
				iteration_improvement = iteration_improvement+1

		G = self.build_model_edges(feature_vectors, R, S, labels)

		return [R,S, G]

	def build_model_edges(self, feature_vectors, R,S, labels):

		#Search for lateral edges between components
		G = [] #blank lateral edges
		O = feature_vectors
		best_score, nodes = self.score(O,R,S, labels, G)

		#For cycle search later
		visited_hash = {}
		for node in nodes:
			visited_hash[node.label] = False

		#Search through all non-root nodes
		for x in range(len(nodes)):
			for y in range(len(nodes)):
				#If this is a value new edge
				if x != y and not (nodes[x] in nodes[y].children):
					G_new = G 
					G_new.append([nodes[x].label, nodes[y].label])
					#If there are no cycles
					if not numpy_helpers.graph_cycle_search(G_new, nodes[x], nodes,  visited_hash):
						#And if this edge improves the score
						new_score, nodes = self.score(O,R,S, labels, G_new)
						if (new_score > best_score):
							G = G_new
		return G

	#Given a model structure and training set, generate samples
	def generate(self, R,S, G, labels, f_v, pca):

		reload(feature_vectors)
		reload(mesh_processing)

		#Feature vectors from raw feature vectors
		fv_processor = MeshFeatureVectorProcessor()
		O = fv_processor.modify_feature_vectors(f_v, R, S, labels, G)
		#Max occurence for discrete distribition prior
		N = [ len(O[0]["N_{}".format(label)]) for label in labels]
		C = [ len(O[0]["C_{}".format(label)]) for label in labels]
		D = [ len(O[0]["D_{}".format(label)]) for label in labels]

		#Train
		opt, error, nodes = self.build_node_tree(R,S,N,C,D, labels, len(O), G)
		self.train(O, opt, error, nodes, 100)
		ops.reset_default_graph() 

		#Sample
		node_queue = collections.deque()
		node_queue.append(nodes[0])
		sample_outputs = {}

		#Make sure node is only sampled after its parents
		while len(node_queue) > 0:
			node = node_queue.pop()
			all_parents_found = all((parent.label in sample_outputs) for parent in node.parents)
				
			if (all_parents_found):
				sample = node.sample(sample_outputs)

				#Reconstruct from inverse transform
				if node.is_continuous:
					sample_outputs[node.label] = pca[node.label[2:]].inverse_transform(sample)
				else:
					sample_outputs[node.label] = sample

				for child in node.children:
					node_queue.append(child)
			else:
				node_queue.append(node)


		processor = mesh_processing.MeshProcessor()

		component_list = []

		#Assemble from components
		for label in labels:

			#Component copies
			number = sample_outputs["N_{}".format(label)]
			style = sample_outputs["S_{}".format(label)]

			#Polygon of correct style 
			p = mesh.Mesh.load_scene("output/{}.off".format(label))[0]
			poly = mesh_processing.MeshProcessor().create_polyhedron(p)

			#Continuous features
			continous_features = sample_outputs["C_{}".format(label)]
			for _ in range(max(int(round(number)), 1)):
				offset = _ * (1000+257+12+257+16) #feature vector length
				offset_end = (_+1) * (1000+257+12+257+16)
				continous_subfeatures = continous_features[offset:offset_end]
				orientation_matrix = MeshFeatureVectorProcessor().orientation_matrix_from_feature_vector(continous_subfeatures)
				poly = processor.transform(poly, orientation_matrix)

				component_list.append(poly)

		print component_list

		#Get open edges
		#match eve placement ry component (energy minimizer)
		#Energy minimize totalof components
		#for non-matched components: create hole
		#Stich using triangles
		output = processor.stitch(component_list)
		return output

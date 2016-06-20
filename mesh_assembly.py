import mesh
import mesh_library
import mesh_processing

# import the packages
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import scipy.stats as stats
import time
import random

import CGAL
from CGAL import CGAL_Kernel
from CGAL import CGAL_Polyhedron_3
from CGAL import CGAL_Polygon_mesh_processing

Vertex = CGAL.CGAL_Kernel.Point_3
Vector_3 = CGAL.CGAL_Kernel.Vector_3
BoundingBox = CGAL.CGAL_Polygon_mesh_processing.BoundingBox

Library = mesh_library.MeshLibrary()

class MeshAssembler(object):
	#Create one of these objects for each class of 3D model

	def __init__(self):

		self.max_components = 30
		self.features_per_vec = 3 * 4 + 2
		self.vector_length = self.features_per_vec * self.max_components
		return None

	# Make a mutlilevel perceptron - used for D_pre, D1, D2, G networks
	def generate_mlp(self, input, output_dim):
	    # construct learnable parameters within local scope
	    w1=tf.get_variable("w0", [input.get_shape()[1], 20], initializer=tf.random_normal_initializer())
	    b1=tf.get_variable("b0", [20], initializer=tf.constant_initializer(0.0))
	    w2=tf.get_variable("w1", [20, 20], initializer=tf.random_normal_initializer())
	    b2=tf.get_variable("b1", [20], initializer=tf.constant_initializer(0.0))
	    w3=tf.get_variable("w2", [20,output_dim], initializer=tf.random_normal_initializer())
	    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
	    # nn operators
	    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
	    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
	    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
	    return fc3, [w1,b1,w2,b2,w3,b3]

	def pre_train(self, D, feature_vectors):
		batch=tf.Variable(0)
		
		obj_d=tf.reduce_mean(tf.log(D))
		opt_d=tf.train.GradientDescentOptimizer(0.01).minimize(1-obj_d,global_step=batch,var_list=theta_d)
		sess=tf.InteractiveSession()
		
		tf.initialize_all_variables().run()
		for vector in feature_vectors:
			sess.run(opt_d, {x_node: vector}) 

	def generative_adversarial_net(self, feature_vectors):
		batch=tf.Variable(0)

		with tf.variable_scope("G"):
			self.y_node = tf.placeholder(tf.float32, shape= (len(feature_vectors[0]),1) )
			self.G, theta_g = self.generate_mlp(self.y_node,1)

		with tf.variable_scope("D") as scope:
			self.z_node = tf.placeholder(tf.float32, shape= (len(feature_vectors[0]),1) )
			D1, theta_d = self.generate_mlp(self.z_node,1)
			scope.reuse_variables() #to use variables from G
			D2, theta_d = self.generate_mlp(self.G,1)

		#pre-train
		self.pre_train(D1, feature_vectors[:10])

		obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
		self.opt_d=tf.train.GradientDescentOptimizer(0.01).minimize(1-obj_d,global_step=batch,var_list=theta_d)

		obj_g=tf.reduce_mean(tf.log(D2))
		self.opt_g=tf.train.GradientDescentOptimizer(0.01).minimize(1-obj_g,global_step=batch,var_list=theta_g)

		sess=tf.InteractiveSession()
		tf.initialize_all_variables().run()

		for vector in feature_vectors:
			#Train on features + noise vectors
			z = np.reshape(np.array(vector),(len(vector),1))
			y = np.random.random((len(vector),1))
			sess.run(self.opt_d, {self.y_node: y, self.z_node: z}) 
			y= np.random.random((len(vector),1))
			sess.run(self.opt_g, {self.y_node: y}) 

		return None

	def generate_feature_vector(self, count):
		y= np.random.random((len(self.vector_length),1))

		sess=tf.InteractiveSession()
		tf.initialize_all_variables().run()

		feature_vectors = []
		for _ in range(count):
			output = sess.run(self.opt_g, {self.y_node: y}) 
			feature_vectors.append(output)

		return feature_vectors

	def encode_feature_vector(self, mesh, component_list):		
		feature_vector_components = []

		for poly in component_list[:self.max_components]:
			#add the bounding box transform to the feature vector
			bounding_box = BoundingBox(poly)
			features = [bounding_box.get_origin(), bounding_box.get_x_axis(), bounding_box.get_y_axis(), bounding_box.get_z_axis()]
			for vector in features:
				feature_vector_components.append(vector.x())
				feature_vector_components.append(vector.y())
				feature_vector_components.append(vector.z())

			#two identifiers for shape style
			mesh_class, mesh_style = Library.component_id(poly)
			feature_vector_components.append(mesh_class)
			feature_vector_components.append(mesh_style)

		#pad with zeros
		for _ in range(self.max_components - len(component_list)):
			for n in range(14):
				feature_vector_components.append(0)

		feature_vector = np.array(feature_vector_components)

		return feature_vector


	def decode_feature_vector(self, feature_vector):

		polygon_list = []

		for _ in range(self.max_components):
			offset = _ * self.max_components

			translation = Vector_3(feature_vector[offset], feature_vector[offset+1], feature_vector[offset+2])
			x_axis = Vector_3(feature_vector[offset+3*1], feature_vector[offset+3*1+1], feature_vector[offset+3*1+2])
			y_axis = Vector_3(feature_vector[offset+3*2], feature_vector[offset+3*2+1], feature_vector[offset+3*2+2])
			z_axis = Vector_3(feature_vector[offset+3*3], feature_vector[offset+3*3+1], feature_vector[offset+3*3+2])

			bounding_box = BoundingBox(translation, x_axis, y_axis, z_axis)

			poly = Library.retrive_component(vector[offset+24],vector[offset+25])
			
			new_poly = mesh_processing.MeshProcessor.align(poly, bounding_box)
			polygon_list.append(new_poly)

		return polygon_list

	def training_set(self, polygons):	
		feature_vectors = []	
		for poly in polygons:
			poly_list = mesh_processing.MeshProcessor().segmentation(poly)
			vec = self.encode_feature_vector(poly,poly_list)
			feature_vectors.append(vec)

		self.generative_adversarial_net(feature_vectors)

	def generate_meshes(self, count):
		vectors = generate_feature_vector(count)
		polygon_lists = [self.decode_feature_vector(vec) for vec in vectors]
		polygons = [self.stitch(poly_list) for poly_list in polygon_lists]

		return polygons

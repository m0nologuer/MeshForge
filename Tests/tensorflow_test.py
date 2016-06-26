# import the packages
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import scipy.stats as stats
import time
import random


class MeshAssembler(object):
	#Create one of these objects for each class of 3D model

	def __init__(self):

		self.max_components = 10
		self.features_per_vec = 3 * 4 + 2
		self.vector_length = self.features_per_vec * self.max_components
		
		self.generative_adversarial_net()

		return None

	# Make a mutlilevel perceptron - used for D_pre, D1, D2, G networks
	def generate_mlp(self, input, output_dim):
	    # construct learnable parameters within local scope
	    size = input.get_shape()[1]
	    w1=tf.get_variable("w0", [size, size], initializer=tf.random_normal_initializer())
	    b1=tf.get_variable("b0", [size], initializer=tf.constant_initializer(0.0))
	    w2=tf.get_variable("w1", [size, size], initializer=tf.random_normal_initializer())
	    b2=tf.get_variable("b1", [size], initializer=tf.constant_initializer(0.0))
	    w3=tf.get_variable("w2", [size,output_dim], initializer=tf.random_normal_initializer())
	    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
	    # nn operators
	    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
	    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
	    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
	    return fc3, [w1,b1,w2,b2,w3,b3]

	#For teaching the neural net about the basic structure of the feature vector
	def fake_feature_vector(self):

		feature_vector_components = np.array([])

		components = np.random.randint(self.max_components)
		for _ in range(components):
			#generate a plausible bounding box transform to the feature vector
			origin = np.random.random(3)
			x_axis = np.random.random(3) - origin
			y_axis = np.random.random(3) - origin
			z_axis = np.cross(x_axis,y_axis)*np.random.random()*2 #orthogonal, but scaled randomly

			feature_vector_components = np.append(feature_vector_components, origin)
			feature_vector_components = np.append(feature_vector_components, x_axis)
			feature_vector_components = np.append(feature_vector_components, y_axis)
			feature_vector_components = np.append(feature_vector_components, z_axis)

			#two identifiers for shape style
			object_type = np.random.randint(100)
			style = np.random.randint(100)

			feature_vector_components = np.append(feature_vector_components,object_type)
			feature_vector_components = np.append(feature_vector_components,style)

		#then pad with zeroes
		zeros = (self.max_components - components)*self.features_per_vec
		feature_vector = np.append(feature_vector_components,np.zeros(zeros))
		return feature_vector

	def fill_feature_vector(number):
		feature_vector = np.full_like(np.arange(self.vector_length, dtype=np.float32), number)


	def pre_train(self, D, theta_d, feature_vectors):
		batch=tf.Variable(0)
		
		x_node = tf.placeholder(tf.float32, shape= (1, self.vector_length) )

		with tf.variable_scope("D") as scope:
			scope.reuse_variables()
			obj_d=tf.reduce_mean(tf.log(D))
			opt_d=tf.train.AdamOptimizer(1e-4).minimize(1-obj_d,global_step=batch,var_list=theta_d)

		sess=tf.InteractiveSession()
		
		tf.initialize_all_variables().run()
		for vector in feature_vectors:
			sess.run(opt_d, feed_dict={x_node: vector}) 

	def generative_adversarial_net(self):
		
		batch=tf.Variable(0)

		with tf.variable_scope("G"):
			self.y_node = tf.placeholder(tf.float32, shape= (1, self.vector_length) )
			self.G, theta_g = self.generate_mlp(self.y_node,self.vector_length)

		with tf.variable_scope("D") as scope:
			self.z_node = tf.placeholder(tf.float32, shape= (1,self.vector_length) )
			D1, theta_d = self.generate_mlp(self.z_node,1)
			scope.reuse_variables() #to use variables from G
			D2, theta_d = self.generate_mlp(self.G,1)

		self.obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
		self.opt_d=tf.train.AdamOptimizer(0.1).minimize(1-self.obj_d,global_step=batch,var_list=theta_d)

		self.obj_g=tf.reduce_mean(tf.log(D2))
		self.opt_g=tf.train.AdamOptimizer(0.1).minimize(1-self.obj_g,global_step=batch,var_list=theta_g)

	def train(self, feature_vectors):

		sess=tf.InteractiveSession()
		tf.initialize_all_variables().run()
		
		for vector in feature_vectors:
			#Train on features + noise vectors
			z = np.reshape(np.array(vector),(1,self.vector_length))
			y = np.random.random((1,self.vector_length))
			sess.run(self.opt_d, feed_dict={self.y_node: y, self.z_node: z}) 
			y= np.random.random((1,self.vector_length))
			sess.run(self.opt_g, feed_dict={self.y_node: y}) 

	def generate_feature_vector(self, count):
		y= np.random.random((1,self.vector_length)).astype(np.float32)

		opt_gen=tf.identity(self.G)

		sess=tf.InteractiveSession()
		tf.initialize_all_variables().run()

		feature_vectors = []
		for _ in range(count):
			output = sess.run(opt_gen, feed_dict={self.y_node: y}) 
			feature_vectors.append(output)

		return feature_vectors


import mesh_processing
import numpy_helpers
import numpy as np
import os
from os import listdir
import django
import feature_vectors
import math
import random
import scipy
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA

from django.db import connection

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mesh_database.settings")
django.setup()

from mesh_database.models import Mesh

MeshProcessor = mesh_processing.MeshProcessor()
MeshFeatureVectorProcessor = feature_vectors.MeshFeatureVectorProcessor()

MESH_FOLDER = "output/MeshDatabase/"

class MeshLibrary(object):
	#Create one of these objects for each class of 3D model

	# Simple clustering for classifying meshes
	# Class = broad categories of objects
	# Style = n variations on that object

	def __init__(self):
		self.threshold = 0.3

		self.index_set = False
		np.set_printoptions(threshold=10000) #for saving strings


		return None

	# Create an in-memory index of dimenstion-reducted feature vectors for matching
	# models to category classes
	def build_top_level_pca_index(self, test_feature_vectors):

		if (len(Mesh.objects.all()) == 0):
			self.index_set = False
			return None

		#Pick a representative mesh from each top level type
		top_level_count = max([m.model_type for m in Mesh.objects.all()])+1
		example_meshes = [Mesh.objects.filter(model_type=_) for _ in range(top_level_count+1)]
		index_feature_vectors = [np.fromstring(meshes[0].feature_vector, sep=',') for meshes in example_meshes if len(meshes) > 0]
		
		index_feature_vectors.append(test_feature_vectors[0])
		index_feature_vectors = np.reshape(np.array(index_feature_vectors),(len(index_feature_vectors),-1))

		#Up to five vectors from each for the basis
		basis_meshes = [m  for meshes in example_meshes for m in meshes[:min(5,len(meshes))]]
		basis_feature_vectors = [np.fromstring(m.feature_vector, sep=',') for m in basis_meshes]
		
		for vec in test_feature_vectors:
			basis_feature_vectors.append(vec)
		basis_feature_vectors = np.reshape(np.array(basis_feature_vectors),(len(basis_feature_vectors),-1))

		#Calculate feature vectors & fit principal component analysis
		pca = MeshFeatureVectorProcessor.create_basis_set(basis_feature_vectors)
		top_level_reduced_fv = MeshFeatureVectorProcessor.transform(pca, index_feature_vectors)

		self.index_set = True
		
		return pca, top_level_reduced_fv

	#Create an in-memory index of dim red feature vectors for mathing within category classes
	def build_type_pca_index(self, model_type, test_feature_vectors):

		#Pick a representative mesh from each style for the type
		style_count = max([mesh.model_type for mesh in Mesh.objects.filter(model_type=model_type)])
		example_meshes = [Mesh.objects.filter(model_type=model_type, model_style=_) for _ in range(style_count+1)]
		index_feature_vectors = [np.fromstring(meshes[0].feature_vector, sep=',') for meshes in example_meshes if len(meshes) > 0]
		index_feature_vectors.append(test_feature_vectors[0])
		index_feature_vectors = np.reshape(np.array(index_feature_vectors),(len(index_feature_vectors),-1))

		#Up to five vectors from each for the basis
		basis_meshes = [m  for meshes in example_meshes for m in meshes[:min(5,len(meshes))]]
		basis_feature_vectors = [np.fromstring(m.feature_vector, sep=',') for m in basis_meshes]
		for vec in test_feature_vectors:
			basis_feature_vectors.append(vec)
		basis_feature_vectors = np.reshape(np.array(basis_feature_vectors),(len(basis_feature_vectors),-1))

		#Calculate feature vectors & fit principal component analysis
		pca = MeshFeatureVectorProcessor.create_basis_set(basis_feature_vectors)
		reduced_fv = MeshFeatureVectorProcessor.transform(pca, index_feature_vectors)

		return pca, reduced_fv

	def add_files_from_folder(self,folder, segmented):
		polys = MeshProcessor.load(folder)
		category = str.split(folder,'/')[-1] #take last folder directory name
		max_clusters = int(math.sqrt(len(polys)))

		#Segment each polygon if needed
		if (segmented == bool(1)):
			segments = []
			for poly in polys:
				segments.append(MeshProcessor.segmentation(poly))
			max_clusters = max_clusters*(len(segments)/len(polys))
			polys = segments

		#Cluster models
		poly_features = [MeshFeatureVectorProcessor.feature_vectors(poly) for poly in polys]
		clustering = numpy_helpers.cluster(poly_features, max_clusters)

		#Split into clusters 
		for _ in range(1, max(clustering)):
			cluster_polys = [polys[x] for x in range(len(polys)) if clustering[x] == _]
			feature_vectors = [poly_features[cluster_polys.index(poly)] for poly in cluster_polys]
			#Fit with existing cluster, or create own?
			model_type, model_style = self.match_cluster(feature_vectors)

			#Add to database
			new_folder_name = MESH_FOLDER + "type{}/style{}/".format(model_type, model_style)
			MeshProcessor.write(cluster_polys, new_folder_name)
			for _ in range(len(cluster_polys)):
				filename = new_folder_name + "{}.off".format(_)
				#String feature vec
				features = ','.join([str(x) for x in feature_vectors[_]])
				faces = cluster_polys[_].size_of_facets()
				mesh_model = Mesh(filename=filename, category=category, model_type=model_type,
					model_style=model_style, face_count=faces, feature_vector=features,
					generated=False, fragment=segmented)
				mesh_model.save()

	def add_mesh(self,mesh, name):
		#Add mesh singularly
		feature_vec = MeshFeatureVectorProcessor.feature_vectors(poly)
		features = np.array_str(feature_vec)
		faces = mesh.size_of_facets()
		model_type, model_style = self.match_single(feature_vec)

		new_folder_name = MESH_FOLDER + "type{}/style{}/".format(model_type, model_style)
		MeshProcessor.write([mesh], new_folder_name)

		mesh_model = Mesh(filename=filename, category=name, model_type=model_type,
					model_style=model_style, face_count=faces, feature_vector=features,
					generated=False, fragment=segmented)
		mesh_model.save()


	#Search for the best category for a mesh...
	def match_single(self,feature_vector):

		feature_vector = np.reshape(feature_vector, (1, -1))

		#Use top level fv index
		top_pca, top_reduced_fv = self.build_top_level_pca_index([feature_vector])
		mod_fvs = MeshFeatureVectorProcessor.transform(top_pca, [feature_vector])
		type_distances = np.sum(pairwise.cosine_similarity(mod_fvs, top_reduced_fv), 0)
		model_type = np.argmax(type_distances[:-1]) #Exclude control feature vectors


		#Build index for type & find best fit style
		pca, reduced_fv = self.build_type_pca_index(model_type)
		mod_type_fv = MeshFeatureVectorProcessor.transform(pca, [feature_vector])
		style_distances = [pairwise.cosine_similarity(mod_type_fv, fv) for fv in reduced_fv]
		model_style = style_distances.index(max(style_distances))

		#If this is too dissimilar from the other meshes, create a new type
		min_similarity = min([pairwise.cosine_similarity(fv1,fv2) for fv1 in reduced_fv for fv2 in reduced_fv])
		if (max(style_distances) < min_similarity*0.9):
			style_count = max([mesh.model_type for mesh in Mesh.objects.filter(model_type=model_type)])
			model_style = style_count + 1
			self.new_type = True

		return model_type, model_style

	def match_cluster(self,feature_vectors):

		print "	Index", self.index_set
		#First entry
		if (self.index_set == False):
			model_type = 0 #create new type
			model_style = 0
			self.index_set = True
			return model_type, model_style

		feature_vectors = np.reshape(feature_vectors, (len(feature_vectors), -1))

		#Build top level fv index, with last vector as control vector
		top_pca, top_reduced_fv = self.build_top_level_pca_index(feature_vectors)
		mod_fvs = MeshFeatureVectorProcessor.transform(top_pca, feature_vectors)
		type_distances = np.sum(pairwise.cosine_similarity(mod_fvs, top_reduced_fv), 0)
		model_type = np.argmax(type_distances[:-1]) #Exclude control feature vectors

		print "cluster feature vectors", mod_fvs
		print "type index feature vectors", top_reduced_fv

		print "Model type", model_type, "of", (len(type_distances)-1)
		print "Max type", max(type_distances), "self", type_distances[:-1]

		#Build index for type & find best fit style
		pca, reduced_fv = self.build_type_pca_index(model_type, feature_vectors)
		mod_type_fvs = MeshFeatureVectorProcessor.transform(pca, feature_vectors)
		style_distances = np.sum(pairwise.cosine_similarity(mod_type_fvs, reduced_fv),0)
		model_style = np.argmax(style_distances[:-1])

		print "cluster type feature vectors", mod_type_fvs
		print "style index feature vectors", top_reduced_fv

		print "Model style", model_style, "of", (len(style_distances)-1)
		print "Max model", max(style_distances), "self", style_distances[:-1]

		#GOAL: If this is too dissimilar from the other meshes, create a new group
		#Consider sum of minimum similarity between meshes in this cluster
		ingroup_distance = style_distances[-1]
		print "min similarity ingroup", ingroup_distance
		if (max(style_distances[:-1]) < ingroup_distance):
			#Decide between creating a ingroup_distance style or a whole type entirely
			non_self_max_type_score = max(type_distances[:-1])
			self_type_score = type_distances[-1]
			print "sts", self_type_score, "nsmts", non_self_max_type_score
			if (non_self_max_type_score > self_type_score*0.9): #If this type fits well
				model_style = len(style_distances[:-1]) #Just create a new style
			else:
				#Otherwise, create a new type
				model_type = len(type_distances[:-1]) 
				model_style = 0

		return model_type, model_style


	def retrive_component(self, model_type, model_style, array=False):
		if array:
			#All such components
			mesh_models = Mesh.objects.filter(model_type=model_type, model_style=model_style)
			meshes = []
			for model in mesh_models:
				m = mesh_processing.mesh.Mesh.load_scene(model.filename)[0]
				meshes.append(MeshProcessor.create_polyhedron(m))
			return meshes
		else:
			mesh_models = Mesh.objects.filter(model_type=model_type, model_style=model_style)
			location = random.choice(mesh_models).filename
			m = mesh_processing.mesh.Mesh.load_scene(location)[0]
			return MeshProcessor.create_polyhedron(m)

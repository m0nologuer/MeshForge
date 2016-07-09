import mesh
import mesh_processing

import numpy as np
import os
from os import listdir


MeshProcessor = mesh_processing.MeshProcessor()

MESH_FOLDER = "MeshDatabase/"

class MeshLibrary(object):
	#Create one of these objects for each class of 3D model

	# Simple clustering for classifying meshes
	# Class = broad categories of objects
	# Style = n variations on that object

	def __init__(self):
		self.threshold = 0.3
		return None

	def add_files_from_folder(self,folder, split):
		polys = MeshProcessor.load(folder)
		#Break up the polygons and cluster them 

		if (split == bool(1)):
			segments = []
			for poly in polys:
				segments = MeshProcessor.segmentation(poly)
				cluster = MeshProcessor.cluster(segments)
				self.write_from_cluster(segments, cluster)
		else:
			cluster = MeshProcessor.cluster(polys)
			self.write_from_cluster(polys, cluster)

	def consolidate(self,max_folders):
		folders = listdir(MESH_FOLDER)
		folders.remove('.DS_Store')

		for folder in folders[max_folders:]:
			files = listdir(MESH_FOLDER+folder)
			for file in files:
				path = MESH_FOLDER + folder + "/" + file
				m = mesh.Mesh.load_scene(path)[0]
				file_poly = MeshProcessor.create_polyhedron(m)
				new_folder_name, dist = self.component_class(file_poly, max_folders)
				
				new_folder_name = MESH_FOLDER + new_folder_name + "/"
				new_folder_size = len(listdir(new_folder_name))
				MeshProcessor.write([file_poly], new_folder_name, new_folder_size)

	def write_from_cluster(self,polys, cluster):

		for _ in range(max(cluster)):
			cluster_polys = MeshProcessor.polygons_in_cluster(polys,cluster,_)
			if len(cluster_polys) > 0:
				folder, dist = self.component_class(cluster_polys[0])

				#if the cluster is more than half the folder size
				#or if the match is very weak (or non existent)
				if (dist == None or (len(listdir(folder)) > 4 and len(listdir(folder)) < len(cluster_polys)*2) or dist > self.threshold):
					new_folder_name = MESH_FOLDER + "type{}/".format(len(listdir(MESH_FOLDER)))
					os.makedirs(new_folder_name)
					MeshProcessor.write(cluster_polys, new_folder_name)
				else:
				#otherwise, append to existing folder
					old_folder_size = len(listdir(folder))
					MeshProcessor.write(cluster_polys, folder, old_folder_size)

	#Very rudimentary search through classes
	def component_class(self,poly, max_folders = None):
		folders = listdir(MESH_FOLDER)
		folders.remove('.DS_Store')

		min_distance = None
		best_folder = ""

		if (max_folders != None): #Allows for limit on search
			folders = folders[:max_folders]

		if (folders == None):
			return best_folder, min_distance

		for folder in folders:
			#check against first example in each class
			path = MESH_FOLDER + folder + "/0.off" 
			print path
			m = mesh.Mesh.load_scene(path)[0]
			folder_poly = MeshProcessor.create_polyhedron(m)
			dist = MeshProcessor.similarity_metric(poly, folder_poly)
			if (min_distance==None or dist < min_distance):
				min_distance = dist
				best_folder = folder

		return best_folder, min_distance

	def component_id(self, poly):

		best_folder, dist = self.component_class(poly)

		files = listdir(MESH_FOLDER + best_folder)
		if '.DS_Store' in files:
			files.remove('.DS_Store')

		matching_file = ""
		class_id = int(best_folder[4:])

		#Check against all files in the class
		for file in files:
			path = MESH_FOLDER + best_folder + "/" + file
			print path
			m = mesh.Mesh.load_scene(path)[0]
			file_poly = MeshProcessor.create_polyhedron(m)
			dist = MeshProcessor.similarity_metric(poly, file_poly)
			if (dist == 0):
				matching_file = file
				file_id = int(best_folder[:1])
				return class_id, file_id

		#If we don't have a match, write it in the correct folder
		counter = len(files)
		matching_file =  MESH_FOLDER + best_folder + "/{}.off".format(counter)
		poly.write_to_file(matching_file)

		return class_id, counter #need to process this

	def retrive_component(self, class_id, style_id):
		path = MESH_FOLDER + "type{}/{}.off".format(class_id, id)
		mesh = mesh.Mesh().load_scene(path)[0]
		return MeshProcessor.create_polyhedron(mesh)


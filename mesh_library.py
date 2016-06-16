import mesh
import mesh_processing

import numpy as np
from os import listdir


MeshProcessor = mesh_processing.MeshProcessor()

MESH_FOLDER = "MeshDatabase/"

class MeshLibrary(object):
	#Create one of these objects for each class of 3D model

	# Simple clustering for classifying meshes
	# Class = broad categories of objects
	# Style = n variations on that object

	def __init__(self):
		return none

	def sort_folder(folder, split):
		polys = MeshProcessor.load(folder)
		cluster = MeshProcessor.cluster(polys)

		for _ in max(cluster):
			polys = MeshProcessor.polygons_in_cluster

	#Very rudimentary search through components
	def component_id(self, poly):
		folders = listdir(MESH_FOLDER)

		min_distance = None
		best_folder = ""

		for folder in folders:
			#check against first example in each class
			path = MESH_FOLDER + folder + "0.off" 
			mesh = mesh.Mesh().load_scene(path)[0]
			folder_poly = MeshProcessor.create_polyhedron(mesh)
			dist = MeshProcessor.similarity_metric(poly, folder_poly)
			if (min_distance==None or dist < min_distance):
				min_distance = dist
				best_folder = folder

		MESH_FOLDER = MESH_FOLDER + best_folder
		files = listdir(MESH_FOLDER)
		matching_file = ""
		found = bool(0)

		#Check against all files in the class
		for file in files:
			mesh = mesh.Mesh().load_scene(file)[0]
			file_poly = MeshProcessor.create_polyhedron(mesh)
			dist = MeshProcessor.similarity_metric(poly, file_poly)
			if (dist == 0):
				matching_file = file
				found = bool(1)

		#If we don't have a match, write it in the correct folder
		if (found == bool(0)):
			counter = len(files)
			poly.write_file(matching_file.format(counter))
			matching_file =  MESH_FOLDER + "{}.off"

		return matching_file #need to process this

	def retrive_component(self, class_id, style_id):
		path = MESH_FOLDER + "class{}/{}.off".format(class_id, id)
		mesh = mesh.Mesh().load_scene(path)[0]
		return MeshProcessor.create_polyhedron(mesh)


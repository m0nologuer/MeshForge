#Basic utilities such as segmentation

import mesh
import numpy as np
import scipy
from scipy import cluster
from emd import emd
from os import listdir
import os
from numpy import random
import math

import sys
sys.path.append("../Opensource/cgal-swig-bindings/build-python/")

import CGAL
from CGAL import CGAL_Kernel
from CGAL import CGAL_Polyhedron_3
from CGAL import CGAL_Polygon_mesh_processing

Vertex = CGAL.CGAL_Kernel.Point_3
Vector = CGAL.CGAL_Kernel.Vector_3
Polyhedron = CGAL.CGAL_Polyhedron_3.Polyhedron_3
Mod = CGAL.CGAL_Polyhedron_3.Polyhedron_modifier
BoundingBox = CGAL.CGAL_Polygon_mesh_processing.BoundingBox

class MeshProcessor(object):
	"""Class for CGAL polyhedral mesh operations"""

	def __init__(self):
		return None

	def create_polyhedron(self, mesh):
		
		#Build surface
		f_count = mesh.mesh.faces.size
		v_count = mesh.mesh.vertices.size
		m = Mod()
		m.begin_surface(v_count, f_count)

		for v in mesh.mesh.vertices:
			m.add_vertex(Vertex(float(v[0]),float(v[1]),float(v[2])))

		for f in mesh.mesh.faces:
			m.begin_facet()
			m.add_vertex_to_facet(int(f[0]))
			m.add_vertex_to_facet(int(f[1]))
			m.add_vertex_to_facet(int(f[2]))
			m.end_facet()

		m.end_surface()

		poly = Polyhedron()
		poly.delegate(m)

		#Remove duplicate vertices
		poly = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(poly).stitch_borders()

		return poly

	def simplify_mesh(self, poly):
		util = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(poly)
		util.simplify_mesh(1000)
		simple_poly = util.get_mesh()
		return simple_poly

	def segmentation(self, poly, adjacanacy=False):

		#Split
		segmenter = CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly)
		poly_list = [x for x in segmenter.segmentation()]

		return poly_list

	def adjacancy(self, poly, poly2):
		#adjacancy of connected components
		segmenter = CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly)

		return segmenter.adjacent(poly, poly2)				

	def get_matrix(self, bounding_box):
		translation = bounding_box.get_origin()
		x_axis = bounding_box.get_x_axis()
		y_axis = bounding_box.get_y_axis()
		z_axis = bounding_box.get_z_axis()

		matrix = np.matrix([[x_axis.x(), x_axis.y(), x_axis.z(), translation.x()],
			[y_axis.x(), y_axis.y(), y_axis.z(), translation.y()],
			[z_axis.x(), z_axis.y(), z_axis.z(), translation.z()],
			[0,0,0,1]])
		return matrix
	
	def transform(self, poly, matrix):

		old_bounding_box = BoundingBox(poly)
		old_matrix = self.get_matrix(old_bounding_box)

		t_mat = (matrix * old_matrix.I).tolist()

		x_axis = Vector(t_mat[0][0],t_mat[0][1],t_mat[0][2])
		y_axis = Vector(t_mat[1][0],t_mat[1][1],t_mat[1][2])
		z_axis = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		translation = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		processor = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(poly)
		new_poly =processor.transform_mesh(translation, x_axis, y_axis, z_axis)

		return new_poly


	def align(self, poly, bounding_box):

		old_bounding_box = BoundingBox(poly)
		old_matrix = self.get_matrix(old_bounding_box)
		new_matrix = self.get_matrix(bounding_box)

		t_mat = (matrix * old_matrix.I).tolist()

		x_axis = Vector(t_mat[0][0],t_mat[0][1],t_mat[0][2])
		y_axis = Vector(t_mat[1][0],t_mat[1][1],t_mat[1][2])
		z_axis = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		translation = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		processor = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(poly)
		new_poly =processor.transform_mesh(translation, x_axis, y_axis, z_axis)

		return new_poly

	def stitch(self, polygon_list):
		processor = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(polygon_list[0])
		poly_list = CGAL.CGAL_Polygon_mesh_processing.Polyhedron_list()
		for poly in polygon_list:
			poly_list.push_back(poly)
		return processor.concatenate_mesh(poly_list)
		

#IO helper functions
	def write(self,polygon_list, folder, offset=0):

		#Make chain of directories
		folder_path = folder
		folder_create_list = []
		while not os.path.exists(folder_path):
			folder_create_list.append(folder_path)
			folder_path = "/".join(str.split(folder_path,"/")[:-1])
		folder_create_list.reverse()

		for f in folder_create_list:
			if not os.path.exists(f):
				os.mkdir(f) #take name of last folder

		offset = len(os.listdir(folder))

		for _ in range(len(polygon_list)):
			location = folder + "{}.off".format(_+offset)
			polygon_list[_].write_to_file(location)
		
		return polygon_list

	def load(self,folder):
		files = listdir(folder)
		polygon_list = []
		for file in files:
			if (file[-4:]=='.off'):
				m = mesh.Mesh.load_scene(folder+file)[0]
				poly = self.create_polyhedron(m)
				polygon_list.append(poly)
		
		return polygon_list




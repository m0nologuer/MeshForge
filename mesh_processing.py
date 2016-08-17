import mesh
import remesh_voxel

import numpy as np
import scipy
from scipy import cluster
from emd import emd
from os import listdir
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
PolySignature = CGAL.CGAL_Polygon_mesh_processing.PolySignature
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

	def segmentation(self, poly):
		poly_list = [x for x in CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly).segmentation()]

		return poly_list

	def signature(self,poly,normalize=True):
		#Run the CGAL algorithm, to gen points per cell
		#Put resuts from non-empty cells into numpy arrays
		sig = PolySignature(poly)
		points = []
		weights = []
		for _ in range(15625):
			if sig.weight_at(_) != 0:
				pos = sig.point_at(_)
				w = sig.weight_at(_)
				points.append([pos.x(), pos.y(), pos.z()])
				weights.append(w)

		if (normalize):
			#pick 500 points at random
			indices = np.random.permutation(range(len(points)))[:500]
			points = [points[_] for _ in indices]
			weights = [weights[_] for _ in indices]

			weights = np.array(weights)
			weights = np.divide(weights,weights.sum()) #normalize

		signature = [np.array(points), weights]
		return signature
		

	def similarity_metric(self,poly1,poly2):

		sig1 = self.signature(poly1)
		sig2 = self.signature(poly2)

		dist = emd(sig1[0], sig2[0], sig1[1], sig2[1]);

		return dist

	def similarity(self,sig1,sig2):
		dist = emd(sig1[0], sig2[0], sig1[1], sig2[1]);
		return dist

	def cluster(self, polygon_list):
		
		polygon_list = [self.simplify_mesh(poly) for poly in polygon_list]
		signatures = [self.signature(poly) for poly in polygon_list]

		distance_mat = np.zeros((len(polygon_list), len(polygon_list)))
		for _ in range(len(polygon_list)):
			for _x in range(_):
				sig = signatures[_]
				sig2 = signatures[_x]
				distance_mat[_][_x] = self.similarity(sig, sig2)
				distance_mat[_x][_] = distance_mat[_][_x]
			distance_mat[_][_] = 0

		cluster_count = int(math.sqrt(len(polygon_list)))

		linkage = scipy.cluster.hierarchy.centroid(distance_mat)
		clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.2)
		return clusters

	def polygons_in_cluster(self, polygon_list, clusters, id):
		select_polygons = []
		for _ in range(len(polygon_list)):
			if (clusters[_] == id):
				select_polygons.append(polygon_list[_])
		return select_polygons

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
		
	def align(self, poly, bounding_box):

		old_bounding_box = BoundingBox(poly)
		old_matrix = self.get_matrix(old_bounding_box)
		new_matrix = self.get_matrix(bounding_box)

		t_mat = (old_matrix.I * new_matrix).tolist()

		x_axis = Vector(t_mat[0][0],t_mat[0][1],t_mat[0][2])
		y_axis = Vector(t_mat[1][0],t_mat[1][1],t_mat[1][2])
		z_axis = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		translation = Vector(t_mat[2][0],t_mat[2][1],t_mat[2][2])

		processor = CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly)
		new_poly =processor.transform_mesh(translation, x_axis, y_axis, z_axis)

		return new_poly

	def stitch(self, polygon_list):
		processor = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(polygon_list[0])
		return processor.concatenate_mesh(polygon_list)


	def feature_vectors(self, poly):
		#Create feature vector from polyhedron
		remesher = remesh_voxel.RemeshVoxel()
		res = 10

		#Voxel vector representation
		mat = np.reshape(remesher.voxelize(poly, res),(-1))

		#Salient points histogram
		hist_points = self.signature(poly, False)[1]
		hist = np.histogram(np.log(hist_points),128)
		hist_features = np.append(hist[0],hist[1])

		#bounding box features
		bounding_box = remesh_voxel.BoundingBox(poly)
		box = self.get_matrix(bounding_box)

		vector = np.append(mat, hist_features)
		vector = np.append(vector, box)
		vector = np.reshape(np.array(vector),(-1))

		return vector

#IO helper functions
	def write(self,polygon_list, folder, offset=0):
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




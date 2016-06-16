import mesh

import numpy as np
import scipy
from scipy import cluster
from emd import emd
from os import listdir

import sys
sys.path.append("../Opensource/cgal-swig-bindings/build-python/")

import CGAL
from CGAL import CGAL_Kernel
from CGAL import CGAL_Polyhedron_3
from CGAL import CGAL_Polygon_mesh_processing

Vertex = CGAL.CGAL_Kernel.Point_3
Polyhedron = CGAL.CGAL_Polyhedron_3.Polyhedron_3
Mod = CGAL.CGAL_Polyhedron_3.Polyhedron_modifier
PolySignature = CGAL.CGAL_Polygon_mesh_processing.PolySignature

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
		poly = CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly).stitch_borders()

		return poly

	def segmentation(self, poly):
		poly_list = [x for x in CGAL.CGAL_Polygon_mesh_processing.Mesh_segmenter(poly).segmentation()]

		return poly_list

	def signature(self,poly):
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
		weights = np.array(weights)
		weights = np.divide(weights,weights.sum()) #normalize
		signature = [np.array(points), weights]
		return signature

	def similarity_metric(self,poly1,poly2):

		sig1 = self.signature(poly1)
		sig2 = self.signature(poly2)

		dist = emd(sig1[0], sig2[0], sig1[1], sig2[1]);

		return dist

	def cluster(self, polygon_list):
		distance_mat = np.zeros((len(polygon_list), len(polygon_list)))
		for _ in range(len(polygon_list)):
			for _x in range(len(polygon_list)):
				poly = polygon_list[_]
				poly2 = polygon_list[_x]
				distance_mat[_][_x] = self.similarity_metric(poly, poly2)
		
		linkage = scipy.cluster.hierarchy.centroid(distance_mat)
		clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.05)
		return clusters

	def polygons_in_cluster(self, polygon_list, clusters, id):
		select_polygons = []
		for _ in range(len(polygon_list)):
			if (clusters[_] == id):
				select_polygons.append(polygon_list[_])
		return select_polygons

#IO helper functions
	def write(self,polygon_list, folder):
		for _ in range(len(polygon_list)):
			location = folder + "{}.off".format(_)
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




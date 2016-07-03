import mesh
import mesh_processing

import pylib
from pylib import inrimage

import numpy as np
import math

import CGAL
from CGAL import CGAL_Kernel
from CGAL import CGAL_Polyhedron_3
from CGAL import CGAL_Surface_mesher
from CGAL import CGAL_Polygon_mesh_processing
from CGAL import CGAL_AABB_tree


C2T3 = CGAL.CGAL_Surface_mesher.Complex_2_in_triangulation_3
Tr = CGAL.CGAL_Surface_mesher.Surface_mesh_default_triangulation_3
Surface = CGAL.CGAL_Surface_mesher.Implicit_surface_Gray_level_image_3
Criteria = CGAL.CGAL_Surface_mesher.Surface_mesh_default_criteria_3
Voxel = CGAL.CGAL_Surface_mesher.Gray_level_image_3
Sphere = CGAL.CGAL_Kernel.Sphere_3

Polyhedron = CGAL.CGAL_Polyhedron_3.Polyhedron_3
Point = CGAL.CGAL_Kernel.Point_3
Segment = CGAL.CGAL_Kernel.Segment_3

Manifold_Tag = CGAL.CGAL_Surface_mesher.MANIFOLD_TAG
Mesher = CGAL.CGAL_Surface_mesher.make_surface_mesh
Output = CGAL.CGAL_Surface_mesher.output_surface_facets_to_polyhedron

BoundingBox = CGAL.CGAL_Polygon_mesh_processing.BoundingBox
AABB_handle = CGAL.CGAL_AABB_tree.AABB_tree_Polyhedron_3_Facet_handle

class RemeshVoxel(object):
	def __init__(self):
		return None

	def voxelize(self,poly, resolution):
		bounding_box = BoundingBox(poly)
		tree = AABB_handle(poly.facets())

		origin = bounding_box.get_origin()
		x_axis = bounding_box.get_x_axis()
		y_axis = bounding_box.get_y_axis()
		z_axis = bounding_box.get_z_axis()

		def get_value(x,y,z):
			#Look at the center of the box, 
			point = origin + x_axis * ((x+0.5)/resolution)
			point = point + y_axis * ((y+0.5)/resolution)
			point = point + z_axis * ((z+0.5)/resolution)

			interior_score = 0
			#create line segments in each direction & test
			for a in range(-1,1):
				for b in range(-1,1):
					for c in range(-1,1):
						point_2 = point + x_axis*a*0.5/resolution
						point_2 = point_2 + y_axis*b*0.5/resolution
						point_2 = point_2 + z_axis*c*0.5/resolution

						if tree.do_intersect(Segment(point, point_2)):
							interior_score = interior_score + 1.0/27.0

			return interior_score;

		mat = np.empty((resolution,resolution,resolution))

		for x in range(resolution):
			for y in range(resolution):
				for z in range(resolution):
					mat[x][y][z] = get_value(x,y,z)
		return mat

	def write_inr(self, mat, resolution, filename):
		settings = {}
		settings["VX"] = 1 
		settings["VY"] = 1 
		settings["VZ"] = 1 
		settings["TX"] = 1 
		settings["TY"] = 1 
		settings["TZ"] = 1 
		settings["SCALE"] = 2**0
		settings["CPU"] = 'decm'

		inrimage.write_inrimage(mat,settings, filename)

	def remesh(self, inr_file, bounding_box, iso_value, angular, radius, distance):
		c2t3 = C2T3(Tr())
		criteria = Criteria(angular, radius, distance)
		voxels = Voxel(inr_file, iso_value)

		b = bounding_box
		diag = (b.get_x_axis()+b.get_y_axis()+b.get_z_axis())

		boundary = Sphere(bounding_box.get_origin(), math.sqrt(diag.squared_length())/2.0)
		surface = Surface(voxels, boundary)

		Mesher(c2t3, surface, criteria)
		poly = Polyhedron()
		Output(c2t3, poly)
		return poly
		

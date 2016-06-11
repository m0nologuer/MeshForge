import mesh

import sys
sys.path.append("../Opensource/cgal-swig-bindings/build-python/")
import CGAL
from CGAL import *

Vertex = CGAL.CGAL_Kernel.Point_3
Polyhedron = CGAL.CGAL_Polyhedron_3.Polyhedron_3
Mod = CGAL.CGAL_Polyhedron_3.Polyhedron_modifier


class MeshSegmenter(object):
	"""Train model to automatically segment meshes"""

	def __init__(self):
		return None

	def create_polyhedron(self, mesh):
		
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

		return poly

	def create_mesh(self, poly):
		return poly


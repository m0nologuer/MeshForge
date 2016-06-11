from pyassimp import *

class Mesh(object):
	"""Base class for a 3D model"""
	def __init__(self, imported_mesh):
		super(Mesh, self).__init__()
		self.mesh = imported_mesh

	@staticmethod
	def load_scene(filename):
		scene = load(filename)
		meshes = [Mesh(imported_mesh) for imported_mesh in scene.meshes]
		return meshes

	def add_mesh_labels(self, filename):
		f = open(filename)
		entries = f.read().split("\n")
		labels = zip(entries[0::2], [x.split() for x in entries[1::2]]) 
		self.labels = [[label[0],[int(x) for x in label[1]]] for label in labels]

	

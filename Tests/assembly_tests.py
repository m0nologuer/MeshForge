import mesh_assembly
import unittest

MeshAssembler = mesh_assembly.MeshAssembler

class AssemblyTests(unittest.TestCase):
	def train_structure(self):
		
		assembler = MeshAssembler()
		self.assertTrue(poly.is_valid())
		#segment
		poly_list = mesh_processing.MeshProcessor().segmentation(poly)
		for poly in poly_list:
			self.assertTrue(poly.is_valid())
		pass


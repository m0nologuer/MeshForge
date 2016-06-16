import mesh_processing
import unittest


class MeshTests(unittest.TestCase):
	def test_segment_polyhedron(self):
		#load mesh
		mesh = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/21.off")[0]
		m = mesh_processing.MeshProcessor()
		poly = m.create_polyhedron(mesh)
		self.assertTrue(poly.is_valid())
		#segment
		poly_list = mesh_processing.MeshProcessor().segmentation(poly)
		for poly in poly_list:
			self.assertTrue(poly.is_valid())
		pass

	def test_mesh_similarity(self):
		#load mesh
		cup = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/21.off")[0]
		human = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Human/11.off")[0]
		cup2 = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/25.off")[0]
		human2 = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Human/17.off")[0]

		process = mesh_processing.MeshProcessor()
		poly_cup = process.create_polyhedron(cup)
		poly_human = process.create_polyhedron(human)
		poly_cup2 = process.create_polyhedron(cup2)
		poly_human2 = process.create_polyhedron(human)
		
		d1 = process.similarity_metric(poly_cup, poly_cup2)
		d2 = process.similarity_metric(poly_cup, poly_human)
		d3 = process.similarity_metric(poly_cup2, poly_human)
		d4 = process.similarity_metric(poly_human, poly_human2)
		pass

if __name__ == '__main__':
    unittest.main()



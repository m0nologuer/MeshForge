import mesh_segmenter

class MeshTests(unittest.TestCase):
    def load_polyhedron_test(self):
    	mesh = mesh_segmenter.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/21.off")[0]
        m = mesh_segmenter.MeshSegmenter()
        poly = m.create_polyhedron(mesh)
        self.assertIsTrue(poly.is_valid())

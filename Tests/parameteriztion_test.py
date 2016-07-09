import sys
sys.path.append("../Opensource/cgal-swig-bindings/build-python/")
sys.path.append("../MeshForge")

import CGAL
from CGAL import CGAL_Kernel
from CGAL import CGAL_Polyhedron_3
from CGAL import CGAL_Polygon_mesh_processing

import mesh_processing

torus = mesh_processing.mesh.Mesh.load_scene("../Datasets/Torus.obj")[0]
poly = mesh_processing.MeshProcessor().create_polyhedron(torus)
cutter = CGAL.CGAL_Polygon_mesh_processing.Mesh_cuts(poly)
x = cutter.parameterize(10,30)

#util = CGAL.CGAL_Polygon_mesh_processing.Mesh_util(poly)
#util.subdivide_mesh(2)
#x = util.get_mesh()
#x.write_to_file("new.off")
#cutter.get_cuts()

#cup = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/21.off")[0]
#poly = mesh_processing.MeshProcessor().create_polyhedron(cup)
#cutter = CGAL.CGAL_Polygon_mesh_processing.Mesh_cuts(poly)

#x = cutter.get_cuts()

#x = cutter.parameterize(10,30)


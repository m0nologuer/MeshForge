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
cutter.get_cuts()
x = cutter.parameterize(10,5)

size = 64
pixel_grid = []
for x in range(size):
	pixels_row = []
	for y in range(size):
		point = cutter.sample_geometry(x*0.125*0.125,y*0.125*0.125)
		pixels_row.append([numpy.uint16(point.x()*50+50), numpy.uint16(point.y()*50+50), numpy.uint16(point.z()*50+50)])
	pixel_grid.append(pixels_row)


with open('foo_color.png', 'wb') as f:
    writer = png.Writer(width=64, height=64, bitdepth=16)
    # Convert z to the Python list of lists expected by
    # the png writer.
    z2list = numpy.reshape(pixel_grid,(-1, 64*3)).tolist()
    writer.write(f, z2list)

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


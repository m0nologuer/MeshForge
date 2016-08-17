import sys
sys.path.append("../MeshForge")


import mesh_processing
torus = mesh_processing.mesh.Mesh.load_scene("../Datasets/Torus.obj")[0]
poly = mesh_processing.MeshProcessor().create_polyhedron(torus)
import remesh_voxel
remesher = remesh_voxel.RemeshVoxel()
mat = remesher.voxelize(poly, 10)
remesher.write_inr(mat,10,"out.inr")
bounding_box = remesh_voxel.BoundingBox(poly)
poly = remesher.remesh("out.inr",bounding_box,0.9,30.,0.2,0.2)

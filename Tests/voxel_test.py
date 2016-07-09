import sys
sys.path.append("../MeshForge")


import mesh_processing
cup = mesh_processing.mesh.Mesh.load_scene("../Datasets/Labelled_Segmented_Models/Cup/21.off")[0]
poly = mesh_processing.MeshProcessor().create_polyhedron(cup)
import remesh_voxel
remesher = remesh_voxel.RemeshVoxel()
mat = remesher.voxelize(poly, 20)
remesher.write_inr(mat,20,"out.inr")
bounding_box = remesh_voxel.BoundingBox(poly)
poly = remesher.remesh("out.inr",bounding_box,0.9,30,1,1)

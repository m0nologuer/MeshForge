import assembly_model
import mesh_processing
import model_feature_vectors
processor = mesh_processing.MeshProcessor()
tables = processor.load("../Datasets/Labelled_Segmented_Models/Table/")
ass = assembly_model.AssemblyModel()
fv_processor = model_feature_vectors.ModelFeatureVectorProcessor()

data, labels, pca = fv_processor.create_feature_vectors(tables)
ass.build_component_style_model(data, labels)

reload(assembly_model)
ops.reset_default_graph()
ass = assembly_model.AssemblyModel()
ass.reload()
ass.generate(2, [1, 1, 1, 1, 1] , [], labels, data, pca).write_to_file("out.off")

import assembly_model
import mesh_processing
import numpy
from numpy import *
data = [{'C_type5': array([ -3.57860652e-01,  -3.55400984e-01,   2.65421191e-16]), 'C_type4': array([ -5.00000000e-01,  -5.53055849e-16,   1.09395663e-16]), 'C_type6': array([ -5.00000000e-01,   8.52277660e-15,   6.32543900e-17]), 'C_type1': array([ -2.00000000e+00,   1.61150790e-15,  -5.59551353e-18]), 'C_type0': array([ -1.43200250e+00,   1.60133772e-01,  -8.60958532e-16]), 'C_type3': array([ -2.00000000e+00,  -1.96631796e-15,   2.49763251e-17]), 'C_type2': array([ -4.09026923e-01,   4.08580961e-01,   2.47478030e-17]), 'N_type0': array([ 1.,  0.,  0.]), 'N_type1': array([ 0.,  1.,  0.]), 'N_type2': array([ 0.,  1.,  0.]), 'N_type3': array([ 0.,  1.,  0.]), 'N_type4': array([ 1.,  0.,  0.]), 'N_type5': array([ 0.,  1.,  0.]), 'N_type6': array([ 1.,  0.,  0.])}, {'C_type5': array([  1.00000000e+00,   6.15251856e-02,  -2.66550643e-16]), 'C_type4': array([ -5.00000000e-01,  -5.53055849e-16,   1.09395663e-16]), 'C_type6': array([  1.00000000e+00,  -1.70483940e-14,  -7.59696491e-18]), 'C_type1': array([  1.00000000e+00,  -8.16478683e-16,   2.56461037e-18]), 'C_type0': array([  4.32002498e-01,  -6.85646913e-01,   2.41543198e-15]), 'C_type3': array([  1.00000000e+00,   9.20718165e-16,  -1.02175875e-17]), 'C_type2': array([ -5.90973077e-01,  -3.61855007e-01,   2.74877384e-16]), 'N_type0': array([ 0.,  1.,  0.]), 'N_type1': array([ 1.,  0.,  0.]), 'N_type2': array([ 0.,  1.,  0.]), 'N_type3': array([ 1.,  0.,  0.]), 'N_type4': array([ 1.,  0.,  0.]), 'N_type5': array([ 1.,  0.,  0.]), 'N_type6': array([ 0.,  0.,  1.])}, {'C_type5': array([ -6.42139348e-01,   2.93875798e-01,   4.65898900e-17]), 'C_type4': array([  1.00000000e+00,   1.12312880e-15,  -1.29654119e-16]), 'C_type6': array([ -5.00000000e-01,   8.52277660e-15,   6.32543900e-17]), 'C_type1': array([  1.00000000e+00,  -8.16478683e-16,   2.56461037e-18]), 'C_type0': array([  1.00000000e+00,   5.25513141e-01,  -1.51056414e-15]), 'C_type3': array([  1.00000000e+00,   9.20718165e-16,  -1.02175875e-17]), 'C_type2': array([  1.00000000e+00,  -4.67259538e-02,  -2.54548831e-16]), 'N_type0': array([ 0.,  1.,  0.]), 'N_type1': array([ 1.,  0.,  0.]), 'N_type2': array([ 1.,  0.,  0.]), 'N_type3': array([ 1.,  0.,  0.]), 'N_type4': array([ 0.,  1.,  0.]), 'N_type5': array([ 0.,  1.,  0.]), 'N_type6': array([ 1.,  0.,  0.])}]
labels = ['type0', 'type1', 'type2', 'type3', 'type4', 'type5', 'type6']
ass = assembly_model.AssemblyModel()
ass.build_component_style_model(data, labels)


import mesh_library
lib = mesh_library.MeshLibrary()
lib.add_files_from_folder("../Datasets/Labelled_Segmented_Models/Human/", False)

mesh_library.Mesh.objects.all().delete()


m = Mesh(model_type=1, model_style=1, face_count=200, generated= False)
m.save()
 

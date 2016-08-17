import assembly_model
ass = assembly_model.AssemblyModel()
R = 4
S = [1,3]
labels = ["arm", "leg"]
opt, error, nodes = ass.build_node_tree(R,S,labels)
training_set = ass.fake_feature_vectors(R,S,labels,100)
x = ass.train(training_set,opt, error, nodes,100)
for node in nodes:
	node.copy_variables_out()

nodes[0].joint_probability({"R":0})
nodes[1].joint_probability({"R":1, "S_arm":0})
nodes[2].joint_probability({"R":1, "S_arm":0, "C_arm":[0,1]})

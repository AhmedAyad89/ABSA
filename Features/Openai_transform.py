from depend.openAI.encoder import Model
import json

model = None

#takes in a json_dict, a field name.
#encodes the field using the openai rnn and adds the results to the dict
def openai_transform(json_dict, field_name):
	global model
	if model is None:
		model = Model()

	features = []
	for i in sorted(json_dict):
		features.append(json_dict[i][field_name])

	vecs = model.transform(features)

	for counter, key in enumerate(sorted(json_dict)):
		json_dict[key][field_name+'-openai_vec'] = vecs[counter].tolist()

	return json_dict

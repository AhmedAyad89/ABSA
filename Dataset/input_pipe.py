import json
from Dataset.dataset_utils import *
from itertools import compress

#read the dataset, gets the text and labels
#encodes the labels
def prep_semeval_aspects(domain='laptop', single=False, extra_train_features=None, extra_test_features=None):

	train_path = 'Dataset/Semeval_'+domain+'_train.json'
	test_path = 'Dataset/Semeval_'+domain+'_test.json'
	with open(train_path, 'r') as infile:
		train_data = json.load(infile)
	with open(test_path, 'r') as infile:
		test_data = json.load(infile)


	size = len(train_data)
	train_data = [train_data[str(x)] for x in range(size)]
	size = len(test_data)
	test_data = [test_data[str(x)] for x in range(size)]
	train_text = [x['sentence'] for x in train_data]
	test_text = [x['sentence'] for x in test_data]
	train_vecs =  [x['sentence-openai_vec'] for x in train_data]
	test_vecs = [x['sentence-openai_vec'] for x in test_data]
	
	train_vecs, test_vecs = load_extra_features(extra_train_features, extra_test_features, train_vecs, test_vecs)

	train_labels, encoded_train_labels, test_labels, encoded_test_labels, encoder, labeling = \
		get_labels(train_data=train_data, test_data=test_data, single=single, subtask=[True, True, False])

	dict = {}
	dict['train_data'] = train_text
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = encoded_train_labels
	dict['test_data'] = test_text
	dict['test_labels'] = test_labels
	dict['test_vecs'] = test_vecs
	dict['encoded_test_labels'] = encoded_test_labels
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	return dict

def get_labels(train_data, single=False, test_data=None, subtask=[True, True, False]):
	print('getting labels')
	train_labels = [x['label'] for x in train_data]
	if test_data is not None:
		test_labels = [x['label'] for x in test_data]
	else:
		test_labels = []

	if single:
		for i in range(len(train_labels)):
			train_labels[i] = train_labels[i][0]
		for i in range(len(test_labels)):
			test_labels[i] = test_labels[i][0]
	else:
		irrelevant = ([['NA', 'NA', 'NA']], [['not relevant', 'not relevant', 'not relevant']], [['relevant', 'relevant', 'relevant']])
		for i in range(len(train_labels)):
			if train_labels[i] in irrelevant:
				train_labels[i] = []
		for i in range(len(test_labels)):
			if test_labels[i] in irrelevant:
				test_labels[i] = []

	encoded_labels, encoder, labeling = encode_labels \
		(labels=train_labels + test_labels, scheme=[True, True, False])
	encoded_train_labels = encoded_labels[:len(train_labels)]
	encoded_test_labels = encoded_labels[len(train_labels):]

	if single:
		train_labels = [list(compress(x, subtask)) for x in train_labels]
		test_labels = [list(compress(i, subtask)) for i in test_labels]
	else:
		new_test = []
		for i, entry in enumerate(test_labels):
			tmp = []
			for a in entry:
				tmp.append(list(compress(a, subtask)))
			new_test.append(tmp)
		new_train = []
		for i, entry in enumerate(train_labels):
			tmp = []
			for a in entry:
				tmp.append(list(compress(a, subtask)))
			new_train.append(tmp)
		train_labels = new_train
		test_labels = new_test

		return train_labels, encoded_train_labels, test_labels, encoded_test_labels, encoder, labeling

def load_extra_features(train_feat, test_feat, train_vecs, test_vecs):
	for file in train_feat:
		with open(file, 'rb') as infile:
			features = pickle.load(infile)
			train_vecs = np.concatenate((train_vecs, features), axis=1)
	for file in test_feat:
		with open(file, 'rb') as infile:
			features = pickle.load(infile)
			test_vecs = np.concatenate((test_vecs, features), axis=1)

	return train_vecs, test_vecs


if __name__ == '__main__':
	prep_semeval_aspects()
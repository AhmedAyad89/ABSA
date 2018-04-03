import json
import pickle
from Dataset.dataset_utils import *
from itertools import compress
from os import path
from math import ceil
import numpy as np
from Features.Openai_transform import *
from Features.bag_of_words import *


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
	
	train_vecs, test_vecs = load_extra_features(extra_train_features, train_vecs, extra_test_features, test_vecs)

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

def prep_organic_aspects(single=False, rel_filter=True, bow_features=True, merged=True, extra_features=None):
	train_path = 'Dataset/Organic_train_test.json'
	with open(train_path, 'r') as infile:
		train_data = json.load(infile)

	size = len(train_data)
	train_data = [train_data[str(x)] for x in range(size)]
	if rel_filter:
		irrelevant = ([['not relevant', 'not relevant', 'not relevant']], [['relevant', 'relevant', 'relevant']])
		train_data = [x for x in train_data if x['label'] not in irrelevant ]
		print('remaining ',len(train_data))
	train_text = [x['sentence'] for x in train_data]
	train_vecs = [x['sentence-openai_vec'] for x in train_data]

	if extra_features is not None:
		train_vecs, _= load_extra_features(extra_features, train_vecs)

	if bow_features:
		features = bow_clusters_features(data = train_text)
		train_vecs = np.concatenate((train_vecs, features), axis=1)

	label = 'label'
	if merged:
		label = 'merged_label'
	train_labels, encoded_train_labels, test_labels, encoded_test_labels, encoder, labeling = \
		get_labels(train_data=train_data, test_data=None, single=single,  subtask=[True, True, False], label=label)

	dict = {}
	dict['train_data'] = train_text
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = encoded_train_labels
	dict['test_labels'] = test_labels
	dict['encoded_test_labels'] = encoded_test_labels
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	return dict


def get_labels(train_data, single=False, test_data=None, subtask=[True, True, False], label='label'):
	print('getting labels')
	train_labels = [x[label] for x in train_data]
	if test_data is not None:
		test_labels = [x[label] for x in test_data]
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
		(labels=train_labels + test_labels, scheme=subtask)
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

def load_extra_features(train_feat, train_vecs, test_feat=None, test_vecs=None):
	for file in train_feat:
		with open(file, 'rb') as infile:
			features = pickle.load(infile)
			train_vecs = np.concatenate((train_vecs, features), axis=1)
	if test_feat is not None:
		for file in test_feat:
			with open(file, 'rb') as infile:
				features = pickle.load(infile)
				test_vecs = np.concatenate((test_vecs, features), axis=1)

	return train_vecs, test_vecs

def read_sentiment_file(data_path, pickle_path, labeling=None, encoder=None):
	with open(data_path, 'r') as infile:
		data = json.load(infile)

	size = len(data)
	train_data = [data[str(x)] for x in range(size)]

	train_text = [x['sentence'] for x in train_data]
	train_labels = [x['label'] for x in train_data]

	train_text, train_labels = merge_sentence_with_aspect(train_text, train_labels)

	encoded_labels, encoder, labeling = encode_labels \
		(labels=train_labels, scheme=[True], labeling=labeling, encoder=encoder)

	#The training vectors are loaded from pickle files to save time, they are the openAI transforms of the train/test text
	basepath = path.dirname(__file__)
	filepath = path.abspath(path.join(basepath, "..",pickle_path))
	with open(filepath, 'rb') as infile:
		train_vecs = pickle.load(infile)
	return train_text, train_vecs, train_labels, encoded_labels, encoder, labeling

def prep_sentiment(domain, test_set_partition = None):
	if domain in ['laptop', 'restaurant']:
		train_path = 'Dataset/Semeval_' + domain + '_train.json'
		test_path = 'Dataset/Semeval_' + domain + '_test.json'
		train_feats =  "Features/sentiment_transforms/semeval_"+domain+"-sent-openAI-data"
		test_feats = "Features/sentiment_transforms/semeval_"+domain+"-sent_test-openAI-data"
	elif domain == 'organic':
		train_path = 'Dataset/Organic_train_test.json'
		train_feats = 'Features/sentiment_transforms/organic_sent_merged_openAI-data'
	else:
		print('domain invalid')
		return 0

	train_text, train_vecs, train_labels, encoded_train_labels, encoder, labeling= \
		read_sentiment_file(train_path, train_feats)

	if domain in ['laptop', 'restaurant']:
		test_text, test_vecs, test_labels, encoded_test_labels, _, _ = \
			read_sentiment_file(test_path, test_feats, labeling, encoder)
	elif test_set_partition is not None:
		dataset_size = len(train_text)
		test_size = ceil(test_set_partition * dataset_size)
		test_idx =  np.random.randint(0,dataset_size, test_size)
		final_train_text = []
		final_train_labels = []
		final_train_vecs = []
		final_encoded_train_labels = []
		test_text = []
		test_labels=[]
		test_vecs=[]
		encoded_test_labels=[]
		for i in range(dataset_size):
			if i in test_idx:
				test_text.append(train_text[i])
				test_vecs.append(train_vecs[i])
				encoded_test_labels.append(encoded_train_labels[i])
				test_labels.append(train_labels[i])
			else:
				final_train_text.append(train_text[i])
				final_train_labels.append(train_labels[i])
				final_encoded_train_labels.append(encoded_train_labels[i])
				final_train_vecs.append(train_vecs[i])
		train_labels = final_train_labels
		train_vecs = final_train_vecs
		encoded_train_labels = final_encoded_train_labels
		train_text = final_train_text

	dict = {}
	dict['train_data'] = train_text
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = np.asarray(encoded_train_labels)
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	if domain in ['laptop', 'restaurant'] or test_set_partition is not None:
		dict['test_data'] = test_text
		dict['test_labels'] = test_labels
		dict['test_vecs'] = test_vecs
		dict['encoded_test_labels'] = np.asarray(encoded_test_labels)
	return dict

def	merge_sentence_with_aspect(train_sentences, train_labels, test_sentences=None, test_labels=None):
	new_data = []
	new_labels = []
	for i, sentence in enumerate(train_sentences):
		for label in train_labels[i]:
			if 'NA' in label or 'relevant' in label or 'not relevant' in label:
				continue
			l =  ' '.join(label[:2])
			l = l.lower()
			l = l.replace('_', ' ')
			new_data.append(sentence +' '+ l)
			new_labels.append(label[2:])
	train_data = new_data
	train_labels = new_labels


	return train_data, train_labels

def create_sentiment_features(data_path, out_path, label='label'):
	with open(data_path, 'r') as infile:
		data = json.load(infile)

	size = len(data)
	train_data = [data[str(x)] for x in range(size)]
	train_text = [x['sentence'] for x in train_data]
	train_labels = [x[label] for x in train_data]
	train_text, train_labels = merge_sentence_with_aspect(train_text, train_labels)

	train_vecs = openai_sentence_transform(train_text)

	with open(out_path, 'wb') as outfile:
		pickle.dump(train_vecs, outfile)

if __name__ == '__main__':
	basepath = path.dirname(__file__)
	filepath = path.abspath(path.join(basepath, "..",'Features/sentiment_transforms/organic_sent_merged_openAI-data'))
	create_sentiment_features('Organic_train_test.json', filepath, label='merged_label' )
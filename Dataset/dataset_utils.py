import json
import pickle
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from itertools import compress
from Features.Openai_transform import openai_transform
from read_organic import parse_hannah_csv, parse_hannah_legend, hannah_map_labels


##takes data as a list of lists and returns linear list
#takes multi-labels as a list of lists of lists and returns a linear single-label list
#takes only the first label of each sentence
def flatten_single_label(pickle_data=None, pickle_labels =None, data=None, labels=None):
	if pickle_data is not None :
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if pickle_labels is not None:
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels = []
	for i, review in enumerate(data):
		for j, sentence in enumerate(review):
			new_data.append(sentence)
			new_labels.append(labels[i][j][0])
			#print(sentence, labels[i][j][0], '\n----------\n')

	# for i in range(len(new_data)):
	# 	print(new_data[i], new_labels[i])
	# 	print('\n-----------------\n',i)
	print(len(new_data), len(new_labels))
	return new_data, new_labels


# this one returns labels as a list of lists
def flatten_multi_label(pickle_data=None, pickle_labels=None, data=None, labels=None):
	if pickle_data is not None:
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if  pickle_labels is not None:
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels = []
	review_lengths = []
	# for i, label in enumerate(labels):
		# for j, e in enumerate(label):
			# if e == [['NA', 'NA', 'NA']]:
			# 	labels[i][j] = []
	for i, review in enumerate(data):
		l = 0
		for j, sentence in enumerate(review):
			l += 1
			new_data.append(sentence)
			new_labels.append(labels[i][j])
		review_lengths.append(l)
	return new_data, new_labels


def encode_labels(pickle_labels= None, labels=None, one_hot= True,
									scheme=[True,True,True], labeling=None, encoder=None):
	if pickle_labels is not None:
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (labeling is None):
		labeling=[]

	L=[]
	single = True
	for i in labels:
		if any(isinstance(a, list) for a in i) :
			single =False
			break

	if single:	#single label
		for e in labels:
			#print(e)
			#e[0] = entities.index(e[0])
			#e[1] = attrs.index(e[1])
			#e[2] = polarity.index(e[2]) - 1
			filtered = list(compress(e,scheme))
			#print(filtered)
			if (filtered not in labeling):
				labeling.append(filtered)
			L.append(labeling.index(filtered))
			# print('-----------single--------------\n')
		if (one_hot):
			if encoder is None:
				encoder = LabelBinarizer()
				L = encoder.fit_transform(np.asarray(L).reshape(len(L), 1))
			else:
				L = encoder.transform(np.asarray(L).reshape(len(L), 1))
		return L, encoder, labeling
	else:																		#multi-label
		for s in labels:
			S=[]
			for e in s:
				#e[0]= entities.index(e[0])
				#e[1] = attrs.index(e[1])
				#e[2]=polarity.index(e[2])-1
				filtered = list(compress(e, scheme))
				#print(filtered)
				if (filtered not in labeling):
					labeling.append(filtered)
				S.append(labeling.index(filtered))
			L.append(S)
			# print('-------------------------\n')
		if (one_hot):
			# for lab in L:
			# 	temp = np.zeros(len(labeling))
			# 	for i in lab:
			# 		temp[i] = 1
			# 	oh.append( temp )
			sum=0
			for entry in L:
				sum += len(entry)
			print('avg length', sum/ len(L))
			if encoder is None:
				encoder = MultiLabelBinarizer()
				L = encoder.fit_transform(L)
			else:
				L = encoder.transform(L)
			return L, encoder, labeling

	return L,0, labeling

#convert and save a flattned dataset as json
#label is [enttiy, attr, sentiment]
def dataset_to_json(data, labels,  dataset_name, merged_labels=None):
	json_dict = {}
	for i, sentence in enumerate(data):
		d = {}
		d['sentence'] = sentence
		d['label'] = labels[i]
		if merged_labels is not None:
			d['merged_label'] = merged_labels[i]
		json_dict[str(i)] = d

	with open(dataset_name, 'w', encoding='utf-8') as outfile:
		json.dump(json_dict, outfile)


def create_organic_json():
	#create organic json
	data, labels = parse_hannah_csv('Organic/19-03-05_comments_HD.csv')
	entity_dict, attr_dict = parse_hannah_legend('Organic/legend.csv')
	sent_dict = {'p': 'Positive', 'n': 'Negative', '0': 'Neutral',
								'': 'not relevant', 'NA': 'not relevant', 'rel': 'relevant'}
	full_labels, merged_labels = hannah_map_labels(labels, entity_dict, attr_dict, sent_dict)

	dataset_to_json(data, full_labels, 'Organic_train_test.json', merged_labels)


def create_semeval_json():
	#laptop
	data, labels = flatten_multi_label('SemEval/semevalLaptop-rawTextData', 'SemEval/semevalLaptop-rawTextLabels')
	dataset_to_json(data, labels, 'Semeval_laptop_train.json')
	data, labels = flatten_multi_label('SemEval/semevalLaptop_test-rawTextData', 'SemEval/semevalLaptop_test-rawTextLabels')
	dataset_to_json(data, labels, 'Semeval_laptop_test.json')
	# restaurant
	data, labels = flatten_multi_label('SemEval/semevalRestaurant-rawTextData', 'SemEval/semevalRestaurant-rawTextLabels')
	dataset_to_json(data, labels, 'Semeval_restaurant_train.json')
	data, labels = flatten_multi_label('SemEval/semevalRestaurant_test-rawTextData', 'SemEval/semevalRestaurant_test-rawTextLabels')
	dataset_to_json(data, labels, 'Semeval_restaurant_test.json')


#for each json file, add the open AI transform of the text sentences
def add_openai_sentence_vecs_(path=None):
	if path is None:
		path = os.getcwd()
	for filename in glob.glob(os.path.join(path, '*.json')):
		with open(filename, 'r') as file:
			data = json.load(file)
		updated  = openai_transform(data, 'sentence')
		with open(filename, 'w') as outfile:
			json.dump(updated, outfile)


if __name__ == '__main__':
	print(0)

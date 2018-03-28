import json
import pickle
from read_organic import parse_hannah_csv, parse_hannah_legend, hannah_map_labels
from itertools import compress


#takes data as a list of lists and returns linear list
#takes multi-labels as a list of lists of lists and returns a linear single-label list
#takes only the first label of each sentence
def flatten_single_label(pickle_data=None, pickle_labels =None, data=None, labels=None):
	if(pickle_data is not None):
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if (pickle_labels is not None):
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels=[]
	for i,review in enumerate(data):
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
def flatten_multi_label(pickle_data=None, pickle_labels =None, data=None, labels=None):
	if(pickle_data is not None):
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if (pickle_labels is not None):
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels=[]
	review_lengths =[]
	# for i, label in enumerate(labels):
		# for j, e in enumerate(label):
			# if e == [['NA', 'NA', 'NA']]:
			# 	labels[i][j] = []
	for i,review in enumerate(data):
		l = 0
		for j, sentence in enumerate(review):
			l+=1
			new_data.append(sentence)
			new_labels.append(labels[i][j])
		review_lengths.append(l)
	return new_data, new_labels


#convert and save a flattned dataset as json
#label is [enttiy, attr, sentiment]
def dataset_to_json(data, labels,  dataset_name, merged_labels = None):

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


if __name__ == '__main__':
	print(0)
	#create organic json
	# data, labels = parse_hannah_csv('Organic/19-03-05_comments_HD.csv')
	# entity_dict, attr_dict = parse_hannah_legend('Organic/legend.csv')
	# sent_dict = { 'p': 'Positive', 'n': 'Negative', '0':'Neutral',
	# 							'':'not relevant', 'NA':'not relevant', 'rel': 'relevant' }
	# full_labels, merged_labels = hannah_map_labels(labels, entity_dict, attr_dict, sent_dict)
	#
	# dataset_to_json(data, full_labels, 'Organic_train_test.json', merged_labels)


	#Create Semeval json
	#laptop
	# data, labels = flatten_multi_label('SemEval/semevalLaptop-rawTextData', 'SemEval/semevalLaptop-rawTextLabels')
	# dataset_to_json(data, labels, 'Semeval_laptop_train.json')
	# data, labels = flatten_multi_label('SemEval/semevalLaptop_test-rawTextData', 'SemEval/semevalLaptop_test-rawTextLabels')
	# dataset_to_json(data, labels, 'Semeval_laptop_test.json')
	#restaurant
	# data, labels = flatten_multi_label('SemEval/semevalRestaurant-rawTextData', 'SemEval/semevalRestaurant-rawTextLabels')
	# dataset_to_json(data, labels, 'Semeval_Restaurant_train.json')
	# data, labels = flatten_multi_label('SemEval/semevalRestaurant_test-rawTextData', 'SemEval/semevalRestaurant_test-rawTextLabels')
	# dataset_to_json(data, labels, 'Semeval_Restaurant_test.json')
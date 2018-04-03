import numpy as np
import copy
from sklearn import metrics


def single_label_metrics(pred, labels, encoded_labels, labeling, data):
	try:
		print('micro', metrics.f1_score(encoded_labels, pred, average='micro'))
	# print('macro', metrics.f1_score(l, pred, average='macro'))
	# print('weight', metrics.f1_score(l, pred, average='weighted'))
	except Exception as e:
		print(e)
	print('f1 samples', metrics.f1_score(encoded_labels, pred, average='samples'))
	print('precision samples', metrics.precision_score(encoded_labels, pred, average='samples'))
	print('recall samples', metrics.recall_score(encoded_labels, pred, average='samples'))
	decoded_pred = np.where(pred == 1)[1]
	raw_pred=[]
	for i in range(len(labels)):
		if labels[i] == ['not relevant', 'not relevant']:
			continue
		raw_pred.append(labeling[decoded_pred[i]])
		print(i, '- ', data[i])
		print('Prediction: ', (labeling[decoded_pred[i]]), decoded_pred[i])
		print('Label: ', labels[i])
		print('\n', '\n--------------------------\n')
	return raw_pred


def multi_label_metrics(pred, labels, encoded_labels, labeling, data, mute=True):
	try:
		print('micro', metrics.f1_score(encoded_labels, pred, average='micro'))
		# print('f1 samples', metrics.f1_score(encoded_labels, pred, average='samples'))
		# print('precision samples', metrics.precision_score(encoded_labels, pred, average='samples'))
		# print('recall samples', metrics.recall_score(encoded_labels, pred, average='samples'))
		print('accuracy', metrics.accuracy_score(encoded_labels, pred))
		acc = metrics.accuracy_score(encoded_labels, pred)
		f1 = metrics.f1_score(encoded_labels, pred, average='micro')
	except Exception as e:
		print('accuracy', metrics.accuracy_score(encoded_labels, pred))
		print(e)

	decoded_pred = []
	for p in pred:
		decoded_pred.append(np.where(p == 1)[0])
	if not mute:
		for i in range(len(labels)):
			try:
				if (not labels[i]) and (not decoded_pred[i]):
					continue
			except:
				p=4
			print(i, '- ', data[i])
			print('Prediction: ',  [labeling[x] for x in decoded_pred[i]] )
			print('Label: ', labels[i])
			print('\n', '\n--------------------------\n')
	return acc, f1


def print_raw_predictions(pred, labels, encoded_labels, labeling, data, mute=True):
	decoded_pred = []
	for p in pred:
		decoded_pred.append(np.where(p == 1)[0])
	if not mute:
		for i in range(len(labels)):
			print(i, '- ', data[i])
			print('Prediction prob: ', [(labeling[x],pred[i][x]) for x in range(len(pred[i]))])
			print('Label: ', labels[i])
			print('\n', '\n--------------------------\n')


def fold_data_dict(data_dict, start, stop):

	folded = copy.deepcopy(data_dict)
	folded['test_labels'] = data_dict['train_labels'][start : stop]
	folded['encoded_test_labels'] = data_dict['encoded_train_labels'][start : stop]
	folded['test_data'] = data_dict['train_data'][start : stop]
	folded['test_vecs'] = data_dict['train_vecs'][start : stop]

	if start !=0  and stop < len(data_dict['train_labels']):
		folded['train_labels'] =  np.concatenate([data_dict['train_labels'][:start] , data_dict['train_labels'][stop:]])
		folded['encoded_train_labels'] = np.concatenate([data_dict['encoded_train_labels'][:start] , data_dict['encoded_train_labels'][stop:]])
		folded['train_data'] = np.concatenate([data_dict['train_data'][:start] , data_dict['train_data'][stop:]])
		folded['train_vecs'] = np.concatenate([data_dict['train_vecs'][:start] , data_dict['train_vecs'][stop:]])
	elif start == 0:
		folded['train_labels'] = data_dict['train_labels'][stop:]
		folded['encoded_train_labels'] = data_dict['encoded_train_labels'][stop:]
		folded['train_data'] = data_dict['train_data'][stop:]
		folded['train_vecs'] = data_dict['train_vecs'][stop:]
	else:
		folded['train_labels'] = data_dict['train_labels'][:start]
		folded['encoded_train_labels'] = data_dict['encoded_train_labels'][:start]
		folded['train_data'] = data_dict['train_data'][:start]
		folded['train_vecs'] = data_dict['train_vecs'][:start]

	return folded
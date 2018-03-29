import numpy as np
from sklearn.utils import shuffle as skshuffle
import json
import csv
from sklearn.preprocessing import scale
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
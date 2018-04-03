from utils import *
from Dataset.input_pipe import *
from Learning.tf_multipath_classifier import *


def config_graph():
	paths = []

	path = {}
	path['input_dim'] = 4116
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.16)
	path['input'] = 'organic'
	paths.append(path)

	path = {}
	path['name'] = 'aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [11], batch_norm=False, activation=None)
	path['optimizer'] = tf.train.AdamOptimizer(name='optimizer', learning_rate=0.0001 , beta1=0.92 , beta2=0.9999)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)

	return paths


org_dict_full = prep_organic_aspects()
dataset_size = len(org_dict_full['train_data'])

folds = 10
fold_size= ceil(dataset_size / folds)
avg_f1 = 0
for f in range(0,folds):
	fold_start = f * fold_size
	fold_end = min((f+1) * fold_size, dataset_size )
	print(fold_start, fold_end)
	org_dict = fold_data_dict(org_dict_full, fold_start, fold_end )

	datasets = []
	dataset = {}
	dataset['name'] = 'organic'
	# dataset['holdout'] = 50
	dataset['batch_size'] = 10
	dataset['features'] = org_dict['train_vecs']
	dataset['type'] = tf.float32
	dataset['tasks'] = [{'name' : 'aspects', 'features' : org_dict['encoded_train_labels'], 'type': tf.float32}]
	datasets.append(dataset)

	paths = config_graph()
	params = {}
	params['train_iter'] = 4001

	model = TfMultiPathClassifier(datasets, paths, params)

	model.train()
	model.save()

	y = model.get_prediciton('aspects', org_dict['test_vecs'])
	x = model.get_prediciton('aspects', org_dict['train_vecs'])

	multi_label_metrics(x, org_dict['train_labels'], org_dict['encoded_train_labels'],
											org_dict['labeling'], org_dict['train_data'] )

	_, f1 = multi_label_metrics(y, org_dict['test_labels'], org_dict['encoded_test_labels'],
											org_dict['labeling'], org_dict['test_data'], mute=True )
	avg_f1 +=f1

avg_f1 = avg_f1 / folds
print('\n--------------------------------------------------------------------------\nAverage F1 score:', avg_f1)
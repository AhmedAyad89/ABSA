from utils import *
from Dataset.input_pipe import *
from Learning.tf_multipath_classifier import *
from math import ceil

def config_graph():
	paths = []

	path = {}
	path['input_dim'] = 4096
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.9,
																			 ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.9, scope=None))
	path['input'] = 'sent'
	paths.append(path)

	path = {}
	path['name'] = 'sent'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [3], batch_norm=False, activation=None)
	path['loss'] = loss_map('softmax')
	path['predictor'] = softmax_predictor()
	path['optimizer'] = tf.train.AdamOptimizer(name='sent_optimizer', learning_rate=0.0001, beta1=0.92)
	paths.append(path)

	return paths

organic_dict_full = prep_sentiment(domain='organic', test_set_partition=None)

dataset_size = len(organic_dict_full['train_data'])
print(dataset_size)

folds = 7
fold_size= ceil(dataset_size / 7)
avg_acc = 0
for f in range(0,folds):
	fold_start = f * fold_size
	fold_end = min((f+1) * fold_size, dataset_size )
	print(fold_start, fold_end)
	organic_dict = fold_data_dict(organic_dict_full, fold_start, fold_end )

	datasets=[]
	dataset={}
	dataset['name'] = 'organic_sentiment'
	# dataset['holdout'] = 30
	dataset['batch_size'] = 10
	dataset['features'] = organic_dict['train_vecs']
	dataset['type'] = tf.float32
	dataset['tasks'] = [{'name' : 'sent', 'features' : organic_dict['encoded_train_labels'], 'type': tf.float32}]
	datasets.append(dataset)

	paths = config_graph()
	params={}
	params['train_iter'] = 3000
	M = TfMultiPathClassifier(datasets, paths, params)

	M.save()
	M.train()

	x = M.get_prediciton('sent', organic_dict['test_vecs'])
	y = M.get_prediciton('sent', organic_dict['train_vecs'])


	multi_label_metrics(y, organic_dict['train_labels'], organic_dict['encoded_train_labels'], organic_dict['labeling'], organic_dict['train_data'])
	avg_acc += multi_label_metrics(x, organic_dict['test_labels'], organic_dict['encoded_test_labels'], organic_dict['labeling'], organic_dict['test_data'], mute=True)

avg_acc = avg_acc / folds

print('average accuracy', avg_acc)
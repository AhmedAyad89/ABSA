from utils import *
from Dataset.input_pipe import *
from Learning.tf_multipath_classifier import *


def config_graph():
	paths = []

	path = {}
	path['input_dim'] = 4096
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.1)
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


semeval_dict = prep_sentiment(domain='laptop')


datasets=[]
dataset={}
dataset['name'] = 'semeval_sentiment'
dataset['holdout'] = 100
dataset['batch_size'] = 150
dataset['features'] = semeval_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'sent', 'features' : semeval_dict['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

paths = config_graph()
params={}
params['train_iter'] = 2000
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('sent', semeval_dict['test_vecs'])
y = M.get_prediciton('sent', semeval_dict['train_vecs'])

for i in range(50):
	print(y[i], semeval_dict['encoded_train_labels'][i])

multi_label_metrics(y, semeval_dict['train_labels'], semeval_dict['encoded_train_labels'], semeval_dict['labeling'], semeval_dict['train_data'])
multi_label_metrics(x, semeval_dict['test_labels'], semeval_dict['encoded_test_labels'], semeval_dict['labeling'], semeval_dict['test_data'], mute=False)



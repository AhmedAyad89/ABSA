from utils import *
from Dataset.input_pipe import *
from Learning.tf_multipath_classifier import *


def config_graph():
	paths = []

	path = {}
	path['input_dim'] = 4096
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.16)
	path['input'] = 'semeval_laptop'
	paths.append(path)

	path = {}
	path['name'] = 'aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [88], batch_norm=False, activation=None)
	path['optimizer'] = tf.train.AdamOptimizer(name='optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)

	return paths


lap_dict = prep_semeval_aspects(domain='laptop')

datasets = []
dataset = {}
dataset['name'] = 'semeval_laptop'
# dataset['holdout'] = 200
dataset['batch_size'] = 650
dataset['features'] = lap_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'aspects', 'features' : lap_dict['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

paths = config_graph()
params = {}
params['train_iter'] = 2500

model = TfMultiPathClassifier(datasets, paths, params)

model.train()
model.save()

y = model.get_prediciton('aspects', lap_dict['test_vecs'])

multi_label_metrics(y, lap_dict['test_labels'], lap_dict['encoded_test_labels'],
										lap_dict['labeling'], lap_dict['test_data'], mute=False )
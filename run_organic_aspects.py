from utils import *
from Dataset.input_pipe import *
from Learning.tf_multipath_classifier import *


def config_graph():
	paths = []

	path = {}
	path['input_dim'] = 4127
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.4, noise=False, noise_std=0.16)
	path['input'] = 'semeval_laptop'
	paths.append(path)

	path = {}
	path['name'] = 'aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [88], batch_norm=False, activation=None)
	path['optimizer'] = tf.train.AdamOptimizer(name='optimizer', learning_rate=0.0001 , beta1=0.92 , beta2=0.9999)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)

	return paths

#These features are obtained by training a classifier to predict attribute/entity separately,
#its predictions are then usedd as input features to the final aspect classifier
extra_train = ['Features/pred_features/lap_attrs_train_predictions_67',
							 'Features/pred_features/lap_entities_train_predictions_67']
extra_test = ['Features/pred_features/lap_attrs_test_predictions_67',
							'Features/pred_features/lap_entities_test_predictions_67']
lap_dict = prep_semeval_aspects(domain='laptop', extra_train_features=extra_train, extra_test_features=extra_test)

datasets = []
dataset = {}
dataset['name'] = 'semeval_laptop'
# dataset['holdout'] = 200
dataset['batch_size'] = 700
dataset['features'] = lap_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'aspects', 'features' : lap_dict['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

paths = config_graph()
params = {}
params['train_iter'] = 2750

model = TfMultiPathClassifier(datasets, paths, params)

model.train()
model.save()

y = model.get_prediciton('aspects', lap_dict['test_vecs'])
x = model.get_prediciton('aspects', lap_dict['train_vecs'])

multi_label_metrics(x, lap_dict['train_labels'], lap_dict['encoded_train_labels'],
										lap_dict['labeling'], lap_dict['train_data'] )

multi_label_metrics(y, lap_dict['test_labels'], lap_dict['encoded_test_labels'],
										lap_dict['labeling'], lap_dict['test_data'], mute=False )
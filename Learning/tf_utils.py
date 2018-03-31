import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import tensorflow as tf
import itertools
from sklearn import metrics


def loss_exec(loss_fn):
		def custom(pred, labels):
			i = tf.reduce_mean(loss_fn(logits=pred, labels=labels))
			return i
		return custom


def loss_map(type, weight=1):
	if type == 'softmax':
		def fn(pred, labels):
			i = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels) )
			return i * weight
	if type == 'sigmoid':
		def fn(pred, labels):
			i = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=labels) )
			return i * weight
	# if type == 'sparsemax':
	# 	def fn(pred, labels):
	# 		return tf.reduce_mean( tf.contrib.sparsemax )
	return fn


def construct_path(name, layers, batch_norm=False, dropout=False, dropout_rate=0.5, noise=False, noise_std=0.1, ker_reg=None, activation=tf.nn.relu):
	def compute_path(input, phase):
		with tf.variable_scope(name + '/layers', reuse=tf.AUTO_REUSE ):
			tmp = input
			for num, layer in enumerate(layers):
				if noise:
					add = tf.random_normal(shape=tf.shape(tmp), mean=0.0, stddev=noise_std, dtype=tf.float32)
					tmp = tf.cond(phase, lambda: tf.add(tmp, add), lambda: tmp)
				if dropout:
					tmp = tf.layers.dropout(tmp, rate=dropout_rate, training=phase)
					# tmp = tf.cond(phase, lambda: tmp, lambda: tf.scalar_mul(1-dropout_rate, tmp))
				tmp = tf.layers.dense(tmp, layer, activation=None, name=str(num),
															kernel_regularizer= ker_reg, reuse=tf.AUTO_REUSE, kernel_initializer=tf.contrib.layers.xavier_initializer() )
				if batch_norm:
					tmp = tf.layers.batch_normalization(tmp, training=True)
				if activation is not None:
					tmp = activation(tmp)

		return tmp
	return compute_path


def softmax_threshold_layer(name, size):
	def fn(logits, phase):
		with tf.variable_scope(name+ '/softmax_thresholding_layer', reuse=tf.AUTO_REUSE):
			b = tf.get_variable(name='bias', shape=(size))
			tmp = tf.nn.softmax(logits)
			tmp= tf.add(tmp,b)
		return tmp
	return fn


def softmax_predictor():
	def predict(logits):
		x = tf.nn.softmax(logits)
		return tf.one_hot(tf.argmax(x, 1), logits.get_shape()[1])
	return predict


def thresholded_softmax_predictor():
	def predict(logits):
		x = tf.nn.softmax(logits)
		thresh = tf.constant(0.12, shape= logits.get_shape()[1:], dtype=tf.float32)
		return tf.greater(x, thresh )
	return predict


def sigmoid_predictor():
	def predict(logits):
		x = tf.nn.sigmoid(logits)
		return tf.round(x)
	return predict


def sparsemax_layer():
	def fn(logits, phase):
		return tf.contrib.sparsemax(logits)
	return fn


#this function takes network predictions and labels with one-hot
def f1_from_integrated_predictions(pred, labels):
	# try:
	# 	print('micro', metrics.f1_score(labels, pred, average='micro'))
	# 	print('macro', metrics.f1_score(labels, pred, average='macro'))
	# 	print('weight', metrics.f1_score(labels, pred, average='weighted'))
	# except:
	# 	print('x')
	try:
		print('samples', metrics.f1_score(labels, pred, average='micro'))
	except Exception as a:
		print(a)
		print(labels)
		print(pred)

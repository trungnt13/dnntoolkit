def vggnet(params_path, api='lasagne'):
	''' Pretrained VGGnet from:
		- https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb
		- Input: images of size (3,224,224)
		- Output: 1000 categories

	Parameters
	----------
	params_path : str
		path to pretrained model file (https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl)

	Returns
	-------
	model : object
		- lasagne: output layer of the network
		- blocks: b
		- keras: a
	'''
	#####################################
	# 1. Create model.
	import lasagne
	from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
	from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
	from lasagne.layers import MaxPool2DLayer as PoolLayer
	from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer

	net = {}
	net['input'] = InputLayer((None, 3, 224, 224))
	net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
	net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
	net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)

	net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
	net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)

	net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)

	net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)

	net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
	net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)

	net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
	net['drop6'] = DropoutLayer(net['fc6'], p=0.5)

	net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
	net['drop7'] = DropoutLayer(net['fc7'], p=0.5)

	# output
	net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
	output_layer = net['fc8']

	#####################################
	# 2. load params.

	import pickle

	model = pickle.load(open(params_path))
	CLASSES = model['synset words']
	MEAN_IMAGE = model['mean image']

	lasagne.layers.set_all_param_values(output_layer, model['values'])

	return output_layer

def alexnet(params_path, api='lasagne'):
	pass
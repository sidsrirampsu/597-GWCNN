from tensorflow import keras
from matplotlib import pyplot
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from numpy import expand_dims
import tensorflow as tf
import numpy as np
import cv2

# load the model
model = keras.models.load_model('C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/new_model.h5')
print(model.summary())

# summarize filter shapes
conv_layer_num = []
for i in range(len(model.layers)):
	layer = model.layers[i]
	#weights, bias= layer.get_weights()
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	conv_layer_num.append(i)
	print(i, layer.name, layer.output.shape)

	#normalize filter values between  0 and 1 for visualization
	#f_min, f_max = weights.min(), weights.max()
	#filters = (weights - f_min) / (f_max - f_min)  
		
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(inputs=model.inputs, outputs=successive_outputs)
#visualization_model = tf.keras.models.Model(input_size = (128,128,1), inputs = model.input, outputs = successive_outputs)

img_path = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/dataset/00028_HG_0_0_0.07_n10_p17.png'
#Load the input image
img = load_img(img_path, target_size=(128, 128))
print(img.shape)
x   = img_to_array(img)
x = x[..., np.newaxis]
print(x.shape)
successive_feature_maps = visualization_model.predict(x)

# Retrieve are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)

square = 12
ix = 1
for fmap in feature_map:
# plot all 64 maps in an 8x8 squares

	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			print(fmap.shape)
			pyplot.imshow(fmap[:, :], cmap='gray')
			ix += 1
# show the figure
pyplot.show()

'''
conv_layer_num = []
for i in range(len(model.layers)):
layer = model.layers[i]
# check for convolutional layer
if 'conv' not in layer.name:
	continue
# get filter weights
conv_layer_num.append(i)
print(i, layer.name, layer.output.shape)



outputs = [model.layers[i].output for i in conv_layer_num]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
layer1 = Model(inputs=model.input, outputs= model.layers[0].output)
#model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img_name = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/dataset/00028_HG_0_0_0.07_n10_p17.png'
#img = load_img(img_name, target_size=(128,128))
# convert the image to an array
#img = img_to_array(img)
img_arr = cv2.imread(img_name, 0)#[..., ::-1]
img_arr = img_arr[..., np.newaxis]
# print(img_arr.shape)

# expand dimensions so that it represents a single 'sample'
img = expand_dims(img_arr, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img_arr)
#feature_maps2 = layer2.predict(feature_maps)
feature_maps.reshape(1, 128, 128, 128)
print(feature_maps)
temp = feature_maps[:, :, 0, :]
print(temp.shape)
# plot the output from each block
square = 12
ix = 1
for fmap in temp:
# plot all 64 maps in an 8x8 squares

for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		print(fmap.shape)
		pyplot.imshow(fmap[:, :], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

'''
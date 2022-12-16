# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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


# %%
model = keras.models.load_model('C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/new_model.h5')
print(model.summary())


# %%
# summarize filter shapes
conv_layer_num = []
for i in range(len(model.layers)):
	layer = model.layers[i]

	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	conv_layer_num.append(i)
	print(i, layer.name, layer.output.shape)
print(conv_layer_num)
	#normalize filter values between  0 and 1 for visualization
	#f_min, f_max = weights.min(), weights.max()
	#filters = (weights - f_min) / (f_max - f_min)  


# %%
successive_outputs = []
for i in conv_layer_num:
    successive_outputs.append(model.layers[i].output)
visualization_model = Model(inputs=model.inputs, outputs=successive_outputs)
print(model.inputs)
#visualization_model = tf.keras.models.Model(input_size = (128,128,1), inputs = model.input, outputs = successive_outputs)


# %%
img_path = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/dataset/00052_HG_3_5_0.11_n01_p07.png'
img_temp = (img_path.split('/')[-1]).split(".")
img_name = img_temp[0]+img_temp[1]

# %%
img = load_img(img_path, target_size=(128, 128))
print(img.size)
img_data = img.getdata()
x   = img_to_array(img )


# %%
temp = x.dot([1/3, 1/3, 1/3])
temp = temp[np.newaxis, ...]


# %%
print(temp.shape)


# %%
successive_feature_maps = visualization_model.predict(temp)


# %%
layer_names = [layer.name for layer in model.layers]


# %%
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)


# %%
square = 12


# %%
counter = 0
for ind, fmap in enumerate(successive_feature_maps):
	ix = 0
	max_ix=fmap.shape[-1]
	square = int(np.ceil(np.sqrt(max_ix)))
	fig, ax = pyplot.subplots(square, square, sharex=True, sharey=True, facecolor="white", edgecolor="white")
	for row in range(square):
		for col in range(square):
			if ix == max_ix:
				break
			# specify subplot and turn of axis
			ax[row, col].set_xticks([])
			ax[row, col].set_yticks([])
			# ax[row*col].set_facecolor('black')
			# plot filter channel in grayscale
			ax[row ,col].imshow(fmap[0, :, :, ix], cmap='gray')
			ix += 1
	file_name = 'C:/Users/siddh/Desktop/GW CNN/output_plots/feature maps/' + f'{img_name}_conv_layer{ind+1}.png'
	pyplot.savefig(file_name, dpi=500)
	print(file_name, "saved")
	del ax, fig




import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os
import numpy as np
import splitfolders
splitfolders.ratio('/home/siddhika/dataset/', output="Output", seed=1337, ratio=(.8, 0.1,0.1))
import shutil

# Creating Train / Val / Test folders 
root_dir = '/home/siddhika/dataset/'

os.makedirs("Output" +'/train')
os.makedirs("Output" +'/val')
os.makedirs("Output" +'/test')


val_ratio = 0.15
test_ratio = 0.15

# Creating partitions of the data after shuffeling

src = root_dir # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                           [int(len(allFileNames)* 0.7), 
                                                               int(len(allFileNames)* 0.85)])


#train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
#test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "Output/train")

for name in val_FileNames:
    shutil.copy(name, "Output/val")

for name in test_FileNames:
    shutil.copy(name, "Output/test")

#print(train_FileNames)
train_files = os.listdir("Output/train")
train_y = []
for name in train_files:
  name = name[9]+"."+name[11]
  name = float(name)
  train_y.append(name)
#print(train_y)

test_files = os.listdir("Output/test")
test_y = []
for name in test_files:
  name = name[9]+"."+name[11]
  name = float(name)
  test_y.append(name)
#print(test_y)

val_files = os.listdir("Output/val")
val_y = []
for name in val_files:
  name = name[9]+"."+name[11]
  name = float(name)
  val_y.append(name)
#print(val_y)

#Data Preparation

img_size = 128
def get_data(data_dir, labels):
    data = [] 
    for index, img in enumerate(os.listdir(data_dir)):
        try:
          img_arr = cv2.imread(os.path.join(data_dir, img), 0)#[..., ::-1]
          img_arr = img_arr[..., np.newaxis]
        
          #resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
          data.append([img_arr, labels[index]])
        except Exception as e:
          print(e)
    return np.array(data)

#img_arr = cv2.imread('Output/train/01338_HG_4_3_0.50_n01_n30.png', 0)
#print(img_arr)

train = get_data('Output/train', train_y)
test = get_data('Output/test', test_y)
val = get_data('Output/val', val_y)

#plt.figure(figsize = (5,5))
#plt.imshow(train[11][0])
#plt.title(train[11][0])

#train[1][0].shape

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(str(int(label*10)))
y_train_cat=tf.keras.utils.to_categorical(y_train)


for feature, label in val:
  x_val.append(feature)
  y_val.append(str(int(label*10)))
y_val_cat=tf.keras.utils.to_categorical(y_val)

for feature, label in test:
  x_test.append(feature)
  y_test.append(str(int(label*10)))
y_test_cat=tf.keras.utils.to_categorical(y_test)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train_cat = np.array(y_train_cat)

x_val.reshape(-1, img_size, img_size, 1)
y_val_cat = np.array(y_val_cat)

#print(y_val_cat.shape)
#print(y_train_cat.shape)

#x_train.shape

model = Sequential()
model.add(Conv2D(32,(3, 3),padding="same", activation="relu", input_shape=(128, 128, 1)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(56, activation="softmax"))

model.summary()

opt = Adam(learning_rate=0.0001)
model.compile(optimizer = opt , loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train_cat,epochs = 50 , validation_data = (x_val, y_val_cat))



model.evaluate(x_test,y_test_cat)

predict = model.predict(x_test)
classes = np.argmax(predict, axis = 1)
print(classes)

y_test

root_dir = '/content/gdrive/My Drive/GW_Input_Data_2'

val_files = os.listdir(root_dir)
val_new_y = []

for img_file in val_files:
  img_file = img_file[3]+"."+img_file[5]
  img_file = float(img_file)
  val_new_y.append(img_file)

def get_val_data(data_dir, labels):
  data = []
  for index, img in enumerate (os.listdir(data_dir)):
    try:
      img_arr = cv2.imread(os.path.join(data_dir, img))[...,::-1] #convert BGR to RGB format
      #resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
      data.append([img_arr, labels[index]])
    except Exception as e:
      print(e)
  return np.array(data)

val_data = get_val_data(root_dir, val_new_y)

val_data

new_x_val = []
new_y_val = []

for feature, label in val_data:
  new_x_val.append(feature)
  new_y_val.append(str(int(label*10)))
y_test_cat=tf.keras.utils.to_categorical(new_y_val)

#normalising

new_x_val = np.array(new_x_val) / 255
new_y_val = np.array(new_y_val)

#model.evaluate(new_x_val,y_test_cat)

#y_test_cat

#new_y_val

#model.predict_classes(new_x_val)

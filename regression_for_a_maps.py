from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow 
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import shutil
import numpy as np
from keras.models import Model
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers.core import Dropout
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv2D , MaxPool2D 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# load dataset
root_dir = '/home/siddhika/dataset/'
#root_dir = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/dataset/normal'
allFileNames = os.listdir(root_dir)

#os.makedirs("Output" +'/train')
#os.makedirs("Output" +'/val')
#os.makedirs("Output" +'/test')

val_ratio = 0.15
test_ratio = 0.15

# Creating partitions of the data after shuffeling

src = root_dir # Folder to copy images from


np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)* 0.7), int(len(allFileNames)* 0.85)])

'''
# to filter 0, 1 and 2 modes to check biasing 
filteredFiles = []
for name in (allFileNames):
  if int(name[9]) < 3 and int(name[11]) < 3:
    filteredFiles.append(name)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(filteredFiles), [int(len(filteredFiles)* 0.7), int(len(filteredFiles)* 0.85)])
'''


print(allFileNames)

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


for name in (test_FileNames):
  shutil.copy('/home/siddhika/dataset/'+name, '/home/siddhika/gw-modal-decomposition/Output/test2')


#name = train_FileNames[0]
#print(name)
#temm = name[9]
#temn = name[11]
#xoff = name[18:21]
#yoff = name[22:25]

def labelling (fileName):
  
  yParam = []
  for image in fileName:
    #print(image)
    labels = []
    #print(int(image[9]))
    labels.append(int(image[9]))
    labels.append(int(image[11]))
    if image[18] == 'p':
      labels.append(64 + int(image[19:21]))

    else:
      labels.append(64 + int("-"+image[19:21]))
    
    if image[22] == 'p':
      labels.append(64 + int("+"+image[23:25]))

    else:
      labels.append(64 + int("-"+image[23:25]))
    yParam.append(labels)
  yParam = np.array([np.array(x) for x in yParam])
  return yParam

#Y Labels
yTrain = labelling(train_FileNames)
yTest = labelling(test_FileNames)
yVal = labelling(val_FileNames)

#train_FileNames = [src + name for name in train_FileNames.tolist()]
#val_FileNames = [src + name for name in val_FileNames.tolist()]
#test_FileNames = [src+ name for name in test_FileNames.tolist()]
#len(yTest)

#type(yVal[0])

img_size = 128
def getdata(data_dir):
  data=[]
  for img in data_dir:
    img_arr = cv2.imread(os.path.join(src, img), 0)#[..., ::-1]
    print(img_arr)
    img_arr = img_arr[..., np.newaxis]
    data.append(img_arr)
  return data

training = getdata(train_FileNames)
testing = getdata(test_FileNames)
validation = getdata(val_FileNames)

#print(len(validation))

#X Data

x_train = []
x_val = []
x_test = []

# Normalize the data
x_train = np.array(training) / 255
x_val = np.array(validation) / 255
x_test = np.array(testing) / 255

x_train.reshape(-1, img_size, img_size, 1)
x_test.reshape(-1, img_size, img_size, 1)
x_val.reshape(-1, img_size, img_size, 1)

#len(x_train)

#x_test.shape 

def model_def(dropout_rate = 0.1):

  model = Sequential()
  
  model.add(Conv2D(128,(3, 3),padding="same", activation="relu", input_shape=(128, 128, 1)))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(dropout_rate))
  
  model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2))) 

  model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(dropout_rate))

  model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2)))

  model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(dropout_rate))

  model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis= 1))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(dropout_rate)) 

  model.add(Flatten())
  model.add(Dense(8, activation='relu'))
  model.add(BatchNormalization(axis= 1))
  model.add(Dense(4, activation='relu'))

  model.summary()

  opt = adam(learning_rate=0.001)
  model.compile(optimizer = opt , loss ='mean_squared_error' , metrics = ['accuracy'])
  return model

#GRIDSEARCH

model = KerasClassifier(build_fn = model_def)

#dropout_rate = [0.2, 0.4, 0.6]
#batch_size = [10, 70, 150]
#epochs =[20, 50, 70]

#param_grid = dict(dropout_rate = dropout_rate, batch_size = batch_size, epochs = epochs)

#grid = GridSearchCV(estimator = model, param_grid = param_grid)
#grid_result = grid.fit(x_train, yTrain )
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#model.set_params(**grid_result.best_params_)

history = model.fit(x_train, yTrain, epochs = 50, validation_data = (x_val, yVal))

#model.evaluate(x_train, yTrain)

#seed = 7
#np.random.seed(seed)
#kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
#results = cross_val_score(model, x_train, yTrain, cv=kfold)
#print(results.mean())

#saving the model
model.model.save('new_model.h5')
print("Saved model to disk")
print(history.history.keys())

# summarize history for accuracy
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Val Accuracy'], loc='upper left')

# summarize history for loss
plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'Val Loss'], loc='upper left')

plt.savefig("result.png")

#predict = model.predict(x_test)
#print(predict)
print(yTest)


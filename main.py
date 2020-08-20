# Author ~ Eeshan Narula
# Github : https://github.com/eeshannarula/trash_classification/blob/master

# importing libraries
import numpy as np 
# keras machine learning library
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# cv2 for image rendering
import cv2,os

# function for loading images
def loadImages(folder):
    images = []
    for file in os.listdir(folder):  # listdir lists the name of all the directory in a folder
        img = cv2.imread(os.path.join(folder,file)) # imread return the pixel array
        if img is not None:
            images.append(cv2.resize(img,(100,100))) # resize all the images to same size
    return images


# all the paths to the data
paths = [
"/Users/eeshannarula/Downloads/datas/trash-dataset/cardboard",
"/Users/eeshannarula/Downloads/datas/trash-dataset/glass",
"/Users/eeshannarula/Downloads/datas/trash-dataset/metal",
"/Users/eeshannarula/Downloads/datas/trash-dataset/paper",
"/Users/eeshannarula/Downloads/datas/trash-dataset/plastic"
]

#  class for all the catigories
class trash_cat:
    def __init__(self,data,label):
        self.images = data
        self.label = label
        self.targets = self.makeTragets()

    def makeTragets(self):
        target = [0] * 6
        target[self.label] = 1
        return [target] * len(self.images)

types = []

for cat in paths:
    types.append(trash_cat(loadImages(cat),paths.index(cat)))

# concat all types of trash into traning set and its targets
def concat(array):
    xs = []
    ys = []
    for cat in array:
        for img in cat.images:
            xs.append(img)
        for target in cat.targets:
            ys.append(target)    
    return np.divide(np.array(xs),255),np.array(ys)            

xs,ys = concat(types)

# building the model
model = Sequential()

model.add(Conv2D(kernel_size = [3,3],filters = 8,activation = 'relu',input_shape = [100,100,3]))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(kernel_size = [3,3],filters = 16,activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(kernel_size = [3,3],filters = 32,activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(6,activation = 'sigmoid'))
# we would be using SGD optimizer
sgd = SGD(lr = 0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(sgd,loss = 'categorical_crossentropy')
# traning the model
model.fit(xs,ys,epochs=25,batch_size=100,shuffle=True)
#save the model
model.save('trashnet.h5')

# -*- coding: utf-8 -*-
"""
CONTENTS ::
1 ) Importing Various Modules

2 ) Preparing the Data

3 ) Modelling

4 ) Evaluating the Model Performance

5 ) Visualizing Predictons on the Validation Set
"""
#==============================================================================
# User imput parameters
rootPath = '/home/jahaniam/Desktop/PlantGrowthClassification/'

dataPath = 'data_sample_test/'
weights_path = 'transfer_learning_weights/'
weights_save_path = 'transfer_learning_save_weights/'
saveWeightsFileName = 'weights.hdf5'
bestWeightsFileName = "weights.best.hdf5"
X=[]
Z=[]
IMG_SIZE=256
batch_size=8
num_epochs=50
#==============================================================================
#1 ) Importing Various Modules.
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#dl libraraies
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau,  ModelCheckpoint, LearningRateScheduler, EarlyStopping

# specifically for cnn
from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.regularizers import l2
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os  
import shutil                 
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from subprocess import check_output
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#TL specific modules
from tensorflow.keras.applications.vgg16 import VGG16
#==============================================================================
#2 ) Preparing the Data
#2.1) Making the functions to get the training and validation set from the Images
def assign_label(img,flower_type):
    return flower_type

def move_file(fileName, source_directory):
    destination_directory = rootPath + dataPath + '/rejects'
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    shutil.move(source_directory+'/'+fileName, destination_directory)

def make_train_data_r0(flower_type,DIR):
    files = os.listdir(DIR)
    i = 0
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)                
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        try:
            X.append(cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC))
            #img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            #X.append(np.array(img))
            Z.append(str(label))            
        except:
            print( "Resize error!")
            f = files[i]            
            move_file(f, DIR)            
            print(f + ' moved to rejects folder.')               
        i = i + 1        

def make_train_data(flower_type,DIR):
    print(flower_type)
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)                
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        Z.append(str(label))

def read_data(filePath):
    nb_categories = 0
    print("Reading images from the following directories:")
    filePath = filePath 
    data_folders =  os.listdir(filePath)
    print(data_folders)
    for folder in data_folders:
        if folder != 'rejects':
            make_train_data(folder, filePath + folder)
            print(len(X))
            nb_categories = nb_categories + 1
    
    return  nb_categories      
            
# Read the data - create feature matrix and target vector
nb_categories=read_data(rootPath + dataPath)
#==============================================================================
#2.2 ) Visualizing some Random Images
# fig,ax=plt.subplots(5,2)
# fig.set_size_inches(15,15)
# for i in range(5):
#     for j in range (2):
#         l=rn.randint(0,len(Z))
#         ax[i,j].imshow(X[l])
#         ax[i,j].set_title('Flower: '+Z[l])
        
# plt.tight_layout()

#2.3 ) Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,nb_categories)
X=np.array(X)
X=X/255

#2.4 ) Splitting into Training and Validation Sets
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

#2.5 ) Setting the Random Seeds
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
#==============================================================================
#3 ) Modelling
#3.1 ) Specifying the Base Modell
# Transfer learning refers to using a pretrained model on some other task for your own task.
# Hence we need to specify the particular model which we are deploying in our task and thus
# needs to specify the base model. 
# In our case we are using the VGG16 model from the tensorflow.keras.Applications library as the base
# model.
''''Arguments
include_top: whether to include the 3 fully-connected layers at the top of the network.
weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
input_shape: optional shape tuple, only to be specified if include_top is False 
(otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format)
or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels,
and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
pooling: Optional pooling mode for feature extraction when include_top is False.
None means that the output of the model will be the 4D tensor output of the last convolutional layer.
'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
'max' means that global max pooling will be applied.
classes: optional number of classes to classify images into, only to be specified
if include_top is True, and if no weights argument is specified.

BREAKING IT DOWN--
1) Firstly we import the VGG16 module from the Keras library.

2) Next we need to specify if we want to use the fully connected layers of the VGG16 module
or own layers. Since our task is different and we have only 5 target classes we need
to have our own layers and I have specified the 'include_top' arguement as 'False'.

3) Next we need to specify the weights to be used by the model.
Since I want it to use the weights it was trained on in ImageNet competition, 
I have loaded the weights from the corressponding file. You can directly specify
the weights arguement as 'imagenet' in VGG16( ) but it didn't work in my case so
I have to explicitily load the weghts from a file.

4) Lastly we just need to specify the shape of the imput that our model need to
expect and also specify the 'pooling' type.'''
base_model=VGG16(include_top=False, weights=None,input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='max')
fileName = os.listdir(rootPath + weights_path)
base_model.load_weights(rootPath + weights_path + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.summary()
# Note that this is NOT the summary of our model and this is the summary or the
# ARCHITECTURE of the VGG16 model that we are deploying as the base model.

#3.2 ) Adding our Own Fully Connected Layers
#Now we need to add at the top of the base model some fully connected layers.
# Also we can use the BatchNormalization and the Dropout layers as usual in case 
# we want to.

#For this I have used a Keras sequential model and build our entire model on top of it;
# comprising of the VGG model as the base model + our own fully connected layers.

model=Sequential()
model.add(base_model)

model.add(Dense(256,activation='relu'))
model.add(Dense(nb_categories,activation='softmax'))

# 3.3 ) Data Augmentation to prevent Overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(x_train)

# 3.4 ) Using a Learning Rate Annealer & the Summary
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
model.summary()

# This is now the complete summary of our model that we shall use to classify the images.

# 3.5 ) Compiling & Training the Model

# 3.5.1 ) USING BASE MODEL AS A FEATURE EXTRACTOR
'''While using transfer learning in VGG16; we have basically have 3 main approaches-->

1) To use the pretrained model as a feature extractor and just train your classifier on top of it. In this method we do not tune any weights of the model.

2) Fine Tuning- In this approach we tune the weights of the pretrained model. This can be done by unfreezing the layers that we want to train.In that case these layers will be initialised with their trained weights on imagenet.

3) Lasty we can use a pretrained model.

Note that in this section I have used the first approach ie I have just use the conv layers and added my own fully connected layers on top of VGG model. Thus I have trained a classifier on top of the CNN codes.''' 
base_model.trainable=False # setting the VGG model to be untrainable.
weight_saver = ModelCheckpoint('set_a_weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
# saves the model weights after each epoch if the validation loss decreased
weights_save_filepath = rootPath + weights_path + saveWeightsFileName
#checkpoint = ModelCheckpoint(filepath = weights_save_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint= ModelCheckpoint(filepath= weights_save_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=0, mode='auto', baseline=None)
##early_stopping = EarlyStopping(monitor='val_loss', patience=2)
callbacks_list = [checkpoint,early_stopping]
##3.5.2 ) FINE TUNING BY UNFREEZING THE LAST BLOCK OF VGG16
##In this section I have done fine tuning. To see the effect of the fine tuning I have first unfreezed the last block of the VGG16 model and have set it to trainable.
#for i in range (len(base_model.layers)):
#    print (i,base_model.layers[i])
  
# for layer in base_model.layers[15:]:
#     layer.trainable=True
# for layer in base_model.layers[0:15]:
#     layer.trainable=False
    

# #3.5.3) UNFREEZING THE LAST 2 BLOCKS
# # Similarly unffreezing the last 2 blocks of the VGG16model. 
# for i in range (len(base_model.layers)):
#     print (i,base_model.layers[i])
    
# for layer in base_model.layers[11:]:
#     layer.trainable=True
# for layer in base_model.layers[0:11]:
#     layer.trainable=False    

# #3.5.4) UNFREEZING THE LAST 3 BLOCKS
# # Similarly unffreezing the last 2 blocks of the VGG16model. 
# for i in range (len(base_model.layers)):
#     print (i,base_model.layers[i])
    
# # for layer in base_model.layers[7:]:
for layer in base_model.layers[:]:
    layer.trainable=True
# for layer in base_model.layers[0:7]:
#     layer.trainable=False      

## load weights
#weights_best_filepath = rootPath + weights_save_path + bestWeightsFileName    
#model.load_weights(weights_best_filepath)  
model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = num_epochs, validation_data = (x_test,y_test),
                              callbacks=callbacks_list,
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

#===================================
## plot model loss of with training and test sets vs Epochs
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.savefig('Loss.png')

#plt.show()

## plot model accuracty with training and test sets vs Epochs
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.savefig('acc.png')
plt.show()
##==============================================================================


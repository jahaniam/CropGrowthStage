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

weight_file='trained_weights/weights_best.hdf5'

X=[]
Z=[]
IMG_SIZE=256
batch_size=128
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

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,nb_categories)
X=np.array(X)
X=X/255


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
#==============================================================================
base_model=VGG16(include_top=False, weights=None,input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='max')

base_model.summary()

model=Sequential()
model.add(base_model)

model.add(Dense(256,activation='relu'))
model.add(Dense(nb_categories,activation='softmax'))
model.load_weights(weight_file)
datagen.fit(X)

# 3.4 ) Using a Learning Rate Annealer & the Summary
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
model.summary()

predicted_classes = model.predict_classes(X)
print("predicted_classes:",predicted_classes)
gt=np.argmax(Y,axis=1)

accuracy2 = np.sum(gt==predicted_classes)/np.size(gt)
print("test accuracy:",accuracy2)


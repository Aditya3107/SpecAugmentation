############################ IMPORT LIBRARIES #############################################
import librosa
import argparse
import keras
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import glob
from sklearn.metrics import confusion_matrix
from SpecAugment import spec_augment_pytorch

import os
import sys
import warnings
import IPython.display as ipd
import seaborn as sns
import glob
import pickle
import json
import torch
from tqdm import tqdm
input_duration=3

#FOR CREATING MODEL IMPORT LIBS
# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

#sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

############################## Exploring the dataset ###########################################

TESS = 'TESS emotion datasets/'
dir_list = os.listdir(TESS)
dir_list.sort()
#print(dir_list)

path = []
emotion = []
gender = []


for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname: 
        part = f.split(".")[0].split('_')
        tmp = part[0]
        if tmp == 'OAF':
            tmp = 'male'
        else : 
            tmp = 'female'
        gender.append(tmp)
        emotion.append(part[2])
        path.append(TESS + i + '/' + f)

TESS_df = pd.DataFrame(emotion)
TESS_df = pd.concat([pd.DataFrame(gender), TESS_df], axis = 1)
TESS_df.columns = ['gender','emotion']
TESS_df['labels'] = TESS_df.gender + "_" + TESS_df.emotion
TESS_df['source'] = TESS
TESS_df = pd.concat([TESS_df, pd.DataFrame(path,columns = ['path'])], axis = 1)
TESS_df = TESS_df.drop(['gender', 'emotion','source'], axis = 1)
TESS_df["labels"].replace({"male_ps": "male_surprise", "female_ps": "female_surprise"}, inplace=True)  
from sklearn.utils import shuffle
TESS_df = TESS_df.sample(frac = 1,random_state= 42) 
TESS_df_new = TESS_df

################################# Introduce a callback #####################################
ACCURACY_THRESHOLD = 0.99

# Implement callback function to stop training
# when accuracy reaches e.g. ACCURACY_THRESHOLD = 0.95
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):   
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

################################# Train Test Split ############################################

X_train, X_test, y_train, y_test = train_test_split(TESS_df_new, TESS_df_new.labels, test_size = 0.25, shuffle = True, random_state = 42)

################################# X train Feature Extraction ###################################

#X_train feature extraction
X_train_features = pd.DataFrame(columns = ['features'])
counter = 0
for index,path in enumerate(X_train.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration = 2.5, sr = 22050*2, offset = 0.5)
    sample_rate = np.array(sample_rate)
    #taking mean of MFCC 
    mfccs = np.mean(librosa.feature.melspectrogram(y = X, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000), axis = 0)
    X_train_features.loc[counter] = [mfccs]
    counter = counter+1

X_train_features = (pd.DataFrame(X_train_features['features']. values.tolist())).fillna(0)

################################## X test feature extraction #######################################

X_test_features = pd.DataFrame(columns = ['features'])
counter = 0
for index,path in enumerate(X_test.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration = 2.5, sr = 22050*2, offset = 0.5)
    sample_rate = np.array(sample_rate)
    #taking mean of MFCC 
    mfccs = np.mean(librosa.feature.melspectrogram(y = X, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000), axis = 0)
    X_test_features.loc[counter] = [mfccs]
    counter = counter+1
X_test_features = X_test_features.fillna(0)
X_test_features = (pd.DataFrame(X_test_features['features']. values.tolist())).fillna(0)

################################### Normalization #########################################

mean = np.mean(X_train_features, axis=0)
std = np.std(X_train_features, axis=0)

X_train_features = ((X_train_features - mean)/std).fillna(0)
X_test_features = ((X_test_features - mean)/std).fillna(0)

X_train = np.array(X_train_features)
y_train = np.array(y_train)
X_test = np.array(X_test_features)
y_test = np.array(y_test)

################################ Model Generation ##########################################

callbacks = myCallback()
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

#pickle lb object for future use
filename = 'labels'
outfile = open(filename, 'wb')
pickle.dump(lb,outfile)
outfile.close()

X_train = np.expand_dims(X_train, axis = 2)
X_test = np.expand_dims(X_test, axis = 2)


model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(14)) # Target class number
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
#opt = keras.optimizers.Adam(lr=0.000001)
#opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
print(model.summary())

#Compile Model
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model_history=model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), callbacks=[callbacks])

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel('epochs')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel('epochs')
plt.legend(['train','test'],loc = 'upper left')
plt.show()


model_name = 'Emotion_recognition_without_specAugment.h5'
save_dir = os.path.join(os.getcwd(),'saved_models')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir,model_name)
model.save(model_path)
print("save model and weight at %s" %model_path)
#save model to disk
model_json = model.to_json()
with open("model_json.json", "w") as file:
    file.write(model_json)

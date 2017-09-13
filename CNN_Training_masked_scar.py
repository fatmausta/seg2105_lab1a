# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:53:54 2017

@author: fusta
"""
import numpy as np
import keras
import SimpleITK
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy
import SimpleITK as sitk
K.set_image_dim_ordering('th')
from skimage.util import view_as_windows
from skimage.util import view_as_blocks
import random
import sys
#PARAMETERS TO ADJUST
nclasses = 2
randomly_drop = 0
#desired ratio of true positives, for scar in this case
desired_ratio_balance = 0.30
datapopfraction = 0.80
scar_definition_ratio = 0.75
patch_size = 1
window_size = 5
filter_size = 2
epochs = 30
skip = 4
modelname= 'CNN_scar_1.h5'
pid_train = np.array(['0329','0364','0417'])#, '0424', '0450', '0473', '0485','0493', '0494', '0495', '0515', '0519', '0529', '0546', '0562', '0565', '0574', '0578', '0587', '0591'])
#datapath = 'DataCNNScarNorm/' #for sharcnet work directory
datapath = 'C:\\Users\\fusta\\Dropbox\\1_Machine_Learning\\DataCNNScarNorm\\'
#TRAINING
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
def PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip, scar_definition_ratio):  
    pads = []
    LGE_patches_scar = []
    LGE_windows_scar = []
    LGE_patches_bg = []
    LGE_windows_bg = []
    LGE_patches_arr = []
    LGE_windows_arr = []
    LGE_padded_slice = []
#    pid = '0329'
#    pid = '0364'
#    pid = '0417'
    for pid in pid_train:
        LGE = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-LGE-cropped.mhd')
        scar = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-scar-cropped.mhd')
        myo = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-myo-cropped.mhd')      
        #convert a SimpleITK object into an array
        LGE_3D = SimpleITK.GetArrayFromImage(LGE)
        myo_3D = SimpleITK.GetArrayFromImage(myo)
        scar_3D = SimpleITK.GetArrayFromImage(scar) 
        #masking the LGE
        LGE_3D = np.multiply(LGE_3D,myo_3D)        
        d_LGE = LGE_3D.shape[0]
        h_LGE = LGE_3D.shape[1]
        w_LGE = LGE_3D.shape[2] 
        #calculate the amount of padding for height and width of a slice for patches
        w_pad=patch_size-(w_LGE%patch_size)      
        h_pad=patch_size-(h_LGE%patch_size)    
        pads.append((h_pad,w_pad))
        all_slice = range(0, d_LGE, skip)#15,5)#30,60,2)   
#        sl=35
        for sl in all_slice:   
            #pad your images            
            LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
            scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
            #patches as blocks 
            LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
            LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size))
            #pad the images one more time before making windows
            padding = int((window_size - patch_size)/2)
            LGE_padded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
            #windows as overlaping blocks
            LGE_windows = view_as_windows(LGE_padded_slice, (window_size,window_size), step=patch_size)
            LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))
            LGE_padded_slice=None
            #remove samples from outside of myocardium. 
            rang=[]
            for r in range(0,len(LGE_windows)):
                if(np.sum(LGE_windows[r])==0):
                    rang.append(r)
            LGE_patches = np.delete(LGE_patches, rang, axis = 0) 
            LGE_windows = np.delete(LGE_windows, rang, axis = 0)
            LGE_patches_arr.extend(LGE_patches)
            LGE_windows_arr.extend(LGE_windows)
    LGE_patches_arr = np.asarray(LGE_patches_arr)
    LGE_windows_arr = np.asarray(LGE_windows_arr)
    #1) SEPERATE SCAR FROM BACKGROUND
    for r in range(0,len(LGE_patches_arr)):
        if(np.sum(LGE_patches_arr[r])>=patchsize_sq*scar_definition_ratio):#scar 
            LGE_patches_scar.append(LGE_patches_arr[r])
            LGE_windows_scar.append(LGE_windows_arr[r])
        else: #background
            LGE_patches_bg.append(LGE_patches_arr[r])
            LGE_windows_bg.append(LGE_windows_arr[r])
    LGE_patches_scar = np.asarray(LGE_patches_scar)
    LGE_windows_scar = np.asarray(LGE_windows_scar)
    LGE_patches_bg = np.asarray(LGE_patches_bg)
    LGE_windows_bg = np.asarray(LGE_windows_bg)        
    #2) CALCULATE AMOUNT OF DATA TO BE DROPPED
    ratio_imbalance = len(LGE_patches_scar)/(len(LGE_patches_arr)) 
    #formula to decide how many samples to drop 
    controlled_datapopnumber = (desired_ratio_balance-ratio_imbalance)*len(LGE_patches_arr)/desired_ratio_balance                       
    if controlled_datapopnumber>0 and controlled_datapopnumber<len(LGE_patches_bg):
        if len(LGE_patches_bg)>len(LGE_patches_scar):
            print('too little scar samples, deleting from bg samples')
            randomrange=random.sample(range(0, len(LGE_patches_bg)), int(controlled_datapopnumber))
            #delete from background
            LGE_patches_bg = np.delete(LGE_patches_bg, randomrange, axis = 0) 
            LGE_windows_bg = np.delete(LGE_windows_bg, randomrange, axis = 0)   
        else:#more scar than bg, so drop some from the scar region
            print('too little bg samples, deleting from scar samples')
            randomrange=random.sample(range(1, len(LGE_patches_scar)), int(controlled_datapopnumber))
            #delete from scar samples
            LGE_patches_scar = np.delete(LGE_patches_scar, randomrange, axis = 0) 
            LGE_windows_scar = np.delete(LGE_windows_scar, randomrange, axis = 0)                   
    #combine left-over desired scar and background patches together
    LGE_patches_arr = np.concatenate((LGE_patches_scar,LGE_patches_bg),axis=0)
    LGE_windows_arr = np.concatenate((LGE_windows_scar,LGE_windows_bg),axis=0)
    print('number of scar samples in the training data: %d' % len(LGE_patches_scar))
    print('number of bg samples in the training data: %d' % len(LGE_patches_bg))
    print('scar is %d percent of entire data' %  ( len(LGE_patches_scar) / len(LGE_patches_arr)*100))
    print('background is %d percent of entire data' %  ( len(LGE_patches_bg) / len(LGE_patches_arr)*100))    
    #LGE_patches_scar=None
    #LGE_patches_bg=None
    #LGE_windows_scar=None
    #LGE_windows_bg=None
    #calculate the label values for the patches
    LGE_patches_label = np.empty(LGE_patches_arr.shape[0])
    for p in range(0,len(LGE_patches_arr)):            
        if numpy.sum(LGE_patches_arr[p])/patchsize_sq>=scar_definition_ratio:
            label=numpy.reshape(1, (1,1))
        else:
            label=numpy.reshape(0, (1,1))
        LGE_patches_label[p] = label
        #making your window  intensities a single row
    LGE_windows_single_row = numpy.reshape(LGE_windows_arr,(LGE_windows_arr.shape[0], window_size*window_size))  
    training_data= list(zip(numpy.uint8(LGE_windows_single_row),numpy.uint8(LGE_patches_label)))
    print('\n\nsize of training data %d'%len(training_data))        
    return training_data, pads
    numpy.savetxt('training.csv', training_data ,fmt='%s', delimiter=',' ,newline='\r\n') 
#Dice Calculation
def DiceIndex(BW1, BW2):
    BW1 = BW1.astype('float32')
    BW2 = BW2.astype('float32')
    #elementwise multiplication
    t= (np.multiply(BW1,BW2))
    total = np.sum(t)
    DI=2*total/(np.sum(BW1)+np.sum(BW2))
    DI=DI*100
    return DI

def runCNNModel(dataset_training, pads, epochs, patch_size, window_size, nclasses, datapath):
    # preprocessing
    X_training = np.zeros((len(dataset_training),windowsize_sq)).astype('int16')
    Y_training = np.zeros((len(dataset_training),1)).astype('int16')
    for p in range(0,len(dataset_training)):
        X_training[p]=dataset_training[p][0]
        Y_training[p]=dataset_training[p][1]
    #count the samples with scar in it
    X_training_scar = X_training[np.where(Y_training==1)]
    X_training_bg = X_training[np.where(Y_training==0)]
    print('\ntotal number of samples: %d' % len(X_training))        
    print('\nnumber of scar samples in the training data: %d' % len(X_training_scar))
    print('background is %d percent of entire data' %  ( len(X_training_bg) / len(X_training)*100))    
    print('scar is %d percent of entire data' %  ( len(X_training_scar) / len(X_training)*100))
    #Reshape my dataset for my model       
    X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
    X_training = X_training.astype('float32')
    X_training /= 255
    Y_training = np_utils.to_categorical(Y_training, nclasses)
    model = Sequential()
    model.add(Convolution2D(16, filter_size, filter_size, activation='relu', input_shape=(1,window_size,window_size), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, filter_size, filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_training, Y_training, epochs=epochs, batch_size=100, shuffle=True, verbose = 2)     
    model.summary()
    #save your model
    model.save(modelname)#path to  save  "C:\Users\fusta\Dropbox\1_Machine_Learning\Machine Learning\KerasNN\Neural_Network_3D_Scar\2D\Data Augmentation\Model.h5"    
    y_pred_scaled_cropped = []#.append(y_pred_scaled[p][:-pads[p+len(pid_train)][0],:-pads[p+len(pid_train)][1]])
    return y_pred_scaled_cropped

#to do a rough segmentation, save the ,model
(dataset_training, pads) = PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip, scar_definition_ratio)
y_pred_scaled_cropped = runCNNModel(dataset_training, pads, epochs, patch_size, window_size, nclasses, datapath)

#to do a finer segmentation, save the mpodel
#patch_size = 2
#window_size = 16
#patchsize_sq = np.square(patch_size)
#windowsize_sq = np.square(window_size)
#numpy.random.seed(windowsize_sq-1)
#modelname= 'CNN_scar_2.h5'
#(dataset_training, pads) = PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip)
#y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, pads, epochs, patch_size, window_size, nclasses, datapath)

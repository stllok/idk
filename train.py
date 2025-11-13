from platform import python_version
import os
import shutil
import numpy as np
import pandas as pd
import PIL.Image
from matplotlib import pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
from pathlib import Path
import zipfile
import shutil
import threading
from tqdm import tqdm
import random
import re

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import *


from pathlib import Path
import random
import cv2
import numpy as np
import shutil

SrcFolder = Path('./cleaned_data')
DstFolder = Path('./Processed_Handwritten_Data')

if DstFolder.exists():
    shutil.rmtree(DstFolder)

DstFolder.mkdir(parents=True)


def apply_random_transform(image: cv2.Mat) -> cv2.Mat:
    """
    Apply random transformations to an image: rotation, shearing, scaling
    """
    rows, cols = image.shape[:2]

    # Random rotation (-180 to 180 degrees)
    angle = random.uniform(-15, 15)
    rotation_matrix = cv2.getRotationMatrix2D((cols, rows), angle, 1)

    # # Random scaling (0.75 to 1.25)
    scale = random.uniform(0.8, 1.05)
    scaled_matrix = rotation_matrix.copy()
    scaled_matrix[0, 0] *= scale
    scaled_matrix[1, 1] *= scale

    # # Random shearing
    shear_x = random.uniform(-0.1, 0.1)
    shear_y = random.uniform(-0.1, 0.1)
    shear_matrix = np.float32([[1, 0, 0], [0, 1, 0]])

    # Apply transformations
    rotated = cv2.warpAffine(image, scaled_matrix, (cols, rows), borderValue=(255, 255, 255))
    transformed = cv2.warpAffine(rotated, shear_matrix, (cols, rows), borderValue=(255, 255, 255))

    return transformed

def process_and_save_images(n=10):
    """
    Process images in the source folder, apply transformations, and save to destination folder
    """
    CharFolders = [f for f in SrcFolder.iterdir() if f.is_dir()][:n]
    for CharFolder in CharFolders:
        DstCharFolder = DstFolder / CharFolder.name
        DstCharFolder.mkdir(parents=True, exist_ok=True)
        ImgFiles = list(CharFolder.glob('*.png'))
        print(ImgFiles)
        for ImgFile in ImgFiles:
            image = cv2.imread(str(ImgFile), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(DstCharFolder / f"{ImgFile.stem}_0.png"), image)  # Save original image
            transformed_images = [apply_random_transform(image) for _ in range(200)]
            for i, transformed_image in enumerate(transformed_images):
                DstImgFile = DstCharFolder / f"{ImgFile.stem}_{i + 1}.png"
                cv2.imwrite(str(DstImgFile), transformed_image)


process_and_save_images(5)

Num_Classes = len(list( DstFolder.iterdir()))
Image_Size = ( 300, 300 )
Epochs = 25
Batch_Size = 8

Train_Data_Genetor = ImageDataGenerator( rescale = 1./255, validation_split = 0.2,
                                         width_shift_range = 0.05,
                                         height_shift_range = 0.05,
                                         zoom_range = 0.1,
                                         horizontal_flip = False )

Train_Generator = Train_Data_Genetor.flow_from_directory( DstFolder ,
                                                          target_size = Image_Size,
                                                          batch_size = Batch_Size,
                                                          class_mode = 'categorical',
                                                          shuffle = True,
                                                          subset = 'training' )

Val_Data_Genetor = ImageDataGenerator( rescale=1./255, validation_split = 0.2 )

Val_Generator = Train_Data_Genetor.flow_from_directory( DstFolder ,
                                                        target_size = Image_Size,
                                                        batch_size = Batch_Size,
                                                        class_mode = 'categorical',
                                                        shuffle = True,
                                                        subset = 'validation' )

CNN = Sequential( name = 'CNN_Model' )
CNN.add( Conv2D( 5, kernel_size = (2,2), padding = 'same',  activation='relu',
                 input_shape = (Image_Size[0],Image_Size[1],3), name = 'Convolution' ) )
CNN.add( MaxPooling2D( pool_size = (2,2),  name = 'Pooling' ) )
CNN.add( Flatten( name = 'Flatten' ) )
CNN.add( Dropout( 0.5, name = 'Dropout_1') )
CNN.add( Dense( 512, activation = 'relu', name = 'Dense' ) )
CNN.add( Dropout( 0.5, name = 'Dropout_2') )
CNN.add( Dense( Num_Classes, activation = 'relu', name = 'Softmax' ) )
CNN.summary()
CNN.compile( optimizer = Adam(),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'] )

History = CNN.fit( Train_Generator,
                   steps_per_epoch = Train_Generator.samples//Batch_Size,
                   validation_data = Val_Generator,
                   validation_steps = Val_Generator.samples//Batch_Size,
                   epochs = Epochs )

Train_Accuracy = History.history['accuracy']
Val_Accuracy = History.history['val_accuracy']
Train_Loss = History.history['loss']
Val_Loss = History.history['val_loss']
epochs_range = range(Epochs)

plt.figure( figsize=(14,4) )
plt.subplot( 1,2,1 )
plt.plot( range( len(Train_Accuracy) ), Train_Accuracy, label='Train' )
plt.plot( range( len(Val_Accuracy) ), Val_Accuracy, label='Validation' )
plt.legend( loc='lower right' )
plt.title( 'Accuracy' )

plt.subplot( 1,2,2 )
plt.plot( range( len(Train_Loss) ), Train_Loss, label='Train' )
plt.plot( range( len(Val_Loss) ), Val_Loss, label='Validation' )
plt.legend( loc='upper right' )
plt.title( 'Loss')

plt.show()

CNN.save( 'CNN_Model.keras' )
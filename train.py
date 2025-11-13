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

'''
繁體中文顯示設定
'''
from matplotlib.font_manager import FontProperties

default_type = findfont( FontProperties( family=FontProperties().get_family() ) )
ttf_path = Path('/'.join( default_type.split('/')[:-1] ))  # 預設字型的資料夾路徑


DisplayChinese = Path("./matplotlib_Display_Chinese_in_Colab")
if not DisplayChinese.exists(  ):
    !git clone https://github.com/YenLinWu/matplotlib_Display_Chinese_in_Colab --depth 1

msj_name = ""
for item in DisplayChinese.glob( '*.ttf' ):
    msj_ttf_path = item.absolute()
    msj_name = msj_ttf_path.name


try:
    shutil.move( msj_ttf_path, ttf_path )
except:
    pass
finally:
    shutil.rmtree( DisplayChinese )
font = FontProperties( fname=ttf_path/msj_name )

CurrentFolder = Path(".")
OutputFolder = Path('./cleaned_data')

Path(OutputFolder).mkdir(exist_ok=True)

ImageList = list(OutputFolder.glob('**/*.png'))

def CreateMoveImageThread(img_path: Path):
    w = img_path.name.split('_')[0]
    (OutputFolder / w).mkdir(exist_ok=True) # Create the new word folder in OutputPath.
    shutil.move(img_path, OutputFolder / w / img_path.name)

MoveJoinSet: list[threading.Thread] = [threading.Thread(target=CreateMoveImageThread, args=(i,)) for i in tqdm(ImageList, desc="Spawnning task")]

for thread in tqdm(MoveJoinSet, desc="Starting job"):
    thread.start()

for thread in tqdm(MoveJoinSet, desc="Moving images"):
    thread.join()


for folder in OutputFolder.iterdir():
  if not re.match(r"[0-9]",folder.name):
    continue
  shutil.rmtree(folder)

print( 'Data Deployment completed.' )

a=0
b=0

for item in OutputFolder.iterdir():
  a += 1
  b += len(list(item.iterdir()))

print('總共: ' + str(a) + '個字, ' + str(b) + '張圖片')

from pathlib import Path
import random
import cv2
import numpy as np
import shutil

SrcFolder = OutputFolder
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

def Loading_Image( image_path ):
    img = load_img( image_path )
    img = tf.constant( np.array(img) )
    return img

def Show( image, title=None ) :
    if len( image.shape )>3 :
        image = tf.squeeze( image, axis=0 )

    plt.imshow( image )
    if title:
        plt.title( title, fontproperties=font )

img_list: list[Path] = []
for folder_path in DstFolder.iterdir():
    file_names = [item.name for item in  folder_path.iterdir()]
    for i in range(5) :
        img_list.append( folder_path / file_names[i] )

plt.gcf().set_size_inches( (12,12) )
for i in range(20):
    plt.subplot(4,5,i+1)
    title = img_list[i].name.split('_')[-3]
    img = Loading_Image( img_list[i] )
    Show( img, title )

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
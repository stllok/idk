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

TestDir = Path("./test")
DstFolder = Path('./Processed_Handwritten_Data')

if TestDir.exists():
    shutil.rmtree(TestDir)
TestDir.mkdir()

for folder in DstFolder.iterdir():
  for img_file in folder.iterdir():
    shutil.copyfile(img_file,TestDir /  img_file.stem)

from keras.preprocessing import image

# load trained model
model = load_model('CNN_Model.keras')

chars = list([x.name for x in DstFolder.iterdir()][::-1])
 
print(chars)

# predict all photos (loop though the folder)
for img_file in [random.choice(list(TestDir.iterdir())) for _ in range(30)]:
    img = image.load_img(str(img_file), target_size=(300, 300))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = list(sorted(filter(lambda x: x[1] > 0, zip(chars, model.predict(x).tolist()[0])), key=lambda x: x[1]))
    print(pred)

    char, max_prob = pred[-1]

    print(f"src img: {img_file.name.split("_")[0] }\t detected: {char} with pedict {max_prob*100}%\tCorrect: {img_file.name.split("_")[0] == char}")

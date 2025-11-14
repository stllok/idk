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

chars = list(set([x.name.split("_")[0] for x in TestDir.iterdir()]))

print(chars)

# predict all photos (loop though the folder)
for img_file in [random.choice(list(TestDir.iterdir())) for _ in range(5)]:
    img = image.load_img(str(img_file), target_size=(300, 300))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = model.predict(x)

    max_value = pred[0]
    max_index = 0

    for i in range(1, len(pred)):
        if pred[i] > max_value:
            max_value = pred[i]
            max_index = i

    plt.gcf().set_size_inches((20,2))
    print(f"src img: {img_file.name.split("_")[0] }\t detected: {chars[max_index]}")
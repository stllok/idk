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
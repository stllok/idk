## Before read

The model (aka current version of script) only training 1000 due to the resource limitation

By using Google colab maximum training is ~40 words then it will forced shutdown due to the OOM occur (google colab free-tier only provided 12GB memory).


## Training Machine

Motherboard: Supermicro SYS-440P-TNRT

CPU: Intel Xeon Platinum 8372HC * 2

GPU: NVIDIA RTX A6000 * 2

RAM: 3TB

OS: Debian GNU/Linux 12 (bookworm)

Python version 3.11.2

CUDAToolKit  13.0

CUDNN 8.5.0

Library used (from `pyproject.toml`)

```
    "ipykernel>=7.1.0",
    "keras>=3.12.0",
    "matplotlib>=3.10.7",
    "numpy>=2.3.4",
    "opencv-python>=4.11.0.86",
    "pandas>=2.3.3",
    "pillow>=12.0.0",
    "requests>=2.32.5",
    "scipy>=1.16.3",
    "tensorflow[and-cuda]>=2.20.0",
    "tqdm>=4.67.1",
```

## Step

### Method on generate various image sample

Using script `download.sh` to prepare raw dataset.

Then using script `prepare.py` to processing the word.

By using the `train.py`, it will do the following job:

### Process image:
- random rotate -15 - 15 degree

- shearing rows cols with 0.9 - 1.1

with repeat the generation by 5 times to generate ~200 words

Then, with using `ImageDataGenerator` I can easily wrap all filltered image to the data model.

After that, start the training with cost about a day for the training.

For testing, use the script `test_model.py` to test the word detecting by randomly pick the image from the trained dataset.

## Reference

https://github.com/chenkenanalytic/handwritting_data_all

https://github.com/AI-FREE-Team/Handwriting-Chinese-Characters-Recognition

## AI model used in this project

Qwen3-coder

Kimi-K2-Instruct-0905

## AI tools used in this project

Cline

Context7

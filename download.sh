#!/bin/bash

git clone https://github.com/chenkenanalytic/handwritting_data_all.git --depth 1
cat ./handwritting_data_all/all_data.zip* > ./handwritting_data_all/all_data.zip
rm -rf cleaned_data/
LC_ALL=C unzip ./handwritting_data_all/all_data.zip -d "."

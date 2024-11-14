#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/9 8:50
import json
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import radiomics
from bin import resample_file
from radiomics import featureextractor

file_paths = ["./data/RECT_LABEL/extracted_features_rect1.csv", "./data/RECT_LABEL/extracted_features_rect2.csv",
              "./data/RECT_LABEL/extracted_features_rect3.csv", "./data/RECT_LABEL/extracted_features_rect4.csv",]

dfs = [pd.read_csv(file) for file in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv('./data/RECT_LABEL/rect_combined_total_features.csv', index=False)
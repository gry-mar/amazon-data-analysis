import os
import yaml
import pandas as pd
import json
from sklearn.model_selection import train_test_split

"""
File for segmentation to train and test data
"""

params = yaml.safe_load(open('params.yaml'))['segmentation']

data_path = os.path.join('data', 'segmentation')
os.makedirs(data_path, exist_ok=True)

file = params['datasets']

df = pd.read_csv(file)
print(df.head())
df = df.drop('Unnamed: 0', axis=1)
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['overall'])
train.to_csv(os.path.join(data_path, 'train.csv'))
test.to_csv(os.path.join(data_path, 'test.csv'))


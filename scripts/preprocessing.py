
import os
import yaml
import pandas as pd
import json
params = yaml.safe_load(open('params.yaml'))['preprocessing']

# create folder to save file
data_path = os.path.join('data', 'preprocessing')
os.makedirs(data_path, exist_ok=True)

file = params['datasets']
data = []
for line in open(file, "r"):
    data.append(json.loads(line))

df = pd.json_normalize(data)


# adding new cols not affecting train and test data
scores = {1.0: "negative", 2.0: "negative", 3.0: "neutral", 4.0: "positive", 5.0: "positive"}
df['overall'].replace(scores, inplace=True)
datetime_col = pd.to_datetime(df['unixReviewTime'], unit='s')
df['weekday'] = datetime_col.dt.weekday
df['VerifiedAndVoted'] = (df['verified'] == True) & (df['vote'].notnull())
df['word_count'] = df['reviewText'].str.split().str.len()
df['ReviewLength'] = df['reviewText'].astype(str).fillna('').apply(len)
print(df.head())
df.to_csv(os.path.join(data_path, 'Luxury_Beauty_5.csv'))
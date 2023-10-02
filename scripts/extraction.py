"""
Extract sentiment from dataset
"""
import json
import pandas as pd
import os
import yaml
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
import spacy
from spacy.tokens import Token
from spacy.matcher import Matcher
from spacytextblob.spacytextblob import SpacyTextBlob
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer

params = yaml.safe_load(open('params.yaml'))['extraction']
test_name = params['datasets']['test']
train_name = params['datasets']['train']
# create folder to save file
data_path = os.path.join('data', 'extraction')
os.makedirs(data_path, exist_ok=True)


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def extract(df):
    df_new = df[['overall', 'verified', 'reviewText', 'summary', 'vote', 'weekday', 'VerifiedAndVoted', 'word_count', 
                    'ReviewLength' ]]
    return df_new

def save(df, name):
    print(df.head())
    df.to_json(os.path.join(data_path, name))

def handle_null(df):
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df.fillna((df[col].median()), inplace=True)
        
    df['summary'] = df['summary'].fillna('').astype(str)
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    return df


def clean_col(column):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("spacytextblob")
    clean_text = []
    for doc in nlp.pipe(column):
        words = [token.text for token in doc if token._.blob.sentiment_assessments.assessments]
        words = ' '.join(words).strip()
        clean_text.append(words.lower())

    return clean_text

def text_clean(df):
    df['summary'] = clean_col(df['summary'])
    df['reviewText'] = clean_col(df['reviewText'])
    return df



def process_df(df, filename):
    df_new = extract(df)
    df_new = handle_null(df_new)
    df_new = text_clean(df_new)
    print(df_new.head())
    save(df_new, filename)


test = load_data(test_name)
train = load_data(train_name)
process_df(test, "test.json")
process_df(train, "train.json")

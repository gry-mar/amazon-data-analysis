"""
Model evaluation
"""
import os
from sklearn.experimental import enable_halving_search_cv
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
import json
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import time

params = yaml.safe_load(open('params.yaml'))['evaluation']


train_name = params['datasets']['train']
test_name = params['datasets']['test']
strategy = params['strategy']


# create folder to save file
data_path = os.path.join('data', 'evaluation')
os.makedirs(data_path, exist_ok=True)


def load_data(filepath):
    with open(filepath) as file:
        data = json.load(file)
        df = pd.DataFrame.from_dict(data)
        return df


# pipeline elements
cat_preprocess = Pipeline([('ohe', OneHotEncoder())])
review_preprocess = Pipeline([('bow_review', CountVectorizer())])
summary_preprocess = Pipeline([('bow_summary', CountVectorizer())])
scaler = Pipeline([('scaler', MinMaxScaler())])



train = load_data(train_name)
test = load_data(test_name)
data_selector = params['data_selector']
classifier = params['classifier']


# # feature selection (text/other/all)
if data_selector =="text":
    train = train.drop(['vote', 'verified', 'VerifiedAndVoted', 'weekday', 'word_count', 'ReviewLength'], axis=1 )
    preprocess = ColumnTransformer([
    ('review_preprocess', review_preprocess, 'reviewText'),
    ('summary_preprocess', summary_preprocess, 'summary'),
    ])
elif data_selector =="other":
    train = train[['vote', 'verified', 'weekday', 'word_count', 'ReviewLength', 'overall']]
    preprocess = ColumnTransformer([
    ('cat_preprocess', cat_preprocess, ['weekday', 'VerifiedAndVoted', 'verified']),
    ('scaler', scaler, ['word_count', 'ReviewLength', 'vote'])
    ])
else:
   preprocess = ColumnTransformer([
    ('cat_preprocess', cat_preprocess, ['weekday', 'VerifiedAndVoted', 'verified']),
    ('review_preprocess', review_preprocess, 'reviewText'),
    ('summary_preprocess', summary_preprocess, 'summary'),
    ('scaler', scaler, ['word_count', 'ReviewLength', 'vote'])
    ])


X_train= train.drop('overall', axis=1)
X_test = test.drop('overall', axis=1)
y_train = train['overall']
y_test = test['overall']

# choose classifier
if classifier == "dummy":
    if strategy == "uniform":
        clf = DummyClassifier(strategy="uniform")
    else:
        clf = DummyClassifier()
    
elif classifier == "svm":
    clf = SVC()
else:
    clf = RandomForestClassifier()

pipeline = Pipeline([
                ('preprocess', preprocess),
                ('clf', clf)])
    
if data_selector=='text' or data_selector=='all':
    param_grid = [
            {
                "preprocess__summary_preprocess__bow_summary__max_features": [50, 100, 200, 400],
                "preprocess__review_preprocess__bow_review__max_features": [50, 100, 200, 400]
            }]
else:
    param_grid = []

halvingrandom = HalvingRandomSearchCV(pipeline, param_distributions=param_grid, cv=5, verbose=1, n_jobs=-1,
                                        scoring='f1_macro')



start_time = time.time()
halvingrandom.fit(X_train, y_train)
score_hr = halvingrandom.best_estimator_.score(X_test,y_test)
print(halvingrandom.best_estimator_)
print(f'Halving random search score:{score_hr}')
end_time = time.time()
t_hr = end_time-start_time
print(f"Halving Random ended. Time passed: {t_hr}")

if data_selector=='text' or data_selector=='all':
    max_review = halvingrandom.best_params_['preprocess__review_preprocess__bow_review__max_features']
    max_summary = halvingrandom.best_params_['preprocess__summary_preprocess__bow_summary__max_features']

    review_preprocess = Pipeline([('bow_review', CountVectorizer(max_features=max_review))])
    summary_preprocess = Pipeline([('bow_summary', CountVectorizer(max_features=max_review))])

if data_selector =="text":
    preprocess = ColumnTransformer([
    ('review_preprocess', review_preprocess, 'reviewText'),
    ('summary_preprocess', summary_preprocess, 'summary'),
    ])
elif data_selector =="other":
    preprocess = ColumnTransformer([
    ('cat_preprocess', cat_preprocess, ['weekday', 'VerifiedAndVoted', 'verified']),
    ('scaler', scaler, ['word_count', 'ReviewLength', 'vote'])
    ])
else:
   preprocess = ColumnTransformer([
    ('cat_preprocess', cat_preprocess, ['weekday', 'VerifiedAndVoted', 'verified']),
    ('review_preprocess', review_preprocess, 'reviewText'),
    ('summary_preprocess', summary_preprocess, 'summary'),
    ('scaler', scaler, ['word_count', 'ReviewLength', 'vote'])
    ])
   
pipeline = Pipeline([
                ('preprocess', preprocess),
                ('clf', clf)])

#apply best to test
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score_final = f1_score(y_test, y_pred, average='macro')
print(f'Final score: {score_final}')


results = {'f1': score_final, 'clf': classifier}


with open(os.path.join(data_path, 'results.json'), "w") as f:
    json.dump(results, f)

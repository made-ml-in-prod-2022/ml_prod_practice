import joblib
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from transformer import Transformer


def create_ohe(df):
    lst_ohe_feat = ['sex', 'restecg', 'slope', 'fbs', 'cp', 'exang', 'thal', 'ca']
    numvars_targ = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'condition']
    lst_ohe_out = []
    for i in lst_ohe_feat:
        tdf = pd.get_dummies(df[i], i)
        lst_ohe_out.append(tdf)

    df_ohe = pd.concat(lst_ohe_out,axis=1)
    df = df[numvars_targ].merge(df_ohe, left_index=True, right_index=True)
    return df


def split_dataset(src_df):
    train, test = train_test_split(src_df,
                                   test_size=TEST_SIZE,
                                   random_state=RANDOM_STATE)
    return train, test


def get_label(src_df, label_name):
    features = src_df.drop(label_name, axis=1)
    label = src_df[label_name]
    return features, label


PATH = r'data/heart_cleveland_upload.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCORING = "neg_root_mean_squared_error"
TESTNAME = "data/test"
FILENAME = "data/Completed_model.joblib"

df = pd.read_csv(PATH)
train, test = split_dataset(df)
test.to_csv(TESTNAME)
train_feat, train_label = get_label(train, 'condition')

model = LogisticRegression()
pipe = make_pipeline(Transformer('thalach'), model)
pipe.fit(train_feat, train_label)

joblib.dump(model, FILENAME)

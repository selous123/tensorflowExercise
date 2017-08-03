# Python deomo code to readin and plot the radar signal data

# create by Fernando from Team Marmot

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_SET_PATH = "../ics/adult.data"
TEST_SET_PATH = "../ics/adult.test"

def read_data_ics():
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
    #Data Frame;
    df_train = pd.read_csv(TRAIN_SET_PATH,names = COLUMNS,skipinitialspace = True);
    df_test = pd.read_csv(TEST_SET_PATH,names = COLUMNS,skipinitialspace=True, skiprows=1)

#    print type(df_train['income_bracket'][0]);
    LABEL_COLUMN = "label"
    df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    FEATURE_COLUMNS  = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country","age", 
                       "education_num", "capital_gain", "capital_loss", "hours_per_week"];

    keys = ['data','labels'];
    train = {}.fromkeys(keys);
    test = {}.fromkeys(keys);
    print train.keys()
    train['data'] = pd.get_dummies(df_train[FEATURE_COLUMNS]).as_matrix();
    test['data'] = pd.get_dummies(df_test[FEATURE_COLUMNS]).as_matrix();
    train['labels'] = df_train['label'].as_matrix();
    test['labels'] = df_test['label'].as_matrix();
    return train,test;


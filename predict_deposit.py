
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./bank-additional-full.csv', sep = ';')

#DROPING THESE THREE FEATURES AS THE NUMBER OF UNKNOWN VALUES ARE VERY HIGH IN THEM ACRESS DATA POINTS.
df.drop(['default', 'loan', 'pdays'], axis = 1, inplace = True)
#REPLACING UNKNOWN VALUES WITH NaN
df.replace(to_replace = 'unknown', value = np.NaN, inplace = True)

#LABEL_COL IS THE LIST OF ALL FEATURES WHOSE DOMAIN IS {Yes, No}
label_col = []
for col in df.columns:
    if(len(df[col].value_counts()) == 2):
        label_col.append(col)

label_col.remove('housing')

#FOR ALL THE FEATURES IN LABEL_COL YES IS REPLACED WITH 1 AND NO IS REPLACED WITH 0
for col in label_col:
    replace = df[col].value_counts().index
    replace_dict = {np.NaN: np.NaN}
    for key in range(len(replace)):
        replace_dict[replace[key]] = key
    df[col] = [replace_dict[x] for x in df[col]]
label_col.append('housing')

enc_list = []
num_list = []
for col in df.columns:
    if(str(np.array(df[col]).dtype) == 'object'):
        enc_list.append(col)
    else:
        num_list.append(col)

#THIS FUNCTION IS USED TO FILL THE MISSING VALUES IN TARGET FEATURE USING DATA IN INPUT_FEATURES
def fill_unknown(data, inp_features, target):
    train_index = data.loc[data[target].notnull()].index
    X_train = data[inp_features].loc[train_index]
    y_train = data[target].loc[train_index]
    pred_index = data.loc[data[target].isnull()].index
    X_pred = data[inp_features].loc[pred_index]
    if(list(pred_index) == []):
        return
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_pred)
    return predictions, pred_index


df1 = df.drop(['y'], axis = 1)

print('Filling unknown values')
for col in tqdm(df1.columns):
    inp = list(set(num_list) - set(col))
    temp = fill_unknown(df, inp, col)
    if(temp != None):
        predictions, pred_index = temp
        df[col].loc[pred_index] = predictions
df['housing'] = [{'yes' : 1, 'no' : 0}[x] for x in df['housing']]

#THIS FUNCTION RE-SAMPLES THE DATA SET SUCH THAT THE NUMBER OF INSTANCES IN ALL THE CALSSES ARE SAME.
def re_sample(df, kn = 5):
    categorical = [i for i in df.columns if(set(df[i].values) != set([0,1]))]
    non_categorical = [i for i in df.columns if(i not in categorical)]
    inp_features = list(df.columns)
    inp_features.remove('y')
    x = df[inp_features]
    y = pd.DataFrame(df.y)
    df  = pd.concat([pd.get_dummies(df[categorical]),df[non_categorical]], axis = 1)
    sm = SMOTE(random_state = 0, k_neighbors = kn)
    x,y = sm.fit_resample(x,y)
    df1 = pd.DataFrame(x, columns = inp_features)
    df2 = pd.DataFrame(y, columns = ['y'])
    df = pd.concat([df1, df2], axis = 1)
    return df

# ONE HOT ENCODING
df_ohe = pd.get_dummies(df)
# RE-SAMPLING THE DATASET TO MAKE IT BALANCED
df_ohe = re_sample(df_ohe)
# RE-SAMPLING INTRODUCES FLOATING NUMBERS FOR THE FEATURES WHOSE VALUES IN THE DOMAIN IS SUPPOSED TO BE INTEGERS

# TO TACKLE THIS ISSUE WE ROUND UP ALL THE FEATURES WHOSE VALUES IN THE DOMAIN IS SUPPOSED TO BE INTEGERS
tb_rounded = list(set(df_ohe.columns) - set(enc_list) - set(num_list) | set(label_col))

for col in tb_rounded:
    df_ohe[col] = [round(x) for x in df_ohe[col]]

def normalize(array):
    array = np.array(array)
    return (array - min(array))/(max(array) - min(array))

for col in df_ohe.columns:
    df_ohe[col] = normalize(df_ohe[col])

df_ohe.to_csv('pre_processed.csv', index = False)
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(df_ohe.drop('y', axis = 1), df_ohe['y'], test_size = test_size)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)

accuracy_score(y_test, log_pred), recall_score(y_test, log_pred), f1_score(y_test, log_pred), precision_score(y_test, log_pred)
print("accuracy score for logistic regression :",accuracy_score(y_test, log_pred))
print("recall score for logistic regression :",recall_score(y_test, log_pred))
print("f1 score for logistic regression :",f1_score(y_test, log_pred))
print("precision score for logistic regression :",precision_score(y_test, log_pred))

neigh_def = KNeighborsClassifier(n_neighbors = 1)
neigh_def.fit(X_train, y_train)
knn_pred = neigh_def.predict(X_test)

accuracy_score(y_test, knn_pred), recall_score(y_test, knn_pred), f1_score(y_test, knn_pred), precision_score(y_test, knn_pred)
print("accuracy score for KNN classifier for k = 1:",accuracy_score(y_test, knn_pred))
print("recall score for KNN classifier for k = 1:",recall_score(y_test, knn_pred))
print("f1 score for KNN classifier for k = 1:",f1_score(y_test, knn_pred))
print("precision score for KNN classifier for k = 1:",precision_score(y_test, knn_pred))

#TO SAVE THE FILE IN A PICKLE FORMAT
def save_to_pickle(filename, model):
    pkl_filename = filename
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

#TO LOAD MODEL FROM A PICKLE FILE
def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

save_to_pickle('log.pkl', logmodel)
save_to_pickle('knn_k1.pkl', neigh_def)

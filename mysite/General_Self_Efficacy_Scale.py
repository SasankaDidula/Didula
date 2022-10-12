# Imported Libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
import mysite.encoders as encoder

warnings.filterwarnings('ignore')
oneH = OneHotEncoder()
import pickle
import dill


def predict_Positive_negative(For_predict_lis):
    with open('models/tree_clf.joblib', 'rb') as io:
        tree_clf = dill.load(io)

    Pkl_Filename = "models/encoder.pkl"
    with open(Pkl_Filename, 'rb') as file:
        Encoder = pickle.load(file)

    train_data_categorical_columns = encoder.GSE_data_categorical_columns
    new_colums = encoder.GSE_new_colums
    data = [{'Q1': For_predict_lis[0], 'Q2': For_predict_lis[1], 'Q3': For_predict_lis[2], 'Q4': For_predict_lis[3],
             'Q5': For_predict_lis[4], 'Q6': For_predict_lis[5], 'Q7': For_predict_lis[6], 'Q8': For_predict_lis[7],
             'Q9': For_predict_lis[8], 'Q10': For_predict_lis[9]}]
    try_df = pd.DataFrame(data)
    try_df = pd.DataFrame(Encoder.transform(try_df[train_data_categorical_columns]).toarray(), columns=new_colums)
    y_predict = tree_clf.predict(try_df)
    x = y_predict.tolist()
    y = x[0]
    # print('Predicted Value :', y)
    return y



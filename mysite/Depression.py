# Imported Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')
le = LabelEncoder()
oneH = OneHotEncoder()
import pickle
import dill
import mysite.encoders as encoder


def predict_depression_status(For_predict_lis):
    with open('models/Depression_svm_clf.joblib', 'rb') as io:
        Depression_svm_clf = dill.load(io)

    Pkl_Filename = "models/Depression_svm_encoder.pkl"
    with open(Pkl_Filename, 'rb') as file:
        Depression_svm_encoder = pickle.load(file)

    train_data_categorical_columns = encoder.Depression_data_categorical_columns
    new_colums = encoder.Depression_new_colums

    data = [{'Q3A': For_predict_lis[0],
             'Q5A': For_predict_lis[1],
             'Q10A': For_predict_lis[2],
             'Q13A': For_predict_lis[3],
             'Q16A': For_predict_lis[4],
             'Q17A': For_predict_lis[5],
             'Q21A': For_predict_lis[6],
             'Q24A': For_predict_lis[7],
             'Q26A': For_predict_lis[8],
             'Q31A': For_predict_lis[9],
             'Q34A': For_predict_lis[10],
             'Q37A': For_predict_lis[11],
             'Q38A': For_predict_lis[12],
             'Q42A': For_predict_lis[13],
             'Extraverted-enthusiastic': For_predict_lis[14],
             'Critical-quarrelsome': For_predict_lis[15],
             'Dependable-self_disciplined': For_predict_lis[16],
             'Anxious-easily upset': For_predict_lis[17],
             'Open to new experiences-complex': For_predict_lis[18],
             'Reserved-quiet': For_predict_lis[19],
             'Sympathetic-warm': For_predict_lis[20],
             'Disorganized-careless': For_predict_lis[21],
             'Calm-emotionally_stable': For_predict_lis[22],
             'Conventional-uncreative': For_predict_lis[23],
             'education': For_predict_lis[24],
             'gender': For_predict_lis[25],
             'engnat': For_predict_lis[26],
             'screensize': For_predict_lis[27],
             'hand': For_predict_lis[28],
             'religion': For_predict_lis[29],
             'orientation': For_predict_lis[30],
             'married': For_predict_lis[31],
             'Age_Groups': For_predict_lis[32]}]

    try_df = pd.DataFrame(data)
    try_df = pd.DataFrame(Depression_svm_encoder.transform(try_df[train_data_categorical_columns]).toarray(),
                          columns=new_colums)
    y_predict = Depression_svm_clf.predict(try_df)
    x = y_predict.tolist()
    y = x[0]
    # print('Predicted Value :', y)
    return y


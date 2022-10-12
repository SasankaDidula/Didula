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


def stress_depression_status(For_predict_lis):
    with open('models/Stress_svm_clf.joblib', 'rb') as io:
        Stress_svm_clf = dill.load(io)

    Pkl_Filename = "models/Stress_svm_encoder.pkl"
    with open(Pkl_Filename, 'rb') as file:
        Stress_svm_encoder = pickle.load(file)

    train_data_categorical_columns = encoder.Stress_data_categorical_columns
    new_colums = encoder.Stress_new_colums

    data = [{'Q1A': For_predict_lis[0],
             'Q6A': For_predict_lis[1],
             'Q8A': For_predict_lis[2],
             'Q11A': For_predict_lis[3],
             'Q12A': For_predict_lis[4],
             'Q14A': For_predict_lis[5],
             'Q18A': For_predict_lis[6],
             'Q22A': For_predict_lis[7],
             'Q27A': For_predict_lis[8],
             'Q29A': For_predict_lis[9],
             'Q32A': For_predict_lis[10],
             'Q33A': For_predict_lis[11],
             'Q35A': For_predict_lis[12],
             'Q39A': For_predict_lis[13],
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
    try_df = pd.DataFrame(Stress_svm_encoder.transform(try_df[train_data_categorical_columns]).toarray(),
                          columns=new_colums)
    y_predict = Stress_svm_clf.predict(try_df)
    x = y_predict.tolist()
    y = x[0]
    # print('Predicted Value :', y)
    return y


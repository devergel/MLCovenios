import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import os.path as osp
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
def main():
    X_train, y_train, X_test, y_test = preprocess()
    model = model(X_train, y_train)
    st.title('Recomendacion de Convenios de Universidades del Exteriror')
    st.sidebar.title('Convenios')
    option = st.sidebar.radio('Opciones:',
                ['Sugerencia de convenios',
                'Segmentacion de Estudiantes'])
    if option=='Sugerencia de convenios':
        name = st.text_input('Sugerencia de convenios')
        if st.button('Recomendar'):
            try:
                y_pred_train_EN = model.predict(X_train)
                report_EN = classification_report(y_train, y_pred_train_EN)
                st.write(report_EN)
            except:
                st.write('Ocurrio un error al analizar la sugenrencias de convenios')
    if option=='Segmentacion de Estudiantes':
        name = st.text_input('Segmentacion de Estudiantes')
        try:
            st.write('Tipos Estudiantes test')
        except:
            st.write('Ocurrio un error al analizar la sugenrencias de convenios')


@st.experimental_singleton
def loadData():
    url = "/Dataset/AcademicMoveWishesOutgoing (Mon May 16 2022).xlsx"
    return pd.read_excel(url)


@st.experimental_singleton
def preprocess():
    data = loadData()
    lb_make = LabelEncoder()
    data_clean = data
    data_clean.drop('Stay wish: ID', inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 1"], inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 2"], inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 3"], inplace=True, axis=1)
    data_clean["Status selection"] = lb_make.fit_transform(data["Status selection"])
    data_clean["Status offer"] = lb_make.fit_transform(data["Status offer"])
    data_clean["Form"] = lb_make.fit_transform(data["Form"])
    data_clean["Start period"] = lb_make.fit_transform(data["Start period"])
    data_clean["Country"] = lb_make.fit_transform(data["Country"])
    data_clean["Stay: Examen 1"] = lb_make.fit_transform(data["Stay: Examen 1"])
    data_clean["Stay: Idioma 1"] = lb_make.fit_transform(data["Stay: Idioma 1"])
    data_clean["Stay: Idioma 2"] = lb_make.fit_transform(data["Stay: Idioma 2"])
    data_clean["Stay: Idioma 3"] = lb_make.fit_transform(data["Stay: Idioma 3"])
    data_clean["Stay: Examen 3"] = lb_make.fit_transform(data["Stay: Examen 3"])
    data_clean["Stay: Examen 2"] = lb_make.fit_transform(data["Stay: Examen 2"])
    data_clean["Institution"] = lb_make.fit_transform(data["Institution"])
    data_clean["Level"] = lb_make.fit_transform(data["Level"])
    data_clean["Stay: Status"] = lb_make.fit_transform(data["Stay: Status"])
    data_clean["Stay: Degree programme"] = lb_make.fit_transform(data["Stay: Degree programme"])
    data_clean["Stay: GPA outgoing"]=data["Stay: GPA outgoing"].str.replace(',','.').astype(float)
    data_clean = data_clean.fillna(0)
    train, test = train_test_split(data_clean, test_size=0.2, random_state=33)
    return train.drop(['Status selection'],axis=1), train['Status selection'], test.drop(['Status selection'],axis=1), test['Status selection']  
 

@st.experimental_singleton
def model(X_train, y_train):
    random = RandomForestClassifier(max_depth=8,random_state=0)
    random.fit(X_train,y_train)

      
if __name__ == '__main__':
    main()
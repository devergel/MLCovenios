import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import os.path as osp
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
def main():
    X_train, y_train, X_test, y_test = preprocess()
    random = model(X_train, y_train)
    st.title('Recomendacion de Convenios de Universidades del Exteriror')
    st.sidebar.title('Convenios')
    option = st.sidebar.selectbox('Opciones:',
                ['Sugerencia de Convenios',
                'Segmentacion de Estudiantes'])
    if option=='Sugerencia de convenios':
        st.subheader("Datos Estudiante")
        with st.form(key="form"):
            col1,col2 = st.beta_columns([3,3])
            
            with col1:
                program = st.sidebar.selectbox('Programa:',
                ['Sugerencia de convenios',
                'Segmentacion de Estudiantes'])
                promedio = st.number_input("Promedio",1,10)
                country = st.sidebar.selectbox('Pais:',
                ['EEUU',
                'Alemania'])
                
            with col2:
                st.write('Idioma Max 3 opciones:')
                languaje = st.multiselect("Idioma Max 3 Opciones")
                if len(languaje) <1:
                    st.write('Selecciona Minimo 1 Idioma')
                elif len(languaje) >3:
                    st.write('Selecciona Maximo 3 Idiomas')
                
                semestre = st.number_input("Semestre",1,10)
            
        if st.form_submit_button(label='Recomendar'):
            try:
                df2 = {'First Name': program, 'Last Name': promedio, 'Country': languaje[0],
                'First Name': languaje[1], 'Last Name': languaje[2], 'Country': country,
                'First Name': semestre, 'Last Name': country}

                df = df.append(df2, ignore_index = True)
                y_pred_train_EN = random.predict_proba(df)
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
    url = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/AcademicMoveWishesOutgoing%20(Mon%20May%2016%202022).xlsx"
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
    return random.fit(X_train,y_train)

      
if __name__ == '__main__':
    main()
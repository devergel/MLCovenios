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
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    data_clean, X_train, y_train, X_test, y_test = preprocess()
    random = model(X_train, y_train)
    st.title('Recomendacion de Convenios de Universidades del Exteriror')
    st.sidebar.title('Convenios')
    option = st.sidebar.selectbox('Opciones:',
                ['Sugerencia de Convenios',
                'Segmentacion de Estudiantes'])
    if option=='Sugerencia de Convenios':
        st.subheader("Datos Estudiante")
        with st.form(key="form"):
            col1,col2 = st.columns([3,3])
            
            with col1:
                program = st.selectbox('Programa:',
                ['Sugerencia de convenios',
                'Segmentacion de Estudiantes'])
                promedio = st.number_input("Promedio:",0.0,10.0,step=0.1,format="%.2f")
                country = st.selectbox('Pais:',
                ['EEUU',
                'Alemania'])
                
            with col2:
                languaje = st.multiselect("Idioma Max 3 Opciones:",
                    ['Español',
                    'Alemania',
                    'Ingles',
                    'Portugal'])
                if len(languaje) <1:
                    st.write('Selecciona Minimo 1 Idioma')
                elif len(languaje) >3:
                    st.write('Selecciona Maximo 3 Idiomas')
                
                semestre = st.number_input("Semestre:",1,10)
                
            submit_button = st.form_submit_button(label='Recomendar')
            
            if submit_button:
                try:
                    try:
                        if len(languaje)>=1:
                            lenguaje_1 = languaje[0]
                        if len(languaje)>=2:
                            lenguaje_2 = languaje[1]
                        if len(languaje)>=3:
                            lenguaje_3 = languaje[2]
                    except:
                        st.write('Ocurrio un error al extraer los idiomas')
                    
                    df2 = {'First Name': program, 'Last Name': promedio, 'Country': languaje_1,
                    'First Name': languaje_2, 'Last Name': languaje_3, 'Country': country,
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
            st.pyplot(clusters(data_clean))
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
    return data_clean, train.drop(['Status selection'],axis=1), train['Status selection'], test.drop(['Status selection'],axis=1), test['Status selection']  
 

@st.experimental_singleton
def model(X_train, y_train):
    random = RandomForestClassifier(max_depth=8,random_state=0)
    return random.fit(X_train,y_train)


@st.experimental_singleton
def clusters(data_clean):
    scaler = StandardScaler()
    datos_scaled = scaler.fit_transform(data_clean)
    
    kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    labels = kmeans.fit_predict(datos_scaled)
    
    y_kmeans = kmeans.predict(datos_scaled)
    data_clean['Cluster']       = y_kmeans
    datos_scaled              = pd.DataFrame(datos_scaled)
    datos_scaled['Cluster']   = y_kmeans        
        
    data_clean["Cluster"].replace({0: 3, 6:5, 7:5, 4:3, 2:3}, inplace=True)
    datos_scaled["Cluster"].replace({0: 3, 6:5, 7:5, 4:3, 2:3}, inplace=True)
    clusters_   = data_clean["Cluster"]
    
    pca = PCA(n_components = 2)
    pca.fit(datos_scaled)

    scores = pca.transform(datos_scaled)

    x,y = scores[:,0] , scores[:,1]
    df_data = pd.DataFrame({'x': x, 'y':y, 'clusters':clusters_})
    grouping_ = df_data.groupby('clusters')
    fig, ax = plt.subplots(figsize=(20, 13))

    names = {1: 'Cluster 1', 
             3: 'Cluster 3',
             5: 'Cluster 5',}

    for name, grp in grouping_:
        ax.plot(grp.x, grp.y, marker='o', label = names[name], linestyle='')
        ax.set_aspect('auto')

    ax.legend()
    return fig
      
if __name__ == '__main__':
    main()
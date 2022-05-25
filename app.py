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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    countryDf, idioma1Df, instDf, programDf, data_clean, data_scaled, X_train, y_train, X_test, y_test = preprocess()
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
                programDf["label"])
                promedio = st.number_input("Promedio:",0.0,10.0,step=0.1,format="%.2f")
                country = st.selectbox('Pais:',
                countryDf["label"])
                
            with col2:
                languaje = st.multiselect("Idioma Max 3 Opciones:",
                    idioma1Df["label"])
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
                    
                    seats_country, seats_index_LT, seats_index_GT, facultades = loadCountryAndInst(country)
                    df = pd.DataFrame()

                    for index, row in seats_index_GT.iterrows():
                        try: 
                          countryA=countryDf[countryDf['label'] == row["Country"]].values.item(0)
                        except:
                          countryDf = countryDf.append({'code': (countryDf['code'].max() + 1), 'label': row["Country"]}, ignore_index = True)
                          countryA= countryDf[countryDf['label'] == row["Country"]].values.item(0)

                        try: 
                          InstitutionA=instDf[instDf['label'] == row["Name"]]["code"].values.item(0)
                        except:
                          instDf = instDf.append({'code': (instDf['code'].max() + 1), 'label': row["Name"]}, ignore_index = True)
                          InstitutionA= instDf[instDf['label'] == row["Name"]].values.item(0)

                        df2 = {'Relation: ID': row["RelationID"], 'Country': countryA, 'Institution': InstitutionA, 'Stay: Degree programme':programDf[programDf['label'] == program]["code"].values.item(0), 'Stay: Semestre actual de estudios': semestre, 'Stay: GPA outgoing': promedio, 
                               'Chinese': lenguaje_1=='Chinese' or lenguaje_2=='Chinese' or lenguaje_3=='Chinese'?1:0, 'Faculty': 2,'English':lenguaje_1=='English' or lenguaje_2=='English' or lenguaje_3=='English'?1:0, 'French':lenguaje_1=='French' or lenguaje_2=='French' or lenguaje_3=='French'?1:0, 'German':lenguaje_1=='German' or lenguaje_2=='German' or lenguaje_3=='German'?1:0, 'Italian':lenguaje_1=='Italian' or lenguaje_2=='Italian' or lenguaje_3=='Italian'?1:0, 'Japanese':lenguaje_1=='Japanese' or lenguaje_2=='Japanese' or lenguaje_3=='Japanese'?1:0, 'Korean':lenguaje_1=='Korean' or lenguaje_2=='Korean' or lenguaje_3=='Korean'?1:0, 'Portuguese':lenguaje_1=='Portuguese' or lenguaje_2=='Portuguese' or lenguaje_3=='Portuguese'?1:0}


                        df = df.append(df2, ignore_index = True)
                    
                    y_pred_GT = random.predict_proba(df)
                    df_y = pd.DataFrame(y_pred)
                    aux = pd.concat([df, df_y], axis=1)
                    aux = aux.sort_values(by = [1], ascending = False).head(3)
                    st.write(aux)
                except:
                    st.write('Ocurrio un error al analizar la sugenrencias de convenios')
    if option=='Segmentacion de Estudiantes':
        try:
            st.pyplot(clusters(data_clean, data_scaled))
        except:
            st.write('Ocurrio un error al analizar la sugenrencias de convenios')


@st.experimental_singleton
def loadCountryAndInst(country):
    url2 = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/Institutions%20(Mon%20May%2016%202022).xlsx"
    countryAndInstDf = pd.read_excel(url2)
    url = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/Flows%20(Mon%20May%2016%202022).xlsx"
    Seats = pd.read_excel(url)
    seats_clean = Seats.copy()
    seats_clean.drop('Total duration', inplace=True, axis=1)
    seats_clean.drop('Time unit', inplace=True, axis=1)

    #Eliminar datos año 1967 
    df_idx=seats_clean[seats_clean["Academic year"]<2018].index
    seats_clean=seats_clean.drop(df_idx)

    #Eliminar registros con Number en 0 o mull
    df_idx=seats_clean[seats_clean["Number"]==0].index
    seats_clean=seats_clean.drop(df_idx)
    seats_clean = seats_clean[seats_clean['Number'].notna()] 

    #Ajustar los valores para los Remaining Status en negativo porque son cupos adicionales
    seats_clean["RemainingSeats"] = seats_clean["Remaining seats"].apply(lambda x: 0 if x < 0 else x)
    seats_clean["aux"] = seats_clean["Remaining seats"].apply(lambda x: (x*(-1)) if x < 0 else 0)
    seats_clean["NumOffered"] = seats_clean["Number"] + seats_clean["aux"] 
    seats_clean[seats_clean["Remaining seats"]<0]
    seats_clean.drop('aux', inplace=True, axis=1)	
    seats_clean.drop('Remaining seats', inplace=True, axis=1)	
    seats_clean.drop('Number', inplace=True, axis=1) 

    #Calcular el índice de disponibilidad
    seats_clean["availabiltyIndex"] = seats_clean["RemainingSeats"] / seats_clean["NumOffered"] 

    #Eliminar columnas index, Academic year y Academin period
    seats_clean.drop('Academic year', inplace=True, axis=1) 
    seats_clean.drop('Academic period', inplace=True, axis=1) 

    #Ordenar por RelationID
    seats_clean = seats_clean.rename(columns={'Relation: Relation ID': 'RelationID'})
    seats_clean = seats_clean.sort_values('RelationID')
    seats_clean = seats_clean.reset_index()

    #Eliminar columnas index RemainingSeats NumOffered
    seats_clean.drop('index', inplace=True, axis=1) 
    seats_clean.drop('RemainingSeats', inplace=True, axis=1) 
    seats_clean.drop('NumOffered', inplace=True, axis=1) 

    seats_clean = seats_clean.groupby(['RelationID']).mean()
    url2 = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/relation_institution.xlsx"
    rel_Institution = pd.read_excel(url2)

    #Quitar las columas que no se necesitan
    rel_Institution.drop('relation_institution.id', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.role.id', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.is_active', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.created_on', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.created_by', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.last_modified_by', inplace=True, axis=1)
    rel_Institution.drop('relation_institution.last_modified_on', inplace=True, axis=1)

    #Quitar los registros con id=1 en relation_institution.institution.id (U Andes es 1)
    rel_Institution = rel_Institution[rel_Institution['relation_institution.institution.id'] != 1]
    rel_Institution = rel_Institution.rename(columns={'relation_institution.relation.id': 'RelationID'})
    rel_Institution = rel_Institution.rename(columns={'relation_institution.institution.id': 'institutionID'})
    
    url3 = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/Institutions%20(Mon%20May%2016%202022).xlsx"
    Institutions = pd.read_excel(url3)

    #Eliminar las columas que no se necesitan>
    Institutions.drop('City', inplace=True, axis=1) 
    Institutions.drop('Language requirement 1', inplace=True, axis=1) 
    Institutions.drop('Language cerf score 1', inplace=True, axis=1) 
    Institutions.drop('Language requirement 2', inplace=True, axis=1) 
    Institutions.drop('Language cerf score 2', inplace=True, axis=1) 
    Institutions.drop('Minimum GPA/4', inplace=True, axis=1)

    Institutions = Institutions.rename(columns={'Institution: ID': 'institutionID'})
    
    # Joins de las 3 tablas seats_clean rel_Institution Institutions
    new = seats_clean.merge(rel_Institution, on='RelationID', how='left')
    new2 = new.merge(Institutions, on='institutionID', how='left')
    seats_index = new2[['RelationID', 'institutionID', 'Country', 'Name', 'availabiltyIndex']]
    seats_index = seats_index[seats_index['institutionID'].notna()]
    
    #Archivo Relacion Convenio - Programa
    url4 = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/RelationsStayOpportunities%20(Mon%20May%2016%202022).xlsx"
    conv_Programa = pd.read_excel(url4)

    #Quitar las columas que no se necesitan
    conv_Programa.drop('Parent relation', inplace=True, axis=1) 
    conv_Programa.drop('Relation type', inplace=True, axis=1) 
    conv_Programa.drop('Direction', inplace=True, axis=1) 
    conv_Programa.drop('Status', inplace=True, axis=1) 
    conv_Programa.drop('Level', inplace=True, axis=1) 
    conv_Programa.drop('External institutions', inplace=True, axis=1) 
    conv_Programa.drop('Frameworks', inplace=True, axis=1)

    conv_Programa = conv_Programa.rename(columns={'Relation ID': 'RelationID'})
    conv_Programa = conv_Programa.rename(columns={'Name': 'ConvenioName'})
    new3 = seats_index.merge(conv_Programa, on='RelationID', how='left')
    seats_index = new3
    if country:
        seats_country = seats_index[seats_index['Country'] == country]
    seats_index_LT = seats_index[seats_index['availabiltyIndex'] < 0.4]
    seats_index_GT = seats_index[seats_index['availabiltyIndex'] >= 0.4]
    
    url5 = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/Courses%20(Tue%20May%2024%202022).xlsx"

    facultades = pd.read_excel(url5)
    return seats_country, seats_index_LT, seats_index_GT, facultades
    

@st.experimental_singleton
def loadData():
    url = "https://raw.githubusercontent.com/devergel/MLCovenios/main/Dataset/AcademicMoveWishesOutgoing.xlsx"
    return pd.read_excel(url)
    
    
def asignar_facultad(row):
  temp_row = facultades.loc[facultades['Name'] == row["Stay: Degree programme"]]
  if len(temp_row) != 0:
    val = temp_row["Sub institution"].iat[0]
    return val
  else:
    return np.nan
    
    
def lable_mobility(row):
  if row["Status selection"] == "Selected":
    return 1
  elif row["Status selection"] == "Rejected":
    return 0
  elif row["Status offer"] == "Offer accepted":
    return 1
  elif row["Status offer"] == "Offer rejected":
    return 1
  elif row["Status offer"] == "No offer":
    return 0
  elif row["Stay: Status"] == "Not accepted":
    return 0
  elif row["Stay: Status"] == "Completed":
    return 1
  elif row["Stay: Status"] == "Planned":
    return 1
  elif row["Stay: Status"] == "Current":
    return 1
  else:
    return -1


@st.experimental_singleton
def preprocess():
    data = loadData()
    lb_make = LabelEncoder()
    data_clean = data

    #Generar la etiqueta Mobility a partir de las condiciones de la función lable_mobility
    data_clean['Mobility'] = data_clean.apply (lambda row: lable_mobility(row), axis=1)
    data_clean = data_clean[data_clean['Mobility'] >= 0]

    #Limpieza de registros de acuerdo a la logica de Ranks
    data_clean_rank1 = data_clean[data_clean['Rank'] == 1]
    data_clean_rank2 = data_clean[data_clean['Rank'] == 2]
    data_clean_rank3 = data_clean[data_clean['Rank'] == 3]
    rank1_pass = data_clean_rank1[data_clean_rank1['Mobility'] == 1]
    for index, row in rank1_pass.iterrows():
      data_clean_rank2 = data_clean_rank2[data_clean_rank2["Stay: ID"] != row["Stay: ID"]]
      data_clean_rank3 = data_clean_rank3[data_clean_rank3["Stay: ID"] != row["Stay: ID"]]

    rank2_pass = data_clean_rank2[data_clean_rank2['Mobility'] == 1]
    for index, row in rank2_pass.iterrows():
      data_clean_rank3 = data_clean_rank3[data_clean_rank3["Stay: ID"] != row["Stay: ID"]]

    data_clean = pd.concat([data_clean_rank1,data_clean_rank2], ignore_index=True)
    data_clean = pd.concat([data_clean,data_clean_rank3], ignore_index=True)
    data_clean = data_clean.reset_index(drop=True)

    #Filtrat los datos por la columna Frameworks para solo tener los datos de Pregrado
    data_clean = data_clean[data_clean["Frameworks"] == "Exchange Student Undergraduate"]
    #Eliminar las columnas que no se requieren o que dan información demás al modelo
    data_clean.drop(['Frameworks'], inplace=True, axis=1)
    data_clean.drop(['Stay wish: ID'], inplace=True, axis=1)
    data_clean.drop(['Person: ID'], inplace=True, axis=1)
    data_clean.drop(['Stay: ID'], inplace=True, axis=1)
    data_clean.drop(["Form"], inplace=True, axis=1)
    data_clean.drop(["Level"], inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 1"], inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 2"], inplace=True, axis=1)
    data_clean.drop(["Stay: Calificación total 3"], inplace=True, axis=1)
    data_clean.drop(["Stay: Examen 1"], inplace=True, axis=1)
    data_clean.drop(["Stay: Examen 2"], inplace=True, axis=1)
    data_clean.drop(["Stay: Examen 3"], inplace=True, axis=1)
    data_clean.drop(["Status selection"], inplace=True, axis=1)
    data_clean.drop(["Status offer"], inplace=True, axis=1)
    data_clean.drop(["Stay: Status"], inplace=True, axis=1)
    data_clean.drop(["Academic year"], inplace=True, axis=1)
    data_clean.drop(["Start period"], inplace=True, axis=1)
    data_clean.drop(["Status nomination"], inplace=True, axis=1)
    data_clean.drop(["Stay opportunity"], inplace=True, axis=1)
    #data_clean.drop(["Relation: ID"], inplace=True, axis=1)
    data_clean.drop(['Rank'], inplace=True, axis=1)
    data_clean = data_clean.reset_index(drop=True)

    #Transformación de , a . decimal para el número del promedio
    data_clean["Stay: GPA outgoing"]=data_clean["Stay: GPA outgoing"].str.replace(',','.').astype(float)

    #Asignar la facultad segun el programa
    data_clean['Faculty'] = data_clean.apply (lambda row: asignar_facultad(row), axis=1)
    data_clean["Faculty"] = lb_make.fit_transform(data_clean["Faculty"])

    #Pasar la columna Country de categórica a numérica
    countryDf=pd.DataFrame()
    countryDf["label"] = data_clean["Country"]
    data_clean["Country"] = lb_make.fit_transform(data_clean["Country"])
    countryDf["code"] = data_clean["Country"]
    countryDf=countryDf.drop_duplicates(['code','label'])[['code','label']]

    #Pasar la Institution: Degree programme de categórica a numérica
    instDf=pd.DataFrame()
    instDf["label"] = data_clean["Institution"]
    data_clean["Institution"] = lb_make.fit_transform(data_clean["Institution"])
    instDf["code"] = data_clean["Institution"]
    instDf=instDf.drop_duplicates(['code','label'])[['code','label']]

    #Pasar la columna 'Stay: Degree programme' de categórica a numérica
    programDf=pd.DataFrame()
    programDf["label"] = data_clean["Stay: Degree programme"]
    data_clean["Stay: Degree programme"] = lb_make.fit_transform(data_clean["Stay: Degree programme"])
    programDf["code"] = data_clean["Stay: Degree programme"]
    programDf=programDf.drop_duplicates(['code','label'])[['code','label']]

    #Tratamiento de idioma
    idioma1Df=pd.DataFrame()
    idioma1Df["label"] = data_clean["Stay: Idioma 1"]
    idioma1Df=idioma1Df.drop_duplicates(['label'])[['label']]
    data_clean["Idiomas"] = data_clean["Stay: Idioma 1"].astype(str) +"|"+data_clean["Stay: Idioma 2"].astype(str)
    data_clean["Idiomas"] = data_clean["Idiomas"].astype(str) +"|"+data_clean["Stay: Idioma 3"].astype(str)
    data_clean.drop(["Stay: Idioma 1"], inplace=True, axis=1)
    data_clean.drop(["Stay: Idioma 2"], inplace=True, axis=1)
    data_clean.drop(["Stay: Idioma 3"], inplace=True, axis=1)
    data_clean['Idiomas']=data_clean['Idiomas'].str.split('|')

    label = data_clean['Idiomas']
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(data=mlb.fit_transform(label),columns=mlb.classes_)
    one_hot.drop(["nan"], inplace=True, axis=1)
    one_hot.drop(["Spanish"], inplace=True, axis=1)

    data_clean.reset_index(drop=True, inplace=True)
    one_hot.reset_index(drop=True, inplace=True)
    data_clean = pd.concat([data_clean, one_hot], axis=1)
    data_clean.drop(["Idiomas"], inplace=True, axis=1)

    data_clean = data_clean.fillna(0)
    scaler = StandardScaler()
    datos_scaled = scaler.fit_transform(data_clean)
    return countryDf, idioma1Df, instDf, programDf, data_clean, datos_scaled, train.drop(['Mobility'],axis=1), train['Mobility'], test.drop(['Mobility'],axis=1), test['Mobility']  
 

@st.experimental_singleton
def model(X_train, y_train):
    random = RandomForestClassifier(max_depth=10,random_state=32, criterion='entropy', max_features=0.5, n_estimators=30)
    return random.fit(X_train,y_train)


@st.experimental_singleton
def clusters(data_clean, datos_scaled):
    
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
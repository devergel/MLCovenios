import streamlit as st
def main():
    st.title('Recomendacion de Convenios de Universidades del Exteriror')
    st.sidebar.title('Convenios')
    option = st.sidebar.radio('Opciones:',
                ['Sugerencia de convenios',
                'Segmentacion de Estudiantes'])
    if option=='Sugerencia de convenios':
        name = st.text_input('Sugerencia de convenios')
        if st.button('Recomendar'):
            try:
              st.write('Recomendaciones')
            except:
              st.write('Ocurrio un error al analizar la sugenrencias de convenios')
    if option=='Segmentacion de Estudiantes':
        name = st.text_input('Segmentacion de Estudiantes')
        try:
            st.write('Tipos Estudiantes')
        except:
            st.write('Ocurrio un error al analizar la sugenrencias de convenios')
            
            
if __name__ == '__main__':
    main()
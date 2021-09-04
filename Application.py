import streamlit as st
import pandas as pd
import plotly.express as px

from App_Functions import PCA_Maker

st.set_page_config(layout='wide')
scatter_col, settings_col = st.beta_columns((4, 1))

scatter_col.title('Multi-Dimensional Analysis')
settings_col.title('Settings')

uploaded_file = settings_col.file_uploader('Choose File')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    pca_data, cat_cols, pca_cols = PCA_Maker(data)

    categorical_variable = settings_col.selectbox('Variable Select', options=cat_cols)
    categorical_variable2 = settings_col.selectbox('Second Variable  Select', options=cat_cols)

    pca1 = settings_col.selectbox('First Principle Component', options=pca_cols, index=0)
    pca_cols.remove(pca1)
    pca2 = settings_col.selectbox('Second Principle Component', options=pca_cols)


    scatter_col.plotly_chart(px.scatter(data_frame=pca_data, x=pca1, y=pca2, color=categorical_variable, template='simple_white', 
                             height=800, hover_data=[categorical_variable2]), use_container_width=True)

else:
    scatter_col.header('Please Choose a File')
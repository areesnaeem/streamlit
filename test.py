import streamlit as st
import seaborn as sns
st.header('This file is made on dated 30-01-2023')
st.text('So far so good')


st.header('soon going on fiverr with this service')

df = sns.load_dataset('iris')

st.write(df.head(10))
st.write(df[['species', 'sepal_length']].head())
st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])

import streamlit as st
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# webapp ka title.
st.markdown('''
# **Exploratory data analysis web application**
 This app is developed by areed naeem called **EDA App** 
''')
# how to upload a file from PC
with st.sidebar.header("upload your dataset (.csv)."):
    uploaded_file = st.sidebar.file_uploader('Upload your data', type=['.csv'])
# profilng for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input Dataset**')
    st.write(df)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)
else:
    st.info('awaiting for CSV file, pls upload.')
    if st.button('Pls click to see example data'):
        # example dataset
        @ st.cache
        def load_dataset():
            a = pd.DataFrame(np.random.rand(100, 5),
                             columns=['A', 'B', 'C', 'D', 'E'])
            return a
        df = load_dataset()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input Dataset**')
        st.write(df)
        st.write('---')
        st.header('**Profiling Report**')
        st_profile_report(pr)

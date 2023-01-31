import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# make containers

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title('kashti ki app')
    st.text("In this project, we'll work on kashti dataset")

with data_sets:
    st.header('Kashti doob gai.')
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.bar_chart(df['sex'].value_counts())
    # can make other plots too
    st.subheader('kashti ki classes')
    st.bar_chart(df['class'].value_counts())
    st.bar_chart(df['age'].sample(100))

with features:
    st.header('These are our app features:')
    st.text("In this project, we'll work on kashti dataset")
    st.markdown('**1. Feature 1:** This is feature # 1.')
    st.markdown('**2. Feature 2:** This is feature # 2.')

with model_training:
    st.header('kashti walon k sath kia hwa? (model training)')
    # making columns
    input, display = st.columns(2)
    # here we are going to chnage in column 1
    max_depth = input.slider(
        'how many people: ', min_value=10, max_value=100, value=25, step=5)
# n estimators
n_est = input.selectbox('How many RF tree should be present?', options=[
                        10, 20, 30, 50, 100, 'No limit'])

# adding list of features
input.write(df.columns)


# if want to get input feature from user

input_feature = input.text_input('which feature do you want to work with')


# machine learning model
model = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth)
# here put a condition as no result when no limit selected
if RandomForestRegressor == 'No limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth)


x = df[[input_feature]]
y = df[['fare']]

model.fit(x, y)
pred = model.predict(x)

# display metrices
display.subheader('the value of mean absolute error is: ')
display.write(mean_absolute_error(y, pred))
display.subheader('the value of mean sqaured error is: ')
display.write(mean_squared_error(y, pred))
display.subheader('the value of mean R square is: ')
display.write(r2_score(y, pred))

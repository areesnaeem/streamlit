import streamlit as st
import plotly.express as px
import pandas as pd


# import data set
st.title('plotly and streamly combine for webapp')
df = px.data.gapminder()
st.write(df.head())
st.write(df.columns)

# summary stat
st.write(df.describe())


# data management
year_option = df['year'].unique().tolist()
year = st.selectbox('which year you want?', year_option, 0)
# df = df[df['year'] == year]
# commented out year option and disables as we want to add animation in our plot, we make change in fig variable
# for plotting
fig = px.scatter(df, x='gdpPercap', y='lifeExp', size='pop', color='continent',
                 hover_name='country', log_x=True, size_max=55, range_x=[100, 1300000], range_y=[20, 100], animation_frame='year', animation_group='country')  # added animation
fig.update_layout(width=800, height=400)
st.write(fig)

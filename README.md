import streamlit as st 
import pandas as pd 
fromsklearn.model_selection import train_test_split
fromsklearn.linear_model import linearRegression
df=pd.read_csv("student_score.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,:-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
st.title("exam score prediction model")
st.write("write the number of hours you studied for exam")

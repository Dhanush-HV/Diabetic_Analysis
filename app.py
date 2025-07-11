
import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title('🩸Diabetes Prediction')
st.sidebar.header('User Input Features')

def user_input_parameters():
  AGE=st.sidebar.number_input('Age',1,100)
  GENDER=st.sidebar.selectbox('Gender',('Male','Female'))
  GENDER=1 if GENDER=='Male' else 0
  EXCESSIVE_URINATION=st.sidebar.selectbox('Excessive Urination',('Yes','No'))
  EXCESSIVE_URINATION=1 if EXCESSIVE_URINATION=='Yes' else 0
  EXCESSIVE_THIRST=st.sidebar.selectbox('Excessive Thirst',('Yes','No'))
  EXCESSIVE_THIRST=1 if EXCESSIVE_THIRST=='Yes' else 0
  SUDDEN_WEIGHT_LOSS=st.sidebar.selectbox('Sudden Weight Loss',('Yes','No'))
  SUDDEN_WEIGHT_LOSS=1 if SUDDEN_WEIGHT_LOSS=='Yes' else 0
  WEAKNESS=st.sidebar.selectbox('Weakness',('Yes','No'))
  WEAKNESS=1 if WEAKNESS=='Yes' else 0
  FEELING_EXCESSIVE_HUNGER=st.sidebar.selectbox('Feeling Excessive Hunger',('Yes','No'))
  FEELING_EXCESSIVE_HUNGER=1 if FEELING_EXCESSIVE_HUNGER=='Yes' else 0
  VISUAL_BLURING=st.sidebar.selectbox('Visual Blurring',('Yes','No'))
  VISUAL_BLURING=1 if VISUAL_BLURING=='Yes' else 0
  HAIR_LOSS=st.sidebar.selectbox('Hair Loss',('Yes','No'))
  HAIR_LOSS=1 if HAIR_LOSS=='Yes' else 0
  data={'Age':AGE,'Gender':GENDER,'Excessive Urination':EXCESSIVE_URINATION,
        'Excessive Thirst':EXCESSIVE_THIRST,'sudden weight loss':SUDDEN_WEIGHT_LOSS,
        'weakness':WEAKNESS,'Feeling Excessive Hungary':FEELING_EXCESSIVE_HUNGER,'visual blurring':VISUAL_BLURING,'Hair Loss':HAIR_LOSS}
  feature=pd.DataFrame(data,index=[0])
  return feature

df=user_input_parameters()
st.subheader('User Input Parameters')
st.write(df)

diabities=Data
diabities=Data.dropna()

#Split the Data
X=diabities.drop(['class','Genital thrush','Itching','Irritability','delayed healing','Muscle weakness','muscle stiffness','Obesity'],axis=1) #Independent Variable
Y=diabities['class'] #Dependent Variable

#Match the order
df=df[X.columns]

#Model
Le=LogisticRegression()
Le.fit(X,Y)

#Do the predection on the user data
predection=Le.predict(df)
st.subheader('Prediction of Diabities')
st.write(predection)

#Check the accuracy of the data
from sklearn.metrics import accuracy_score
st.subheader('Accuracy of the Model')
st.write(str(accuracy_score(Y,Le.predict(X))*100)+'%')


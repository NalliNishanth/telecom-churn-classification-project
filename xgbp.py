#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.title('Model Deployment: XGBClassifier')


st.header('Heatmap')

tlc= pd.read_csv("telecommunications_churn.csv")
correlations = tlc.corr()
fig = plt.figure()
ax = fig.add_subplot(111)  
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.sidebar.header('User Input Parameters')





def user_input_features():
    account_length =st.sidebar.number_input("Insert the length")
    voice_mail_plan=st.sidebar.number_input("1 or 0")
    voice_mail_messages=st.sidebar.number_input("Insert number of voice_mail_messages")
    day_mins=st.sidebar.number_input("Insert day_mins")
    evening_mins=st.sidebar.number_input("Insert evening_mins")
    night_mins=st.sidebar.number_input("Insert night_mins")
    international_mins=st.sidebar.number_input("Insert international_mins")
    customer_service_calls=st.sidebar.number_input("Insert customer_service_calls")
    international_plan=st.sidebar.number_input(" Insert international_plan")
    day_calls=st.sidebar.number_input("Insert day_calls")
    day_charge=st.sidebar.number_input("Insert day_charge")
    evening_calls=st.sidebar.number_input("Insert number of evening_calls")
    evening_charge=st.sidebar.number_input("Insert evening_charge")
    night_calls=st.sidebar.number_input("Insert number of night_calls")
    night_charge=st.sidebar.number_input("Insert night_charges")
    international_calls=st.sidebar.number_input("Insert number of international_calls")
    international_charge=st.sidebar.number_input("Insert international_charges")
    total_charge=st.sidebar.number_input("Insert total_charges")
    data={'account_length':account_length,
        'voice_mail_plan':voice_mail_plan,
         'voice_mail_messages':voice_mail_messages,
         'day_mins':day_mins,
         'evening_mins':evening_mins,
         'night_mins':night_mins,
         'international_mins':international_mins,
         'customer_service_calls':customer_service_calls,
         'international_plan':international_plan,
         'day_calls':day_calls,
         'day_charge':day_charge,
         'evening_calls':evening_calls,
         'evening_charge':evening_charge,
         'night_calls':night_calls,
         'night_charge':night_charge,
         'international_calls':international_calls,
         'international_charge':international_charge,
         'total_charge':total_charge}
    features =pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader('User Input parameters')
st.write(df)

tlc= pd.read_csv("telecommunications_churn.csv")

X = tlc.iloc[:,0:18]
Y = tlc['churn']

model=XGBRFClassifier()
model.fit(X,Y)

prediction=model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)


